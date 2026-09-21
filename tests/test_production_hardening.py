"""
Regression tests for the production-readiness fixes: without these, a
future edit could silently reintroduce any of them (same failure mode as
the bugs documented in docs/ARCHITECTURE.md's "Serving-time correctness
fixes" section -- caught live, not by inspection).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings


def test_production_env_rejects_insecure_default_key(monkeypatch):
    """Settings must refuse to construct with ENV=production and the
    placeholder DDWF_API_KEY still in place -- that placeholder is
    visible in this repo's git history, so it can never be a real
    production secret."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("DDWF_API_KEY", "change-me-to-a-long-random-secret")
    with pytest.raises(ValueError, match="insecure default"):
        Settings()


def test_production_env_rejects_short_key(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("DDWF_API_KEY", "short")
    with pytest.raises(ValueError, match="shorter than 32 chars"):
        Settings()


def test_production_env_accepts_real_key(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("DDWF_API_KEY", "a" * 40)
    Settings()  # should not raise


def test_development_env_allows_default_key(monkeypatch):
    """Local dev / docker-compose without a .env shouldn't be blocked --
    only ENV=production enforces this."""
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.setenv("DDWF_API_KEY", "change-me-to-a-long-random-secret")
    Settings()  # should not raise


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DDWF_API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    from app.main import create_app

    return TestClient(create_app())


def test_rate_limit_blocks_after_threshold(client):
    """RATE_LIMIT_PER_MINUTE=5 in this fixture -- the 6th request from the
    same client within the window must be rejected with 429, and earlier
    ones must succeed (as far as the limiter is concerned; a 401 is fine
    here since no API key is sent -- the point is it's not a 429)."""
    statuses = [client.get("/v1/forecast/does-not-exist").status_code for _ in range(5)]
    assert all(s != 429 for s in statuses)
    assert client.get("/v1/forecast/does-not-exist").status_code == 429


def test_rate_limit_does_not_apply_to_health(client):
    """/health must stay responsive regardless of rate-limit state --
    orchestrators poll it continuously and it must never look unhealthy
    because of caller traffic."""
    for _ in range(20):
        assert client.get("/health").status_code == 200


def test_metrics_requires_api_key(client):
    resp = client.get("/metrics")
    assert resp.status_code == 401

    resp = client.get("/metrics", headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200


def test_unhandled_exception_returns_json_not_plaintext(monkeypatch):
    """A bug anywhere in the forecast flow must surface as a stable JSON
    error body with a request-correlatable header, not FastAPI's default
    plain-text 500."""
    from app.data.external_forecast import CoarseForecastService
    from app.main import create_app

    async def _boom(self, bbox, grid_size=8, forecast_days=16):
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(CoarseForecastService, "get_coarse_patch", _boom)

    # raise_server_exceptions=False: TestClient normally re-raises the
    # handler's exception for easier debugging in tests, which bypasses
    # the very global exception handler this test exists to check runs
    # and returns a real HTTP response to the caller in production.
    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.post(
        "/v1/forecast",
        headers={"X-API-Key": "test-key"},
        json={"lat": 26.9, "lon": 75.8, "horizon_days": 1},
    )
    assert resp.status_code == 500
    body = resp.json()
    assert "detail" in body
    assert "X-Request-ID" in resp.headers


def test_coarse_forecast_cache_evicts_expired_entries():
    """Regression for the unbounded process-local cache: entries older
    than the TTL must be dropped rather than accumulating forever."""
    from app.data.external_forecast import CoarseForecastService

    svc = CoarseForecastService.__new__(CoarseForecastService)  # bypass __init__'s client setup
    svc._cache = {}
    svc._CACHE_TTL_S = CoarseForecastService._CACHE_TTL_S
    svc._CACHE_MAX_ENTRIES = CoarseForecastService._CACHE_MAX_ENTRIES

    stale_key = "stale"
    fresh_key = "fresh"
    svc._cache[stale_key] = (time.monotonic() - svc._CACHE_TTL_S - 10, {"data": np.zeros(1)})
    svc._cache[fresh_key] = (time.monotonic(), {"data": np.zeros(1)})

    CoarseForecastService._evict_expired(svc)

    assert stale_key not in svc._cache
    assert fresh_key in svc._cache
