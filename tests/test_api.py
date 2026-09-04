"""
Smoke tests for the API surface. Run with: pytest -q
These use FastAPI's TestClient and monkeypatch the heavy model services so
CI doesn't need real checkpoints/GPUs to validate routing, schemas, and auth.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DDWF_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch):
    from app.services.downscaler import DownscalerService
    from app.services.global_engine import GlobalEngineService
    from app.services.terrain_fusion import TerrainFusionService

    monkeypatch.setattr(
        GlobalEngineService, "get_trajectory",
        lambda self, cond, force_refresh=False: np.zeros((120, 86, 4, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        TerrainFusionService, "fetch_raster_patch",
        lambda self, bbox, resolution_m=100: np.zeros((8, 16, 16), dtype=np.float32),
    )
    monkeypatch.setattr(
        DownscalerService, "downscale",
        lambda self, coarse_patch, terrain_raster, coarse_tokens, n_channels_out=8, n_steps=None:
            np.zeros((n_channels_out, 16, 16), dtype=np.float32),
    )

    from app.main import create_app

    return TestClient(create_app())


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_forecast_requires_api_key(client):
    resp = client.post("/v1/forecast", json={"lat": 26.9, "lon": 75.8, "horizon_days": 5})
    assert resp.status_code == 401


def test_forecast_happy_path(client):
    resp = client.post(
        "/v1/forecast",
        headers={"X-API-Key": "test-key"},
        json={"lat": 26.9, "lon": 75.8, "horizon_days": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["timeseries"]
    assert body["timeseries"][0]["confidence"] in {"high_skill", "moderate_skill", "low_skill"}


def test_vayu_weather_layer(client):
    resp = client.post(
        "/v1/vayu/weather-layer",
        headers={"X-API-Key": "test-key"},
        json={"lat": 26.9, "lon": 75.8, "horizon_days": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["points"]) > 0
    assert "temperature_c" in body["points"][0]
