"""
Smoke tests for the API surface. Run with: pytest -q
These monkeypatch the coarse-forecast source and heavy model services so
CI doesn't need network access, real checkpoints, or a GPU to validate
routing, schemas, and auth.
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
    from app.data.external_forecast import CoarseForecastService
    from app.services.downscaler import DownscalerService
    from app.services.terrain_fusion import TerrainFusionService

    async def _fake_coarse_patch(self, bbox, grid_size=8, forecast_days=16):
        n_hours = forecast_days * 24
        return {
            "hourly_time": [f"2026-09-0{d+1}T00:00" for d in range(min(forecast_days, 9))],
            "data": np.zeros((n_hours, 8, grid_size, grid_size), dtype=np.float32),
            "variables": ["temperature_2m"] * 8,
        }

    monkeypatch.setattr(CoarseForecastService, "get_coarse_patch", _fake_coarse_patch)

    async def _fake_raster_patch(self, bbox, resolution_m=100, target_size=256):
        return np.zeros((8, 16, 16), dtype=np.float32)

    monkeypatch.setattr(TerrainFusionService, "fetch_raster_patch", _fake_raster_patch)
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
    body = resp.json()
    assert body["status"] == "ok"
    assert body["coarse_forecast_provider"] == "open-meteo"


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


def test_forecast_horizon_capped_at_16_days(client):
    resp = client.post(
        "/v1/forecast",
        headers={"X-API-Key": "test-key"},
        json={"lat": 26.9, "lon": 75.8, "horizon_days": 30},
    )
    assert resp.status_code == 422  # Pydantic rejects horizon_days > 16


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
