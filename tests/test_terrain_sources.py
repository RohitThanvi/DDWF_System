"""Pure-math tests for terrain derivation — no network calls."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.data.terrain_sources import slope_aspect_from_elevation


def test_flat_elevation_has_zero_slope():
    flat = np.full((10, 10), 500.0, dtype=np.float32)
    slope, aspect = slope_aspect_from_elevation(flat, cell_size_m=100.0)
    assert np.allclose(slope, 0.0, atol=1e-4)


def test_linear_ramp_has_uniform_positive_slope():
    # elevation increases by 10m per grid step along axis 1 (x)
    x = np.arange(10, dtype=np.float32) * 10.0
    ramp = np.tile(x, (10, 1))
    slope, aspect = slope_aspect_from_elevation(ramp, cell_size_m=100.0)
    interior = slope[2:-2, 2:-2]
    assert interior.std() < 1e-3  # uniform slope away from edge effects
    assert interior.mean() > 0


def test_worldcover_tile_id_matches_documented_convention():
    from app.data.terrain_sources import worldcover_tile_id

    # Jaipur, Rajasthan -> falls in the 3x3deg tile with SW corner (24N, 75E)
    assert worldcover_tile_id(26.9, 75.8) == "N24E075"
    # Southern hemisphere: ESA's own docs example is "S48E036" for the tile
    # covering 48S-45S, 36E-39E
    assert worldcover_tile_id(-46.5, 37.2) == "S48E036"
    # Western hemisphere
    assert worldcover_tile_id(51.5, -0.1) == "N51W003"


def test_read_lulc_fractions_from_synthetic_raster():
    """Builds a small in-memory GeoTIFF with known class values and checks
    that read_lulc_fractions recovers the correct group fractions — no
    network access needed."""
    import rasterio
    from rasterio.transform import from_bounds as transform_from_bounds

    from app.data.terrain_sources import read_lulc_fractions

    # 4x4 raster: left half built-up (50), right half water (80)
    raw = np.array([
        [50, 50, 80, 80],
        [50, 50, 80, 80],
        [50, 50, 80, 80],
        [50, 50, 80, 80],
    ], dtype=np.uint8)

    bbox = (10.0, 10.0, 10.04, 10.04)  # small AOI matching the raster's extent
    transform = transform_from_bounds(*bbox, raw.shape[1], raw.shape[0])

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=raw.shape[0], width=raw.shape[1], count=1,
            dtype=raw.dtype, crs="EPSG:4326", transform=transform,
        ) as dataset:
            dataset.write(raw, 1)

        with memfile.open() as dataset:
            fractions = read_lulc_fractions(dataset, bbox, target_size=8)

    assert fractions.shape == (4, 8, 8)
    veg, built_up, water, bare = fractions
    # left half of the AOI should read as mostly built-up, right half mostly water
    assert built_up[:, :3].mean() > 0.7
    assert water[:, 5:].mean() > 0.7
    assert veg.mean() < 0.1 and bare.mean() < 0.1


@pytest.mark.anyio
async def test_lst_client_parses_and_scales_response(monkeypatch):
    """Mocks the two ORNL DAAC HTTP calls (dates, subset) and checks the
    scale factor (raw DN * 0.02 -> Kelvin -> Celsius) and fill-value
    handling are correct -- no network access needed."""
    from app.data.terrain_sources import LSTClient

    # 2x2 raw grid: three valid pixels at ~300K, one fill (0) pixel
    raw_data = [15000, 15000, 15000, 0]  # 15000 * 0.02 = 300.0K = 26.85C

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None):
            if url.endswith("/dates"):
                return FakeResponse({"dates": [{"modis_date": "A2024001", "calendar_date": "2024-01-01"}]})
            return FakeResponse({
                "nrows": 2, "ncols": 2,
                "subset": [{"data": raw_data, "modis_date": "A2024001"}],
            })

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    client = LSTClient()
    result = await client.fetch_lst_patch((75.0, 26.0, 75.1, 26.1), target_size=4)

    assert result.shape == (1, 4, 4)
    # valid pixels should read ~26.85C; the fill pixel gets replaced with
    # the patch mean rather than propagating as a discontinuity
    assert 20.0 < result.mean() < 30.0
