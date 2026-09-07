"""Tests for the Open-Meteo coarse-forecast client, in particular the
request-chunking fix for the 414 Request-URI Too Large bug hit when a
grid_size produces more points than fit comfortably in one URL."""
from __future__ import annotations

import numpy as np
import pytest

from app.data.external_forecast import MAX_COORDS_PER_REQUEST, OpenMeteoClient, _grid_points


def test_grid_points_count():
    bbox = (75.0, 26.0, 76.0, 27.0)
    points = _grid_points(bbox, grid_size=4)
    assert len(points) == 16


@pytest.mark.anyio
async def test_fetch_coarse_patch_batches_large_grids(monkeypatch):
    """A grid_size of 32 (1024 points) must not be sent as a single
    1024-coordinate URL -- this is exactly the request shape that produced
    a real 414 from Open-Meteo's server."""
    call_param_sizes = []

    class FakeResponse:
        def __init__(self, n_points):
            self.n_points = n_points
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            # one entry per requested point in this chunk, each with a
            # 2-hour hourly series so the reshape logic has something to do
            return [
                {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                            "temperature_2m": [20.0, 21.0]}}
                for _ in range(self.n_points)
            ]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None):
            n_points = len(params["latitude"].split(","))
            call_param_sizes.append(n_points)
            return FakeResponse(n_points)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    client = OpenMeteoClient()
    result = await client.fetch_coarse_patch(
        (75.0, 26.0, 76.0, 27.0), grid_size=32, forecast_days=1,
        variables=["temperature_2m"],
    )

    # 32*32 = 1024 points must be split across multiple requests, none
    # exceeding MAX_COORDS_PER_REQUEST
    assert len(call_param_sizes) > 1
    assert all(n <= MAX_COORDS_PER_REQUEST for n in call_param_sizes)
    assert sum(call_param_sizes) == 1024

    assert result["data"].shape == (2, 1, 32, 32)  # (n_hours, n_vars, grid, grid)


@pytest.mark.anyio
async def test_fetch_coarse_patch_small_grid_single_request(monkeypatch):
    """A small grid (well under the chunk size) should still work in
    exactly one request -- no behavior change for the common case."""
    call_count = 0

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"hourly": {"time": ["2024-01-01T00:00"], "temperature_2m": [20.0]}}
                for _ in range(4)
            ]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None):
            nonlocal call_count
            call_count += 1
            return FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    client = OpenMeteoClient()
    result = await client.fetch_coarse_patch(
        (75.0, 26.0, 75.1, 26.1), grid_size=2, forecast_days=1, variables=["temperature_2m"],
    )
    assert call_count == 1
    assert result["data"].shape == (1, 1, 2, 2)
