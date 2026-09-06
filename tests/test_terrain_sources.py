"""Pure-math tests for terrain derivation — no network calls."""
from __future__ import annotations

import numpy as np

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
