"""
Tests for app/data/era5_zarr.py. No network access -- the module docstring
explains why the live Google Cloud Storage store itself isn't reachable
from this repo's sandbox/CI; these tests cover the variable-derivation
math and grid-sampling logic directly instead, using small synthetic
in-memory xarray Datasets standing in for the real (2.05 PB) store.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from app.data.era5_zarr import (
    RAW_FIELDS,
    relative_humidity_from_dewpoint,
    sample_grid_from_dataset,
    wind_speed_direction,
)


# --- Variable-derivation math (the part most worth getting right, and
# the part this test suite can actually verify without live data) ---

def test_relative_humidity_is_100_when_saturated():
    """Dewpoint == air temperature means the air is saturated -> RH=100%."""
    t = np.array([20.0, -5.0, 35.0])
    rh = relative_humidity_from_dewpoint(t, t)
    np.testing.assert_allclose(rh, 100.0, atol=0.01)


def test_relative_humidity_decreases_as_dewpoint_depression_increases():
    t = np.array([20.0, 20.0, 20.0])
    td_small_gap = np.array([18.0, 18.0, 18.0])
    td_large_gap = np.array([5.0, 5.0, 5.0])
    rh_small_gap = relative_humidity_from_dewpoint(t, td_small_gap)
    rh_large_gap = relative_humidity_from_dewpoint(t, td_large_gap)
    assert (rh_small_gap > rh_large_gap).all()


def test_wind_direction_from_north():
    """Southward flow (wind blowing FROM the north) -> 0 degrees."""
    speed, direction = wind_speed_direction(u=np.array([0.0]), v=np.array([-5.0]))
    assert direction[0] == pytest.approx(0.0, abs=1e-6)
    assert speed[0] == pytest.approx(5.0 * 3.6, abs=1e-6)


def test_wind_direction_from_east():
    """Westward flow (wind blowing FROM the east) -> 90 degrees."""
    _, direction = wind_speed_direction(u=np.array([-5.0]), v=np.array([0.0]))
    assert direction[0] == pytest.approx(90.0, abs=1e-6)


def test_wind_direction_from_south():
    """Northward flow (wind blowing FROM the south) -> 180 degrees."""
    _, direction = wind_speed_direction(u=np.array([0.0]), v=np.array([5.0]))
    assert direction[0] == pytest.approx(180.0, abs=1e-6)


def test_wind_direction_from_west():
    """Eastward flow (wind blowing FROM the west) -> 270 degrees."""
    _, direction = wind_speed_direction(u=np.array([5.0]), v=np.array([0.0]))
    assert direction[0] == pytest.approx(270.0, abs=1e-6)


def test_wind_speed_converts_ms_to_kmh():
    speed, _ = wind_speed_direction(u=np.array([3.0]), v=np.array([4.0]))  # 3-4-5 triangle -> 5 m/s
    assert speed[0] == pytest.approx(5.0 * 3.6, abs=1e-6)


# --- Grid sampling against a small synthetic Dataset ---

def _synthetic_era5_dataset() -> "xr.Dataset":
    """A tiny (1 time, 4x4 lat/lon) Dataset with known, hand-picked values
    for every RAW_FIELDS variable, standing in for the real ARCO-ERA5
    store. Longitude uses ERA5's native 0..360 convention on purpose, to
    exercise sample_grid_from_dataset's -180..180 -> 0..360 conversion."""
    lats = np.array([-10.0, -5.0, 0.0, 5.0])
    lons = np.array([70.0, 75.0, 80.0, 85.0])  # matches a bbox given in -180..180 with no wraparound needed
    times = np.array([np.datetime64("2020-06-15T12:00:00")])

    data = {}
    for field in RAW_FIELDS:
        # Distinct, deterministic per-field constant so a mixed-up field
        # mapping would fail a test rather than silently averaging out.
        base = {
            "2m_temperature": 300.0,             # K -> 26.85 C
            "2m_dewpoint_temperature": 290.0,    # K -> 16.85 C
            "surface_pressure": 101325.0,        # Pa -> 1013.25 hPa
            "total_precipitation": 0.002,        # m -> 2.0 mm
            "10m_u_component_of_wind": 0.0,
            "10m_v_component_of_wind": -10.0,    # southward flow -> wind FROM north, 36 km/h
            "total_cloud_cover": 0.75,           # fraction -> 75%
            "surface_solar_radiation_downwards": 1_800_000.0,  # J/m^2 -> 500 W/m^2
        }[field]
        data[field] = (("time", "latitude", "longitude"), np.full((1, 4, 4), base, dtype=np.float32))

    return xr.Dataset(data, coords={"time": times, "latitude": lats, "longitude": lons})


def test_sample_grid_shape_and_derived_values():
    ds = _synthetic_era5_dataset()
    bbox = (70.0, -10.0, 85.0, 5.0)  # matches the synthetic dataset's lon/lat span exactly
    when = datetime(2020, 6, 15, 12, tzinfo=timezone.utc)

    out = sample_grid_from_dataset(ds, bbox, grid_size=4, when=when)

    assert set(out.keys()) == {
        "temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m", "cloud_cover", "shortwave_radiation",
    }
    for arr in out.values():
        assert arr.shape == (4, 4)

    np.testing.assert_allclose(out["temperature_2m"], 26.85, atol=0.01)
    np.testing.assert_allclose(out["surface_pressure"], 1013.25, atol=0.01)
    np.testing.assert_allclose(out["precipitation"], 2.0, atol=0.01)
    np.testing.assert_allclose(out["cloud_cover"], 75.0, atol=0.01)
    np.testing.assert_allclose(out["shortwave_radiation"], 500.0, atol=0.01)
    np.testing.assert_allclose(out["wind_speed_10m"], 36.0, atol=0.01)   # 10 m/s * 3.6
    np.testing.assert_allclose(out["wind_direction_10m"], 0.0, atol=0.01)  # southward flow -> from the north


def test_sample_grid_handles_negative_longitude_bbox():
    """A bbox given as e.g. -122..-100 (typical -180..180 convention) must
    still land on the right cells in ERA5's 0..360 longitude coordinate --
    this is the exact conversion most likely to silently go wrong."""
    lats = np.array([30.0, 35.0])
    lons_0_360 = np.array([238.0, 245.0])  # -122, -115 in 0..360
    times = np.array([np.datetime64("2020-06-15T12:00:00")])
    # Realistic placeholder Kelvin/Pa/etc. values for every field except
    # 2m_temperature (made spatially distinct below) -- an unrealistic
    # placeholder like 1.0 Kelvin would still produce a shape-correct
    # result but blow up the Magnus-formula math with nonphysical inputs,
    # which isn't what this test is checking.
    placeholder = {
        "2m_temperature": 290.0, "2m_dewpoint_temperature": 285.0, "surface_pressure": 101325.0,
        "total_precipitation": 0.001, "10m_u_component_of_wind": 1.0, "10m_v_component_of_wind": 1.0,
        "total_cloud_cover": 0.5, "surface_solar_radiation_downwards": 500_000.0,
    }
    data = {
        field: (("time", "latitude", "longitude"), np.full((1, 2, 2), placeholder[field], dtype=np.float32))
        for field in RAW_FIELDS
    }
    # Make one field spatially distinct so we can confirm the right cell was picked
    data["2m_temperature"] = (
        ("time", "latitude", "longitude"),
        np.array([[[280.0, 290.0], [300.0, 310.0]]], dtype=np.float32),
    )
    ds = xr.Dataset(data, coords={"time": times, "latitude": lats, "longitude": lons_0_360})

    out = sample_grid_from_dataset(ds, bbox=(-122.0, 30.0, -115.0, 35.0), grid_size=2,
                                    when=datetime(2020, 6, 15, 12, tzinfo=timezone.utc))
    # (row=lat ascending, col=lon ascending) -> same layout as the source array above
    np.testing.assert_allclose(out["temperature_2m"], np.array([[280.0, 290.0], [300.0, 310.0]]) - 273.15, atol=0.01)
