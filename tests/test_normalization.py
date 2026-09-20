"""Tests for VariableNormalizer -- now load-bearing (training normalizes
the diffusion target, serving denormalizes the sampled output), where it
used to be dead code nothing imported."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.data.normalization import VariableNormalizer


@pytest.fixture
def stats_file(tmp_path):
    stats = {
        "temperature_2m": {"mean": 15.0, "std": 12.0},
        "surface_pressure": {"mean": 1000.0, "std": 40.0},
    }
    path = tmp_path / "stats.json"
    with open(path, "w") as f:
        json.dump(stats, f)
    return str(path)


def test_normalize_denormalize_round_trip(stats_file):
    norm = VariableNormalizer(stats_file)
    raw = np.array([15.0, 27.0, 3.0])  # temperature_2m-ish values
    normalized = norm.normalize("temperature_2m", raw)
    recovered = norm.denormalize("temperature_2m", normalized)
    assert np.allclose(recovered, raw, atol=1e-4)


def test_normalize_centers_and_scales(stats_file):
    norm = VariableNormalizer(stats_file)
    at_mean = norm.normalize("temperature_2m", np.array([15.0]))
    assert abs(at_mean[0]) < 1e-6  # value at the mean normalizes to ~0

    one_std_above = norm.normalize("temperature_2m", np.array([27.0]))  # mean + std
    assert abs(one_std_above[0] - 1.0) < 1e-4


def test_real_data_lands_at_roughly_unit_scale(stats_file):
    """The actual point of normalization: raw values with wildly
    different native scales (pressure ~1000, temperature ~20) should both
    land in a similar, roughly-unit-scale range after normalization --
    this is what makes them safe to mix in one diffusion tensor."""
    norm = VariableNormalizer(stats_file)
    temp_normalized = norm.normalize("temperature_2m", np.array([22.0]))
    pressure_normalized = norm.normalize("surface_pressure", np.array([995.0]))
    assert abs(temp_normalized[0]) < 3
    assert abs(pressure_normalized[0]) < 3


def test_normalize_stack_round_trip(stats_file):
    norm = VariableNormalizer(stats_file)
    var_names = ["temperature_2m", "surface_pressure"]
    raw = np.stack([
        np.full((4, 4), 22.0, dtype=np.float32),
        np.full((4, 4), 995.0, dtype=np.float32),
    ])

    normalized = norm.normalize_stack(var_names, raw)
    assert normalized.shape == raw.shape
    # both channels should now be roughly comparable scale despite very
    # different raw magnitudes (22 vs 995)
    assert abs(normalized[0].mean()) < 3
    assert abs(normalized[1].mean()) < 3

    recovered = norm.denormalize_stack(var_names, normalized)
    assert np.allclose(recovered, raw, atol=1e-3)


def test_shipped_stats_file_covers_all_open_meteo_variables():
    from app.data.external_forecast import OPEN_METEO_VARIABLES

    norm = VariableNormalizer("./data/open_meteo_variable_stats.json")
    for var in OPEN_METEO_VARIABLES:
        assert var in norm.stats, f"{var} missing from data/open_meteo_variable_stats.json"
        assert norm.stats[var]["std"] > 0
