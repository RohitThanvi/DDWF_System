"""
Tests for scripts/build_aoi_pairs_manifest.py's world-coverage additions:
stratified sampling across WORLD_STRATA, and the ocean-skip gate that
avoids spending Open-Meteo/MODIS API budget on water-heavy AOIs. Pure
logic only -- no network access needed.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from scripts.build_aoi_pairs_manifest import (
    WORLD_STRATA,
    _is_mostly_ocean,
    sample_world_aoi,
)


def test_world_strata_are_all_valid_bboxes():
    for name, (min_lon, min_lat, max_lon, max_lat), weight in WORLD_STRATA:
        assert -180 <= min_lon < max_lon <= 180, name
        assert -90 <= min_lat < max_lat <= 90, name
        assert weight > 0, name


def test_sample_world_aoi_stays_within_its_stratum():
    rng = random.Random(0)
    strata_by_name = {name: region for name, region, _ in WORLD_STRATA}
    for _ in range(200):
        name, (lo, la, hi, ha) = sample_world_aoi(size_deg=0.5, rng=rng)
        min_lon, min_lat, max_lon, max_lat = strata_by_name[name]
        assert min_lon <= lo and hi <= max_lon, name
        assert min_lat <= la and ha <= max_lat, name


def test_sample_world_aoi_covers_multiple_strata_over_many_draws():
    """A single global bbox sampled naively would never surface distinct
    strata at all -- this is the actual behavior the stratified sampler
    exists to guarantee versus the old single-region approach."""
    rng = random.Random(1)
    seen = {sample_world_aoi(size_deg=0.5, rng=rng)[0] for _ in range(300)}
    assert len(seen) >= len(WORLD_STRATA) // 2  # broad coverage, not every single one required


def test_sample_world_aoi_is_deterministic_given_seed():
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    draws_a = [sample_world_aoi(0.5, rng_a) for _ in range(20)]
    draws_b = [sample_world_aoi(0.5, rng_b) for _ in range(20)]
    assert draws_a == draws_b


class _FakeLulcClient:
    def __init__(self, water_fraction: float):
        self.water_fraction = water_fraction

    def fetch_lulc_patch(self, bbox, target_size):
        from app.data.terrain_sources import LULC_GROUP_ORDER

        fractions = np.zeros((len(LULC_GROUP_ORDER), target_size, target_size), dtype=np.float32)
        water_idx = LULC_GROUP_ORDER.index("water")
        fractions[water_idx] = self.water_fraction
        fractions[0] = 1.0 - self.water_fraction  # vegetation, arbitrary land bucket
        return fractions


@pytest.mark.asyncio
async def test_is_mostly_ocean_true_above_threshold():
    client = _FakeLulcClient(water_fraction=0.95)
    assert await _is_mostly_ocean((0, 0, 1, 1), client, water_fraction_threshold=0.85) is True


@pytest.mark.asyncio
async def test_is_mostly_ocean_false_below_threshold():
    client = _FakeLulcClient(water_fraction=0.2)
    assert await _is_mostly_ocean((0, 0, 1, 1), client, water_fraction_threshold=0.85) is False
