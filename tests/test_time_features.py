"""Unit tests for app/data/time_features.py -- the shared year/season
conditioning appended to coarse-forecast tokens at both training and
serving time (see that module's docstring for why train/serve must agree
here)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from app.data.time_features import (
    BASELINE_YEAR,
    TIME_FEATURE_DIM,
    append_time_features,
    time_conditioning_features,
)


def test_year_norm_is_zero_at_baseline_year():
    feats = time_conditioning_features(date(BASELINE_YEAR, 1, 1))
    assert feats[0] == pytest.approx(0.0, abs=0.01)


def test_year_norm_increases_with_year():
    early = time_conditioning_features(date(2016, 6, 15))
    late = time_conditioning_features(date(2024, 6, 15))
    assert late[0] > early[0]


def test_day_of_year_cyclical_encoding_wraps_around():
    """Dec 31 and Jan 1 should be close in (sin, cos) space -- a raw
    day-of-year integer would wrongly put them ~365 apart."""
    dec31 = time_conditioning_features(date(2023, 12, 31))[1:]
    jan1 = time_conditioning_features(date(2024, 1, 1))[1:]
    dist = np.linalg.norm(dec31 - jan1)
    assert dist < 0.1


def test_day_of_year_cyclical_encoding_distinguishes_opposite_seasons():
    winter = time_conditioning_features(date(2024, 1, 1))[1:]
    summer = time_conditioning_features(date(2024, 7, 2))[1:]  # ~half a year later
    dist = np.linalg.norm(winter - summer)
    assert dist > 1.5  # opposite points on the unit circle -> distance ~2


def test_append_time_features_shape_and_broadcast():
    tokens = np.random.randn(16, 8).astype(np.float32)
    out = append_time_features(tokens, date(2024, 6, 15))
    assert out.shape == (16, 8 + TIME_FEATURE_DIM)
    np.testing.assert_array_equal(out[:, :8], tokens)
    # same date -> every token gets the identical 3 appended scalars
    for row in out[1:, 8:]:
        np.testing.assert_array_equal(row, out[0, 8:])
