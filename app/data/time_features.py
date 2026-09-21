"""
Temporal conditioning features appended to every coarse-forecast token.

Without this, training on years of historical data doesn't actually give
the downscaler any way to use the extra time span for anything beyond a
bigger, more variable sample of (weather, terrain) -> (hi-res weather)
pairs. The model has no input that tells it *when* a given pair is from,
so a January 2016 sample and a January 2024 sample look conditionally
identical to it except for whatever the weather values themselves happen
to say -- there's no way for it to represent a systematic long-term shift
(e.g. a warming trend in a region's typical temperature_2m) as a function
of time, because time was never one of its inputs. It can only reproduce
whatever's implicitly baked into the coarse forecast (Open-Meteo's own
model output already reflects real recent conditions) and the target
values; it can't learn "this AOI tends to run warmer in later years."

Two scalars fix that:
  - `year_norm`: an inter-annual trend signal. Deliberately linear and
    unbounded rather than one-hot/bucketed by year, so the model can
    extrapolate smoothly a little past the training window's max year
    rather than having an undefined representation for a year it never
    saw a bucket for.
  - `sin(doy)`/`cos(doy)`: cyclical day-of-year encoding, so the model has
    an explicit seasonal-position signal distinct from year (a raw
    day-of-year integer would wrongly imply Dec 31 and Jan 1 are far
    apart; the cyclical encoding puts them next to each other).

This does NOT make DDWF a climate model. It's still a short-horizon
(<=16-day) spatial downscaler conditioned on Open-Meteo's own coarse
forecast for the exact dates requested -- these features let it use
multi-year training data to represent seasonal + long-run-average
conditioning (e.g. "AOIs like this one tend to run warmer in later
years"), not physically model the mechanisms of climate change. Actually
projecting climate change forward (rather than conditioning on a mostly-
historical training window) would need real climate-model inputs (e.g.
CMIP6 scenario output), which is a different, much larger project than
this downscaler.

Shared by training/data.py and app/api/routes/forecast.py so both sides
compute this identically -- the same reason app/models/diffusion_schedule.py
exists as one definition rather than two: a training/serving mismatch here
would silently feed the model a nonsensical time signal at inference
(different scale/meaning than what it was trained on) rather than failing
loudly.
"""
from __future__ import annotations

from datetime import date

import numpy as np

# Roughly the start of the modern high-resolution reanalysis/satellite-LULC
# era this pipeline's other data sources (ESA WorldCover 2021, MODIS) sit
# within -- not physically meaningful beyond "a fixed, arbitrary reference
# point," which is all a linear normalization needs.
BASELINE_YEAR = 2015
# Divisor chosen so ~1-2 decades either side of BASELINE_YEAR keeps
# year_norm within a comfortable, roughly [-1, 2] range for the network --
# not a hard bound (the whole point is the model can still be given a
# year_norm slightly past its training range and extrapolate), just a
# reasonable scale for a fresh model with no other prior on the number.
YEAR_NORM_DIVISOR = 15.0

TIME_FEATURE_DIM = 3  # year_norm, sin(doy), cos(doy)


def time_conditioning_features(d: date) -> np.ndarray:
    """Returns (TIME_FEATURE_DIM,) float32: [year_norm, sin(doy), cos(doy)]."""
    year_norm = (d.year - BASELINE_YEAR) / YEAR_NORM_DIVISOR
    doy = d.timetuple().tm_yday
    angle = 2 * np.pi * doy / 365.25
    return np.array([year_norm, np.sin(angle), np.cos(angle)], dtype=np.float32)


def append_time_features(tokens: np.ndarray, d: date) -> np.ndarray:
    """tokens: (N_tokens, n_vars) raw coarse-forecast tokens for one AOI/day
    -- every token gets the same 3 time scalars appended (the whole AOI
    shares one date), returning (N_tokens, n_vars + TIME_FEATURE_DIM)."""
    feats = time_conditioning_features(d)
    broadcast = np.tile(feats, (tokens.shape[0], 1))
    return np.concatenate([tokens, broadcast], axis=1).astype(np.float32)
