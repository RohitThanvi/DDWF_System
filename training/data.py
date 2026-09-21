"""
Dataset utilities for downscaler training, kept dependency-light on
purpose (torch/numpy only — no hydra/pytorch_lightning/omegaconf).

This split exists so `AOIPairDataset` can be imported and tested (see
tests/test_training_data.py) with just the base `requirements.txt`
installed — the same install the serving app uses — instead of requiring
`requirements-train.txt`. training/train_downscaler.py imports from here
and adds the actual training-orchestration dependencies on top.

The noise schedule itself lives in app/models/diffusion_schedule.py (not
here) so both this module and the serving-time sampler
(app/services/downscaler.py) share exactly one definition — see that
module's docstring for why that used to be two different, inconsistent
schedules.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import torch
from torch.utils.data import IterableDataset

from app.data.external_forecast import OPEN_METEO_VARIABLES
from app.data.normalization import VariableNormalizer
from app.data.time_features import append_time_features
from app.models.diffusion_schedule import cosine_noise_schedule  # noqa: F401 -- re-exported for train_downscaler.py


class AOIPairDataset(IterableDataset):
    """Yields (coarse_patch, terrain_raster, coarse_tokens, hires_target)
    tuples read from a manifest built by scripts/build_aoi_pairs_manifest.py
    (Open-Meteo-historical-based coarse/fine pairs + real elevation/LULC/LST
    terrain — see that script's docstring for exactly what's real vs. a
    documented proxy).

    `hires_target` is normalized per-variable (see app/data/normalization.py)
    before being yielded -- the raw target mixes pressure (~1000), wind
    direction (~0-360), and temperature (~20) in one tensor, and Gaussian
    diffusion implicitly assumes the "signal" being noised is roughly
    unit-scale, comparable to the noise it's mixed with. Training on raw,
    wildly-different-scale values produced numerically unstable samples
    that got *worse*, not better, the more training a from-scratch model
    got early on -- serving-side denormalization is the matching half of
    this fix, in app/services/downscaler.py."""

    def __init__(self, manifest_path: str, patch_size: int, stats_path: str = "./data/open_meteo_variable_stats.json"):
        self.manifest_path = manifest_path
        self.patch_size = patch_size
        self.normalizer = VariableNormalizer(stats_path)
        with open(manifest_path) as f:
            self.manifest: list[dict] = json.load(f)
        if not self.manifest:
            raise ValueError(
                f"{manifest_path} has no entries — run scripts/build_aoi_pairs_manifest.py first "
                "(see docs/TRAINING.md 'Downscaler training data')."
            )

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        entries = self.manifest
        if worker is not None:
            entries = entries[worker.id :: worker.num_workers]

        for entry in entries:
            data = np.load(entry["path"])
            coarse = torch.tensor(data["coarse"], dtype=torch.float32)      # (n_vars, coarse_grid, coarse_grid)
            terrain = torch.tensor(data["terrain"], dtype=torch.float32)    # (8, fine_grid, fine_grid)
            target_raw = data["target"]                                    # (n_vars, fine_grid, fine_grid)
            target = torch.tensor(
                self.normalizer.normalize_stack(OPEN_METEO_VARIABLES[: target_raw.shape[0]], target_raw),
                dtype=torch.float32,
            )

            # coarse-forecast tokens for cross-attention: one token per
            # coarse grid cell, raw feature width = n_vars (see
            # DiffusionDownscaler.raw_token_dim). Left un-normalized --
            # it's a conditioning input attended to, not something the
            # diffusion process noises/denoises, so it doesn't carry the
            # same scale requirement `target` does.
            tokens = coarse.reshape(coarse.shape[0], -1).T  # (coarse_grid*coarse_grid, n_vars)

            # Append year/season conditioning so the model can actually use
            # a multi-year manifest's date spread for something -- without
            # this, every pair looks conditionally identical regardless of
            # which year it's from. See app/data/time_features.py's
            # docstring for exactly what this can and can't represent.
            pair_date = date.fromisoformat(entry["date"])
            tokens = torch.tensor(
                append_time_features(tokens.numpy(), pair_date), dtype=torch.float32
            )

            yield coarse, terrain, tokens, target
