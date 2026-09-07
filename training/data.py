"""
Dataset + noise-schedule utilities for downscaler training, kept dependency-
light on purpose (torch/numpy only — no hydra/pytorch_lightning/omegaconf).

This split exists so `AOIPairDataset` can be imported and tested (see
tests/test_training_data.py) with just the base `requirements.txt`
installed — the same install the serving app uses — instead of requiring
`requirements-train.txt`. training/train_downscaler.py imports from here
and adds the actual training-orchestration dependencies on top.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import IterableDataset


class AOIPairDataset(IterableDataset):
    """Yields (coarse_patch, terrain_raster, coarse_tokens, hires_target)
    tuples read from a manifest built by scripts/build_aoi_pairs_manifest.py
    (Open-Meteo-historical-based coarse/fine pairs + real elevation/LULC/LST
    terrain — see that script's docstring for exactly what's real vs. a
    documented proxy)."""

    def __init__(self, manifest_path: str, patch_size: int):
        self.manifest_path = manifest_path
        self.patch_size = patch_size
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
            target = torch.tensor(data["target"], dtype=torch.float32)     # (n_vars, fine_grid, fine_grid)

            # coarse-forecast tokens for cross-attention: one token per
            # coarse grid cell, raw feature width = n_vars (see
            # DiffusionDownscaler.raw_token_dim)
            tokens = coarse.reshape(coarse.shape[0], -1).T  # (coarse_grid*coarse_grid, n_vars)

            yield coarse, terrain, tokens, target


def cosine_noise_schedule(timesteps: int) -> torch.Tensor:
    s = 0.008
    t = torch.linspace(0, timesteps, timesteps + 1)
    f = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    return alphas_cumprod.clamp(min=1e-5, max=1.0)
