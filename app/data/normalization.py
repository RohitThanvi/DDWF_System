"""
Per-variable normalization, applied identically at train and inference
time. Default stats ship in data/open_meteo_variable_stats.json --
approximate global priors, not computed climatology (see that file's
comment) -- good enough to get the 8 Open-Meteo variables (raw pressure
~1000, raw wind direction 0-360, raw temperature ~20, etc.) onto a
roughly comparable scale before they're diffused together in one tensor,
which is what the diffusion math actually requires. Swap in real computed
climatology (via scripts/build_zarr_store.py-style stats) once there's an
archive to compute it from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VariableNormalizer:
    def __init__(self, stats_path: str | Path):
        with open(stats_path) as f:
            self.stats: dict[str, dict[str, float]] = json.load(f)

    def normalize(self, var_name: str, arr: np.ndarray) -> np.ndarray:
        s = self.stats[var_name]
        return (arr - s["mean"]) / (s["std"] + 1e-8)

    def denormalize(self, var_name: str, arr: np.ndarray) -> np.ndarray:
        s = self.stats[var_name]
        return arr * (s["std"] + 1e-8) + s["mean"]

    def normalize_stack(self, var_names: list[str], arr: np.ndarray) -> np.ndarray:
        """arr: (len(var_names), ...) -- normalizes each channel by its own
        variable's stats. Silently passes through any channel whose name
        isn't in `self.stats` (logged nowhere on purpose -- callers should
        make sure var_names matches what the stats file actually covers;
        this is a convenience for the common "already know it matches"
        case, not a validation layer)."""
        out = arr.copy().astype(np.float32)
        for i, name in enumerate(var_names):
            if name in self.stats:
                out[i] = self.normalize(name, arr[i])
        return out

    def denormalize_stack(self, var_names: list[str], arr: np.ndarray) -> np.ndarray:
        out = arr.copy().astype(np.float32)
        for i, name in enumerate(var_names):
            if name in self.stats:
                out[i] = self.denormalize(name, arr[i])
        return out
