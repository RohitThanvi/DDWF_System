"""
Per-variable z-score normalization using 1991-2020 climatology, applied
identically at train and inference time. Stats are computed offline by
scripts/build_zarr_store.py and shipped as a small JSON alongside the
model checkpoint — never recomputed on the fly.
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
