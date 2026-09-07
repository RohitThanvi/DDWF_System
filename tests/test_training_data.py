"""Tests AOIPairDataset against a small real (synthetic-values, real-shape)
manifest + npz files — proves the training data path actually reads what
scripts/build_aoi_pairs_manifest.py writes, without needing network access."""
from __future__ import annotations

import json

import numpy as np
import pytest

from training.data import AOIPairDataset


@pytest.fixture
def small_manifest(tmp_path):
    coarse_grid, fine_grid, n_vars = 4, 8, 8
    manifest = []
    for i in range(3):
        path = tmp_path / f"pair_{i}.npz"
        np.savez_compressed(
            path,
            coarse=np.random.randn(n_vars, coarse_grid, coarse_grid).astype(np.float32),
            terrain=np.random.randn(8, fine_grid, fine_grid).astype(np.float32),
            target=np.random.randn(n_vars, fine_grid, fine_grid).astype(np.float32),
        )
        manifest.append({"path": str(path), "bbox": [0, 0, 1, 1], "date": "2024-01-01",
                          "coarse_grid": coarse_grid, "fine_grid": fine_grid})

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return str(manifest_path)


def test_aoi_pair_dataset_yields_correct_shapes(small_manifest):
    ds = AOIPairDataset(manifest_path=small_manifest, patch_size=8)
    items = list(ds)
    assert len(items) == 3

    coarse, terrain, tokens, target = items[0]
    assert coarse.shape == (8, 4, 4)
    assert terrain.shape == (8, 8, 8)
    assert tokens.shape == (16, 8)   # coarse_grid*coarse_grid tokens, n_vars features each
    assert target.shape == (8, 8, 8)


def test_aoi_pair_dataset_rejects_empty_manifest(tmp_path):
    manifest_path = tmp_path / "empty.json"
    with open(manifest_path, "w") as f:
        json.dump([], f)

    with pytest.raises(ValueError):
        AOIPairDataset(manifest_path=str(manifest_path), patch_size=8)
