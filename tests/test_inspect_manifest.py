"""
Regression test for scripts/inspect_manifest.py's terrain-specific
degenerate check. Built against a real bug: a manifest pair with all-zero
elevation/LULC/LST (a real ~15% failure rate seen in one --world run, see
scripts/build_aoi_pairs_manifest.py's _is_mostly_ocean docstring) used to
slip past the old combined 24-channel check entirely, because 16 of 24
channels (the coarse+target weather data) were legitimately non-zero and
diluted the 8 all-zero terrain channels well under the flag threshold.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "inspect_manifest", Path(__file__).resolve().parent.parent / "scripts" / "inspect_manifest.py"
)
inspect_manifest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inspect_manifest)


def _write_pair(path: Path, terrain_all_zero: bool) -> None:
    coarse = np.random.uniform(10, 20, size=(8, 4, 4)).astype(np.float32)  # realistic non-zero weather
    target = np.random.uniform(10, 20, size=(8, 8, 8)).astype(np.float32)
    if terrain_all_zero:
        terrain = np.zeros((8, 8, 8), dtype=np.float32)
        terrain[2] = 180.0  # aspect's real degenerate-flat value (atan2(0,-0)==pi), not 0 -- see the
                             # module docstring in build_aoi_pairs_manifest.py's post-fetch check for why
    else:
        terrain = np.random.uniform(0, 100, size=(8, 8, 8)).astype(np.float32)
    np.savez_compressed(path, coarse=coarse, terrain=terrain, target=target)


@pytest.fixture
def manifest_with_one_degenerate_pair(tmp_path):
    good_path = tmp_path / "pair_good.npz"
    bad_path = tmp_path / "pair_ocean.npz"
    _write_pair(good_path, terrain_all_zero=False)
    _write_pair(bad_path, terrain_all_zero=True)

    manifest = [
        {"path": str(good_path), "bbox": [0, 0, 1, 1], "date": "2024-01-01"},
        {"path": str(bad_path), "bbox": [35.5, 24.8, 36.0, 25.3], "date": "2017-07-03"},
    ]
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def test_terrain_only_degenerate_pair_is_flagged(manifest_with_one_degenerate_pair, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["inspect_manifest.py", "--manifest", str(manifest_with_one_degenerate_pair)])
    buf = io.StringIO()
    with redirect_stdout(buf):
        inspect_manifest.main()
    output = buf.getvalue()

    assert "TERRAIN channels" in output
    assert "pair_ocean.npz" in output
    assert "1 likely-degenerate pair(s)" in output
    # the good pair must NOT trip the same warning
    good_section = output.split("pair_ocean.npz")[0]
    assert "TERRAIN channels" not in good_section
