#!/usr/bin/env python3
"""
Sanity-check a manifest built by build_aoi_pairs_manifest.py: loads every
pair, checks shapes are consistent, and prints per-channel stats so you can
eyeball whether the data looks real (varying values, plausible ranges) or
degenerate (all zeros, all identical, NaNs) before spending time training
on it.

Usage:
  python scripts/inspect_manifest.py --manifest ./data/aoi_pairs_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TERRAIN_CHANNEL_NAMES = [
    "elevation", "slope", "aspect",
    "lulc_vegetation", "lulc_built_up", "lulc_water", "lulc_bare_or_snow",
    "lst_celsius",
]


def _stats(name: str, arr: np.ndarray) -> str:
    n_nan = int(np.isnan(arr).sum())
    n_zero = int((arr == 0).sum())
    pct_zero = 100 * n_zero / arr.size
    flag = ""
    if n_nan > 0:
        flag += " ⚠ NaNs present"
    if pct_zero > 95:
        flag += " ⚠ >95% zero (likely a failed/stubbed fetch)"
    return (f"    {name:22s} min={np.nanmin(arr):9.2f}  max={np.nanmax(arr):9.2f}  "
            f"mean={np.nanmean(arr):9.2f}  std={np.nanstd(arr):8.2f}  zero%={pct_zero:5.1f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="./data/aoi_pairs_manifest.json")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    if not manifest:
        print(f"{args.manifest} has 0 entries -- nothing to inspect.")
        return

    print(f"Inspecting {len(manifest)} pair(s) from {args.manifest}\n")

    shapes = {"coarse": set(), "terrain": set(), "target": set()}

    for entry in manifest:
        path = entry["path"]
        print(f"=== {path}  (bbox={entry.get('bbox')}, date={entry.get('date')}) ===")
        try:
            data = np.load(path)
        except Exception as exc:
            print(f"  ⚠ FAILED TO LOAD: {exc}\n")
            continue

        for key in ("coarse", "terrain", "target"):
            arr = data[key]
            shapes[key].add(arr.shape)
            print(f"  {key}: shape={arr.shape}")
            if key == "terrain" and arr.shape[0] == len(TERRAIN_CHANNEL_NAMES):
                for i, name in enumerate(TERRAIN_CHANNEL_NAMES):
                    print(_stats(name, arr[i]))
            else:
                print(_stats(key, arr))
        print()

    print("--- Summary ---")
    for key, s in shapes.items():
        consistent = "OK (all pairs match)" if len(s) == 1 else f"⚠ INCONSISTENT across pairs: {s}"
        print(f"{key} shapes: {consistent}")


if __name__ == "__main__":
    main()
