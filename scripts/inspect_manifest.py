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

from app.data.external_forecast import OPEN_METEO_VARIABLES  # noqa: E402

TERRAIN_CHANNEL_NAMES = [
    "elevation", "slope", "aspect",
    "lulc_vegetation", "lulc_built_up", "lulc_water", "lulc_bare_or_snow",
    "lst_celsius",
]


def _stats(name: str, arr: np.ndarray) -> tuple[str, bool]:
    n_nan = int(np.isnan(arr).sum())
    n_zero = int((arr == 0).sum())
    pct_zero = 100 * n_zero / arr.size
    is_mostly_zero = pct_zero > 95
    flag = ""
    if n_nan > 0:
        flag += " ⚠ NaNs present"
    if is_mostly_zero:
        # Not every high-zero% channel is a bug: precipitation is
        # legitimately all-zero in dry-season data, and minority LULC
        # classes (water, built-up, bare/snow) are legitimately near-zero
        # for a mostly-agricultural or mostly-vegetated AOI. Flag it as
        # worth a look, not as a confident diagnosis either way -- the
        # pair-level rollup below (many channels mostly-zero at once) is
        # the more reliable signal that something actually failed.
        flag += " ⚠ >95% zero"
    line = (f"    {name:22s} min={np.nanmin(arr):9.2f}  max={np.nanmax(arr):9.2f}  "
            f"mean={np.nanmean(arr):9.2f}  std={np.nanstd(arr):8.2f}  zero%={pct_zero:5.1f}{flag}")
    return line, is_mostly_zero


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
    degenerate_pairs = []

    for entry in manifest:
        path = entry["path"]
        print(f"=== {path}  (bbox={entry.get('bbox')}, date={entry.get('date')}) ===")
        try:
            data = np.load(path)
        except Exception as exc:
            print(f"  ⚠ FAILED TO LOAD: {exc}\n")
            continue

        n_channels_total = 0
        n_channels_mostly_zero = 0

        for key in ("coarse", "terrain", "target"):
            arr = data[key]
            shapes[key].add(arr.shape)
            print(f"  {key}: shape={arr.shape}")
            if key == "terrain" and arr.shape[0] == len(TERRAIN_CHANNEL_NAMES):
                names = TERRAIN_CHANNEL_NAMES
            elif key in ("coarse", "target") and arr.shape[0] == len(OPEN_METEO_VARIABLES):
                # These stack 8 weather variables on wildly different scales
                # (pressure ~1000 hPa, humidity 0-100%, wind direction
                # 0-360deg) -- a combined min/max/mean across all of them is
                # close to meaningless. Break out per variable, same as terrain.
                names = OPEN_METEO_VARIABLES
            else:
                names = None

            if names is not None:
                for i, name in enumerate(names):
                    line, is_mostly_zero = _stats(name, arr[i])
                    print(line)
                    n_channels_total += 1
                    n_channels_mostly_zero += is_mostly_zero
            else:
                line, is_mostly_zero = _stats(key, arr)
                print(line)
                n_channels_total += 1
                n_channels_mostly_zero += is_mostly_zero

        # Pair-level rollup: a handful of individually-sparse channels
        # (precipitation on a dry day, a minority LULC class) is normal.
        # Most of a pair's channels being mostly-zero at once is a much
        # stronger, more specific signal that the fetch actually failed
        # for that AOI -- that's the one worth acting on.
        frac_mostly_zero = n_channels_mostly_zero / max(n_channels_total, 1)
        if frac_mostly_zero > 0.75:
            degenerate_pairs.append(path)
            print(f"  ⚠⚠ {n_channels_mostly_zero}/{n_channels_total} channels are >95% zero "
                  f"-- this pair looks largely degenerate, consider excluding it from training")
        print()

    print("--- Summary ---")
    for key, s in shapes.items():
        consistent = "OK (all pairs match)" if len(s) == 1 else f"⚠ INCONSISTENT across pairs: {s}"
        print(f"{key} shapes: {consistent}")
    strata = [entry.get("stratum") for entry in manifest if entry.get("stratum")]
    if strata:
        from collections import Counter
        print("\nCoverage by stratum (world-coverage manifests only):")
        for name, count in Counter(strata).most_common():
            print(f"    {name:22s} {count}")
    if degenerate_pairs:
        print(f"\n⚠ {len(degenerate_pairs)} likely-degenerate pair(s):")
        for p in degenerate_pairs:
            print(f"    {p}")
    else:
        print("\nNo pairs look largely degenerate (individual sparse channels can still be legitimate -- see per-channel output above).")


if __name__ == "__main__":
    main()
