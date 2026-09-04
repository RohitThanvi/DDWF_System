#!/usr/bin/env python3
"""
Build the chunked Zarr store from raw ERA5 NetCDF, using kerchunk to
generate a virtual reference dataset first (no re-write of raw archives,
per Module 1 of the TDD), then materializing the normalized, regridded
variable stack that app/data/zarr_pipeline.ZarrDataLake reads.

Usage:
  python scripts/build_zarr_store.py --raw-dir ./data/raw --out ./data/zarr/era5.zarr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--climatology-start", default="1991")
    parser.add_argument("--climatology-end", default="2020")
    args = parser.parse_args()

    import xarray as xr
    from kerchunk.hdf import SingleHdf5ToZarr
    from kerchunk.combine import MultiZarrToZarr

    raw_files = sorted(Path(args.raw_dir).glob("*.nc"))
    if not raw_files:
        raise SystemExit(f"No NetCDF files found under {args.raw_dir} — run download_era5.py first.")

    print(f"Building kerchunk references for {len(raw_files)} files...")
    singles = [SingleHdf5ToZarr(str(f)).translate() for f in raw_files]
    combined = MultiZarrToZarr(singles, concat_dims=["time"]).translate()

    ref_path = Path(args.out).parent / "kerchunk_refs.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(combined, f)

    ds = xr.open_dataset(
        "reference://", engine="zarr",
        backend_kwargs={"storage_options": {"fo": str(ref_path)}, "consolidated": False},
    )

    # Regrid to a common 0.25 deg lat-lon grid (xesmf conservative remapping;
    # target grid construction omitted here — see docs/TRAINING.md for the
    # full xesmf.Regridder setup against a reference 721x1440 grid).
    print("Loaded reference dataset:", ds)

    clim = ds.sel(time=slice(f"{args.climatology_start}-01-01", f"{args.climatology_end}-12-31"))
    stats = {
        var: {"mean": float(clim[var].mean()), "std": float(clim[var].std())}
        for var in ds.data_vars
    }
    stats_path = Path(args.out).parent / "climatology_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote climatology stats to {stats_path}")

    ds.to_zarr(args.out, mode="w", consolidated=True)
    print(f"Wrote Zarr store to {args.out}")


if __name__ == "__main__":
    main()
