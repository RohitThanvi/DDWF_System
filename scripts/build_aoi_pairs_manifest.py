#!/usr/bin/env python3
"""
Build real (coarse forecast, terrain, hi-res target) training pairs for the
diffusion downscaler, using only free/key-less sources:

  - Coarse forecast + "hi-res ground truth" proxy: Open-Meteo's Historical
    Weather API (archive-api.open-meteo.com), queried at two different grid
    densities over the same AOI/date/hour. The fine-grid query is treated
    as the training target; the coarse-grid query (same underlying blended
    model output, sampled sparser) is the conditioning input. This is a
    genuine, defensible proxy: Open-Meteo already blends local high-
    resolution models (e.g. ICON-D2 in Europe) where available, so denser
    sampling captures real local structure the coarse sampling misses.
    It is NOT the same as true satellite/station ground truth — treat this
    as a first, honest training signal to validate the architecture and
    pipeline, and swap in real station/satellite hi-res data
    (see docs/TRAINING.md) once this path is working end-to-end.
  - Terrain: real elevation/slope/aspect from Open-Meteo's Elevation API
    (Copernicus GLO-90), real land-cover class fractions from ESA
    WorldCover 10m (public S3 COG, no key), and real land surface
    temperature from MODIS MOD11A2 via ORNL DAAC's free key-less subset
    service — see app/data/terrain_sources.py. All three terrain channels
    are real; there is no remaining stub.

Usage:
  python scripts/build_aoi_pairs_manifest.py \
      --region 73.0,24.0,78.0,29.0 \
      --start-date 2024-01-01 --end-date 2024-01-31 \
      --n-aois 40 --coarse-grid 8 --fine-grid 32 \
      --out-dir ./data/aoi_pairs --manifest-out ./data/aoi_pairs_manifest.json

Default region is Rajasthan, India.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# Make `app.*` importable when this script is run directly
# (`python scripts/build_aoi_pairs_manifest.py`) rather than as a module
# from the repo root — direct script execution puts scripts/ on sys.path,
# not the repo root, so `from app.data...` would otherwise fail with
# ModuleNotFoundError regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAJASTHAN_BBOX = (69.5, 23.0, 78.3, 30.2)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _sub_aoi(region: tuple[float, float, float, float], size_deg: float, rng: random.Random) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = region
    lon0 = rng.uniform(min_lon, max_lon - size_deg)
    lat0 = rng.uniform(min_lat, max_lat - size_deg)
    return (lon0, lat0, lon0 + size_deg, lat0 + size_deg)


async def _fetch_historical_grid(
    bbox: tuple[float, float, float, float],
    grid_size: int,
    day: date,
    variables: list[str],
    client,
) -> np.ndarray:
    """Returns (n_vars, grid_size, grid_size) at 12:00 UTC on `day`."""
    from app.data.external_forecast import _grid_points

    points = _grid_points(bbox, grid_size)
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
        "hourly": ",".join(variables),
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "timezone": "UTC",
    }
    resp = await client.get(ARCHIVE_URL, params=params)
    resp.raise_for_status()
    payload = resp.json()
    locations = payload if isinstance(payload, list) else [payload]

    data = np.zeros((len(variables), grid_size, grid_size), dtype=np.float32)
    for idx, loc in enumerate(locations):
        row, col = divmod(idx, grid_size)
        hourly = loc.get("hourly", {})
        for v_idx, var in enumerate(variables):
            series = hourly.get(var)
            if series and len(series) > 12:
                data[v_idx, row, col] = np.nan_to_num(series[12], nan=0.0)  # 12:00 UTC
    return data


async def build_manifest(args: argparse.Namespace) -> None:
    import httpx

    from app.data.external_forecast import OPEN_METEO_VARIABLES
    from app.data.terrain_sources import ElevationClient, LandCoverClient, LSTClient, slope_aspect_from_elevation

    region = tuple(float(v) for v in args.region.split(","))
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    n_days = (end - start).days + 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    elevation_client = ElevationClient()
    lulc_client = LandCoverClient()
    lst_client = LSTClient()

    manifest: list[dict] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(args.n_aois):
            bbox = _sub_aoi(region, size_deg=args.aoi_size_deg, rng=rng)
            day = start + timedelta(days=rng.randrange(n_days))

            try:
                coarse = await _fetch_historical_grid(bbox, args.coarse_grid, day, OPEN_METEO_VARIABLES, client)
                target = await _fetch_historical_grid(bbox, args.fine_grid, day, OPEN_METEO_VARIABLES, client)
                elevation = await elevation_client.fetch_elevation_grid(bbox, grid_size=args.fine_grid)
                lulc_fractions = await asyncio.to_thread(lulc_client.fetch_lulc_patch, bbox, args.fine_grid)
                lst = await lst_client.fetch_lst_patch(bbox, args.fine_grid)
            except Exception as exc:
                print(f"[{i}] skip (fetch failed): {exc}")
                continue

            approx_width_m = abs(bbox[2] - bbox[0]) * 111_000
            cell_size_m = approx_width_m / max(args.fine_grid - 1, 1)
            slope, aspect = slope_aspect_from_elevation(elevation, cell_size_m=cell_size_m)
            terrain = np.concatenate(
                [np.stack([elevation, slope, aspect]), lulc_fractions, lst], axis=0
            ).astype(np.float32)

            pair_path = out_dir / f"pair_{i:05d}.npz"
            np.savez_compressed(pair_path, coarse=coarse, terrain=terrain, target=target)

            manifest.append({
                "path": str(pair_path),
                "bbox": list(bbox),
                "date": day.isoformat(),
                "coarse_grid": args.coarse_grid,
                "fine_grid": args.fine_grid,
            })
            print(f"[{i}] wrote {pair_path} (bbox={bbox}, date={day})")

    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest with {len(manifest)} pairs to {args.manifest_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=",".join(str(v) for v in RAJASTHAN_BBOX),
                         help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--n-aois", type=int, default=40)
    parser.add_argument("--aoi-size-deg", type=float, default=0.5)
    parser.add_argument("--coarse-grid", type=int, default=8)
    parser.add_argument("--fine-grid", type=int, default=32)
    parser.add_argument("--out-dir", default="./data/aoi_pairs")
    parser.add_argument("--manifest-out", default="./data/aoi_pairs_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(build_manifest(args))


if __name__ == "__main__":
    main()
