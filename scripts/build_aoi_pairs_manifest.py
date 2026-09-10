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

Default region is Rajasthan, India. If you see repeated 429s even with the
built-in retry logic, you're likely hitting a temporary rate-limit window
from an earlier unthrottled run rather than the endpoint's actual capacity
(Open-Meteo documents 600 requests/minute, which this script stays well
under once paced) — wait a few minutes, then retry with a slower pace:
  --chunk-delay-s 3 --aoi-delay-s 10 --max-retries 10 --retry-base-delay-s 5
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
    chunk_delay_s: float = 1.5,
    max_retries: int = 8,
    retry_base_delay_s: float = 3.0,
) -> np.ndarray:
    """Returns (n_vars, grid_size, grid_size) at 12:00 UTC on `day`.
    Batches into <=MAX_COORDS_PER_REQUEST-coordinate requests -- the
    archive API hits the same URL-length 414 as the forecast API on
    larger grids (fine_grid=32 -> 1024 points in one URL is well past
    what the server accepts, regardless of Open-Meteo's documented
    1000-location count limit, which is a separate thing from URL length).
    Retries with backoff on 429s and pauses briefly between chunks --
    generous defaults on purpose: this is an offline, one-off data-prep
    script, not a live request, so it's worth waiting several minutes to
    succeed rather than failing fast. `max_retries=8` with
    `retry_base_delay_s=3.0` gives a cumulative backoff window of several
    minutes, which matters because a burst of unthrottled requests (e.g.
    from before this retry logic existed) can get an IP flagged for
    longer than a few seconds of backoff would cover."""
    from app.data.external_forecast import MAX_COORDS_PER_REQUEST, _grid_points
    from app.data.http_utils import get_with_retry

    points = _grid_points(bbox, grid_size)
    locations: list[dict] = []
    for i in range(0, len(points), MAX_COORDS_PER_REQUEST):
        chunk = points[i : i + MAX_COORDS_PER_REQUEST]
        params = {
            "latitude": ",".join(f"{lat:.4f}" for lat, _ in chunk),
            "longitude": ",".join(f"{lon:.4f}" for _, lon in chunk),
            "hourly": ",".join(variables),
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "timezone": "UTC",
        }
        resp = await get_with_retry(
            client, ARCHIVE_URL, params, max_retries=max_retries, base_delay_s=retry_base_delay_s
        )
        payload = resp.json()
        locations.extend(payload if isinstance(payload, list) else [payload])
        if i + MAX_COORDS_PER_REQUEST < len(points):
            await asyncio.sleep(chunk_delay_s)

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
                coarse = await _fetch_historical_grid(
                    bbox, args.coarse_grid, day, OPEN_METEO_VARIABLES, client,
                    chunk_delay_s=args.chunk_delay_s, max_retries=args.max_retries,
                    retry_base_delay_s=args.retry_base_delay_s,
                )
                target = await _fetch_historical_grid(
                    bbox, args.fine_grid, day, OPEN_METEO_VARIABLES, client,
                    chunk_delay_s=args.chunk_delay_s, max_retries=args.max_retries,
                    retry_base_delay_s=args.retry_base_delay_s,
                )
                elevation = await elevation_client.fetch_elevation_grid(
                    bbox, grid_size=args.fine_grid, chunk_delay_s=args.chunk_delay_s,
                    max_retries=args.max_retries, retry_base_delay_s=args.retry_base_delay_s,
                )
                lulc_fractions = await asyncio.to_thread(lulc_client.fetch_lulc_patch, bbox, args.fine_grid)
                lst = await lst_client.fetch_lst_patch(
                    bbox, args.fine_grid, max_retries=args.max_retries, retry_base_delay_s=args.retry_base_delay_s,
                )
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

            if i < args.n_aois - 1:
                await asyncio.sleep(args.aoi_delay_s)

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
    parser.add_argument("--chunk-delay-s", type=float, default=1.5,
                         help="Pause between chunked requests within one AOI fetch. Increase if you keep seeing 429s.")
    parser.add_argument("--aoi-delay-s", type=float, default=5.0,
                         help="Pause between AOIs. Increase if you keep seeing 429s.")
    parser.add_argument("--max-retries", type=int, default=8,
                         help="Retries per request on 429/5xx before giving up on that AOI. This is an offline script -- generous by default.")
    parser.add_argument("--retry-base-delay-s", type=float, default=3.0,
                         help="Base for exponential backoff between retries (ignored if the server sends a Retry-After header).")
    args = parser.parse_args()

    asyncio.run(build_manifest(args))


if __name__ == "__main__":
    main()
