# DDWF System — Architecture

This service implements the AOI-downscaling design from the source TDD as
an independent FastAPI microservice, with one deliberate scope change from
the original design: **Option B**.

## Design decision: Option B (borrow the global engine, train the downscaler)

The original TDD (Module 2) calls for training/serving a Spherical Fourier
Neural Operator as the global 30-day planetary engine. That's a large,
multi-week-to-months undertaking (see `docs/TRAINING.md` "Option A") and,
for the actual goal here — Vayu getting a working weather layer — it's not
where the interesting or necessary work is.

**Option B**: treat "a coarse global forecast" as a solved problem and pull
it from a free, already-operational source (Open-Meteo, no API key, blends
GFS/ICON/ECMWF-open-data). All of DDWF's own training effort goes into the
genuinely novel piece: the **satellite-terrain-conditioned diffusion
downscaler** (Module 3) plus the **zero-shot SIREN implicit terrain
field**. This is a smaller, faster, and more honest scope for an
independent project — real research/engineering work where it matters,
without pretending to have retrained a global NWP model from a laptop.

Practical consequence: horizon is capped at **16 days** (Open-Meteo's free
forecast horizon) instead of 30. See `ForecastRequest.horizon_days`.

The SFNO code (`app/models/sfno.py`, `app/services/global_engine.py`,
`training/train_global_engine.py`) is kept in the repo, fully working and
tested — it's the natural **Option A upgrade path** if there's ever a real
reason to own the global-engine piece instead of depending on Open-Meteo.

```
Open-Meteo (free, no key) ──▶ Coarse forecast patch (per-AOI grid, cached)
                                                                  │
User AOI (lat/lon/bbox) ──▶ Terrain Fusion (DEM+LULC+LST) ───┐   │
                                                              ▼   ▼
                                                  AOI Downscaling Head (Diffusion)
                                                              │
                                                              ▼
                                                  Ensemble / Uncertainty Module
                                                              │
                                                              ▼
                                                  p10 / p50 / p90 Timeseries API
```

## Module -> code map

| Module | Code | Status |
|---|---|---|
| Coarse forecast source | `app/data/external_forecast.py` (Open-Meteo client + cache) | **Default (Option B)** |
| 1. Data Ingestion & Spatiotemporal Pipeline | `app/data/zarr_pipeline.py`, `app/data/feature_store.py`, `app/data/normalization.py`, `scripts/download_era5.py`, `scripts/build_zarr_store.py` | Optional — Option A only |
| 2. Global Planetary Engine (SFNO) | `app/models/sfno.py`, `app/services/global_engine.py`, `training/train_global_engine.py`, `training/losses.py` | Optional — Option A only |
| 3. Satellite AOI & Terrain Downscaling Head | `app/models/diffusion_unet.py`, `app/models/siren.py`, `app/services/downscaler.py`, `app/services/terrain_fusion.py`, `training/train_downscaler.py` | **Default — this is what DDWF trains** |
| 4. Uncertainty Estimation & Ensembling | `app/services/ensembler.py` | Default |
| 5. Infrastructure Under Compute Constraints | `Dockerfile`, `docker-compose.yml`, `app/core/config.py`, `training/configs/*.yaml` | Default |
| 6. End-to-End Integration Flow | `app/api/routes/forecast.py`, `app/main.py` | Default |

## Why FastAPI as an independent service

- **Clean boundary.** Vayu never imports DDWF code or touches its model
  weights; it calls `POST /v1/forecast` or `POST /v1/vayu/weather-layer`
  over HTTP with an API key. DDWF can be redeployed, rescaled, or have its
  models swapped without touching Vayu at all.
- **Different scaling profile.** DDWF's downscaler is GPU-bound; Vayu is a
  lightweight FastAPI + React app on Render/Vercel. Coupling them in one
  deployable would force Vayu's ops to inherit GPU/latency requirements for
  no reason.
- **Reusability.** Any other consumer gets the same contract Vayu gets —
  one weather engine to maintain, not one embedded per caller.

## Serving-time notes

- **Coarse patch construction.** Open-Meteo is a point API; DDWF tiles the
  AOI bbox with an N x N grid of sample points and batches them into one
  request (`app/data/external_forecast.py:_grid_points`), building a
  pseudo-raster coarse patch out of point forecasts. This is coarser than a
  true 25km gridded NWP output but free and zero-maintenance.
- **Terrain reads.** Elevation comes from Open-Meteo's point API the same
  way. Land cover comes from a genuinely different pattern: ESA WorldCover
  ships as 3x3-degree Cloud-Optimized GeoTIFFs on a public S3 bucket, and
  `LandCoverClient` reads only the AOI's windowed bytes via GDAL's
  `/vsicurl/` driver — no full-tile download despite tiles running into the
  hundreds of MB. An AOI that straddles a WorldCover tile boundary is
  approximated using only the tile containing the AOI center; see
  `LandCoverClient`'s docstring. Land surface temperature comes from a
  third pattern again: ORNL DAAC's free MODIS/VIIRS subset REST API
  (`LSTClient`) — a two-call flow (list available composite dates, then
  fetch the most recent one) since MOD11A2 is an 8-day composite, not a
  live reading.
- **Per-AOI caching.** `CoarseForecastService` caches by rounded bbox +
  grid size + horizon, so repeated queries for the same AOI within a
  request burst don't re-hit Open-Meteo.
- **Depthwise spectral weights** in the (optional, Option A) SFNO — a
  dense `(in, out, modes, modes)` weight is only tractable for small mode
  counts; at `l_max=180` it's tens of GB per block, so `SpectralConv` is
  depthwise with channel-mixing left to the surrounding 1x1 convs.

## Known gaps you should close before this is production-real

1. `TerrainFusionService.fetch_raster_patch` now fetches **real**
   elevation/slope/aspect (Open-Meteo Elevation API, Copernicus GLO-90),
   **real** land-cover class fractions from ESA WorldCover 10m (public S3
   COG, HTTP range requests, no key), and **real** land surface
   temperature from MODIS MOD11A2 via ORNL DAAC's free key-less subset
   service. All 8 terrain channels are now real — there is no remaining
   stub channel. (MODIS LST initially looked like it would need a NASA
   Earthdata login, the usual path for MODIS data, but ORNL DAAC's
   subsetting service turned out to be genuinely key-less too.)
2. The downscaler and SIREN terrain field ship with **randomly-initialized
   weights** — `scripts/build_aoi_pairs_manifest.py` +
   `training/train_downscaler.py` now form a real, runnable pipeline to
   train them; see `docs/TRAINING.md` for the honest scope of the training
   *signal* itself (the fine-grid-Open-Meteo-as-target proxy is real but
   imperfect — read the script's docstring before trusting the result
   operationally).
3. `app/api/routes/forecast.py`'s tokenization of the coarse patch into
   cross-attention tokens is a crude reshape (one token per coarse grid
   cell, raw feature width = number of coarse variables) — functional and
   dimensionally consistent with the model (`DiffusionDownscaler.raw_token_dim`),
   but a learned patch-embedding would likely do better; worth revisiting
   once there's a trained checkpoint to benchmark against.
4. `EnsemblerService` runs a tiny 3-seed ensemble inline per request for
   demo purposes; production should precompute a larger diffusion-seed
   ensemble and cache it per AOI per cycle, same pattern as the coarse
   forecast cache.
