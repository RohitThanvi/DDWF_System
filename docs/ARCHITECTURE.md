# DDWF System — Architecture

This service implements the Global 30-Day DDWF + Satellite AOI Downscaling
design from the source TDD, as an independent FastAPI microservice. It has
no dependency on Project Vayu's codebase — Vayu is just an HTTP client of
this service (see `app/api/routes/vayu_integration.py`).

```
ERA5/CMIP6/MERRA-2 ──▶ Zarr Data Lake ──▶ Global Engine (SFNO) ──▶ 30-day trajectory cache
                                                                          │
User AOI (lat/lon/bbox) ──▶ Terrain Fusion (DEM+LULC+LST) ───┐           │
                                                              ▼           ▼
                                                  AOI Downscaling Head (Diffusion)
                                                              │
                                                              ▼
                                                  Ensemble / Uncertainty Module
                                                              │
                                                              ▼
                                                  p10 / p50 / p90 Timeseries API
```

## Module -> code map

| TDD Module | Code |
|---|---|
| 1. Data Ingestion & Spatiotemporal Pipeline | `app/data/zarr_pipeline.py`, `app/data/feature_store.py`, `app/data/normalization.py`, `scripts/download_era5.py`, `scripts/build_zarr_store.py` |
| 2. Global Planetary Engine (SFNO) | `app/models/sfno.py`, `app/services/global_engine.py`, `training/train_global_engine.py`, `training/losses.py` |
| 3. Satellite AOI & Terrain Downscaling Head | `app/models/diffusion_unet.py`, `app/models/siren.py`, `app/services/downscaler.py`, `app/services/terrain_fusion.py`, `training/train_downscaler.py` |
| 4. Uncertainty Estimation & Ensembling | `app/services/ensembler.py` |
| 5. Infrastructure Under Compute Constraints | `Dockerfile`, `docker-compose.yml`, `app/core/config.py`, `training/configs/*.yaml` (LoRA/DeepSpeed/precision settings) |
| 6. End-to-End Integration Flow | `app/api/routes/forecast.py`, `app/main.py` |

## Why FastAPI as an independent service

- **Clean boundary.** Vayu never imports DDWF code or touches its model
  weights; it calls `POST /v1/forecast` or `POST /v1/vayu/weather-layer`
  over HTTP with an API key. DDWF can be redeployed, rescaled, or have its
  models swapped without touching Vayu at all.
- **Different scaling profile.** DDWF is GPU-bound and cache-heavy (one
  expensive global rollout per forecast cycle, shared across all callers).
  Vayu is a lightweight FastAPI + React app on Render/Vercel. Coupling them
  in one deployable would force Vayu's ops to inherit DDWF's GPU/latency
  requirements for no reason.
- **Reusability.** Any other consumer (a CLI, a notebook, another product)
  gets the same contract Vayu gets — there's exactly one weather engine to
  maintain.

## Serving-time shortcuts (and why they're safe)

- **FFT fallback instead of true spherical harmonics at inference**
  (`app/models/sfno.py: FFTFallbackSHT`). Training uses the true SHT via
  `torch-harmonics` (pole-distortion-free); serving uses a plain 2D FFT
  because it's dramatically cheaper to ship and run, and the model has
  already learned to compensate for the transform it was trained with —
  as long as train and serve use the *same* transform. **This means the
  served checkpoint must be trained with `sht_backend: fft` if you want
  serving to exactly match training**, or you accept the small approximation
  gap from swapping SHT->FFT post-training. `training/configs/sfno_base.yaml`
  defaults to the true SHT (`harmonics`) for training quality; flip it to
  `fft` if you'd rather train/serve consistently on the cheap transform.
- **Depthwise spectral weights** (`SpectralConv`) instead of a dense
  `(in, out, modes, modes)` tensor — the textbook FNO formulation is only
  tractable for small mode counts; at `l_max=180` a dense weight is tens of
  GB per block. Channel mixing happens in the surrounding 1x1 convs instead.
- **Trajectory caching per forecast cycle**
  (`app/services/cache.py`) — the expensive 120-step SFNO rollout runs once
  per cycle (default 6h TTL), not per user request. AOI downscaling is the
  only per-request-expensive step, and it's the one already designed for
  low latency (8-16 step distilled DDIM sampler).

## Known gaps you should close before this is production-real

1. `GlobalEngineService._load_initial_condition` and
   `TerrainFusionService.fetch_raster_patch` return zero arrays — wire them
   to `ZarrDataLake.latest_analysis_state()` and a real DEM/LULC/LST reader.
2. `app/api/routes/forecast.py`'s tokenization of the coarse patch into
   cross-attention tokens is a placeholder reshape — replace with a real
   patch-embedding (e.g. a small conv stem + flatten, trained jointly with
   the downscaler).
3. `training/train_downscaler.py: AOIPairDataset` raises
   `NotImplementedError` — you need a real paired coarse/hi-res tile
   manifest before that script runs (see `docs/TRAINING.md`).
4. `EnsemblerService` runs a tiny 3-member ensemble inline per request for
   demo purposes; production should precompute the full N (IC) x M
   (diffusion-seed) ensemble grid offline, once per cycle, alongside the
   trajectory cache.
