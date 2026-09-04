# DDWF System

**Global 30-Day Deep-learning Weather Forecast + Satellite AOI Downscaling**
— an independent FastAPI microservice implementing the architecture in
`DDWF-System-Architecture-TDD.md`, built to be called by
[Project Vayu](https://vayu-geop.vercel.app) (or anything else) over a
plain HTTP API.

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

Full design rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
API contract: [`docs/API.md`](docs/API.md).
How to actually train real weights: [`docs/TRAINING.md`](docs/TRAINING.md).

## Stack

- **Serving:** FastAPI + Uvicorn, Redis-backed trajectory cache, Prometheus
  metrics, structured logging
- **Global engine (Module 2):** Spherical Fourier Neural Operator (SFNO) —
  spectral-conv blocks, FiLM-conditioned on GHG/ENSO/MJO indices,
  autoregressive 6h-hop rollout to 30 days
- **Downscaling head (Module 3):** conditional diffusion U-Net (DDPM/EDM-
  style) with cross-attention to the coarse forecast, distilled DDIM
  sampler for low-latency serving, plus a SIREN implicit terrain field for
  zero-shot unmapped-AOI inference
- **Ensembling (Module 4):** two-tier IC + diffusion-seed perturbation,
  aggregated to p10/p50/p90 with an explicit low-skill flag past Day 15
- **Training:** PyTorch + PyTorch Lightning + Hydra, LoRA fine-tuning on
  top of an open foundation model (Prithvi WxC / ClimaX) rather than
  training from scratch — see `docs/TRAINING.md`

## Quickstart (API only, no trained weights needed)

```bash
git clone https://github.com/RohitThanvi/DDWF_System.git
cd DDWF_System
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set a real DDWF_API_KEY

uvicorn app.main:app --reload
```
Then:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/forecast \
  -H "X-API-Key: <your key from .env>" \
  -H "Content-Type: application/json" \
  -d '{"lat": 26.9, "lon": 75.8, "horizon_days": 7}'
```
Without trained checkpoints in `checkpoints/`, the service still runs and
returns a valid, correctly-shaped response — it just isn't meteorologically
meaningful yet. `/health` tells you honestly whether real weights are
loaded (`global_engine_loaded` / `downscaler_loaded`).

## Docker

```bash
docker compose up --build
```

## Calling it from Vayu

```python
resp = await httpx.AsyncClient(base_url=DDWF_BASE_URL).post(
    "/v1/vayu/weather-layer",
    headers={"X-API-Key": DDWF_API_KEY},
    json={"lat": lat, "lon": lon, "horizon_days": 14},
)
```
See [`docs/API.md`](docs/API.md) for the full contract and response shape.
DDWF is fully decoupled from Vayu's codebase — Vayu never imports this
repo, it's just an HTTP client, same as any other consumer would be.

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```
Tests mock the model services so they run fast without checkpoints or a
GPU — they check routing, auth, schemas, and model-architecture shapes
(`tests/test_api.py`, `tests/test_models.py`).

## Repo layout

```
app/
  api/         FastAPI routes + Pydantic schemas (the public contract)
  core/        config, logging, auth
  models/      SFNO, diffusion U-Net, SIREN — pure PyTorch, no I/O
  services/    stateful wrappers around models (caching, ensembling, terrain fusion)
  data/        Zarr/feature-store access layer
training/      Lightning + Hydra training scripts, loss functions, configs
scripts/       ERA5 download + Zarr store build utilities
docs/          architecture, API reference, training guide
tests/         pytest suite (API + model shape smoke tests)
```

## Status

Architecture, API surface, and model code are implemented and tested
(`pytest -q` passes end-to-end including a real forward pass of the
full-scale SFNO). **No trained checkpoints ship in this repo** — see
`docs/TRAINING.md` for the realistic path from here to real weights.
