# DDWF System

**Satellite AOI Weather Downscaling** — an independent FastAPI microservice
built to be called by [Project Vayu](https://vayu-geop.vercel.app) (or
anything else) over a plain HTTP API.

Design note: this runs **Option B** — the coarse global forecast comes
from [Open-Meteo](https://open-meteo.com) (free, no API key, blends
GFS/ICON/ECMWF-open-data), and DDWF's own training effort goes entirely
into the genuinely novel piece: a satellite-terrain-conditioned diffusion
downscaler + a zero-shot implicit terrain field. Full rationale in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The original TDD's SFNO
global-engine design (`docs/ARCHITECTURE.md` "Option A") is implemented,
tested, and kept as an optional upgrade path — not required to run this.

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

Full design rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
API contract: [`docs/API.md`](docs/API.md).
How to actually train the downscaler: [`docs/TRAINING.md`](docs/TRAINING.md).

## Stack

- **Serving:** FastAPI + Uvicorn, per-AOI coarse-forecast cache, Prometheus
  metrics, structured logging
- **Coarse forecast:** Open-Meteo grid client (`app/data/external_forecast.py`)
  — no training, no API key, up to 16 days out
- **Downscaling head (Module 3, the trainable core):** conditional
  diffusion U-Net (DDPM/EDM-style) with cross-attention to the coarse
  forecast, distilled DDIM sampler for low-latency serving, plus a SIREN
  implicit terrain field for zero-shot unmapped-AOI inference. All 8
  terrain conditioning channels are real and key-less: elevation/slope/
  aspect (Open-Meteo Elevation API, Copernicus GLO-90), land-cover class
  fractions (ESA WorldCover 10m, S3), and land surface temperature (MODIS
  MOD11A2 via ORNL DAAC's free subset service).
- **Ensembling (Module 4):** diffusion-seed perturbation aggregated to
  p10/p50/p90 with an explicit low-skill flag past Day 10
- **Optional upgrade path:** the original SFNO global engine (Module 2) —
  implemented and tested, not required for default operation. See
  `docs/TRAINING.md` "Option A".

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
The coarse forecast and terrain elevation are both real (pulled live from
Open-Meteo — internet access required for `/v1/forecast` to work). The
downscaled output isn't meteorologically meaningful yet without a trained
downscaler checkpoint — `/health`'s `downscaler_loaded` tells you honestly
whether one is present.

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
9 tests, all passing, no network/GPU/checkpoints required — they mock the
coarse-forecast source and model services to check routing, auth, schemas,
and real model-architecture forward passes (including the full-scale SFNO,
for the optional Option A path).

## Repo layout

```
app/
  api/         FastAPI routes + Pydantic schemas (the public contract)
  core/        config, logging, auth
  models/      diffusion U-Net, SIREN, SFNO (optional) — pure PyTorch, no I/O
  services/    stateful wrappers around models (downscaling, ensembling, terrain fusion)
  data/        Open-Meteo forecast + elevation clients (default), Zarr/feature-store access layer (optional path)
training/      Lightning + Hydra training scripts, loss functions, configs
scripts/       AOI training-pair builder (Open-Meteo + elevation), ERA5 download + Zarr utilities (optional path)
docs/          architecture, API reference, training guide
tests/         pytest suite (API, model shape, and training-data-pipeline smoke tests)
```

## Status

Architecture, API surface, and model code are implemented and tested
(`pytest -q` passes end-to-end — 17 tests, including a real read of the
training-data pipeline, a real forward pass through the actual serving
code path, and real (mocked-network) parsing of the MODIS LST response
schema). **No trained downscaler checkpoint ships in this repo** —
`scripts/build_aoi_pairs_manifest.py` + `training/train_downscaler.py` are
a real, runnable pipeline to build one; see `docs/TRAINING.md` for the
honest ~1.5-2 week path from here to real weights. All 8 terrain
conditioning channels are real, key-less data (elevation/slope/aspect,
land-cover fractions, land surface temperature) — the coarse forecast
proxy used as the training target is the one remaining documented
approximation, not a fabricated stub.
