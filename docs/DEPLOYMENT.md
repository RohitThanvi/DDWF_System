# DDWF System — Deployment Guide

## Production readiness checklist

Fixes made to get this service safe to run with real traffic, and what's
still on you:

**Fixed in code:**
- The Docker image previously never copied `data/` (only `app/` and
  `checkpoints/`) — `VariableNormalizer` needs
  `data/open_meteo_variable_stats.json` at first-request time, so every
  real forecast request 500'd in the container despite `/health` passing
  and all local tests being green (tests run from the repo root, where
  `data/` naturally exists). Fixed; `docker compose up --build` now
  actually serves a working `/v1/forecast`.
- `downscaler.downscale()` (synchronous, CPU-bound PyTorch inference) was
  called directly inside an `async def` route, blocking the whole asyncio
  event loop — including `/health` — for the full duration of every
  forecast. Now runs via `asyncio.to_thread`; the container's `HEALTHCHECK`
  stays responsive under concurrent load.
- API key comparison used `!=` (timing side-channel on a static bearer
  token); switched to `hmac.compare_digest`.
- `DDWF_API_KEY` silently defaulted to a placeholder that's visible in
  this repo's history — `Settings` now refuses to start with `ENV=production`
  and the default/short key. **You must set a real
  `DDWF_API_KEY`** (`openssl rand -hex 32`) before deploying with `ENV=production`.
- No rate limiting existed at all. Added a basic in-memory limiter
  (`RATE_LIMIT_PER_MINUTE`, default 60/client) — see
  `app/core/rate_limit.py`'s docstring for its single-process limitation
  and the Redis-backed upgrade path if you ever run more than one worker
  or replica.
- Unhandled exceptions returned FastAPI's default plain-text 500 with no
  consistent shape; added a global handler returning stable JSON, plus an
  `X-Request-ID` response header for correlating a caller's error report
  with server logs.
- `CoarseForecastService`'s process-local cache grew forever (one entry
  per distinct AOI ever queried, no eviction) — now TTL'd (1h) and capped
  (2000 entries).
- The container ran as root; now runs as an unprivileged `ddwf` user.
- `/metrics` was the one unauthenticated route on the service; now behind
  the same `X-API-Key` check as everything else.

**Still your call before going live:**
- TLS termination (put this behind a load balancer/ingress that
  terminates HTTPS — the service itself speaks plain HTTP).
- Real load testing — the rate limiter's default of 60 req/min/client is a
  starting guess, not a measured number.
- If you ever scale to more than one Uvicorn worker or more than one
  instance, migrate the rate limiter (and ideally the coarse-forecast
  cache) to Redis — both are documented as single-process-only above.
- Secrets management — `DDWF_API_KEY` still goes through a plain env var
  here; fine for Render/Cloud Run's own secret env-var handling, worth
  revisiting if you adopt a dedicated secrets manager later.

## The short answer on Render's free tier: no

Render's free web service instance gives you **512MB RAM and 0.1 CPU**.
Measured on this exact codebase:

| Step | RSS memory |
|---|---|
| Python interpreter, nothing imported | 9 MB |
| after `import torch` | **497 MB** |
| after `import rasterio` | 518 MB |
| after `import fastapi, uvicorn` | 532 MB |
| after instantiating the default `DiffusionDownscaler` | 664 MB |

`import torch` alone blows past the 512MB ceiling before the app has
served a single request, let alone loaded a model or handled a diffusion
sampling pass. This isn't a "might be tight" situation — it will not boot.
The 0.1 CPU allocation is a second, independent problem: diffusion
sampling is CPU-bound, and 0.1 CPU (a tenth of one core) would make even a
successful request take minutes.

**Minimum that actually works on Render:** the **Starter** plan
($7/month, 512MB RAM/0.5 CPU) is still tight — torch alone eats the whole
RAM budget with ~0MB left for the model, request buffers, or the OS. The
**Standard** plan (2GB RAM/1 CPU) is the realistic minimum for this
service. `render.yaml` in this repo defaults to `starter` as a floor, but
expect to bump it to `standard` if you see out-of-memory restarts.

## Free-tier-friendly alternatives (verify current limits yourself — these
## change; this list reflects each provider's usual practice, not a guarantee)

- **Hugging Face Spaces** (free CPU tier) — historically ships considerably
  more RAM per instance than Render's free tier, and supports a plain
  Dockerfile (this repo's `Dockerfile` should work with minimal changes:
  Spaces expects the app on port 7860 by default — override with `EXPOSE
  7860` and `--port 7860`, or set Render's usual `$PORT` handling aside).
  Good fit for a public demo of this exact service.
- **Google Cloud Run** — pay-per-use with a genuine free monthly quota,
  scale-to-zero, and configurable memory up to several GB per instance.
  Requires a GCP account and `gcloud` CLI or dashboard deploy from the
  same `Dockerfile`.
- **Oracle Cloud "Always Free"** — free-forever compute VMs (including
  ARM-based shapes with several GB of RAM) if you want a persistent VM
  instead of a PaaS. More setup (you manage the box), no torch-memory
  surprises once provisioned.

## Reducing DDWF's own footprint (worth doing regardless of host)

- Skip `prometheus-fastapi-instrumentator` in a memory-constrained deploy —
  small but non-zero savings.
- Don't run multiple Uvicorn workers (`--workers 1`) — each worker pays the
  ~500MB torch tax independently.
- If you don't need LULC in a given deployment, you can avoid importing
  `rasterio` by not calling `TerrainFusionService` — but this touches
  `/v1/forecast`'s core flow, so this is a "know what you're trading off"
  change, not a config flag today.
- None of this gets you under ~550-600MB. There is no configuration of
  this service that fits Render's free 512MB tier — the fix is more RAM,
  not less code.

## Deploying to Render (once you're on Starter/Standard, or an alternative
## host that provisions similarly)

1. Push this repo to GitHub (already done — `RohitThanvi/DDWF_System`).
2. In the Render dashboard: **New > Blueprint**, connect the repo. Render
   reads `render.yaml` automatically.
3. Set the `DDWF_API_KEY` environment variable in the dashboard (it's
   marked `sync: false` in `render.yaml` on purpose — don't commit secrets).
4. Deploy. Watch the build logs — the Docker image build pulls
   `libgdal-dev` and the full `requirements.txt`, so the first build takes
   a few minutes.
5. Once live, confirm:
   ```bash
   curl https://<your-service>.onrender.com/health
   ```
6. If you see repeated restarts in the Render dashboard's event log, that's
   the OOM kill described above — bump the plan, don't debug the code.

## Deploying without a trained checkpoint

This is fine and expected initially — `/health`'s `downscaler_loaded:
false` will be visible to anyone hitting the endpoint, same as local dev.
Nothing about deployment requires a trained checkpoint to exist first.
