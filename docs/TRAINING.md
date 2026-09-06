# DDWF System — Training Guide

This is the honest, step-by-step version of "how do I actually get real
weights into `checkpoints/diffusion_downscaler_v1.pt`". Nothing here can
run inside a sandboxed chat environment — no GPU, no multi-TB data access,
no days-long job scheduler. This is written for you to run on real
infrastructure (a free-tier GPU notebook, a rented cloud GPU, or a
university cluster).

**Scope note (read this first):** DDWF runs in **Option B** — the coarse
global forecast comes from Open-Meteo (free, already trained, no training
needed on your end), and DDWF's own training effort is entirely spent on
the **diffusion AOI downscaler + SIREN terrain field** (Module 3). That's
the plan below. Option A (train the SFNO global engine yourself) is
documented at the end for completeness, but it is **not required** and not
the recommended path — it's a multi-week undertaking that duplicates what
Open-Meteo already gives you for free.

---

## Option B (default): train the downscaler + SIREN terrain field

### 1. Get a GPU
- **Kaggle notebooks** — free P100/T4, 30h/week quota. Genuinely enough
  for this: the downscaler is a modest U-Net operating on small AOI
  patches (256x256 or less), not a 721x1440 global grid.
- **Google Colab** (free or Pro) — same ballpark, easy to get started.
- **Vast.ai / RunPod** — rent an A10/RTX 4090 by the hour if you want
  faster iteration; this model is small enough that you don't need an A100.

### 2. Get the data — this is the actual hard part
You need **paired tiles**: (coarse forecast patch, terrain raster, hi-res
ground truth) for the same place and time. Three pieces:

- **Coarse forecast patch** — call the same Open-Meteo grid endpoint DDWF
  uses at serving time (`app/data/external_forecast.py`), but for *past*
  dates, using Open-Meteo's [historical/archive
  API](https://open-meteo.com/en/docs/historical-weather-api) (also free).
  This gives you the exact input distribution the model will see in
  production — no train/serve mismatch.
- **Terrain raster (DEM/LULC/LST)**:
  - DEM: Copernicus GLO-30 (free, global, 30m) —
    https://registry.opendata.aws/copernicus-dem/
  - LULC: ESA WorldCover (free, global, 10m) — https://esa-worldcover.org
  - LST: MODIS LST (free via NASA Earthdata) —
    https://appeears.earthdatacloud.nasa.gov
- **Hi-res ground truth** — the thing you're actually training the model
  to produce. Realistic sources:
  - Station observations near your AOI (IMD for India, NOAA ISD globally)
    — sparse but real
  - A higher-resolution reanalysis product (ERA5-Land, 9km) as a training
    *proxy* for "hi-res truth" where station density is too low — this is
    standard practice when true 100m verification data isn't available

Build a manifest (AOI bbox, date, paths to the three pieces above) and
implement `AOIPairDataset.__iter__` in `training/train_downscaler.py`
against it — it currently raises `NotImplementedError` on purpose, since
there's no way to fabricate real geospatial training pairs without your
actual data sources wired up.

**Start small.** A few hundred AOI/date pairs around Jaipur/Rajasthan (or
wherever you have station data access) is a reasonable first dataset —
you're validating the architecture and pipeline before scaling up
geographic/temporal coverage.

### 3. Train

```bash
pip install -r requirements-train.txt
python -m training.train_downscaler --config-name diffusion_downscaler
```

Key knobs in `training/configs/diffusion_downscaler.yaml`:
- `data.aoi_patch_size` — start at 128 or 256, not larger, for faster iteration
- `implicit_field.enabled: true` — jointly trains the SIREN terrain field
  so it works for unmapped coordinates without a cached raster tile
- `diffusion.sampling_steps_eval: 16` — matches the serving-time distilled
  DDIM sampler in `app/services/downscaler.py`

Watch the training MSE in MLflow (`mlflow ui --backend-store-uri ./mlruns`).
This is standard DDPM epsilon-prediction training — loss should decrease
steadily; if it plateaus early, check your data pairing before assuming
architecture problems.

### 4. Evaluate before trusting anything

- Hold out some AOI/date pairs entirely from training; compare predicted
  vs. ground-truth fields on those.
- **CRPS** and **rank histograms** for the ensemble output (multiple
  diffusion-sampling seeds) — flat rank histograms mean well-calibrated
  spread; U-shaped means under-dispersed (too confident).
- **Zero-shot island validation** — before trusting the SIREN implicit-
  field path anywhere with no cached terrain tile, check predictions
  against any available buoy/station data near that location.

### 5. Ship the checkpoint

```bash
cp <your best checkpoint>.pt checkpoints/diffusion_downscaler_v1.pt
docker compose up --build
curl http://localhost:8000/health   # downscaler_loaded should now read true
```

### Realistic timeline (Option B)

| Stage | Effort |
|---|---|
| Open-Meteo historical pulls + terrain raster sourcing | ~3-5 days |
| Pairing pipeline (`AOIPairDataset` implementation) | ~3-5 days |
| First training run + iteration | ~3-7 days (small model, fast iteration) |
| Eval (CRPS, held-out comparison, zero-shot check) | ~2-3 days |

Total: roughly **2-3 weeks** part-time to a first real, evaluated
checkpoint — the service is usable end-to-end (against Open-Meteo +
untrained downscaler) the entire time, so nothing about Vayu integration
is blocked on this finishing.

---

## Option A (not required): train the SFNO global engine yourself

Only relevant if there's a concrete reason to stop depending on Open-Meteo
— e.g. needing forecast variables or spatial resolution it doesn't offer,
or wanting to own the full pipeline for research reasons. This is the
original TDD Module 2 scope and is meaningfully bigger than Option B.

### Get the data
1. Register for a free Copernicus Climate Data Store (CDS) account:
   https://cds.climate.copernicus.eu
2. Install the CDS API client, save your key to `~/.cdsapirc`.
3. `python scripts/download_era5.py --year 2023 --month 01 --out ./data/raw`
   — **start with one month**; a full multi-decade 0.25° archive is tens
   of TB.
4. `python scripts/build_zarr_store.py --raw-dir ./data/raw --out ./data/zarr/era5.zarr`

### Get a foundation model to fine-tune (don't train from scratch)
- Prithvi WxC (NASA/IBM): https://huggingface.co/ibm-nasa-geospatial
- ClimaX (Microsoft Research): https://github.com/microsoft/ClimaX

You'll need to write a key-remapping function from the foundation model's
state_dict into `app/models/sfno.py`'s parameter names before applying
LoRA — their checkpoint keys won't match this repo's module names 1:1.

### Train
```bash
python -m training.train_global_engine --config-name sfno_base
```
`peft.enabled: true` with `lora_rank: 8-16` keeps this a fine-tune, not a
from-scratch pretrain — fits on a single A10.

### Realistic timeline (Option A, on top of Option B's work)
4-6 additional weeks: ERA5 access/download (~1 day), Zarr build (~1-2
days), LoRA fine-tune to a first usable checkpoint (~1-2 weeks, mostly GPU
queue time on free tiers), eval (~2-3 days) — and you'd still need Option
B's downscaler trained on top, since the global engine only produces the
coarse forecast, not the AOI-resolution output.
