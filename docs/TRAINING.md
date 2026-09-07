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

- **Coarse forecast patch** — `scripts/build_aoi_pairs_manifest.py` handles
  this for you: it queries Open-Meteo's free [historical/archive
  API](https://open-meteo.com/en/docs/historical-weather-api) at two grid
  densities over the same AOI/date/hour — the sparse query becomes the
  conditioning input, the dense query becomes the training target (see the
  script's docstring for exactly why this is a defensible, if imperfect,
  proxy pairing, and what it is not a substitute for).
- **Terrain raster (DEM/LULC/LST)** — all three are fetched for real by the
  same script: DEM (elevation, slope, aspect) from Open-Meteo's Elevation
  API (Copernicus GLO-90), LULC (vegetation/built-up/water/bare-or-snow
  fractions) from ESA WorldCover 10m read directly off its public S3
  bucket, and LST from MODIS MOD11A2 via ORNL DAAC's free subset REST API
  (`app/data/terrain_sources.py`). None of the three needs an account or
  key — MODIS LST looked at first like it would need a NASA Earthdata
  login (the usual path for MODIS data), but ORNL DAAC's subsetting
  service turned out to be genuinely key-less.
- **Hi-res ground truth** — Option B's default is the dense Open-Meteo
  query above (a real, if approximate, signal). For a genuinely better
  target once the pipeline is proven out: station observations near your
  AOI (IMD for India, NOAA ISD globally), or a higher-resolution reanalysis
  product (ERA5-Land, 9km) as a training proxy where station density is
  too low.

**Build the manifest:**
```bash
pip install -r requirements-train.txt
python scripts/build_aoi_pairs_manifest.py \
    --region 69.5,23.0,78.3,30.2 \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --n-aois 40 --coarse-grid 8 --fine-grid 32 \
    --out-dir ./data/aoi_pairs --manifest-out ./data/aoi_pairs_manifest.json
```
**Windows PowerShell:** the `\` line-continuation above is bash/zsh syntax
and does nothing useful in PowerShell (it'll try to run each line as a
separate command). Either use PowerShell's backtick continuation or just
put it all on one line:
```powershell
python scripts/build_aoi_pairs_manifest.py --region 69.5,23.0,78.3,30.2 --start-date 2024-01-01 --end-date 2024-01-31 --n-aois 40 --coarse-grid 8 --fine-grid 32 --out-dir ./data/aoi_pairs --manifest-out ./data/aoi_pairs_manifest.json
```
The default region is Rajasthan — change `--region` for wherever you have
the best ground-truth station access. **Start with `--n-aois 40` or so** to
validate the pipeline end-to-end before scaling up to hundreds/thousands of
pairs; each AOI/date makes 3 API calls (2x Open-Meteo archive + 1x
elevation), so be considerate of Open-Meteo's free-tier fair-use limits
when scaling.

`AOIPairDataset` in `training/train_downscaler.py` reads this manifest
directly — no further implementation needed to get training running
end-to-end; `tests/test_training_data.py` exercises the exact read path
against synthetic-but-correctly-shaped data if you want to confirm it
works before spending API calls.

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
| Run `build_aoi_pairs_manifest.py` for a first small batch (~40 pairs) | ~1 day (mostly API call latency, not implementation — the pipeline is already built) |
| Review pairs, decide if/when to swap in real station ground truth | ~1-2 days |
| Scale up manifest to a few hundred/thousand pairs | ~2-3 days |
| First training run + iteration | ~3-7 days (small model, fast iteration) |
| Eval (CRPS, held-out comparison, zero-shot check) | ~2-3 days |

Total: roughly **1.5-2 weeks** part-time to a first real, evaluated
checkpoint (down from the original ~2-3 week estimate now that the data
pipeline is implemented, not just designed) — the service is usable
end-to-end (against Open-Meteo + untrained downscaler) the entire time, so
nothing about Vayu integration is blocked on this finishing.

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
