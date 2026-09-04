# DDWF System — Training Guide

This is the honest, step-by-step version of "how do I actually get real
weights into `checkpoints/sfno_global_v1.pt` and `checkpoints/diffusion_downscaler_v1.pt`".
Nothing here can run inside a sandboxed chat environment — no GPU, no
multi-TB data access, no days-long job scheduler. This is written for you
to run on real infrastructure (a cloud GPU box, a university cluster, or a
rented A100/H100 node).

---

## 0. Reality check on scope

Training a global weather model from scratch is a *large* undertaking —
GraphCast/Pangu/FourCastNet-class efforts used dozens of A100s for days to
weeks. You do not need to do that. Module 5 of the TDD already prescribes
the realistic path for a compute-constrained team:

> Start from an open foundation model (**Prithvi WxC** or **ClimaX**)
> rather than training SFNO from scratch, and fine-tune with **LoRA**
> adapters on the spectral-conv and attention projection layers only.

So the real plan is: **fine-tune, don't pretrain.** Everything below
assumes that.

---

## 1. Get access to a GPU

Cheapest realistic options for a student/independent-builder budget, roughly
ordered by cost:
- **Kaggle notebooks** — free P100/T4, 30h/week GPU quota. Enough for small
  LoRA fine-tunes on cropped/subsampled data, not full 0.25° global runs.
- **Google Colab Pro/Pro+** — A100 access, pay-as-you-go, good for the
  downscaling head (Module 3) which is much smaller than the global engine.
- **Vast.ai / RunPod / Lambda Cloud** — rent an A10/A100/H100 by the hour.
  An A10 (24GB) is enough for LoRA fine-tuning + inference of the global
  engine per Module 5's own target hardware; an A100 (40-80GB) makes the
  downscaler training faster.
- **University compute** — SKIT's Centre of Excellence in AI/ML, or ISRO/
  STPI contacts from your internships, may have GPU allocations worth asking
  about given this is a legitimate research-grade project.

## 2. Get the data

### 2a. ERA5 reanalysis (global engine training/fine-tuning target)
1. Register for a free Copernicus Climate Data Store (CDS) account:
   https://cds.climate.copernicus.eu
2. Install the CDS API client and save your key to `~/.cdsapirc` (the CDS
   site walks you through this after registration).
3. Run:
   ```bash
   python scripts/download_era5.py --year 2023 --month 01 --out ./data/raw
   ```
   Loop over the months/years you want. **Start small** — one month of
   pressure-level + surface data is already several GB; a full multi-decade
   archive at 0.25° is tens of TB. For fine-tuning, 1-3 years is a
   reasonable starting point.
4. Build the Zarr store:
   ```bash
   python scripts/build_zarr_store.py --raw-dir ./data/raw --out ./data/zarr/era5.zarr
   ```
   This also writes `data/climatology_stats.json`, consumed by
   `app/data/normalization.py`.

### 2b. Foundation model weights (Prithvi WxC / ClimaX)
- Prithvi WxC (NASA/IBM): https://huggingface.co/ibm-nasa-geospatial
- ClimaX (Microsoft Research): https://github.com/microsoft/ClimaX
Download the pretrained checkpoint, and adapt
`training/train_global_engine.py`'s `GlobalEngineLightningModule` to load
it into `SFNO`'s parameter names before applying LoRA (the exact mapping
depends on which base model you pick — their state_dict keys won't match
`app/models/sfno.py` 1:1, so you'll write a small key-remapping function;
this is normal for any "adapt a foundation model to a custom head" project).

### 2c. Terrain + satellite AOI data (downscaling head)
- **DEM:** Copernicus GLO-30 (free, global, 30m) —
  https://registry.opendata.aws/copernicus-dem/
- **LULC:** ESA WorldCover (free, global, 10m) —
  https://esa-worldcover.org
- **LST:** MODIS LST product (free via NASA Earthdata) —
  https://appeears.earthdatacloud.nasa.gov
- **Ground-truth hi-res target:** station observations (e.g. IMD for India,
  NOAA ISD globally) or a higher-res reanalysis (ERA5-Land at 9km) as a
  training proxy for "hi-res truth", since true 100m observational
  coverage is sparse. This coarse-to-fine *pairing* is the actual hard part
  of Module 3 — plan real time for it.

## 3. Train the global engine (fine-tune, LoRA)

```bash
pip install -r requirements-train.txt
python -m training.train_global_engine --config-name sfno_base
```
Key knobs in `training/configs/sfno_base.yaml`:
- `data.zarr_store_uri` -> point at your `data/zarr/era5.zarr`
- `peft.enabled: true`, `peft.lora_rank: 8-16` -> only adapters train,
  base weights frozen (fast, cheap, fits on one A10)
- `curriculum.stages` -> scheduled sampling (1-step -> 4-step -> 12-step
  rollout) as specified in Module 2; reduce epoch counts for a first run
- `precision: bf16-mixed`, `strategy: deepspeed_stage_3_offload` -> only
  needed if you're on <8 GPUs and hitting OOM

Watch `LatWeightedRMSE` and `SpectralBandpassLoss` in MLflow
(`mlflow ui --backend-store-uri ./mlruns`) — spectral loss dropping while
RMSE plateaus is the classic sign the model is learning to sharpen small-
scale structure instead of just blurring toward the mean.

## 4. Train the downscaling head

```bash
python -m training.train_downscaler --config-name diffusion_downscaler
```
Before this runs, you must implement `AOIPairDataset.__iter__` in
`training/train_downscaler.py` against your real paired coarse/hi-res tile
manifest — it currently raises `NotImplementedError` on purpose, since
there's no way to fabricate real geospatial training pairs without your
actual data sources wired up. The pairing logic is: for each AOI tile,
grab the coincident coarse global-engine trajectory patch (or ERA5 patch
at train time), the DEM/LULC/LST raster at target resolution, and the
hi-res ground truth (station obs or ERA5-Land proxy) at the same
valid_time.

## 5. Evaluate before trusting anything

- **CRPS** (Continuous Ranked Probability Score) and **rank histograms**
  for the ensemble module — flat rank histograms mean well-calibrated
  spread; U-shaped means under-dispersed (ensemble too confident).
- **Skill vs. lead time** — plot RMSE/ACC (anomaly correlation) by day;
  expect meaningful skill to fall off by Day 10-15, per the TDD's own
  "Known Limitations" — that's physics, not a training bug.
- **Zero-shot island validation** — before trusting the SIREN implicit-
  field path operationally anywhere, check predictions against any
  available buoy/station data near that location, exactly as the TDD
  flags.

## 6. Ship the checkpoint

```bash
cp <your best checkpoint>.pt checkpoints/sfno_global_v1.pt
cp <your best downscaler checkpoint>.pt checkpoints/diffusion_downscaler_v1.pt
docker compose up --build
curl http://localhost:8000/health   # global_engine_loaded / downscaler_loaded should now read true
```

## Realistic timeline

| Stage | Effort |
|---|---|
| CDS registration + first month of ERA5 downloaded | ~1 day |
| Zarr store + climatology stats built | ~1-2 days |
| Global engine LoRA fine-tune, first usable checkpoint | ~1-2 weeks (mostly GPU queue time on free tiers) |
| Terrain/satellite data sourcing + pairing pipeline | ~1-2 weeks (this is the part people underestimate) |
| Downscaler training, first usable checkpoint | ~1 week once pairs exist |
| End-to-end eval (CRPS, skill-vs-lead-time) | ~2-3 days |

Total: roughly **4-6 weeks** of part-time work to a first real, if modest,
checkpoint — not a weekend project, but very doable incrementally, and the
service runs and serves an API the whole time (with the honest
`*_loaded: false` health signal) so Vayu integration work isn't blocked on
training finishing.
