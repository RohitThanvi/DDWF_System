"""Shape/forward-pass sanity tests for the model modules (tiny configs,
CPU-only — these are architecture smoke tests, not accuracy tests)."""
from __future__ import annotations

from datetime import date

import numpy as np
import torch

from app.models.diffusion_unet import DiffusionDownscaler
from app.models.siren import TerrainSIREN
from app.models.sfno import SFNO


def test_sfno_forward_shape():
    model = SFNO(
        in_channels=4, out_channels=4, hidden_channels=16, n_blocks=2,
        l_max=8, cond_dim=8, sht_backend="fft", nlat=32, nlon=64,
    )
    x = torch.randn(1, 4, 32, 64)
    cond = torch.randn(1, 8)
    out = model(x, cond)
    assert out.shape == (1, 4, 32, 64)


def test_sfno_rollout_shape():
    model = SFNO(
        in_channels=4, out_channels=4, hidden_channels=16, n_blocks=2,
        l_max=8, cond_dim=8, sht_backend="fft", nlat=32, nlon=64,
    )
    x0 = torch.randn(1, 4, 32, 64)
    cond = torch.randn(1, 8)
    traj = model.rollout(x0, cond, steps=3)
    assert traj.shape == (1, 3, 4, 32, 64)


def test_terrain_siren_forward_shape():
    model = TerrainSIREN(coord_dim=4, raster_feat_dim=8, hidden_dim=32, n_layers=3, out_dim=16)
    lat = torch.tensor([26.9, -12.3])
    lon = torch.tensor([75.8, 130.0])
    feats = torch.randn(2, 8)
    out = model(lat, lon, feats)
    assert out.shape == (2, 16)


def test_diffusion_downscaler_forward_shape():
    model = DiffusionDownscaler(
        in_channels=4, cond_raster_channels=4, base_channels=8,
        channel_mults=(1, 2), raw_token_dim=10, context_dim=16, time_dim=32,
    )
    noisy = torch.randn(1, 4, 32, 32)
    terrain = torch.randn(1, 4, 32, 32)
    tokens = torch.randn(1, 10, 10)  # raw_token_dim=10, projected internally to context_dim=16
    t = torch.tensor([500])
    out = model(noisy, terrain, tokens, t)
    assert out.shape == (1, 4, 32, 32)


def test_downscaler_service_real_forward_pass():
    """End-to-end check of the actual serving code path (DownscalerService),
    not just the raw model — this is what catches token-dimension mismatches
    between app/data/external_forecast.py's token width and the model's
    configured raw_token_dim (a real bug this test was added to prevent
    regressing)."""
    from app.services.downscaler import RAW_TOKEN_DIM, DownscalerService

    svc = DownscalerService.__new__(DownscalerService)  # skip __init__'s settings/device boilerplate
    svc.settings = type("S", (), {"diffusion_sampling_steps": 2, "downscaler_checkpoint": "/nonexistent.pt"})()
    svc.device = torch.device("cpu")
    from app.data.external_forecast import OPEN_METEO_VARIABLES
    from app.data.time_features import append_time_features

    svc.model = DiffusionDownscaler(raw_token_dim=RAW_TOKEN_DIM, base_channels=8, channel_mults=(1, 2))

    from app.models.diffusion_schedule import cosine_noise_schedule
    from app.services.downscaler import DIFFUSION_TRAIN_TIMESTEPS

    svc.alphas_cumprod = cosine_noise_schedule(DIFFUSION_TRAIN_TIMESTEPS)

    from app.data.normalization import VariableNormalizer

    svc.normalizer = VariableNormalizer("./data/open_meteo_variable_stats.json")

    coarse_patch = torch.randn(len(OPEN_METEO_VARIABLES), 4, 4).numpy()
    terrain_raster = torch.randn(8, 16, 16).numpy()
    coarse_tokens = coarse_patch.reshape(coarse_patch.shape[0], -1).T  # (16, n_vars) — real shape from forecast.py
    coarse_tokens = append_time_features(coarse_tokens, date(2024, 6, 15))  # -> (16, n_vars + TIME_FEATURE_DIM), same as forecast.py

    out = DownscalerService.downscale(svc, coarse_patch, terrain_raster, coarse_tokens, n_channels_out=8, n_steps=2)
    assert out.shape == (8, 16, 16)
    assert np.isfinite(out).all()  # regression check for the earlier numerically-unstable sampler
    # x0-clipping bounds the normalized output to +/-X0_CLIP_VALUE std-devs
    # per variable, so even the widest-variance shipped stat (shortwave
    # radiation, std=250) can't produce anything close to the ~-12500
    # seen before the fix -- this bound is deliberately loose (an order
    # of magnitude beyond any plausible real value) so it only fails if
    # the blowup regresses, not on ordinary variation.
    assert np.abs(out).max() < 10_000


def test_training_config_raw_token_dim_matches_serving():
    """training/configs/diffusion_downscaler.yaml's raw_token_dim and
    app/services/downscaler.py's RAW_TOKEN_DIM must agree -- if a model is
    trained against one value and served with the other, every real
    forecast request would either crash (shape mismatch) or silently feed
    the model garbage in whichever extra/missing columns exist. This is
    the same train/serve-mismatch failure mode documented in
    app/models/diffusion_schedule.py, just for token width instead of the
    noise schedule."""
    import yaml

    from app.services.downscaler import RAW_TOKEN_DIM

    with open("training/configs/diffusion_downscaler.yaml") as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["raw_token_dim"] == RAW_TOKEN_DIM
