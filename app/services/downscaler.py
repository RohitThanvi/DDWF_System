"""
Runs the diffusion downscaling head: coarse global forecast patch + terrain
fusion -> hi-res (100m-1km) AOI fields, via a distilled few-step DDIM
sampler (see Module 5: 8-16 steps for ~2-4s/query on an A10/L4).
"""
from __future__ import annotations

import numpy as np
import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.data.external_forecast import OPEN_METEO_VARIABLES
from app.data.normalization import VariableNormalizer
from app.models.diffusion_schedule import cosine_noise_schedule
from app.models.diffusion_unet import DiffusionDownscaler

log = get_logger(__name__)

DIFFUSION_TRAIN_TIMESTEPS = 1000  # must match training/configs/diffusion_downscaler.yaml: diffusion.train_timesteps
X0_CLIP_VALUE = 5.0  # clip the predicted x0 to +/-5 (normalized-data std-devs) at every sampling step


class DownscalerService:
    _instance: "DownscalerService | None" = None

    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device(self.settings.device)
        self.model = DiffusionDownscaler(raw_token_dim=len(OPEN_METEO_VARIABLES))
        self._load_weights()
        self.model.to(self.device).eval()
        self.alphas_cumprod = cosine_noise_schedule(DIFFUSION_TRAIN_TIMESTEPS).to(self.device)
        self.normalizer = VariableNormalizer(self.settings.variable_stats_path)

    @classmethod
    def instance(cls) -> "DownscalerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_weights(self) -> None:
        try:
            state = torch.load(self.settings.downscaler_checkpoint, map_location="cpu")
            self.model.load_state_dict(state)
            log.info("downscaler.weights_loaded", path=self.settings.downscaler_checkpoint)
        except FileNotFoundError:
            log.warning("downscaler.weights_missing", path=self.settings.downscaler_checkpoint)

    @torch.no_grad()
    def downscale(
        self,
        coarse_patch: np.ndarray,      # (C_coarse, h, w) bicubic-upsampled coarse forecast
        terrain_raster: np.ndarray,    # (8, H, W) DEM/slope/aspect/LULC/LST at target res
        coarse_tokens: np.ndarray,     # (N, context_dim) flattened coarse-forecast tokens for cross-attn
        n_channels_out: int = 8,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """DDIM sampling loop (deterministic, eta=0), `n_steps` denoising
        steps (defaults to settings.diffusion_sampling_steps). Uses the
        same cosine noise schedule the model is trained under
        (app/models/diffusion_schedule.py) -- an earlier version of this
        sampler used a different, ad-hoc schedule that both mismatched
        training and was numerically unstable.

        Also clips the predicted x0 to +/-X0_CLIP_VALUE at every step
        ("static thresholding" -- standard practice in essentially every
        real DDPM/DDIM implementation, not something specific to this
        codebase). Without it, the x0-reconstruction formula
        `(x_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)` divides by a
        number approaching zero as t -> T, so *any* imperfection in
        eps_pred -- inevitable with an undertrained model, and never
        exactly zero even with a well-trained one -- gets amplified into
        a runaway value that compounds across the remaining steps. This
        was the actual cause of the wildly-out-of-range (thousands of
        degrees) output seen during early testing; the schedule fix alone
        made the numbers more correct but not bounded."""
        n_steps = n_steps or self.settings.diffusion_sampling_steps
        h, w = terrain_raster.shape[-2:]

        terrain_t = torch.tensor(terrain_raster, dtype=torch.float32, device=self.device).unsqueeze(0)
        tokens_t = torch.tensor(coarse_tokens, dtype=torch.float32, device=self.device).unsqueeze(0)

        x_t = torch.randn(1, n_channels_out, h, w, device=self.device)
        timesteps = torch.linspace(DIFFUSION_TRAIN_TIMESTEPS - 1, 0, n_steps, device=self.device).long()

        for i, t in enumerate(timesteps):
            t_batch = t.expand(1)
            eps_pred = self.model(x_t, terrain_t, tokens_t, t_batch)

            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=self.device)

            x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-X0_CLIP_VALUE, X0_CLIP_VALUE)
            x_t = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev).sqrt() * eps_pred

        result = x_t.squeeze(0).cpu().numpy()
        # x_t is in normalized space (see training/data.py's matching
        # normalize_stack call on the training target) -- denormalize back
        # to real physical units before returning, since API callers
        # expect actual temperature/pressure/etc., not z-scores.
        return self.normalizer.denormalize_stack(OPEN_METEO_VARIABLES[:n_channels_out], result)
