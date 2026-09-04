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
from app.models.diffusion_unet import DiffusionDownscaler

log = get_logger(__name__)


class DownscalerService:
    _instance: "DownscalerService | None" = None

    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device(self.settings.device)
        self.model = DiffusionDownscaler()
        self._load_weights()
        self.model.to(self.device).eval()

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
        """DDIM sampling loop, `n_steps` denoising steps (defaults to
        settings.diffusion_sampling_steps)."""
        n_steps = n_steps or self.settings.diffusion_sampling_steps
        h, w = terrain_raster.shape[-2:]

        terrain_t = torch.tensor(terrain_raster, dtype=torch.float32, device=self.device).unsqueeze(0)
        tokens_t = torch.tensor(coarse_tokens, dtype=torch.float32, device=self.device).unsqueeze(0)

        x_t = torch.randn(1, n_channels_out, h, w, device=self.device)
        timesteps = torch.linspace(999, 0, n_steps, device=self.device).long()

        for t in timesteps:
            t_batch = t.expand(1)
            eps_pred = self.model(x_t, terrain_t, tokens_t, t_batch)
            # Simplified DDIM update (alpha schedule owned by the training
            # config — swap in the real noise schedule used at train time).
            alpha = 1.0 - (t.float() / 1000.0)
            x_t = (x_t - (1 - alpha).sqrt() * eps_pred) / alpha.sqrt().clamp(min=1e-4)

        return x_t.squeeze(0).cpu().numpy()
