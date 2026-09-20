"""
The noise schedule for the diffusion downscaler — shared between training
(training/data.py's DownscalerLightningModule) and serving
(app/services/downscaler.py's DDIM sampling loop).

This used to be defined twice: once here (nowhere, originally) and once
in training/data.py, with serving using a different, ad-hoc "alpha = 1 -
t/1000" formula instead of importing the real one. That mismatch was a
real bug — the model is trained under the cosine schedule below, so
sampling with a different schedule at serving time is denoising against a
noise process the model was never trained to invert, on top of being
numerically unstable on its own (dividing by alpha.sqrt() near t=999,
where alpha≈0.001, amplifies whatever's in the numerator by ~30x in the
very first sampling step). Living in app/models/ (not training/) means
serving code can depend on it without pulling in the training package.
"""
from __future__ import annotations

import torch


def cosine_noise_schedule(timesteps: int) -> torch.Tensor:
    """Returns alphas_cumprod, shape (timesteps + 1,), indexed by integer
    timestep 0..timesteps inclusive (index 0 = no noise, index
    `timesteps` = pure noise). Standard cosine schedule (Nichol & Dhariwal,
    'Improved DDPM')."""
    s = 0.008
    t = torch.linspace(0, timesteps, timesteps + 1)
    f = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    return alphas_cumprod.clamp(min=1e-5, max=1.0)
