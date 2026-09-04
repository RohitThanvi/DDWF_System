"""
Loss terms for the global engine (Module 2):

L = lambda1 * LatWeightedRMSE
  + lambda2 * SpectralBandpassLoss
  + lambda3 * PerceptualLoss
  + lambda4 * PhysicsResidual
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def lat_weights(nlat: int, device=None) -> torch.Tensor:
    lat_deg = torch.linspace(-90, 90, nlat, device=device)
    w = torch.cos(torch.deg2rad(lat_deg))
    return (w / w.mean()).clamp(min=1e-3)


class LatWeightedRMSE(nn.Module):
    """Weights each latitude row by cos(lat) to correct for grid-area
    distortion (equirectangular grids over-represent the poles)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B, C, H, W)
        w = lat_weights(pred.shape[-2], device=pred.device).view(1, 1, -1, 1)
        se = (pred - target) ** 2
        return torch.sqrt((se * w).mean() + 1e-12)


class SpectralBandpassLoss(nn.Module):
    """Penalizes energy loss at high wavenumbers (models tend to blur out
    small-scale structure under plain RMSE) by comparing radially-binned
    2D FFT power spectra of prediction vs. target."""

    def __init__(self, n_bands: int = 8):
        super().__init__()
        self.n_bands = n_bands

    def _radial_power(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft2(x, norm="ortho")
        power = (xf.real ** 2 + xf.imag ** 2)
        h, w = power.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
        )
        r = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
        r_max = r.max()
        bands = (r / r_max * (self.n_bands - 1)).long().clamp(0, self.n_bands - 1)
        out = torch.zeros(*power.shape[:-2], self.n_bands, device=x.device)
        for b in range(self.n_bands):
            mask = (bands == b)
            out[..., b] = power[..., mask].mean(dim=-1) if mask.any() else 0.0
        return out

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._radial_power(pred), self._radial_power(target))


class PerceptualLoss(nn.Module):
    """Pretrained-CNN feature-space loss (LPIPS-style), adapted for
    multi-channel meteorological fields by running a small frozen 2D CNN
    per-variable-group rather than assuming 3-channel RGB input."""

    def __init__(self, feature_extractor: nn.Module | None = None):
        super().__init__()
        self.features = feature_extractor or self._default_extractor()
        for p in self.features.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _default_extractor() -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        pred_flat = pred.reshape(b * c, 1, h, w)
        target_flat = target.reshape(b * c, 1, h, w)
        return F.l1_loss(self.features(pred_flat), self.features(target_flat))


class PhysicsResidual(nn.Module):
    """Soft penalty for violating hydrostatic balance + mass continuity.
    `var_index` maps variable name -> channel index in the model's output
    tensor; wire this up to the true channel layout from
    app/data/zarr_pipeline.GridFeatureSpec / the surface+pressure-level
    variable list once finalized."""

    def __init__(self, u_idx: int, v_idx: int):
        super().__init__()
        self.u_idx, self.v_idx = u_idx, v_idx

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        u, v = pred[:, self.u_idx], pred[:, self.v_idx]
        du_dx = torch.gradient(u, dim=-1)[0]
        dv_dy = torch.gradient(v, dim=-2)[0]
        divergence = du_dx + dv_dy
        return divergence.pow(2).mean()


class GlobalEngineLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.1, lambda3=0.05, lambda4=0.05, u_idx=0, v_idx=1):
        super().__init__()
        self.rmse = LatWeightedRMSE()
        self.spectral = SpectralBandpassLoss()
        self.perceptual = PerceptualLoss()
        self.physics = PhysicsResidual(u_idx, v_idx)
        self.lambdas = (lambda1, lambda2, lambda3, lambda4)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l1, l2, l3, l4 = self.lambdas
        rmse = self.rmse(pred, target)
        spec = self.spectral(pred, target)
        perc = self.perceptual(pred, target)
        phys = self.physics(pred)
        total = l1 * rmse + l2 * spec + l3 * perc + l4 * phys
        return {"total": total, "rmse": rmse, "spectral": spec, "perceptual": perc, "physics": phys}
