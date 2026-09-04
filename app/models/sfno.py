"""
Global Planetary Engine — Spherical Fourier Neural Operator (SFNO).

Real spherical-harmonic transforms (SHT) require `torch-harmonics`
(training-only dependency, see requirements-train.txt) because they need
Legendre-polynomial quadrature weights that are expensive to bundle into
a lightweight inference image. This module defines:

  1. A `SFNOBlock` that is transform-agnostic (works with any SHT/iSHT pair).
  2. A `TorchHarmonicsSHT` adapter used at *training* time.
  3. A `FFTFallbackSHT` adapter used at *inference* time on CPU/edge boxes
     where torch-harmonics + its compiled extensions aren't worth shipping —
     it approximates the spherical transform with a 2D real FFT, which is
     the standard cheap substitute at fixed 0.25 deg / 721x1440 resolution
     (pole distortion re-enters at extreme latitudes, which is why training
     always uses the true SHT; only the served, already-trained weights use
     the FFT fallback for latency).

Swap between them with `sht_backend="harmonics" | "fft"` in SFNO.__init__.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.models.layers import FiLM


class FFTFallbackSHT(nn.Module):
    """Cheap real-FFT stand-in for the spherical harmonic transform pair.
    Used for served inference only — see module docstring."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(x, norm="ortho")

    def inverse(self, x: torch.Tensor, s: tuple[int, int]) -> torch.Tensor:
        return torch.fft.irfft2(x, s=s, norm="ortho")


class TorchHarmonicsSHT(nn.Module):
    """True spherical harmonic transform pair, built lazily so `torch-harmonics`
    stays a training-only import (not required by the serving image)."""

    def __init__(self, nlat: int, nlon: int, l_max: int):
        super().__init__()
        import torch_harmonics as th  # noqa: local import, training-only dep

        self.sht = th.RealSHT(nlat, nlon, lmax=l_max, mmax=l_max, grid="equiangular")
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=l_max, mmax=l_max, grid="equiangular")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sht(x)

    def inverse(self, x: torch.Tensor, s: tuple[int, int]) -> torch.Tensor:
        return self.isht(x)


class SpectralConv(nn.Module):
    """Depthwise (per-channel) learnable complex weighting of the
    (already-transformed) spectral coefficients, truncated to l_max.

    A full dense (in_channels, out_channels, modes, modes) weight — the
    textbook single-mode FNO formulation — is only tractable for small
    `modes` (~12-32). At SFNO's l_max=180 that tensor is tens of GB per
    block, which is why real large-scale spherical FNOs go depthwise here
    and push all channel-mixing into the surrounding 1x1 convs (see
    `SFNOBlock.spatial`) instead of the spectral layer itself.
    """

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / channels
        self.weight = nn.Parameter(
            scale * torch.randn(channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x_ft: torch.Tensor) -> torch.Tensor:
        # x_ft: (B, C, M1, M2) complex
        m1 = min(self.modes, x_ft.shape[-2])
        m2 = min(self.modes, x_ft.shape[-1])
        out = torch.zeros_like(x_ft)
        out[:, :, :m1, :m2] = x_ft[:, :, :m1, :m2] * self.weight[:, :m1, :m2].unsqueeze(0)
        return out


class SFNOBlock(nn.Module):
    """[SpectralConv(l_max) -> GroupNorm -> GELU -> SpatialConv(1x1) -> residual],
    with FiLM injection of the long-term GHG/CO2 conditioning scalar."""

    def __init__(self, channels: int, modes: int, cond_dim: int):
        super().__init__()
        self.spectral = SpectralConv(channels, modes)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.spatial = nn.Conv2d(channels, channels, kernel_size=1)
        self.film = FiLM(cond_dim, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, x_ft: torch.Tensor, cond: torch.Tensor,
                sht: "FFTFallbackSHT | TorchHarmonicsSHT", spatial_shape: tuple[int, int]):
        spec_out = self.spectral(x_ft)
        spatial_from_spec = sht.inverse(spec_out, s=spatial_shape)
        h = self.norm(spatial_from_spec.real if spatial_from_spec.is_complex() else spatial_from_spec)
        h = self.act(h)
        h = self.spatial(h)
        h = self.film(h, cond)
        out = x + h
        return out, sht.forward(out)


class SFNO(nn.Module):
    """
    Input:  (B, C_in=86, H=721, W=1440)  pressure-level + surface vars
    Output: (B, C_out=86, H=721, W=1440) next-6h-step prediction (same var set)

    `cond_dim` conditioning vector carries the GHG/CO2 anomaly scalar (and
    can be extended with ENSO/MJO indices) through FiLM at every block.
    """

    def __init__(
        self,
        in_channels: int = 86,
        out_channels: int = 86,
        hidden_channels: int = 256,
        n_blocks: int = 12,
        l_max: int = 180,
        cond_dim: int = 8,
        sht_backend: str = "fft",
        nlat: int = 721,
        nlon: int = 1440,
    ):
        super().__init__()
        self.spatial_shape = (nlat, nlon)
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [SFNOBlock(hidden_channels, l_max, cond_dim) for _ in range(n_blocks)]
        )
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        if sht_backend == "harmonics":
            self.sht = TorchHarmonicsSHT(nlat, nlon, l_max)
        else:
            self.sht = FFTFallbackSHT()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W); cond: (B, cond_dim) e.g. [ghg_anomaly, oni, rmm1, rmm2, ...]"""
        cond = self.cond_mlp(cond)
        h = self.stem(x)
        h_ft = self.sht.forward(h)
        for block in self.blocks:
            h, h_ft = block(h, h_ft, cond, self.sht, self.spatial_shape)
        return self.head(h)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, cond: torch.Tensor, steps: int) -> torch.Tensor:
        """Autoregressive rollout: `steps` recursive 6h hops.
        Returns (B, steps, C, H, W). Used to produce the 30-day (120-step)
        trajectory cache — call once per forecast cycle, not per user query."""
        self.eval()
        traj = []
        state = x0
        for _ in range(steps):
            state = self.forward(state, cond)
            traj.append(state)
        return torch.stack(traj, dim=1)
