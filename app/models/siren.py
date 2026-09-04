"""
Coordinate-based implicit terrain field (SIREN: sinusoidal-activation MLP).

Query any (lat, lon) — including unmapped islands never seen in training —
and get back a terrain embedding derived from a continuous function fit to
DEM/LULC satellite rasters, instead of a lookup table. This is what makes
zero-shot AOI inference possible: no site-specific retraining, just a
forward pass through this MLP conditioned on the queried coordinate plus
whatever remote-sensing raster patch is available at that coordinate.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self._init_weights(in_f)

    def _init_weights(self, in_f: int) -> None:
        with torch.no_grad():
            bound = 1 / in_f if self.is_first else math.sqrt(6 / in_f) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class TerrainSIREN(nn.Module):
    """
    Input:  concat(sin(lat), cos(lat), sin(lon), cos(lon), raster_patch_features)
    Output: terrain embedding (elevation, slope, aspect, LULC-class logits)
            consumed as one of the conditioning streams by the downscaling U-Net.
    """

    def __init__(
        self,
        coord_dim: int = 4,
        raster_feat_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 5,
        out_dim: int = 64,
        omega_0: float = 30.0,
    ):
        super().__init__()
        in_dim = coord_dim + raster_feat_dim
        layers = [SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0)]
        for _ in range(n_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def encode_coords(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        lat_r, lon_r = torch.deg2rad(lat), torch.deg2rad(lon)
        return torch.stack(
            [torch.sin(lat_r), torch.cos(lat_r), torch.sin(lon_r), torch.cos(lon_r)], dim=-1
        )

    def forward(self, lat: torch.Tensor, lon: torch.Tensor, raster_feats: torch.Tensor) -> torch.Tensor:
        coords = self.encode_coords(lat, lon)
        x = torch.cat([coords, raster_feats], dim=-1)
        return self.out(self.net(x))
