"""
Satellite AOI Downscaling Head — conditional diffusion super-resolution
(DDPM/EDM-style), coarse (~25km global forecast) -> fine (100m-1km AOI).

Conditioning stack (concatenated + cross-attended):
  - Coarse global forecast patch (bicubic-upsampled as init noise prior)
  - DEM/slope/aspect + LULC + LST from `TerrainSIREN` / satellite fusion
  - Diffusion timestep (sinusoidal -> MLP -> FiLM in every ResBlock)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.models.layers import CrossAttention2D, ResBlock, SelfAttention2D, SinusoidalTimeEmbedding


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, attn: bool):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.attn = SelfAttention2D(out_ch) if attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, cond):
        h = self.res(x, cond)
        h = self.attn(h) if not isinstance(self.attn, nn.Identity) else h
        return self.down(h), h  # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int, attn: bool):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + skip_ch, out_ch, cond_dim)
        self.attn = SelfAttention2D(out_ch) if attn else nn.Identity()

    def forward(self, x, skip, cond):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        h = self.res(h, cond)
        return self.attn(h) if not isinstance(self.attn, nn.Identity) else h


class DiffusionDownscaler(nn.Module):
    """
    Encoder: [ResBlock -> SelfAttn (at 32x32, 16x16)] x4, downsample /2 each
    Bottleneck: ResBlock -> CrossAttn(coarse_forecast_tokens) -> ResBlock
    Decoder: mirrored, skip connections
    """

    def __init__(
        self,
        in_channels: int = 8,          # target met vars being downscaled
        cond_raster_channels: int = 8, # DEM/slope/aspect/LULC-onehot/LST
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        context_dim: int = 64,         # TerrainSIREN / coarse-token dim
        time_dim: int = 256,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )

        in_ch = in_channels + cond_raster_channels  # noisy target concat with terrain/LST raster
        self.stem = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        chs = [base_channels * m for m in channel_mults]
        attn_at = {len(chs) - 2, len(chs) - 1}  # attention at the two coarsest encoder scales (32x32, 16x16 in the reference config)

        self.down_blocks = nn.ModuleList()
        prev = base_channels
        for i, ch in enumerate(chs):
            self.down_blocks.append(DownBlock(prev, ch, time_dim, attn=i in attn_at))
            prev = ch

        self.bottleneck_1 = ResBlock(prev, prev, time_dim)
        self.bottleneck_cross_attn = CrossAttention2D(prev, context_dim)
        self.bottleneck_2 = ResBlock(prev, prev, time_dim)

        self.up_blocks = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            skip_ch = ch
            out_ch = chs[i - 1] if i > 0 else base_channels
            self.up_blocks.append(UpBlock(prev, skip_ch, out_ch, time_dim, attn=i in attn_at))
            prev = out_ch

        self.head = nn.Sequential(
            nn.GroupNorm(min(8, prev), prev), nn.SiLU(),
            nn.Conv2d(prev, in_channels, 3, padding=1),
        )

    def forward(
        self,
        noisy_target: torch.Tensor,      # (B, in_channels, H, W) — x_t
        terrain_raster: torch.Tensor,    # (B, cond_raster_channels, H, W) — DEM/LULC/LST
        coarse_forecast_tokens: torch.Tensor,  # (B, N_tokens, context_dim)
        timestep: torch.Tensor,          # (B,)
    ) -> torch.Tensor:
        cond = self.time_embed(timestep)
        x = torch.cat([noisy_target, terrain_raster], dim=1)
        h = self.stem(x)

        skips = []
        for block in self.down_blocks:
            h, skip = block(h, cond)
            skips.append(skip)

        h = self.bottleneck_1(h, cond)
        h = self.bottleneck_cross_attn(h, coarse_forecast_tokens)
        h = self.bottleneck_2(h, cond)

        for block in self.up_blocks:
            h = block(h, skips.pop(), cond)

        return self.head(h)  # predicted noise (epsilon) or v-prediction, per training config
