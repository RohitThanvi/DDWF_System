"""
Shared building blocks used by both the global engine (SFNO) and the
downscaling diffusion U-Net.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: cond -> (scale, shift) applied to x."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, ...)  cond: (B, cond_dim)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        extra_dims = x.dim() - 2
        shape = (scale.shape[0], scale.shape[1]) + (1,) * extra_dims
        return x * (1 + scale.view(*shape)) + shift.view(*shape)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard diffusion timestep embedding (as in DDPM / EDM)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """Conv residual block with GroupNorm + FiLM conditioning, used in the U-Net."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    """Standard multi-head self-attention over flattened spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x).reshape(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        out, _ = self.attn(h_, h_, h_)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class CrossAttention2D(nn.Module):
    """Attends spatial query tokens to an external context sequence
    (used for the downscaler attending to coarse-forecast tokens)."""

    def __init__(self, channels: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.to_context = nn.Linear(context_dim, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        kv = self.to_context(context)
        out, _ = self.attn(q, kv, kv)
        return x + out.transpose(1, 2).reshape(b, c, h, w)
