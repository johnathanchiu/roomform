"""Shared building blocks for the patch-graph ConvFormer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

VOXEL_M = 0.08
PATCH = 8  # 8 * 8 cm = 64 cm global tokens


def _metric_pos(
    shape: tuple[int, int, int], dim: int, device, dtype
) -> torch.Tensor:
    """Deterministic 3-D sinusoidal positions at token centers, in meters."""
    axes = [
        (torch.arange(n, device=device, dtype=torch.float32) + 0.5)
        * (PATCH * VOXEL_M)
        for n in shape
    ]
    xyz = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    nfreq = max(dim // 6, 1)
    # ``logspace.out`` is unavailable on MPS. This is mathematically the same
    # geometric sequence and runs natively on CPU, CUDA, and MPS.
    wavelengths = torch.exp(
        torch.linspace(math.log(40.0), math.log(0.32), nfreq, device=device)
    )
    phase = xyz[:, :, None] * (2.0 * torch.pi / wavelengths)
    pos = torch.cat((phase.sin(), phase.cos()), -1).reshape(len(xyz), -1)
    return F.pad(pos, (0, max(dim - pos.shape[1], 0)))[:, :dim].to(dtype)


class ChannelNorm3d(nn.Module):
    """LayerNorm over channels only; invariant to padded spatial extent."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 4, 1)
        y = F.layer_norm(y, (x.shape[1],), self.weight, self.bias)
        return y.permute(0, 4, 1, 2, 3)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = ChannelNorm3d(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = ChannelNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        # explicit annotation: nn.Module.__call__ returns Any in torch's
        # stubs, which strips IDE type info from chained tensor ops
        y: torch.Tensor = self.conv1(F.gelu(self.norm1(x)))
        if valid is not None:
            y = y * valid.to(y.dtype)
        y = self.conv2(F.gelu(self.norm2(y)))
        out = x + y
        return out * valid.to(out.dtype) if valid is not None else out


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(
        self, x: torch.Tensor, valid_tokens: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, d = x.shape
        qkv: torch.Tensor = self.qkv(self.norm1(x))
        q, k, v = qkv.reshape(b, n, 3, self.heads, d // self.heads).permute(
            2, 0, 3, 1, 4
        )
        # A key mask is enough for attention; invalid query rows are cleared
        # after both residual branches.  This avoids all-masked softmax rows
        # while guaranteeing padded tokens cannot become context.
        mask = (
            valid_tokens[:, None, None, :]
            if valid_tokens is not None
            else None
        )
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, d))
        x = x + self.mlp(self.norm2(x))
        if valid_tokens is not None:
            x = x * valid_tokens[..., None].to(x.dtype)
        return x


class ConvRefine(nn.Module):
    """Residual local cleanup after global attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ChannelNorm3d(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            ChannelNorm3d(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, valid_grid: torch.Tensor | None = None
    ) -> torch.Tensor:
        y: torch.Tensor = x
        for layer in self.body:
            y = layer(y)
            # A two-layer convolution can otherwise write into padding after
            # conv1 and read that value back into the scene at conv2.
            if valid_grid is not None and isinstance(layer, nn.Conv3d):
                y = y * valid_grid.to(y.dtype)
        return x + y
