"""Patch-graph ConvFormer skeleton.

Architecture (per the settled design): local 3D convolutions capture
edges and surfaces -> global attention lets distant evidence interact
(wall segments on opposite sides of an occluder) -> convolutions and a
full-resolution decoder emit, per cell, three independent structure
probabilities (wall/floor/ceiling) and thirteen forward-edge
connectivity probabilities.

This file is the config-driven scaffold: shapes, heads, and forward
signature are contract-stable. The authoritative trained implementation
migrates in from the internal repo behind this interface; internals may
be replaced wholesale as long as ``forward`` keeps its signature and
``ModelConfig`` describes the checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from roomform.model.config import ModelConfig


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class PatchGraphConvFormer(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        stem = [conv_block(cfg.in_channels, cfg.stem_dim)]
        stem += [
            conv_block(cfg.stem_dim, cfg.stem_dim)
            for _ in range(cfg.stem_depth - 1)
        ]
        self.stem = nn.Sequential(*stem)
        self.to_tokens = nn.Conv3d(
            cfg.stem_dim, cfg.attn_dim, cfg.attn_pool, stride=cfg.attn_pool
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.attn_dim,
            nhead=cfg.attn_heads,
            dim_feedforward=4 * cfg.attn_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.attn = nn.TransformerEncoder(layer, cfg.attn_depth)
        self.from_tokens = nn.ConvTranspose3d(
            cfg.attn_dim, cfg.stem_dim, cfg.attn_pool, stride=cfg.attn_pool
        )
        self.fuse = conv_block(2 * cfg.stem_dim, cfg.head_dim)
        self.node_head = nn.Conv3d(cfg.head_dim, cfg.node_classes, 1)
        self.edge_head = nn.Conv3d(cfg.head_dim, cfg.edge_offsets, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, in_channels, X, Y, Z] -> (node_logits [B,3,X,Y,Z],
        edge_logits [B,13,X,Y,Z]). Spatial dims must be divisible by
        attn_pool."""
        local = self.stem(x)
        t = self.to_tokens(local)
        tok_shape = t.shape[2:]
        seq = t.flatten(2).transpose(1, 2)
        seq = self.attn(seq)
        t = seq.transpose(1, 2).reshape(t.shape[0], -1, *tok_shape)
        glob = self.from_tokens(t)
        h = self.fuse(torch.cat([local, glob], 1))
        return self.node_head(h), self.edge_head(h)
