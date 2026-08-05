"""Model configuration — the shape contract for the patch-graph net.

The numbers here mirror docs/data-contracts.md: observable input
channels in, node/edge probability grids out. The architecture reads
everything from this config so checkpoints can pin their exact shape
(a checkpoint npz/pt carries its ModelConfig as JSON metadata).
"""
from __future__ import annotations

from pydantic import BaseModel

N_EDGE_OFFSETS = 13   # forward half of the 26-neighborhood (contract)
N_NODE_CLASSES = 3    # wall, floor, ceiling


class ModelConfig(BaseModel):
    vox_m: float = 0.08
    in_channels: int = 6          # occ, gray, |nrm| xyz, log-density
    stem_dim: int = 64            # local conv feature width
    stem_depth: int = 2           # conv blocks before attention
    attn_dim: int = 384           # global attention width
    attn_depth: int = 8
    attn_heads: int = 6
    attn_pool: int = 4            # spatial downsample factor into attention
    head_dim: int = 128           # decoder width
    node_classes: int = N_NODE_CLASSES
    edge_offsets: int = N_EDGE_OFFSETS
