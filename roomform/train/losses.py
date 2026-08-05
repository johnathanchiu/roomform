"""Patch-graph training losses.

Node loss: per-class BCE on wall/floor/ceiling probabilities.
Edge loss: BCE on the 13 forward-edge connectivity channels, applied
only where at least one endpoint is structure GT (connectivity of
empty-empty pairs carries no signal and would swamp the loss).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def node_loss(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: float
) -> torch.Tensor:
    """logits/target [B,3,X,Y,Z]; full-grid BCE with positive weighting."""
    pw = torch.full((), pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits, target.float(), pos_weight=pw
    )


def edge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    node_target: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """logits/target [B,13,X,Y,Z]; masked to cells where any node class
    is GT-positive (the graph only exists on structure)."""
    mask = (node_target.any(dim=1, keepdim=True)).float()
    pw = torch.full((), pos_weight, device=logits.device)
    per_voxel = F.binary_cross_entropy_with_logits(
        logits, target.float(), pos_weight=pw, reduction="none"
    )
    denom = mask.sum().clamp(min=1.0) * logits.shape[1]
    return (per_voxel * mask).sum() / denom


def total_loss(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    node_target: torch.Tensor,
    edge_target: torch.Tensor,
    *,
    node_pos_weight: float,
    edge_pos_weight: float,
    edge_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ln = node_loss(node_logits, node_target, node_pos_weight)
    le = edge_loss(edge_logits, edge_target, node_target, edge_pos_weight)
    loss = ln + edge_weight * le
    return loss, {"node": float(ln), "edge": float(le)}
