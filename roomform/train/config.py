"""Training configuration. Experiments vary THESE fields (via
research/<exp>/config.yaml overlays) — never forked code."""

from __future__ import annotations

from pydantic import BaseModel

from roomform.model.config import ModelConfig


class TrainConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    epochs: int = 60
    batch_size: int = 2  # whole-room grids are large
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_min_factor: float = 0.05  # cosine floor as a fraction of lr
    node_pos_weight: float = 4.0
    edge_pos_weight: float = 4.0
    edge_loss_weight: float = 1.0
    seed: int = 7
    val_every: int = 3
    grad_clip: float = 1.0
