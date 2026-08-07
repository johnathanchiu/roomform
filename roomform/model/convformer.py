"""Patch-graph model assembled from explicit architectural stages."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from roomform.model.architecture import (
    EvidenceAdapter,
    FeatureDecoder,
    FeatureEncoder,
    GlobalLocalRefiner,
    PatchGraphHeads,
)

EDGE_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
    and (dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0))
)

LEGACY_STAGE_BY_ROOT = {
    "stem": "encoder",
    "enc0": "encoder",
    "down1": "encoder",
    "enc1": "encoder",
    "down2": "encoder",
    "enc2": "encoder",
    "down3": "encoder",
    "attention_in": "refiner",
    "middle_conv": "refiner",
    "attention_out": "refiner",
    "up2": "decoder",
    "dec2": "decoder",
    "up1": "decoder",
    "dec1": "decoder",
    "up0": "decoder",
    "dec0": "decoder",
    "node_head": "heads",
    "edge_head": "heads",
    "offset_head": "heads",
    "refinement_stem": "heads",
    "opening_refine": "heads",
    "opening_head": "heads",
    "opening_evidence_stem": "heads",
}


class PatchGraphConvFormer(nn.Module):
    """Raw 8 cm evidence -> semantic surface nodes and learned graph edges."""

    def __init__(
        self,
        in_ch: int = 6,
        base: int = 48,
        depth: int = 6,
        heads: int = 8,
        predict_offsets: bool = False,
        predict_openings: bool = False,
        opening_visibility: bool = False,
        isolated_refinement: bool = False,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least two")
        c0, c1, c2, c3 = base, 2 * base, 4 * base, 8 * base
        if c3 % heads:
            raise ValueError("bottleneck width must be divisible by heads")
        visibility_channels = 1 if opening_visibility else 0
        offset_channels = 3 if predict_offsets and isolated_refinement else 0
        backbone_channels = in_ch - offset_channels - visibility_channels
        if backbone_channels <= 0:
            raise ValueError("no backbone evidence channels remain")
        channels = (c0, c1, c2, c3)
        self.inputs = EvidenceAdapter(
            backbone_channels=backbone_channels,
            opening_visibility=opening_visibility,
            isolated_refinement=predict_offsets and isolated_refinement,
        )
        self.encoder = FeatureEncoder(backbone_channels, channels)
        self.refiner = GlobalLocalRefiner(c3, depth, heads)
        self.decoder = FeatureDecoder(channels)
        self.heads = PatchGraphHeads(
            channels=c0,
            edge_channels=len(EDGE_OFFSETS),
            opening_evidence_channels=(
                backbone_channels + visibility_channels
            ),
            predict_offsets=predict_offsets,
            predict_openings=predict_openings,
            isolated_refinement=isolated_refinement,
        )

    def opening_logits(
        self, features: torch.Tensor, evidence: torch.Tensor
    ) -> torch.Tensor:
        return self.heads.opening_logits(features, evidence)

    def forward(
        self,
        grid: torch.Tensor,
        valid_voxels: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        inputs = self.inputs(grid, valid_voxels)
        x0, x1, x2, bottleneck = self.encoder(inputs.backbone, inputs.valid)
        bottleneck = self.refiner(bottleneck, inputs.valid[3])
        features = self.decoder(bottleneck, (x0, x1, x2), inputs.valid)
        return self.heads(features, inputs, return_features)

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load both the released flat checkpoints and staged checkpoints."""
        remapped = {}
        for key, value in state_dict.items():
            root = key.split(".", 1)[0]
            stage = LEGACY_STAGE_BY_ROOT.get(root)
            remapped[f"{stage}.{key}" if stage else key] = value
        return super().load_state_dict(remapped, strict=strict, assign=assign)
