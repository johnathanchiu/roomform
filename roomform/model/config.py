"""Model configuration — the shape contract for the patch-graph net.

Field names match the ``config`` dict stored in training checkpoints
(``{"model": state_dict, "config": {...}}``), so a checkpoint's config
loads directly: ``ModelConfig(**ckpt["config"])``. Extra training-only
keys (lr, batch, init_checkpoint, ...) are ignored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

N_EDGE_OFFSETS = 13  # forward half of the 26-neighborhood (contract)
N_NODE_CLASSES = 3  # wall, floor, ceiling


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    vox_m: float = 0.08
    in_ch: int = 6  # occ, color, |nrm| xyz, log-density (+3 local offsets)
    color_mode: str = "grayscale"  # "grayscale" (1ch) or "rgb" (3ch)
    base: int = 48  # feature-pyramid width; global stage is 8 * base
    depth: int = 6  # attention blocks in the bottleneck
    heads: int = 8
    predict_offsets: bool = False  # sub-voxel surface refinement head
    predict_openings: bool = False  # door/window channels
    opening_visibility: bool = False  # extra visibility input channel
    isolated_refinement: bool = False  # offsets fed via separate stem

    def build(self):
        from roomform.model.convformer import PatchGraphConvFormer

        return PatchGraphConvFormer(
            in_ch=self.in_ch,
            base=self.base,
            depth=self.depth,
            heads=self.heads,
            predict_offsets=self.predict_offsets,
            predict_openings=self.predict_openings,
            opening_visibility=self.opening_visibility,
            isolated_refinement=self.isolated_refinement,
        )
