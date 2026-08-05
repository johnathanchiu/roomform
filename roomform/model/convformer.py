"""Global ConvFormer with a dense local decoder for learned surface graphs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from roomform.model.blocks import (
    AttentionBlock,
    ConvRefine,
    ResidualBlock,
    _metric_pos,
)

EDGE_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
    and (dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0))
)


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
        predict_micro: bool = False,
        opening_visibility: bool = False,
        isolated_refinement: bool = False,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least two")
        c0, c1, c2, c3 = base, 2 * base, 4 * base, 8 * base
        if c3 % heads:
            raise ValueError("bottleneck width must be divisible by heads")
        self.isolated_refinement = isolated_refinement
        self.opening_visibility = opening_visibility
        visibility_channels = 1 if opening_visibility else 0
        offset_channels = 3 if predict_offsets and isolated_refinement else 0
        stem_channels = in_ch - offset_channels - visibility_channels
        if stem_channels <= 0:
            raise ValueError("no backbone evidence channels remain")
        self.backbone_in_ch = stem_channels
        self.stem = nn.Conv3d(stem_channels, c0, 3, padding=1)
        self.refinement_stem = (
            nn.Conv3d(3, c0, 3, padding=1, bias=False)
            if predict_offsets and isolated_refinement
            else None
        )
        self.enc0 = ResidualBlock(c0)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.enc1 = ResidualBlock(c1)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(c2)
        self.down3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        split = depth // 2
        self.attention_in = nn.ModuleList(
            AttentionBlock(c3, heads) for _ in range(split)
        )
        self.middle_conv = ConvRefine(c3)
        self.attention_out = nn.ModuleList(
            AttentionBlock(c3, heads) for _ in range(depth - split)
        )
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = ResidualBlock(c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = ResidualBlock(c1)
        self.up0 = nn.ConvTranspose3d(c1, c0, 2, stride=2)
        self.dec0 = ResidualBlock(c0)
        self.node_head = nn.Conv3d(c0, 3, 1)
        self.edge_head = nn.Conv3d(c0, len(EDGE_OFFSETS), 1)
        self.offset_head = nn.Conv3d(c0, 3, 1) if predict_offsets else None
        # Door/window probabilities are independent of the structural node
        # softmax: an opening is evidence that subtracts from latent wall
        # support, not a fourth kind of surface node.
        self.opening_refine = (
            nn.Sequential(
                nn.Conv3d(c0, c0, 3, padding=1),
                nn.GELU(),
                nn.Conv3d(c0, c0, 3, padding=1),
                nn.GELU(),
            )
            if predict_openings
            else None
        )
        self.opening_head = nn.Conv3d(c0, 2, 1) if predict_openings else None
        self.opening_evidence_stem = (
            nn.Conv3d(
                stem_channels + visibility_channels,
                c0,
                3,
                padding=1,
                bias=False,
            )
            if predict_openings
            else None
        )
        self.micro_factor = 4
        self.micro_classes = 5  # wall, floor, ceiling, door, window
        self.micro_head = (
            nn.Sequential(
                nn.Linear(c0, 2 * c0),
                nn.GELU(),
                nn.Linear(2 * c0, self.micro_classes * self.micro_factor**3),
            )
            if predict_micro
            else None
        )
        if self.offset_head is not None:
            # A newly enabled refinement head must reproduce the established
            # voxel-center baseline before it has learned any displacement.
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)
        if self.refinement_stem is not None:
            nn.init.zeros_(self.refinement_stem.weight)
        if self.opening_head is not None:
            nn.init.zeros_(self.opening_head.weight)
            nn.init.constant_(self.opening_head.bias, -4.0)
            nn.init.zeros_(self.opening_evidence_stem.weight)

    def opening_logits(
        self, features: torch.Tensor, evidence: torch.Tensor
    ) -> torch.Tensor:
        if (
            self.opening_head is None
            or self.opening_refine is None
            or self.opening_evidence_stem is None
        ):
            raise RuntimeError("opening prediction is not enabled")
        local = self.opening_evidence_stem(evidence)
        return self.opening_head(self.opening_refine(features + local))

    def decode_micro(
        self, features: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Decode selected 8 cm cells into semantic 4^3 grids at 2 cm.

        `indices` is [N,4] in (batch,x,y,z) order. Keeping this sparse avoids
        materializing hundreds of fine-grid channels over the entire room.
        """
        if self.micro_head is None:
            raise RuntimeError("micro refinement is not enabled")
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError("indices must have shape [N,4]")
        dense = features.permute(0, 2, 3, 4, 1)
        b, x, y, z = indices.long().unbind(1)
        selected = dense[b, x, y, z]
        logits = self.micro_head(selected)
        return logits.reshape(
            -1,
            self.micro_classes,
            self.micro_factor,
            self.micro_factor,
            self.micro_factor,
        )

    @staticmethod
    def _pad(grid: torch.Tensor) -> torch.Tensor:
        pads = [(8 - size % 8) % 8 for size in grid.shape[-3:]]
        return F.pad(grid, (0, pads[2], 0, pads[1], 0, pads[0]))

    @staticmethod
    def _mask(valid: torch.Tensor | None, factor: int):
        if valid is None:
            return None
        return F.max_pool3d(valid.float(), factor, stride=factor) > 0

    @staticmethod
    def _run_block(
        x: torch.Tensor, block: ResidualBlock, valid: torch.Tensor | None
    ) -> torch.Tensor:
        return block(x, valid)

    def forward(
        self,
        grid: torch.Tensor,
        valid_voxels: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        refinement_input = None
        backbone_input = grid[:, : self.backbone_in_ch]
        if self.opening_visibility:
            opening_input = torch.cat((backbone_input, grid[:, -1:]), 1)
        else:
            opening_input = backbone_input
        opening_input = self._pad(opening_input)
        if self.refinement_stem is not None:
            start = self.backbone_in_ch
            refinement_input = self._pad(grid[:, start : start + 3])
        x = self._pad(backbone_input)
        valid0 = (
            self._pad(valid_voxels.float())
            if valid_voxels is not None
            else None
        )
        valid1, valid2, valid3 = (
            self._mask(valid0, factor) for factor in (2, 4, 8)
        )
        x0 = self._run_block(self.stem(x), self.enc0, valid0)
        x1 = self._run_block(self.down1(x0), self.enc1, valid1)
        x2 = self._run_block(self.down2(x1), self.enc2, valid2)
        x3 = self.down3(x2)
        shape = tuple(x3.shape[-3:])
        tokens = x3.flatten(2).transpose(1, 2)
        valid_tokens = valid3[:, 0].flatten(1) if valid3 is not None else None
        tokens = (
            tokens
            + _metric_pos(
                shape, tokens.shape[-1], tokens.device, tokens.dtype
            )[None]
        )
        for block in self.attention_in:
            tokens = block(tokens, valid_tokens)
        x3 = tokens.transpose(1, 2).reshape(len(tokens), -1, *shape)
        x3 = self.middle_conv(x3, valid3)
        tokens = x3.flatten(2).transpose(1, 2)
        for block in self.attention_out:
            tokens = block(tokens, valid_tokens)
        x3 = tokens.transpose(1, 2).reshape(len(tokens), -1, *shape)
        y2 = self._run_block(self.up2(x3) + x2, self.dec2, valid2)
        y1 = self._run_block(self.up1(y2) + x1, self.dec1, valid1)
        y0 = self._run_block(self.up0(y1) + x0, self.dec0, valid0)
        node, edge = self.node_head(y0), self.edge_head(y0)
        result = [node, edge]
        if self.offset_head is None:
            if self.opening_head is not None:
                result.append(self.opening_logits(y0, opening_input))
            if return_features:
                result.append(y0)
            return tuple(result)
        # Offset is expressed in voxel widths from the cell center. Keeping it
        # inside the cell makes the representation identifiable.
        offset_features = y0
        if refinement_input is not None:
            # This branch cannot perturb node/edge logits. It supplies precise
            # observed point placement while y0 supplies frozen global context.
            offset_features = offset_features + self.refinement_stem(
                refinement_input
            )
        offset = 0.5 * torch.tanh(self.offset_head(offset_features))
        result.append(offset)
        if self.opening_head is not None:
            result.append(self.opening_logits(y0, opening_input))
        if return_features:
            result.append(y0)
        return tuple(result)
