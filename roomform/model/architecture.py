"""Composable stages of the patch-graph ConvFormer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from roomform.model.blocks import (
    AttentionBlock,
    ConvRefine,
    ResidualBlock,
    _metric_pos,
)

ValidityLevels = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


def pad_to_factor(grid: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Pad a spatial grid so each axis is divisible by ``factor``."""
    pads = [(factor - size % factor) % factor for size in grid.shape[-3:]]
    return F.pad(grid, (0, pads[2], 0, pads[1], 0, pads[0]))


def downsample_validity(
    valid: torch.Tensor | None, factor: int
) -> torch.Tensor | None:
    if valid is None:
        return None
    return F.max_pool3d(valid.float(), factor, stride=factor) > 0


@dataclass(frozen=True)
class PreparedInputs:
    """Padded evidence streams consumed by the independent model stages."""

    backbone: torch.Tensor
    openings: torch.Tensor
    refinement: torch.Tensor | None
    valid: ValidityLevels


class EvidenceAdapter:
    """Split observable evidence into backbone and optional head inputs."""

    def __init__(
        self,
        backbone_channels: int,
        opening_visibility: bool,
        isolated_refinement: bool,
    ) -> None:
        self.backbone_channels = backbone_channels
        self.opening_visibility = opening_visibility
        self.isolated_refinement = isolated_refinement

    def __call__(
        self,
        grid: torch.Tensor,
        valid_voxels: torch.Tensor | None,
    ) -> PreparedInputs:
        backbone = grid[:, : self.backbone_channels]
        openings = (
            torch.cat((backbone, grid[:, -1:]), 1)
            if self.opening_visibility
            else backbone
        )
        refinement = None
        if self.isolated_refinement:
            start = self.backbone_channels
            refinement = pad_to_factor(grid[:, start : start + 3])

        valid0 = (
            pad_to_factor(valid_voxels.float())
            if valid_voxels is not None
            else None
        )
        valid = (
            valid0,
            downsample_validity(valid0, 2),
            downsample_validity(valid0, 4),
            downsample_validity(valid0, 8),
        )
        return PreparedInputs(
            backbone=pad_to_factor(backbone),
            openings=pad_to_factor(openings),
            refinement=refinement,
            valid=valid,
        )


class FeatureEncoder(nn.Module):
    """Encode native voxel evidence into a four-level feature pyramid."""

    def __init__(
        self, in_channels: int, channels: tuple[int, int, int, int]
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stem = nn.Conv3d(in_channels, c0, 3, padding=1)
        self.enc0 = ResidualBlock(c0)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.enc1 = ResidualBlock(c1)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(c2)
        self.down3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        valid: ValidityLevels,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid0, valid1, valid2, _ = valid
        x0: torch.Tensor = self.enc0(self.stem(x), valid0)
        x1: torch.Tensor = self.enc1(self.down1(x0), valid1)
        x2: torch.Tensor = self.enc2(self.down2(x1), valid2)
        x3: torch.Tensor = self.down3(x2)
        return x0, x1, x2, x3


class GlobalLocalRefiner(nn.Module):
    """Alternate global token reasoning with local 3-D refinement."""

    def __init__(self, channels: int, depth: int, heads: int) -> None:
        super().__init__()
        split = depth // 2
        self.attention_in = nn.ModuleList(
            AttentionBlock(channels, heads) for _ in range(split)
        )
        self.middle_conv = ConvRefine(channels)
        self.attention_out = nn.ModuleList(
            AttentionBlock(channels, heads) for _ in range(depth - split)
        )

    @staticmethod
    def _to_tokens(grid: torch.Tensor) -> torch.Tensor:
        return grid.flatten(2).transpose(1, 2)

    @staticmethod
    def _to_grid(
        tokens: torch.Tensor, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(len(tokens), -1, *shape)

    def forward(
        self, grid: torch.Tensor, valid: torch.Tensor | None
    ) -> torch.Tensor:
        shape = tuple(grid.shape[-3:])
        tokens = self._to_tokens(grid)
        valid_tokens = valid[:, 0].flatten(1) if valid is not None else None
        positions = _metric_pos(
            shape, tokens.shape[-1], tokens.device, tokens.dtype
        )
        tokens = tokens + positions[None]
        for block in self.attention_in:
            tokens = block(tokens, valid_tokens)

        grid = self.middle_conv(self._to_grid(tokens, shape), valid)
        tokens = self._to_tokens(grid)
        for block in self.attention_out:
            tokens = block(tokens, valid_tokens)
        return self._to_grid(tokens, shape)


class FeatureDecoder(nn.Module):
    """Restore native voxel resolution while merging encoder evidence."""

    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = ResidualBlock(c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = ResidualBlock(c1)
        self.up0 = nn.ConvTranspose3d(c1, c0, 2, stride=2)
        self.dec0 = ResidualBlock(c0)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        valid: ValidityLevels,
    ) -> torch.Tensor:
        x0, x1, x2 = skips
        valid0, valid1, valid2, _ = valid
        y2: torch.Tensor = self.dec2(self.up2(bottleneck) + x2, valid2)
        y1: torch.Tensor = self.dec1(self.up1(y2) + x1, valid1)
        return self.dec0(self.up0(y1) + x0, valid0)


class PatchGraphHeads(nn.Module):
    """Project shared voxel features into the patch-graph contracts."""

    def __init__(
        self,
        channels: int,
        edge_channels: int,
        opening_evidence_channels: int,
        predict_offsets: bool,
        predict_openings: bool,
        isolated_refinement: bool,
    ) -> None:
        super().__init__()
        self.node_head = nn.Conv3d(channels, 3, 1)
        self.edge_head = nn.Conv3d(channels, edge_channels, 1)
        self.offset_head = (
            nn.Conv3d(channels, 3, 1) if predict_offsets else None
        )
        self.refinement_stem = (
            nn.Conv3d(3, channels, 3, padding=1, bias=False)
            if predict_offsets and isolated_refinement
            else None
        )
        self.opening_refine = (
            nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.GELU(),
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.GELU(),
            )
            if predict_openings
            else None
        )
        self.opening_head = (
            nn.Conv3d(channels, 2, 1) if predict_openings else None
        )
        self.opening_evidence_stem = (
            nn.Conv3d(
                opening_evidence_channels,
                channels,
                3,
                padding=1,
                bias=False,
            )
            if predict_openings
            else None
        )
        self._initialize_optional_heads()

    def _initialize_optional_heads(self) -> None:
        if self.offset_head is not None:
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)
        if self.refinement_stem is not None:
            nn.init.zeros_(self.refinement_stem.weight)
        if self.opening_head is not None:
            nn.init.zeros_(self.opening_head.weight)
            nn.init.constant_(self.opening_head.bias, -4.0)
            assert self.opening_evidence_stem is not None
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

    def forward(
        self,
        features: torch.Tensor,
        inputs: PreparedInputs,
        return_features: bool,
    ) -> tuple[torch.Tensor, ...]:
        result = [self.node_head(features), self.edge_head(features)]
        if self.offset_head is not None:
            offset_features = features
            if inputs.refinement is not None:
                assert self.refinement_stem is not None
                offset_features = offset_features + self.refinement_stem(
                    inputs.refinement
                )
            result.append(0.5 * torch.tanh(self.offset_head(offset_features)))
        if self.opening_head is not None:
            result.append(self.opening_logits(features, inputs.openings))
        if return_features:
            result.append(features)
        return tuple(result)
