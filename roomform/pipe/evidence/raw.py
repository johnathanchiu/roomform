"""Raw point-cloud -> dense observable evidence channels.

Ported verbatim from the training pipeline so inference sees exactly
the channel construction the checkpoints were trained on: 2 cm dedup
(max 3 points per cell), 8 cm voxelization, and per-voxel means.
Channel order: occupancy, grayscale, |normal| xyz, log-density
(+ mean sub-voxel offset xyz when ``include_local_offsets``).
"""

from __future__ import annotations

import numpy as np

VOX = 0.08


def raw_point_evidence(
    raw,
    shape,
    color_mode: str = "grayscale",
    include_local_offsets: bool = False,
):
    """Voxelize observable point attributes into a dense 8 cm field."""
    if color_mode not in ("grayscale", "rgb"):
        raise ValueError("color_mode must be 'grayscale' or 'rgb'")
    keys = set(raw.files) if hasattr(raw, "files") else set(raw)
    points = raw["pts"].astype(np.float32)
    normals = raw["pts_normal"].astype(np.float32)
    colors = (
        raw["pts_color"].astype(np.float32) if "pts_color" in keys else None
    )
    if len(points):
        cells = np.floor(points / 0.02).astype(np.int64)
        order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
        sorted_cells = cells[order]
        new = np.ones(len(order), bool)
        new[1:] = (sorted_cells[1:] != sorted_cells[:-1]).any(1)
        rank = np.arange(len(order)) - np.maximum.accumulate(
            np.where(new, np.arange(len(order)), 0)
        )
        keep = order[rank < 3]
        points, normals = points[keep], normals[keep]
        colors = colors[keep] if colors is not None else None
    indices = np.floor(points / VOX).astype(np.int64)
    shape_array = np.asarray(shape, np.int64)
    valid = np.all((indices >= 0) & (indices < shape_array), axis=1)
    indices, normals = indices[valid], normals[valid]
    colors = colors[valid] if colors is not None else None
    nvoxels = int(np.prod(shape))
    flat = (
        np.ravel_multi_index(indices.T, shape)
        if len(indices)
        else np.empty(0, np.int64)
    )

    count = np.bincount(flat, minlength=nvoxels).astype(np.float32)
    occupied = count > 0
    normal_sum = np.zeros((3, nvoxels), np.float32)
    for axis in range(3):
        np.add.at(
            normal_sum[axis], flat, np.abs(normals[:, axis].astype(np.float32))
        )
    normal_sum /= np.maximum(count[None], 1.0)
    normal_sum /= np.maximum(
        np.linalg.norm(normal_sum, axis=0, keepdims=True), 1e-6
    )

    if colors is None:
        rgb = np.zeros((3, len(flat)), np.float32)
    else:
        rgb = colors.astype(np.float32)
        if rgb.size and float(np.nanmax(rgb)) > 1.5:
            rgb /= 255.0
        rgb = np.clip(rgb, 0.0, 1.0).T
    color_sum = np.zeros((3, nvoxels), np.float32)
    for channel in range(3):
        np.add.at(color_sum[channel], flat, rgb[channel])
    color_sum /= np.maximum(count[None], 1.0)

    local_sum = np.zeros((3, nvoxels), np.float32)
    if include_local_offsets:
        local = points[valid] / VOX - indices.astype(np.float32) - 0.5
        for axis in range(3):
            np.add.at(local_sum[axis], flat, local[:, axis])
        local_sum /= np.maximum(count[None], 1.0)

    occupancy = occupied.astype(np.float32)[None]
    density = np.clip(np.log1p(count) / 7.0, 0.0, 1.0)[None]
    if color_mode == "grayscale":
        color = (
            0.2126 * color_sum[0]
            + 0.7152 * color_sum[1]
            + 0.0722 * color_sum[2]
        )[None]
    else:
        color = color_sum
    channels = (occupancy, color, normal_sum, density)
    if include_local_offsets:
        channels = (*channels, local_sum)
    return (
        np.concatenate(channels, axis=0)
        .reshape((-1,) + tuple(shape))
        .astype(np.float32)
    )
