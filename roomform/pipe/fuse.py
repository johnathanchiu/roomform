"""Fusion: PatchGraph shell + SceneObjects -> SceneDocument with QA.

Per-object checks: wall_leak_pts (predicted-wall cells inside the box,
4cm interior margin) and floor_support_m (box bottom vs shell floor
height). Everything operates in the grid frame.
"""

from __future__ import annotations

import json
import os

import numpy as np

from roomform.contracts import ObjectQA, PatchGraph, SceneDocument, SceneObject


def _box_mask(
    pts: np.ndarray, center, size, heading: float, margin: float = 0.0
) -> np.ndarray:
    c, s = np.cos(-heading), np.sin(-heading)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    local = (pts - np.asarray(center)) @ rot.T
    half = np.asarray(size) / 2 + margin
    return np.all(np.abs(local) <= half, axis=1)


def fuse(
    shell: PatchGraph,
    objects: list[SceneObject],
    source_scan: str,
    frame_shift: tuple[float, float, float],
    out_json: str,
) -> SceneDocument:
    d = np.load(shell.npz_path)
    node = d["node_probs"] > shell.node_threshold
    if "agent_fill" in d.files:
        fill = d["agent_fill"].astype(bool)
        if fill.shape == node.shape[1:]:
            migrated = np.zeros_like(node)
            migrated[0] = fill
            fill = migrated
        if fill.shape != node.shape:
            raise ValueError(
                f"agent_fill shape {fill.shape} does not match {node.shape}"
            )
        node |= fill
    wall_pts = np.argwhere(node[0]) * shell.vox_m
    floor_pts = np.argwhere(node[1]) * shell.vox_m
    floor_z = (
        float(np.percentile(floor_pts[:, 2], 20)) if len(floor_pts) else None
    )

    for obj in objects:
        leak = (
            int(
                _box_mask(
                    wall_pts, obj.center, obj.size, obj.heading, margin=-0.04
                ).sum()
            )
            if len(wall_pts)
            else 0
        )
        obj.qa = ObjectQA(
            wall_leak_pts=leak,
            floor_support_m=(
                round(obj.center[2] - obj.size[2] / 2 - floor_z, 3)
                if floor_z is not None
                else None
            ),
        )

    doc = SceneDocument(
        source_scan=os.path.basename(source_scan),
        frame_shift=frame_shift,
        shell=shell,
        objects=objects,
        floor_z=floor_z,
    )
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(doc.model_dump(), f, indent=1)
    return doc
