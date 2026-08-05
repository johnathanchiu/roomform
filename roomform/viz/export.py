"""SceneDocument -> GLB export: evidence, shell nodes, object boxes.

One self-contained artifact any glTF viewer opens; the repo's own
answer to "let me see the scene" without a running service.
"""

from __future__ import annotations

import json

import numpy as np
import trimesh

from roomform.contracts import SceneDocument

CLASS_COLORS = {
    "wall": [70, 140, 220, 255],
    "floor": [120, 200, 140, 255],
    "ceiling": [190, 190, 120, 255],
}


def _box_outline(center, size, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    hx, hy, hz = np.asarray(size) / 2
    corners = np.array(
        [
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    corners = corners @ rot.T + np.asarray(center)
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    t = np.linspace(0.0, 1.0, 10)[:, None]
    return np.concatenate(
        [corners[a] + t * (corners[b] - corners[a]) for a, b in edges]
    )


def export_glb(
    doc: SceneDocument,
    out_glb: str,
    evidence_npz: str | None = None,
    max_points: int = 300_000,
) -> str:
    scene = trimesh.Scene()
    vox = doc.shell.vox_m

    if evidence_npz:
        occ = np.load(evidence_npz)["occ"] > 0
        pts = np.argwhere(occ) * vox
        if len(pts) > max_points:
            pts = pts[:: len(pts) // max_points + 1]
        scene.add_geometry(
            trimesh.PointCloud(pts, colors=[150, 150, 150, 255]),
            node_name="evidence",
        )

    d = np.load(doc.shell.npz_path)
    node = d["node_probs"] > doc.shell.node_threshold
    for k, (name, color) in enumerate(CLASS_COLORS.items()):
        pts = np.argwhere(node[k]) * vox
        if not len(pts):
            continue
        if len(pts) > max_points:
            pts = pts[:: len(pts) // max_points + 1]
        scene.add_geometry(
            trimesh.PointCloud(pts, colors=color),
            node_name=f"shell-{name}",
        )

    ok, flagged = [], []
    for obj in doc.objects:
        frame = _box_outline(obj.center, obj.size, obj.heading)
        (flagged if (obj.qa.get("wall_leak_pts") or 0) > 50 else ok).append(
            frame
        )
    if ok:
        scene.add_geometry(
            trimesh.PointCloud(np.concatenate(ok), colors=[230, 160, 60, 255]),
            node_name="objects",
        )
    if flagged:
        scene.add_geometry(
            trimesh.PointCloud(
                np.concatenate(flagged), colors=[230, 60, 60, 255]
            ),
            node_name="objects-flagged",
        )

    scene.export(out_glb)
    return out_glb


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("scene_json")
    ap.add_argument("out_glb")
    ap.add_argument("--evidence", default="")
    args = ap.parse_args()
    with open(args.scene_json) as f:
        doc = SceneDocument(**json.load(f))
    path = export_glb(doc, args.out_glb, args.evidence or None)
    print(path)


if __name__ == "__main__":
    main()
