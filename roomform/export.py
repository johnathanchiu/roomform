"""Scene exports: pipeline artifacts -> display geometry.

Two consumers, one module:

  glb      SceneDocument -> self-contained scene.glb (evidence points,
           shell points, object box outlines) — any glTF viewer opens it.
  objects  artifacts/<scene> -> per-object evidence-cluster GLBs under
           artifacts/<scene>/objects/; the editor makes an object
           draggable only when its mesh exists.

  uv run python -m roomform.export glb SCENE_JSON OUT_GLB [--evidence NPZ]
  uv run python -m roomform.export objects SCENE_DIR

Frames: GLB/ply assets are z-up grid-frame meters. Per-object GLBs are
object-local z-up (center-subtracted, heading-unrotated) so the editor
places them with the object's scene.json transform and saved poses
survive reloads.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import trimesh

from roomform.contracts import SceneDocument, SceneObject
from roomform.planes import planar_shell

CLASS_COLORS = {
    "wall": [70, 140, 220, 255],
    "floor": [120, 200, 140, 255],
    "ceiling": [190, 190, 120, 255],
}


def _boundary_nodes(patchgraph, threshold: float) -> np.ndarray:
    """Threshold model output and merge provenance-separated agent fills."""
    node = patchgraph["node_probs"] > threshold
    if "agent_fill" in patchgraph.files:
        fill = patchgraph["agent_fill"].astype(bool)
        if fill.shape == node.shape[1:]:
            migrated = np.zeros_like(node)
            migrated[0] = fill
            fill = migrated
        if fill.shape != node.shape:
            raise ValueError(
                f"agent_fill shape {fill.shape} does not match {node.shape}"
            )
        node |= fill
    return node


def _cell_points(
    cells: np.ndarray, vox: float, offsets: np.ndarray | None = None
) -> np.ndarray:
    """Convert integer cells to learned surface points in grid-frame meters."""
    points = (cells.astype(np.float64) + 0.5) * vox
    if offsets is not None and len(cells):
        points += (
            np.stack([offsets[a][tuple(cells.T)] for a in range(3)], axis=1)
            * vox
        )
    return points


# ---------------------------------------------------------------- glb ----


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
    include_shell: bool | None = None,
    include_planar: bool = False,
) -> str:
    """include_shell=None auto-skips untrained shells (a random-init
    net marks ~half of ALL cells -> a solid glowing cube that buries
    everything else)."""
    if include_shell is None:
        include_shell = "RANDOM-INIT" not in (doc.shell.model_id or "")
    scene = trimesh.Scene()
    vox = doc.shell.vox_m

    if evidence_npz:
        occ = np.load(evidence_npz)["occ"] > 0
        pts = _cell_points(np.argwhere(occ), vox)
        if len(pts) > max_points:
            pts = pts[:: len(pts) // max_points + 1]
        scene.add_geometry(
            trimesh.PointCloud(pts, colors=[150, 150, 150, 255]),
            node_name="evidence",
        )

    d = np.load(doc.shell.npz_path)
    node = _boundary_nodes(d, doc.shell.node_threshold)
    if not include_shell:
        node = node & False
    for k, (name, color) in enumerate(CLASS_COLORS.items()):
        cells = np.argwhere(node[k])
        pts = _cell_points(cells, vox, d.get("offsets"))
        if not len(pts):
            continue
        if len(pts) > max_points:
            pts = pts[:: len(pts) // max_points + 1]
        scene.add_geometry(
            trimesh.PointCloud(pts, colors=color),
            node_name=f"shell-{name}",
        )

    if include_planar:
        # experimental: fitted-plane boundary mesh (roomform.planes) —
        # off by default until gap post-processing lands; gaps read as
        # holes in a solid surface far more than in a point cloud

        node_probs = d["node_probs"].astype(np.float32)
        if "agent_fill" in d.files:
            node_probs = np.maximum(node_probs, d["agent_fill"])
        planar = planar_shell(
            node_probs,
            vox,
            offsets=d["offsets"].astype(np.float32)
            if "offsets" in d.files
            else None,
            openings=d["openings"].astype(np.float32)
            if "openings" in d.files
            else None,
            node_threshold=doc.shell.node_threshold,
        )
        for cls, mesh in planar.items():
            scene.add_geometry(mesh, node_name=f"boundary-mesh-{cls}")

    if "openings" in d.files:
        op = d["openings"]
        for k, (name, color, thr) in enumerate(
            (
                ("doors", [240, 200, 70, 255], 0.69),
                ("windows", [180, 110, 255, 255], 0.70),
            )
        ):
            cells = np.argwhere(op[k] > thr)
            pts = _cell_points(cells, vox, d.get("offsets"))
            if len(pts):
                scene.add_geometry(
                    trimesh.PointCloud(pts, colors=color),
                    node_name=f"openings-{name}",
                )

    ok, flagged = [], []
    for obj in doc.objects:
        frame = _box_outline(obj.center, obj.size, obj.heading)
        (flagged if obj.qa.leaking else ok).append(frame)
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


# ------------------------------------------------------------ objects ----


def export_object_meshes(scene_dir: str) -> list[str]:
    """Carve each scene.json object's evidence points into a small GLB.

    Writes artifacts/<scene>/objects/object-<i>.glb for every object
    with enough points (labeled points when the pointlabel lifter ran;
    raw evidence points otherwise). Vertices are object-local z-up
    meters — center-subtracted and heading-unrotated — so the editor
    renders them at the object's current transform and edited poses
    stay consistent across save/reload.
    """
    with open(os.path.join(scene_dir, "scene.json")) as fh:
        doc = SceneDocument(**json.load(fh))
    ev = np.load(os.path.join(scene_dir, "evidence.npz"))
    occ = ev["occ"] > 0
    features = ev["features"].astype(np.float32)
    vox = doc.shell.vox_m

    labels_path = os.path.join(scene_dir, "labels.npz")
    if os.path.exists(labels_path):
        lab = np.load(labels_path)
        lab_pts = lab["pts"].astype(np.float32) - np.asarray(
            doc.frame_shift, np.float32
        )
        lab_cls = np.asarray([str(c) for c in lab["classes"][lab["label"]]])
    else:
        lab_pts = (np.argwhere(occ) + 0.5) * vox
        lab_cls = None
    obj_dir = os.path.join(scene_dir, "objects")
    os.makedirs(obj_dir, exist_ok=True)

    def _object_mesh(i: int, o: SceneObject) -> str | None:
        c, s_ = np.asarray(o.center), np.asarray(o.size)
        h = math.cos(o.heading), math.sin(o.heading)
        rel = lab_pts - c
        rot = np.stack(
            [
                rel[:, 0] * h[0] + rel[:, 1] * h[1],
                -rel[:, 0] * h[1] + rel[:, 1] * h[0],
                rel[:, 2],
            ],
            1,
        )
        inside = np.all(np.abs(rot) <= s_ / 2 + 0.06, axis=1)
        if lab_cls is not None:
            inside &= lab_cls == o.cls
        if inside.sum() < 30:
            return None
        vi = np.clip(
            np.floor(lab_pts[inside] / vox).astype(int),
            0,
            np.asarray(occ.shape) - 1,
        )
        col = (
            np.stack(
                [features[k][vi[:, 0], vi[:, 1], vi[:, 2]] for k in (1, 2, 3)],
                1,
            )
            * 255
        ).astype(np.uint8)
        col = np.concatenate([col, np.full((len(col), 1), 255, np.uint8)], 1)
        path = os.path.join(obj_dir, f"object-{i}.glb")
        trimesh.Scene([trimesh.PointCloud(rot[inside], colors=col)]).export(
            path
        )
        return path

    written = []
    for i, o in enumerate(doc.objects):
        path = _object_mesh(i, o)
        if path:
            written.append(path)
    return written


# ----------------------------------------------------------------- cli ----


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    glb = sub.add_parser("glb", help="SceneDocument -> scene.glb")
    glb.add_argument("scene_json")
    glb.add_argument("out_glb")
    glb.add_argument("--evidence", default="")

    obj = sub.add_parser("objects", help="scene dir -> per-object GLBs")
    obj.add_argument("scene_dir")

    args = ap.parse_args()
    if args.cmd == "glb":
        with open(args.scene_json) as fh:
            doc = SceneDocument(**json.load(fh))
        print(export_glb(doc, args.out_glb, args.evidence or None))
    else:
        written = export_object_meshes(args.scene_dir)
        print(f"{len(written)} object meshes -> {args.scene_dir}/objects")


if __name__ == "__main__":
    main()
