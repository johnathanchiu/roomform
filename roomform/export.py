"""Scene exports: pipeline artifacts -> display geometry.

Two consumers, one module:

  glb       SceneDocument -> self-contained scene.glb (evidence points,
            shell points, object box outlines) — any glTF viewer opens it.
  fixtures  artifacts/<scene> -> the demo editor's fixture format
            (input/mesh/shell/editable assets + analysis + index upsert);
            the editor picks new scenes up at runtime, zero code changes.

  uv run python -m roomform.export glb SCENE_JSON OUT_GLB [--evidence NPZ]
  uv run python -m roomform.export fixtures SCENE_DIR [--fixtures DIR]
      [--id ID] [--title TITLE]

Frames: GLB/ply assets are z-up grid-frame meters (the editor's views
declare zUp; fixture assets are additionally origin-centered in xy).
Fixture analysis objects are in the editor's y-up frame: position
[x, height, -y], yaw becomes a quaternion about +Y, bounds [dx, height, dy].
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os

import numpy as np
import trimesh

from roomform.contracts import SceneDocument

MEASURED = [148, 149, 153, 255]  # gray = measured (editor convention)
INFERRED = [245, 158, 11, 255]  # amber = inferred fill
CLASS_COLORS = {
    "wall": [70, 140, 220, 255],
    "floor": [120, 200, 140, 255],
    "ceiling": [190, 190, 120, 255],
}


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
        pts = np.argwhere(occ) * vox
        if len(pts) > max_points:
            pts = pts[:: len(pts) // max_points + 1]
        scene.add_geometry(
            trimesh.PointCloud(pts, colors=[150, 150, 150, 255]),
            node_name="evidence",
        )

    d = np.load(doc.shell.npz_path)
    node = d["node_probs"] > doc.shell.node_threshold
    if not include_shell:
        node = node & False
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

    if "openings" in d.files:
        op = d["openings"]
        for k, (name, color, thr) in enumerate(
            (
                ("doors", [240, 200, 70, 255], 0.69),
                ("windows", [180, 110, 255, 255], 0.70),
            )
        ):
            pts = np.argwhere(op[k] > thr) * vox
            if len(pts):
                scene.add_geometry(
                    trimesh.PointCloud(pts, colors=color),
                    node_name=f"openings-{name}",
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


# ----------------------------------------------------------- fixtures ----


def _surface(matrix: np.ndarray, vox: float, colors: np.ndarray, center):
    """Marching-cubes surface over a voxel mask, vertex-colored from the
    per-voxel color grid — the editor's asset contract is smooth
    triangle meshes, not voxel boxes.

    The mask is display-cleaned first: speck components vanish, one
    closing pass fills pinholes, and Taubin smoothing planes off the
    voxel stairsteps. Display only — pipeline artifacts stay raw."""
    from scipy import ndimage
    from skimage import measure

    raw = matrix
    labels, n = ndimage.label(matrix)
    if n > 1:
        sizes = ndimage.sum_labels(
            np.ones_like(labels), labels, range(1, n + 1)
        )
        keep = np.flatnonzero(sizes >= 12) + 1
        matrix = np.isin(labels, keep)
    matrix = ndimage.binary_closing(np.pad(matrix, 2), iterations=1)[
        2:-2, 2:-2, 2:-2
    ]
    if not matrix.any():
        matrix = raw  # a small object is better chunky than gone
    if not matrix.any():
        return None

    padded = np.pad(matrix, 1)
    verts, faces, _, _ = measure.marching_cubes(
        padded.astype(np.float32), level=0.5
    )
    verts = (verts - 1.0) * vox
    idx = np.clip(
        np.floor(verts / vox).astype(int),
        0,
        np.asarray(matrix.shape) - 1,
    )
    vcol = colors[idx[:, 0], idx[:, 1], idx[:, 2]]
    # marching cubes can land a vertex on an uncolored neighbor cell —
    # snap fully transparent lookups to the nearest colored voxel below
    empty = vcol[:, 3] == 0
    if empty.any():
        idx2 = np.clip(idx[empty] - 1, 0, None)
        vcol[empty] = colors[idx2[:, 0], idx2[:, 1], idx2[:, 2]]
        vcol[vcol[:, 3] == 0] = MEASURED
    mesh = trimesh.Trimesh(
        vertices=verts - [center[0], center[1], 0.0],
        faces=faces,
        vertex_colors=vcol,
        process=False,
    )
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
    return mesh


def _color_grid(shape, mask_colors) -> np.ndarray:
    colors = np.zeros((*shape, 4), np.uint8)
    for mask, color in mask_colors:
        colors[mask] = color
    return colors


def export_fixtures(
    scene_dir: str, fixtures: str, scene_id: str, title: str
) -> str:
    with open(os.path.join(scene_dir, "scene.json")) as fh:
        doc = json.load(fh)
    ev = np.load(os.path.join(scene_dir, "evidence.npz"))
    pg = np.load(os.path.join(scene_dir, "patchgraph.npz"))
    vox = doc["shell"]["vox_m"]
    out = os.path.join(fixtures, scene_id)
    os.makedirs(out, exist_ok=True)

    occ = ev["occ"] > 0
    features = ev["features"].astype(np.float32)
    node = pg["node_probs"] > doc["shell"]["node_threshold"]
    anynode = node.any(0)
    # the editor frames scenes around the origin — center xy, floor at 0
    center = (np.asarray(occ.shape[:2]) * vox / 2.0).tolist()

    # ① input: the full-res RGB display cloud when the evidence stage
    # wrote one; grayscale voxel centers only as a fallback
    src_cloud = os.path.join(scene_dir, "cloud.ply")
    if os.path.exists(src_cloud):
        m = trimesh.load(src_cloud)
        trimesh.PointCloud(
            np.asarray(m.vertices) - [center[0], center[1], 0.0],
            colors=np.asarray(m.visual.vertex_colors),
        ).export(os.path.join(out, "input-cloud.ply"))
    else:
        pts = np.argwhere(occ)
        gray = (features[1][occ] * 255).astype(np.uint8)
        cloud_colors = np.stack([gray, gray, gray, np.full_like(gray, 255)], 1)
        cloud = (pts + 0.5) * vox - [center[0], center[1], 0.0]
        trimesh.PointCloud(cloud, colors=cloud_colors).export(
            os.path.join(out, "input-cloud.ply")
        )

    # ② mesh: measured occupancy only
    _surface(
        occ, vox, _color_grid(occ.shape, [(occ, MEASURED)]), center
    ).export(os.path.join(out, "mesh.glb"))

    # ③ shell: predicted structure, colored per class
    shell_colors = _color_grid(
        occ.shape,
        [(node[k], CLASS_COLORS[n]) for k, n in enumerate(CLASS_COLORS)],
    )
    _surface(anynode, vox, shell_colors, center).export(
        os.path.join(out, "shell.glb")
    )

    # ④ editable: measured gray + completed amber. Object voxels are
    # carved OUT — they ship as per-object measured meshes, and any
    # coincident copy in the background surface would swallow the
    # editor's raycasts and make objects ungrabbable.
    objmask = np.zeros(occ.shape, bool)
    idx = np.argwhere(occ | anynode)
    world = (idx + 0.5) * vox
    for o in doc["objects"]:
        c, s_ = np.asarray(o["center"]), np.asarray(o["size"])
        ch, sh = math.cos(o["heading"]), math.sin(o["heading"])
        rel = world - c
        rot = np.stack(
            [
                rel[:, 0] * ch + rel[:, 1] * sh,
                -rel[:, 0] * sh + rel[:, 1] * ch,
                rel[:, 2],
            ],
            1,
        )
        inside = np.all(np.abs(rot) <= s_ / 2 + 0.04, axis=1)
        objmask[tuple(idx[inside].T)] = True
    editable_mask = anynode | (occ & ~objmask)
    editable_colors = _color_grid(
        occ.shape,
        [
            (anynode & ~occ, INFERRED),
            (occ & ~objmask, MEASURED),
            (anynode & occ, MEASURED),
        ],
    )
    _surface(editable_mask, vox, editable_colors, center).export(
        os.path.join(out, "editable.glb")
    )

    # per-object measured meshes: the editor makes an object draggable
    # only when its proposal carries measured_mesh_uri, so carve each
    # object's points into a small voxel mesh. Labeled points when the
    # pointlabel lifter ran; raw evidence points otherwise.
    labels_path = os.path.join(scene_dir, "labels.npz")
    if os.path.exists(labels_path):
        lab = np.load(labels_path)
        lab_pts = lab["pts"].astype(np.float32) - np.asarray(
            doc["frame_shift"], np.float32
        )
        lab_cls = [str(c) for c in lab["classes"][lab["label"]]]
    else:
        lab_pts = (np.argwhere(occ) + 0.5) * vox
        lab_cls = None
    obj_dir = os.path.join(out, "objects")
    os.makedirs(obj_dir, exist_ok=True)

    def _object_mesh(i: int, o: dict) -> str | None:
        c, s_ = np.asarray(o["center"]), np.asarray(o["size"])
        h = math.cos(o["heading"]), math.sin(o["heading"])
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
            inside &= (
                np.fromiter(
                    (lab_cls[j] == o["cls"] for j in range(len(lab_cls))),
                    bool,
                    len(lab_cls),
                )
                | ~inside
            )  # keep the box filter authoritative
            inside = np.all(np.abs(rot) <= s_ / 2 + 0.06, axis=1) & (
                np.asarray(lab_cls) == o["cls"]
            )
        pts_i = lab_pts[inside]
        if len(pts_i) < 30:
            return None
        grid = np.floor(pts_i / 0.04).astype(np.int64)
        gmin = grid.min(0)
        m = np.zeros(grid.max(0) - gmin + 1, bool)
        m[tuple((grid - gmin).T)] = True
        colors = np.zeros((*m.shape, 4), np.uint8)
        colors[m] = MEASURED
        mesh = _surface(m, 0.04, colors, (0.0, 0.0))
        if mesh is None:
            return None
        mesh.apply_translation(gmin * 0.04 - [center[0], center[1], 0.0])
        path = os.path.join(obj_dir, f"object-{i}.glb")
        mesh.export(path)
        return f"/fixtures/{scene_id}/objects/object-{i}.glb"

    # analysis: objects in the editor's y-up frame
    objects = []
    for i, o in enumerate(doc["objects"]):
        cx, cy, cz = o["center"]
        sx, sy, sz = o["size"]
        mesh_uri = _object_mesh(i, o)
        objects.append(
            {
                "id": f"spatiallm-{i}",
                "label": o["cls"],
                "category": o["cls"],
                "position": [cx - center[0], cz, -(cy - center[1])],
                "rotation_xyzw": [
                    0.0,
                    math.sin(o["heading"] / 2),
                    0.0,
                    math.cos(o["heading"] / 2),
                ],
                "bounds": [sx, sz, sy],
                "confidence": 0.7,
                "method": "spatiallm",
                "source_views": [],
                **(
                    {"measured_mesh_uri": mesh_uri, "measured_mesh_z_up": True}
                    if mesh_uri
                    else {}
                ),
            }
        )
    analysis = {
        "schema": "roomform.splat-analysis.result.v1",
        "scene_id": scene_id,
        "splat_uri": f"/fixtures/{scene_id}/editable.glb",
        "detector": "spatiallm/qwen",
        "coordinate_system": "source-splat",
        "objects": objects,
    }
    with open(os.path.join(out, "analysis.json"), "w") as fh:
        json.dump(analysis, fh, indent=1)

    program = {
        "schema": "roomform.scene-program.v1",
        "dataset_id": scene_id,
        "statements": [],
    }
    with open(os.path.join(out, "scene-program.json"), "w") as fh:
        json.dump(program, fh, indent=1)

    entry = {
        "id": scene_id,
        "title": title,
        "stats": {"objects": len(objects)},
        "assets": {
            "input": f"/fixtures/{scene_id}/input-cloud.ply",
            "mesh": f"/fixtures/{scene_id}/mesh.glb",
            "shell": f"/fixtures/{scene_id}/shell.glb",
            "editable": f"/fixtures/{scene_id}/editable.glb",
            "program": f"/fixtures/{scene_id}/scene-program.json",
            "objectsAnalysis": f"/fixtures/{scene_id}/analysis.json",
        },
        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    index_path = os.path.join(fixtures, "index.json")
    index = {"schema": "roomform.fixtures-index.v1", "scenes": []}
    if os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
    index["scenes"] = [s for s in index["scenes"] if s["id"] != scene_id]
    index["scenes"].append(entry)
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=1)
    return out


# ----------------------------------------------------------------- cli ----

# fixtures dir of a production-editor build; no public default
DEFAULT_FIXTURES = os.environ.get("ROOMFORM_FIXTURES", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    glb = sub.add_parser("glb", help="SceneDocument -> scene.glb")
    glb.add_argument("scene_json")
    glb.add_argument("out_glb")
    glb.add_argument("--evidence", default="")

    fix = sub.add_parser("fixtures", help="scene dir -> demo editor fixture")
    fix.add_argument("scene_dir")
    fix.add_argument(
        "--fixtures",
        default=DEFAULT_FIXTURES,
        required=not DEFAULT_FIXTURES,
        help="editor fixtures dir (or set ROOMFORM_FIXTURES)",
    )
    fix.add_argument("--id", default="")
    fix.add_argument("--title", default="")

    args = ap.parse_args()
    if args.cmd == "glb":
        with open(args.scene_json) as fh:
            doc = SceneDocument(**json.load(fh))
        print(export_glb(doc, args.out_glb, args.evidence or None))
    else:
        name = os.path.basename(os.path.normpath(args.scene_dir))
        scene_id = args.id or f"oss-{name}"
        title = args.title or f"{name} · patch-graph e2e"
        print(export_fixtures(args.scene_dir, args.fixtures, scene_id, title))


if __name__ == "__main__":
    main()
