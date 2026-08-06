"""Export a pipeline scene into the demo editor's fixture format.

  uv run python -m roomform.viz.demo_export artifacts/<scene> \
      [--fixtures <dir>] [--id oss-<scene>] [--title TITLE]

Writes display assets under <fixtures>/<id>/ and upserts the scene's
entry in <fixtures>/index.json — the editor picks it up at runtime
with zero code changes.

Frames: GLB/ply assets are z-up grid-frame meters (the editor's views
declare zUp). analysis.json objects are in the editor's y-up frame:
position [x, height, -y], yaw becomes a quaternion about +Y, bounds
are [dx, height, dy].
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os

import numpy as np
import trimesh

MEASURED = [148, 149, 153, 255]  # gray = measured (editor convention)
INFERRED = [245, 158, 11, 255]  # amber = inferred fill
CLASS_COLORS = {
    "wall": [110, 160, 220, 255],
    "floor": [120, 200, 140, 255],
    "ceiling": [190, 190, 120, 255],
}


def _boxes(matrix: np.ndarray, vox: float, colors: np.ndarray, center):
    grid = trimesh.voxel.VoxelGrid(matrix)
    mesh = grid.as_boxes(colors=colors)
    mesh.apply_scale(vox)
    mesh.apply_translation([-center[0], -center[1], 0.0])
    return mesh


def _color_grid(shape, mask_colors) -> np.ndarray:
    colors = np.zeros((*shape, 4), np.uint8)
    for mask, color in mask_colors:
        colors[mask] = color
    return colors


def export_scene(scene_dir: str, fixtures: str, scene_id: str, title: str):
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

    # ① input: observed cloud, grayscale from evidence
    pts = np.argwhere(occ)
    gray = (features[1][occ] * 255).astype(np.uint8)
    cloud_colors = np.stack([gray, gray, gray, np.full_like(gray, 255)], 1)
    cloud = (pts + 0.5) * vox - [center[0], center[1], 0.0]
    trimesh.PointCloud(cloud, colors=cloud_colors).export(
        os.path.join(out, "input-cloud.ply")
    )

    # ② mesh: measured occupancy only
    _boxes(occ, vox, _color_grid(occ.shape, [(occ, MEASURED)]), center).export(
        os.path.join(out, "mesh.glb")
    )

    # ③ shell: predicted structure, colored per class
    shell_colors = _color_grid(
        occ.shape,
        [(node[k], CLASS_COLORS[n]) for k, n in enumerate(CLASS_COLORS)],
    )
    _boxes(anynode, vox, shell_colors, center).export(
        os.path.join(out, "shell.glb")
    )

    # ④ editable: measured gray + completed amber, one mesh the editor
    # carves into objects using the analysis boxes
    editable_colors = _color_grid(
        occ.shape, [(anynode & ~occ, INFERRED), (occ, MEASURED)]
    )
    _boxes(anynode | occ, vox, editable_colors, center).export(
        os.path.join(out, "editable.glb")
    )

    # analysis: objects in the editor's y-up frame
    objects = []
    for i, o in enumerate(doc["objects"]):
        cx, cy, cz = o["center"]
        sx, sy, sz = o["size"]
        half = math.sin(o["heading"] / 2)
        objects.append(
            {
                "id": f"spatiallm-{i}",
                "label": o["cls"],
                "category": o["cls"],
                "position": [cx - center[0], cz, -(cy - center[1])],
                "rotation_xyzw": [0.0, half, 0.0, math.cos(o["heading"] / 2)],
                "bounds": [sx, sz, sy],
                "confidence": 0.7,
                "method": "spatiallm",
                "source_views": [],
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


DEFAULT_FIXTURES = (
    "/Users/johnathanchiu/Projects/spatial-computing/mono/roomform/"
    "apps/viewer/dist/fixtures"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    ap.add_argument("--id", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.scene_dir))
    scene_id = args.id or f"oss-{name}"
    title = args.title or f"{name} · patch-graph e2e"
    print(export_scene(args.scene_dir, args.fixtures, scene_id, title))


if __name__ == "__main__":
    main()
