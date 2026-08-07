import json

import numpy as np
import trimesh

from roomform.export import _cell_points, export_object_meshes


def test_cell_points_use_centers_and_learned_offsets():
    cells = np.array([[1, 2, 3]])
    offsets = np.zeros((3, 3, 4, 5), dtype=np.float32)
    offsets[:, 1, 2, 3] = [0.25, -0.25, 0.5]

    points = _cell_points(cells, vox=0.08, offsets=offsets)

    np.testing.assert_allclose(points[0], [0.14, 0.18, 0.32])


def test_object_meshes_are_object_local(tmp_path):
    vox = 0.1
    shape = (10, 10, 10)
    occ = np.zeros(shape, np.uint8)
    occ[3:7, 3:7, 3:7] = 1  # 64 evidence points around (0.5, 0.5, 0.5)
    features = np.full((4, *shape), 0.5, np.float32)
    np.savez(tmp_path / "evidence.npz", occ=occ, features=features)
    scene = {
        "version": "0.1.0",
        "source_scan": "synthetic.ply",
        "frame_shift": [0.0, 0.0, 0.0],
        "shell": {
            "npz_path": "unused.npz",
            "vox_m": vox,
            "origin": [0.0, 0.0, 0.0],
            "shape": list(shape),
        },
        "objects": [
            {
                "cls": "box",
                "center": [0.5, 0.5, 0.5],
                "size": [0.5, 0.5, 0.5],
                "heading": 0.3,
            }
        ],
    }
    (tmp_path / "scene.json").write_text(json.dumps(scene))

    written = export_object_meshes(str(tmp_path))

    assert written == [str(tmp_path / "objects" / "object-0.glb")]
    mesh = trimesh.load(written[0])
    pts = np.concatenate(
        [np.asarray(g.vertices) for g in mesh.geometry.values()]
    )
    assert len(pts) >= 30
    # object-local frame: center-subtracted, so the cluster centroid
    # sits at the origin and every point is inside the (padded) box
    np.testing.assert_allclose(pts.mean(0), 0.0, atol=vox)
    assert np.all(np.abs(pts) <= 0.25 + 0.06 + 1e-6)
