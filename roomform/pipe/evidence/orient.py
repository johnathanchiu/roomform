"""Gravity-axis detection and z-up normalization.

Datasets disagree on the up axis (SceneNN ships y-up; Redwood and
ARKitScenes z-up). Everything downstream — the boundary model, the
point-labeling lifter, fusion — assumes z-up, so e2e normalizes the
scan ONCE before any consumer sees it; evidence and lifting then share
one frame and ``frame_shift`` stays trivially consistent.

Detection: a floor is a large axis-aligned planar slab at the low end
of the up axis (and usually a ceiling at the high end). Score each
axis by the number of normal-aligned points in its extreme slabs;
default to z unless another axis wins by a clear margin.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh

# proper rotations mapping <axis>-up -> z-up
_R = {
    0: np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    1: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    2: np.eye(3),
}


def detect_up(pts: np.ndarray, sample: int = 150_000) -> int:
    """Return the gravity axis index (0=x, 1=y, 2=z).

    Known limitation: a scan whose floor is heavily cluttered while
    two facing walls are clean can still read as z-up (seen on one of
    ten benchmark scans) — pass an explicit up axis for those.
    """
    from roomform.pipe.evidence.build import _pca_normals

    if len(pts) > sample:
        pts = pts[:: len(pts) // sample + 1]
    cells = np.floor(pts / 0.04).astype(np.int64)
    _, keep = np.unique(cells, axis=0, return_index=True)
    pts = pts[keep]
    normals = np.abs(_pca_normals(pts))

    ranges = np.zeros(3)
    scores = np.zeros(3)
    for axis in range(3):
        v = pts[:, axis]
        lo, hi = np.percentile(v, [2, 98])
        ranges[axis] = hi - lo
        aligned = normals[:, axis] > 0.9
        slab = 0.10
        scores[axis] = (aligned & (v < lo + slab)).sum() + (
            aligned & (v > hi - slab)
        ).sum()
    # Neither signal alone separates a floor/ceiling pair from a wall
    # pair in a boxy room (slabs tie; ranges tie). Their ratio does:
    # the up axis has the most extreme-slab planar evidence per meter
    # of extent (short axis + strong floor/ceiling). Require a clear
    # margin over z and a plausible ceiling height — a wrong flip is
    # far worse than the z-up default.
    density = scores / np.maximum(ranges, 1e-6)
    best = int(np.argmax(density))
    if (
        best != 2
        and 1.7 <= ranges[best] <= 4.2
        and (density[best] > 1.15 * density[2])
    ):
        return best
    return 2


def ensure_z_up(scan_path: str, out_dir: str, up: str = "auto") -> str:
    """Return a z-up scan path; rewrites into out_dir only if needed."""
    if scan_path.endswith(".npz"):
        return scan_path  # fixture npzs are already normalized
    mesh = trimesh.load(scan_path)
    pts = np.asarray(mesh.vertices, dtype=np.float32)
    axis = "xyz".index(up) if up in "xyz" else detect_up(pts)
    if axis == 2:
        return scan_path
    rotated = pts @ _R[axis].T
    cloud = trimesh.PointCloud(rotated)
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is not None and len(colors) == len(pts):
        cloud.colors = colors
    out = os.path.join(out_dir, "scan-zup.ply")
    cloud.export(out)
    return out
