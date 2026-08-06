"""ScanInput -> EvidenceGrid: voxelize a point cloud into the
observable channels the model was trained on (see evidence/raw.py).

Accepts a .ply point cloud or a .npz already carrying pts / pts_normal
/ pts_color. For plys without normals, normals are estimated by local
PCA over k nearest neighbors after 2 cm dedup — the model only
consumes |normal|, so sign ambiguity is irrelevant.

v0 limitation, by design: without scanner stations there is no free-
space carve; ``sf`` is empty and ``visibility_source="synthesized"``
marks the degradation.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh

from roomform.contracts import EvidenceGrid
from roomform.pipe.evidence.raw import VOX, raw_point_evidence


def _dedup_2cm(pts: np.ndarray, colors: np.ndarray | None):
    """Keep at most one point per 2 cm cell (normal estimation input)."""
    cells = np.floor(pts / 0.02).astype(np.int64)
    _, keep = np.unique(cells, axis=0, return_index=True)
    return pts[keep], colors[keep] if colors is not None else None


def _crop_to_dominant_region(pts: np.ndarray, cell_m: float = 0.5):
    """Keep the dominant dense region of the scan footprint.

    Tripod scans spill sparse long-range returns through doors and
    windows; a percentile bound can't catch spill that is a few
    percent of the cloud. Bin the xy footprint, threshold on density,
    take the largest connected component, and keep points inside its
    bbox (+1 m margin). A clean scan is one component — a no-op.
    """
    from scipy import ndimage

    lo = pts[:, :2].min(0)
    ij = np.floor((pts[:, :2] - lo) / cell_m).astype(np.int64)
    shape = ij.max(0) + 1
    counts = np.zeros(shape, np.int64)
    np.add.at(counts, (ij[:, 0], ij[:, 1]), 1)
    # ponytail: fixed floor + 1% of the densest cell; per-sensor tuning
    # knob if a scanner profile ever needs it
    dense = counts >= max(30, counts.max() // 100)
    labels, n = ndimage.label(dense)
    if n <= 1:
        return np.ones(len(pts), bool)
    best = 1 + np.argmax(ndimage.sum_labels(counts, labels, range(1, n + 1)))
    cells = np.argwhere(labels == best)
    bmin = cells.min(0) * cell_m + lo - 1.0
    bmax = (cells.max(0) + 1) * cell_m + lo + 1.0
    return np.all((pts[:, :2] >= bmin) & (pts[:, :2] <= bmax), axis=1)


def _pca_normals(pts: np.ndarray, k: int = 16) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, nbr = cKDTree(pts).query(pts, k=k, workers=-1)
    nb = pts[nbr]  # [N, k, 3]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0].astype(np.float32)  # smallest-eigenvalue axis


def _load_scan(scan_path: str):
    if scan_path.endswith(".npz"):
        d = np.load(scan_path)
        pts = d["pts"].astype(np.float32)
        colors = d["pts_color"] if "pts_color" in d.files else None
        normals = (
            d["pts_normal"].astype(np.float32)
            if "pts_normal" in d.files
            else None
        )
        return pts, colors, normals
    mesh = trimesh.load(scan_path)
    pts = np.asarray(mesh.vertices, dtype=np.float32)
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is not None and len(colors) == len(pts):
        colors = np.asarray(colors)[:, :3]
    else:
        colors = None
    return pts, colors, None


def build_evidence(
    scan_path: str, out_npz: str, vox_m: float = VOX
) -> EvidenceGrid:
    pts, colors, normals = _load_scan(scan_path)
    keep = _crop_to_dominant_region(pts)
    if not keep.all():
        pts = pts[keep]
        colors = colors[keep] if colors is not None else None
        normals = normals[keep] if normals is not None else None
    # robust grounding: scanner outlier tails (junk points below the
    # real floor) must not set the grid origin — the model was trained
    # with the floor at the grid bottom. Points outside the robust
    # bounds fall off the grid via the voxelizer's validity mask.
    origin = np.percentile(pts, 0.1, axis=0).astype(np.float32)
    top = np.percentile(pts, 99.9, axis=0).astype(np.float32)
    pts = pts - origin
    if normals is None:
        pts, colors = _dedup_2cm(pts, colors)
        normals = _pca_normals(pts)

    # aligned to a multiple of 8 so the UNet needs no crop bookkeeping
    extent = top - origin
    shape = tuple(
        int(v)
        for v in (np.ceil((np.ceil(extent / vox_m) + 1) / 8) * 8).astype(int)
    )
    raw = {"pts": pts, "pts_normal": normals}
    if colors is not None:
        raw["pts_color"] = colors
    # RGB superset (11ch): occ, r, g, b, |nrm| xyz, density, offsets.
    # Grayscale checkpoints get luma computed at load time.
    features = raw_point_evidence(
        raw, shape, "rgb", include_local_offsets=True
    )

    np.savez_compressed(
        out_npz,
        features=features.astype(np.float16),
        occ=(features[0] > 0).astype(np.uint8),
        sf=np.zeros(shape, np.uint8),  # v0: no carve — all non-occ unknown
    )

    # full-res RGB display cloud (grid frame) beside the npz — viewers
    # show this instead of voxel centers
    display_cap = 1_500_000
    stride = max(1, len(pts) // display_cap)
    disp = pts[::stride]
    disp_colors = colors[::stride] if colors is not None else None
    trimesh.PointCloud(disp, colors=disp_colors).export(
        os.path.join(os.path.dirname(out_npz) or ".", "cloud.ply")
    )
    return EvidenceGrid(
        npz_path=out_npz,
        vox_m=vox_m,
        origin=tuple(float(v) for v in origin),
        shape=shape,
        visibility_source="synthesized",
    )
