"""Evidence-snapped wall refinement: voxel bands -> clean lines.

The boundary model predicts walls as 6-8 cm voxel bands: stepped
outlines, half-voxel jitter, soft corners. The raw scan is
millimeter-accurate wherever it has coverage. This stage projects the
model's wall estimates onto that evidence and regularizes the rest:

1. segment wall cells into planar runs (direction-bucketed region
   growing, same scheme as the planar shell),
2. per segment, robust-fit a 2D line to scan points in a band around
   the predicted plane (PCA init + Tukey reweighting); support and
   residual gates keep mirror ghosts and thin walls from dragging the
   fit — ungated segments keep the predicted line,
3. merge collinear neighbors, then intersect adjacent lines to
   re-derive corners exactly (no right-angle prior: curved walls stay
   a chain of intersected facets),
4. project opening cells onto the refined lines as clean u/z
   rectangles.

Output: ``refined.json`` (wall-line graph + per-wall fit report) and
crisp wall quads appended to a GLB for side-by-side inspection.

    uv run python -m roomform.pipe.refine artifacts/<scene>
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from roomform.contracts import SceneDocument

BAND_M = 0.12  # evidence band around the predicted wall plane
MIN_INLIERS = 250  # fewer supporting points -> keep the predicted line
MAX_RMS_M = 0.02  # worse residual -> keep the predicted line
MERGE_ANGLE_DEG = 5.0  # collinear-merge threshold
MERGE_OFFSET_M = 0.05
CORNER_JOIN_M = 0.30  # endpoints closer than this try to intersect
MIN_SEGMENT_CELLS = 24


# ------------------------------------------------------------ segments ----


def _wall_segments(node_wall: np.ndarray, vox: float):
    """Direction-bucketed connected wall segments (cell index sets)."""
    foot = node_wall.any(2)
    fp = np.argwhere(foot)
    if not len(fp):
        return
    tree = cKDTree(fp)
    theta = np.zeros(len(fp))
    for k, nb in enumerate(tree.query_ball_point(fp, r=3.0)):
        q = fp[nb] - fp[nb].mean(0)
        v = np.linalg.eigh(q.T @ q)[1][:, -1]
        theta[k] = np.arctan2(v[1], v[0]) % np.pi
    theta_grid = np.zeros(foot.shape)
    theta_grid[tuple(fp.T)] = theta
    nbins = 12
    bins = np.floor(theta_grid / np.pi * nbins).astype(int) % nbins
    done = np.zeros(node_wall.shape, bool)
    for b in range(nbins):
        sel2d = foot & (
            (bins == b) | (bins == (b + 1) % nbins) | (bins == (b - 1) % nbins)
        )
        sel = node_wall & sel2d[:, :, None] & ~done
        labels, n = ndimage.label(sel, structure=np.ones((3, 3, 3)))
        for lab in range(1, n + 1):
            seg = np.argwhere(labels == lab)
            if len(seg) < MIN_SEGMENT_CELLS:
                continue
            done[tuple(seg.T)] = True
            yield seg


class WallLine:
    """A wall as a 2D line segment with a z-range and a fit report."""

    def __init__(self, seg_cells: np.ndarray, vox: float):
        pts = (seg_cells.astype(np.float64) + 0.5) * vox
        self.center = pts[:, :2].mean(0)
        q = pts[:, :2] - self.center
        self.direction = np.linalg.eigh(q.T @ q)[1][:, -1]
        u = q @ self.direction
        self.u0, self.u1 = float(u.min()), float(u.max())
        self.z0, self.z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
        self.refined = False
        self.rms_before = None
        self.rms_after = None
        self.n_support = 0

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def endpoints(self) -> np.ndarray:
        return np.array(
            [
                self.center + self.direction * self.u0,
                self.center + self.direction * self.u1,
            ]
        )

    def snap(self, cloud_xy: np.ndarray, tree: cKDTree) -> None:
        """Robust line refit from scan points near the predicted line."""
        mid = self.center + self.direction * (self.u0 + self.u1) / 2
        half = (self.u1 - self.u0) / 2 + BAND_M
        idx = tree.query_ball_point(mid, r=half + BAND_M)
        if not idx:
            return
        p = cloud_xy[idx]
        rel = p - self.center
        u, d = rel @ self.direction, rel @ self.normal
        near = (
            (np.abs(d) < BAND_M)
            & (u > self.u0 - BAND_M)
            & (u < self.u1 + BAND_M)
        )
        p = p[near]
        if len(p) < MIN_INLIERS:
            return
        self.rms_before = float(np.sqrt(np.mean(d[near] ** 2)))
        center, direction = self.center, self.direction
        for _ in range(3):  # Tukey-reweighted PCA refit
            rel = p - center
            d = rel @ np.array([-direction[1], direction[0]])
            s = max(1.4826 * np.median(np.abs(d)), 3e-3)
            w = np.clip(1 - (d / (4 * s)) ** 2, 0, None) ** 2
            if w.sum() < MIN_INLIERS / 2:
                return
            center = (p * w[:, None]).sum(0) / w.sum()
            q = (p - center) * np.sqrt(w)[:, None]
            direction = np.linalg.eigh(q.T @ q)[1][:, -1]
        rel = p - center
        d = rel @ np.array([-direction[1], direction[0]])
        keep = np.abs(d) < 3 * max(1.4826 * np.median(np.abs(d)), 3e-3)
        rms = float(np.sqrt(np.mean(d[keep] ** 2)))
        if keep.sum() < MIN_INLIERS or rms > MAX_RMS_M:
            return
        if direction @ self.direction < 0:
            direction = -direction
        u = (p[keep] - center) @ direction
        self.center, self.direction = center, direction
        self.u0, self.u1 = float(u.min()), float(u.max())
        self.refined = True
        self.rms_after = rms
        self.n_support = int(keep.sum())


# --------------------------------------------------------- line network ----


def _merge_collinear(walls: list[WallLine]) -> list[WallLine]:
    """Union nearly-collinear, overlapping neighbors into one line."""
    merged = True
    while merged:
        merged = False
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                a, b = walls[i], walls[j]
                if abs(a.direction @ b.direction) < math.cos(
                    math.radians(MERGE_ANGLE_DEG)
                ):
                    continue
                if abs((b.center - a.center) @ a.normal) > MERGE_OFFSET_M:
                    continue
                ua = np.array([a.u0, a.u1])
                ub = (b.center - a.center) @ a.direction + np.array(
                    [b.u0, b.u1]
                )
                if (
                    ub.min() > ua.max() + CORNER_JOIN_M
                    or ua.min() > ub.max() + CORNER_JOIN_M
                ):
                    continue
                a.u0 = float(min(ua.min(), ub.min()))
                a.u1 = float(max(ua.max(), ub.max()))
                a.z0, a.z1 = min(a.z0, b.z0), max(a.z1, b.z1)
                a.refined = a.refined or b.refined
                walls.pop(j)
                merged = True
                break
            if merged:
                break
    return walls


def _cull_stubs(walls: list[WallLine]) -> list[WallLine]:
    """Drop short unrefined fragments that shadow a refined line.

    Leftover voxel stubs next to an evidence-snapped wall add zigzag
    noise without adding structure; anything short, unsnapped, and
    mostly within a band of a refined line goes."""
    refined = [w for w in walls if w.refined]
    keep = []
    for w in walls:
        if w.refined or (w.u1 - w.u0) > 0.9:
            keep.append(w)
            continue
        mid = w.center + w.direction * (w.u0 + w.u1) / 2
        shadowed = any(
            abs((mid - r.center) @ r.normal) < 3 * MERGE_OFFSET_M
            and r.u0 - CORNER_JOIN_M
            < (mid - r.center) @ r.direction
            < r.u1 + CORNER_JOIN_M
            for r in refined
        )
        if not shadowed:
            keep.append(w)
    return keep


def _intersect_corners(walls: list[WallLine]) -> int:
    """Snap close endpoint pairs to the exact line intersection."""
    joined = 0
    ends = [(w, e) for w in walls for e in (0, 1)]
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            (wa, ea), (wb, eb) = ends[i], ends[j]
            if wa is wb:
                continue
            if abs(wa.direction @ wb.direction) > math.cos(math.radians(20)):
                continue  # near-parallel: no stable intersection
            pa, pb = wa.endpoints()[ea], wb.endpoints()[eb]
            if np.linalg.norm(pa - pb) > CORNER_JOIN_M:
                continue
            # solve wa.center + ta*da = wb.center + tb*db
            A = np.stack([wa.direction, -wb.direction], 1)
            try:
                ta, tb = np.linalg.solve(A, wb.center - wa.center)
            except np.linalg.LinAlgError:
                continue
            old_a = wa.u0 if ea == 0 else wa.u1
            old_b = wb.u0 if eb == 0 else wb.u1
            if (
                abs(ta - old_a) > CORNER_JOIN_M
                or abs(tb - old_b) > CORNER_JOIN_M
            ):
                continue  # intersection would drag a wall far past its span
            if ea == 0:
                wa.u0 = float(ta)
            else:
                wa.u1 = float(ta)
            if eb == 0:
                wb.u0 = float(tb)
            else:
                wb.u1 = float(tb)
            joined += 1
    return joined


# ------------------------------------------------------------- openings ----


def _wall_openings(w: WallLine, opening_cells: np.ndarray, vox: float):
    """Opening cells near this wall -> clean u/z rectangles."""
    if opening_cells is None or not len(opening_cells):
        return []
    p = (opening_cells.astype(np.float64) + 0.5) * vox
    rel = p[:, :2] - w.center
    d, u = rel @ w.normal, rel @ w.direction
    near = (np.abs(d) < 2.5 * vox) & (u > w.u0 - vox) & (u < w.u1 + vox)
    if not near.any():
        return []
    # cluster along u, then take a tight rect per cluster
    us, zs = u[near], p[near][:, 2]
    order = np.argsort(us)
    us, zs = us[order], zs[order]
    rects, start = [], 0
    for k in range(1, len(us) + 1):
        if k == len(us) or us[k] - us[k - 1] > 3 * vox:
            cu, cz = us[start:k], zs[start:k]
            if (cu.max() - cu.min()) * (cz.max() - cz.min()) >= 0.15:
                rects.append(
                    {
                        "u0": float(cu.min()),
                        "u1": float(cu.max()),
                        "z0": float(cz.min()),
                        "z1": float(cz.max()),
                    }
                )
            start = k
    return rects


# ---------------------------------------------------------------- stage ----


def refine_scene(scene_dir: str) -> dict:
    with open(os.path.join(scene_dir, "scene.json")) as fh:
        doc = SceneDocument(**json.load(fh))
    vox = doc.shell.vox_m
    d = np.load(os.path.join(scene_dir, "patchgraph.npz"))
    node = d["node_probs"] > doc.shell.node_threshold
    opening_cells = (
        np.argwhere(d["openings"].max(0) > 0.69)
        if "openings" in d.files
        else None
    )
    cloud = trimesh.load(os.path.join(scene_dir, "cloud.ply"))
    cloud_xy = np.asarray(cloud.vertices, np.float64)[:, :2]
    tree = cKDTree(cloud_xy)

    walls = [WallLine(seg, vox) for seg in _wall_segments(node[0], vox)]
    for w in walls:
        w.snap(cloud_xy, tree)
    walls = _merge_collinear(walls)
    walls = _cull_stubs(walls)
    corners = _intersect_corners(walls)

    refined = [w for w in walls if w.refined and w.rms_after is not None]
    report = {
        "walls": len(walls),
        "refined": len(refined),
        "corners_joined": corners,
        "rms_before_mm": round(
            1e3 * float(np.mean([w.rms_before for w in refined])), 1
        )
        if refined
        else None,
        "rms_after_mm": round(
            1e3 * float(np.mean([w.rms_after for w in refined])), 1
        )
        if refined
        else None,
    }
    out = {
        "schema": "roomform.refined-walls.v0",
        "report": report,
        "walls": [
            {
                "p0": w.endpoints()[0].tolist(),
                "p1": w.endpoints()[1].tolist(),
                "z0": w.z0,
                "z1": w.z1,
                "refined": w.refined,
                "support": w.n_support,
                "rms_mm": round(1e3 * w.rms_after, 1) if w.rms_after else None,
                "openings": _wall_openings(w, opening_cells, vox),
            }
            for w in walls
        ],
    }
    with open(os.path.join(scene_dir, "refined.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    args = ap.parse_args()
    print(json.dumps(refine_scene(args.scene_dir), indent=2))


if __name__ == "__main__":
    main()
