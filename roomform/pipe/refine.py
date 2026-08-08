"""Evidence-snapped wall refinement: room contours -> clean lines.

The boundary model predicts walls as 6-8 cm voxel bands: stepped
outlines, half-voxel jitter, soft corners. Fitting lines to wall-cell
clusters directly inherits their fragmentation — dangling stubs, no
topology. The floor prediction is the better scaffold: it is dense and
closed (the model fills under furniture and clutter), so each room's
floor contour IS the wall line, a closed loop by construction.

1. rooms = connected components of the (hole-filled) floor footprint,
2. trace each room's boundary (Moore neighborhood), simplify with
   Douglas-Peucker — curved walls keep enough vertices to stay a
   facet chain, straight walls collapse to single edges,
3. snap every polygon edge to the raw scan: Tukey-reweighted 2D line
   fit over points in a band around the edge (support + residual
   gates; ungated edges keep the traced line),
4. re-intersect consecutive edges so corners are exact line-line
   intersections — no right-angle prior anywhere,
5. project opening cells onto the refined edges as u/z rectangles.

Output: ``refined.json`` — per-room closed wall loops with a per-edge
fit report.

    uv run python -m roomform.pipe.refine artifacts/<scene>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from roomform.contracts import SceneDocument

BAND_M = 0.12  # evidence band around a traced edge
MIN_INLIERS_PER_M = 90  # supporting points per meter, floor of 50
MAX_RMS_M = 0.02  # worse residual -> keep the traced line
SIMPLIFY_EPS_CELLS = 0.8  # Douglas-Peucker tolerance, in voxels
MIN_ROOM_M2 = 2.0
MIN_EDGE_M = 0.25  # shorter traced edges merge into neighbors
MAX_CORNER_SHIFT_M = 0.35  # cap how far re-intersection may move a vertex


# ------------------------------------------------------- contour tracing ----


def _trace_boundary(mask: np.ndarray) -> np.ndarray:
    """Boundary of one connected component via marching squares.

    Returns the longest closed contour as an (n, 2) array of
    sub-cell coordinates in trace order.
    """
    from skimage import measure

    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.empty((0, 2))
    return max(contours, key=len)[:-1]  # drop duplicated closing vertex


def _douglas_peucker(pts: np.ndarray, eps: float) -> np.ndarray:
    """Iterative DP on a closed polyline (indices kept, order preserved)."""

    def simplify(lo: int, hi: int, keep: np.ndarray) -> None:
        stack = [(lo, hi)]
        while stack:
            a, b = stack.pop()
            if b <= a + 1:
                continue
            seg = pts[b] - pts[a]
            n = np.linalg.norm(seg)
            if n < 1e-9:
                d = np.linalg.norm(pts[a + 1 : b] - pts[a], axis=1)
            else:
                d = np.abs(np.cross(seg / n, pts[a + 1 : b] - pts[a]))
            k = int(np.argmax(d))
            if d[k] > eps:
                keep[a + 1 + k] = True
                stack += [(a, a + 1 + k), (a + 1 + k, b)]

    # anchor at the two most distant points so the closed loop splits well
    far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    keep = np.zeros(len(pts), bool)
    keep[[0, far]] = True
    simplify(0, far, keep)
    # wrap-around half
    keep_r = np.zeros(len(pts), bool)
    keep_r[[0, len(pts) - far]] = True
    simplify(0, len(pts) - far, keep_r)
    keep |= np.roll(keep_r, far)
    return np.nonzero(keep)[0]


# ---------------------------------------------------------------- edges ----


class Edge:
    """One wall edge of a room loop, snappable to scan evidence."""

    def __init__(self, p0: np.ndarray, p1: np.ndarray):
        self.p0, self.p1 = p0.copy(), p1.copy()
        self.refined = False
        self.rms_before = None
        self.rms_after = None
        self.n_support = 0

    @property
    def direction(self) -> np.ndarray:
        d = self.p1 - self.p0
        return d / max(np.linalg.norm(d), 1e-9)

    @property
    def normal(self) -> np.ndarray:
        d = self.direction
        return np.array([-d[1], d[0]])

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def snap(self, cloud_xy: np.ndarray, tree: cKDTree) -> None:
        mid = (self.p0 + self.p1) / 2
        idx = tree.query_ball_point(mid, r=self.length / 2 + BAND_M)
        if not idx:
            return
        p = cloud_xy[idx]
        rel = p - mid
        u, d = rel @ self.direction, rel @ self.normal
        near = (np.abs(d) < BAND_M) & (np.abs(u) < self.length / 2 + BAND_M)
        p = p[near]
        min_inliers = max(50, int(MIN_INLIERS_PER_M * self.length))
        if len(p) < min_inliers:
            return
        self.rms_before = float(np.sqrt(np.mean(d[near] ** 2)))
        center, direction = mid, self.direction
        for _ in range(3):  # Tukey-reweighted PCA refit
            rel = p - center
            dd = rel @ np.array([-direction[1], direction[0]])
            s = max(1.4826 * np.median(np.abs(dd)), 3e-3)
            w = np.clip(1 - (dd / (4 * s)) ** 2, 0, None) ** 2
            if w.sum() < min_inliers / 2:
                return
            center = (p * w[:, None]).sum(0) / w.sum()
            q = (p - center) * np.sqrt(w)[:, None]
            direction = np.linalg.eigh(q.T @ q)[1][:, -1]
        rel = p - center
        dd = rel @ np.array([-direction[1], direction[0]])
        keep = np.abs(dd) < 3 * max(1.4826 * np.median(np.abs(dd)), 3e-3)
        rms = float(np.sqrt(np.mean(dd[keep] ** 2)))
        if keep.sum() < min_inliers or rms > MAX_RMS_M:
            return
        if direction @ self.direction < 0:
            direction = -direction
        # slide endpoints onto the refined line (projection, not clamp:
        # corners are re-derived by intersection right after)
        self.p0 = center + direction * ((self.p0 - center) @ direction)
        self.p1 = center + direction * ((self.p1 - center) @ direction)
        self.refined = True
        self.rms_after = rms
        self.n_support = int(keep.sum())


def _merge_short_edges(edges: list[Edge]) -> list[Edge]:
    """Fold sub-threshold edges into their longer neighbor."""
    out = []
    for e in edges:
        if out and e.length < MIN_EDGE_M:
            out[-1].p1 = e.p1
            continue
        out.append(e)
    if len(out) > 1 and out[0].length < MIN_EDGE_M:
        out[-1].p1 = out[0].p1
        out.pop(0)
    return out


def _reintersect(edges: list[Edge]) -> int:
    """Corner = intersection of consecutive edge lines (capped shift)."""
    joined = 0
    n = len(edges)
    for i in range(n):
        a, b = edges[i], edges[(i + 1) % n]
        A = np.stack([a.direction, -b.direction], 1)
        if abs(np.linalg.det(A)) < 0.17:  # near-parallel (<10 deg)
            mid = (a.p1 + b.p0) / 2
            a.p1, b.p0 = mid.copy(), mid.copy()
            continue
        ta, _ = np.linalg.solve(A, b.p0 - a.p0)
        corner = a.p0 + a.direction * ta
        if np.linalg.norm(corner - a.p1) > MAX_CORNER_SHIFT_M:
            mid = (a.p1 + b.p0) / 2
            a.p1, b.p0 = mid.copy(), mid.copy()
            continue
        a.p1, b.p0 = corner.copy(), corner.copy()
        joined += 1
    return joined


# ------------------------------------------------------------- openings ----


def _edge_openings(e: Edge, opening_cells: np.ndarray, vox: float):
    if opening_cells is None or not len(opening_cells):
        return []
    p = (opening_cells.astype(np.float64) + 0.5) * vox
    mid = (e.p0 + e.p1) / 2
    rel = p[:, :2] - mid
    d, u = rel @ e.normal, rel @ e.direction
    near = (np.abs(d) < 2.5 * vox) & (np.abs(u) < e.length / 2 + vox)
    if not near.any():
        return []
    us, zs = u[near] + e.length / 2, p[near][:, 2]
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

    # rooms from the hole-filled floor footprint; walls thicken the
    # floor edge outward so the traced contour sits ON the wall band
    floor = ndimage.binary_fill_holes(node[1].any(2))
    floor = ndimage.binary_closing(floor, np.ones((3, 3)), iterations=2)
    labels, n = ndimage.label(floor)
    z_lo = (
        float(np.argwhere(node[1])[:, 2].min() + 0.5) * vox
        if node[1].any()
        else 0.0
    )
    z_hi = (
        float(np.argwhere(node[2])[:, 2].mean() + 0.5) * vox
        if node[2].any()
        else 2.6
    )

    rooms = []
    eps = SIMPLIFY_EPS_CELLS
    for lab in range(1, n + 1):
        mask = labels == lab
        if mask.sum() * vox * vox < MIN_ROOM_M2:
            continue
        contour = _trace_boundary(mask)
        if len(contour) < 8:
            continue
        keep = _douglas_peucker(contour, eps)
        verts = (contour[np.sort(keep)] + 0.5) * vox
        edges = [
            Edge(verts[i], verts[(i + 1) % len(verts)])
            for i in range(len(verts))
        ]
        edges = _merge_short_edges(edges)
        for e in edges:
            e.snap(cloud_xy, tree)
        corners = _reintersect(edges)
        rooms.append({"edges": edges, "corners": corners})

    partitions = _interior_walls(node[0], vox, rooms, cloud_xy, tree)
    all_edges = [e for r in rooms for e in r["edges"]] + partitions
    refined = [e for e in all_edges if e.refined and e.rms_after]
    report = {
        "rooms": len(rooms),
        "partitions": len(partitions),
        "edges": len(all_edges),
        "refined": len(refined),
        "rms_before_mm": round(
            1e3 * float(np.mean([e.rms_before for e in refined])), 1
        )
        if refined
        else None,
        "rms_after_mm": round(
            1e3 * float(np.mean([e.rms_after for e in refined])), 1
        )
        if refined
        else None,
    }
    out = {
        "schema": "roomform.refined-walls.v1",
        "report": report,
        "z0": z_lo,
        "z1": z_hi,
        "rooms": [
            {
                "loop": [
                    {
                        "p0": e.p0.tolist(),
                        "p1": e.p1.tolist(),
                        "refined": e.refined,
                        "support": e.n_support,
                        "rms_mm": round(1e3 * e.rms_after, 1)
                        if e.rms_after
                        else None,
                        "openings": _edge_openings(e, opening_cells, vox),
                    }
                    for e in r["edges"]
                ]
            }
            for r in rooms
        ],
        "partitions": [
            {
                "p0": e.p0.tolist(),
                "p1": e.p1.tolist(),
                "refined": e.refined,
                "support": e.n_support,
                "rms_mm": round(1e3 * e.rms_after, 1) if e.rms_after else None,
                "openings": _edge_openings(e, opening_cells, vox),
            }
            for e in partitions
        ],
    }
    with open(os.path.join(scene_dir, "refined.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return report


def _interior_walls(node_wall, vox, rooms, cloud_xy, tree):
    """Partition walls the room loops cannot see.

    Interior walls sit between doorway-connected rooms, so they never
    appear on a floor contour. Fit them as standalone segments from
    wall cells far from every loop edge, and keep one only when the
    scan evidence confirms it (refined) or it is long enough to be
    structure rather than noise."""
    from scipy import ndimage as ndi

    foot = node_wall.any(2)
    cells = np.argwhere(foot)
    if not len(cells):
        return []
    pts = (cells.astype(np.float64) + 0.5) * vox
    edges = [e for r in rooms for e in r["edges"]]
    far = np.ones(len(pts), bool)
    for e in edges:
        mid = (e.p0 + e.p1) / 2
        rel = pts - mid
        u, d = rel @ e.direction, rel @ e.normal
        near = (np.abs(d) < 5 * vox) & (np.abs(u) < e.length / 2 + 5 * vox)
        far &= ~near
    interior = np.zeros_like(foot)
    interior[tuple(cells[far].T)] = True
    labels, n = ndi.label(interior, structure=np.ones((3, 3)))
    out = []
    for lab in range(1, n + 1):
        seg = np.argwhere(labels == lab)
        if len(seg) < 12:
            continue
        p = (seg.astype(np.float64) + 0.5) * vox
        center = p.mean(0)
        q = p - center
        direction = np.linalg.eigh(q.T @ q)[1][:, -1]
        u = q @ direction
        # straightness gate: a chord through an arc or a line chained
        # across scattered cells has high perpendicular residual
        resid = q @ np.array([-direction[1], direction[0]])
        if float(np.sqrt(np.mean(resid**2))) > 1.5 * vox:
            continue
        e = Edge(center + direction * u.min(), center + direction * u.max())
        # a partition lies inside a room by definition; anything outside
        # every loop is window-band or exterior residue
        mid = (e.p0 + e.p1) / 2
        inside = False
        for r in rooms:
            poly = np.array([ed.p0 for ed in r["edges"]])
            j = np.arange(len(poly))
            k = (j + 1) % len(poly)
            cond = (poly[j, 1] > mid[1]) != (poly[k, 1] > mid[1])
            xs = poly[j, 0] + (mid[1] - poly[j, 1]) / (
                poly[k, 1] - poly[j, 1] + 1e-12
            ) * (poly[k, 0] - poly[j, 0])
            if (cond & (mid[0] < xs)).sum() % 2 == 1:
                inside = True
                break
        if not inside:
            continue
        e.snap(cloud_xy, tree)
        if e.refined or e.length > 1.0:
            out.append(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    args = ap.parse_args()
    print(json.dumps(refine_scene(args.scene_dir), indent=2))


if __name__ == "__main__":
    main()
