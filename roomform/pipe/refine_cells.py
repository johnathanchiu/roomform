"""Cell-complex wall refinement: global lines + room labeling.

The contour tracer (refine.py) gets clean envelopes but cannot see
interior partitions: doorway-connected rooms share one floor
component, so shared walls never appear on a contour. This prototype
takes the arrangement view instead:

1. detect global wall lines by sequential RANSAC over the 2D wall
   footprint (inlier removal between rounds), then snap each line to
   the raw scan with the same Tukey-reweighted refit the contour
   version uses,
2. rasterize the *supported* spans of those lines (plus predicted
   door cells) into a barrier mask; rooms = connected components of
   the floor with barriers removed — interior walls now split rooms,
3. vectorize each room boundary by assigning every traced boundary
   vertex to its nearest supporting line; maximal same-line runs
   become edges, and consecutive edges meet at exact line-line
   intersections. Edges between two rooms are the partitions.

No Manhattan or right-angle prior anywhere: a curved wall RANSACs
into a chord chain and vectorizes as a facet polyline.

    uv run python -m roomform.pipe.refine_cells artifacts/<scene>
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

MIN_LINE_CELLS = 14  # RANSAC keeps lines with at least this many cells
RANSAC_TOL_CELLS = 1.2  # inlier band around a candidate line
MAX_LINES = 60
BAND_M = 0.12  # evidence band for the scan snap
MAX_RMS_M = 0.02
MIN_ROOM_M2 = 1.5
ASSIGN_TOL_CELLS = 2.6  # boundary vertex -> line assignment radius
MIN_RUN = 3  # boundary vertices before a run becomes an edge


# ------------------------------------------------------------ line pool ----


class Line:
    """An infinite 2D line with its supporting wall cells."""

    def __init__(
        self, point: np.ndarray, direction: np.ndarray, cells: np.ndarray
    ):
        self.point = point
        self.direction = direction / max(np.linalg.norm(direction), 1e-9)
        self.cells = cells
        self.refined = False
        self.rms_after = None

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def dist(self, pts: np.ndarray) -> np.ndarray:
        return np.abs((pts - self.point) @ self.normal)

    def snap(self, cloud_xy: np.ndarray, tree: cKDTree, vox: float) -> None:
        """Refit against scan points near the supported span."""
        span = self.cells * vox
        u = (span - self.point) @ self.direction
        mid = self.point + self.direction * (u.min() + u.max()) / 2
        half = (u.max() - u.min()) / 2 + BAND_M
        idx = tree.query_ball_point(mid, r=half + BAND_M)
        if not idx:
            return
        p = cloud_xy[idx]
        rel = p - self.point
        uu, dd = rel @ self.direction, rel @ self.normal
        near = (
            (np.abs(dd) < BAND_M)
            & (uu > u.min() - BAND_M)
            & (uu < u.max() + BAND_M)
        )
        p = p[near]
        if len(p) < max(50, 10 * len(self.cells) * vox):
            return
        center, direction = mid, self.direction
        for _ in range(3):
            rel = p - center
            d = rel @ np.array([-direction[1], direction[0]])
            s = max(1.4826 * np.median(np.abs(d)), 3e-3)
            w = np.clip(1 - (d / (4 * s)) ** 2, 0, None) ** 2
            if w.sum() < 25:
                return
            center = (p * w[:, None]).sum(0) / w.sum()
            q = (p - center) * np.sqrt(w)[:, None]
            direction = np.linalg.eigh(q.T @ q)[1][:, -1]
        rel = p - center
        d = rel @ np.array([-direction[1], direction[0]])
        keep = np.abs(d) < 3 * max(1.4826 * np.median(np.abs(d)), 3e-3)
        rms = float(np.sqrt(np.mean(d[keep] ** 2)))
        if rms > MAX_RMS_M:
            return
        if direction @ self.direction < 0:
            direction = -direction
        self.point, self.direction = center, direction
        self.refined = True
        self.rms_after = rms


def _ransac_lines(
    foot_cells: np.ndarray, rng: np.random.Generator
) -> list[Line]:
    """Sequential RANSAC: fit, claim inliers, repeat."""
    remaining = foot_cells.astype(np.float64) + 0.5
    lines: list[Line] = []
    while len(remaining) >= MIN_LINE_CELLS and len(lines) < MAX_LINES:
        best = None
        for _ in range(220):
            a, b = remaining[rng.integers(len(remaining), size=2)]
            d = b - a
            n = np.linalg.norm(d)
            if n < 2.0:  # degenerate or too-local sample
                continue
            d = d / n
            dist = np.abs((remaining - a) @ np.array([-d[1], d[0]]))
            inl = dist < RANSAC_TOL_CELLS
            # connectivity along the line: penalize gappy support so a
            # line cannot stitch two distant parallel walls together
            u = (remaining[inl] - a) @ d
            if inl.sum() < MIN_LINE_CELLS:
                continue
            order = np.sort(u)
            gaps = np.diff(order)
            longest = np.split(
                np.arange(len(order)), np.nonzero(gaps > 6.0)[0] + 1
            )
            seg = max(longest, key=len)
            if len(seg) < MIN_LINE_CELLS:
                continue
            score = len(seg)
            if best is None or score > best[0]:
                keep_idx = np.nonzero(inl)[0][np.argsort(u)[seg]]
                best = (score, a, d, keep_idx)
        if best is None:
            break
        _, a, d, keep_idx = best
        pts = remaining[keep_idx]
        # PCA refit on the claimed cells
        c = pts.mean(0)
        q = pts - c
        d = np.linalg.eigh(q.T @ q)[1][:, -1]
        lines.append(Line(c, d, pts.copy()))
        mask = np.ones(len(remaining), bool)
        mask[keep_idx] = False
        remaining = remaining[mask]
    return lines


# ------------------------------------------------------- rooms + edges ----


def _room_labels(floor: np.ndarray, lines: list[Line], door_cells, vox: float):
    """Rooms = floor components after cutting along supported lines."""
    barrier = np.zeros_like(floor)
    for ln in lines:
        # rasterize the supported span with 1-cell dilation
        ij = np.round(ln.cells - 0.5).astype(int)
        ij = ij[
            (ij[:, 0] >= 0)
            & (ij[:, 1] >= 0)
            & (ij[:, 0] < floor.shape[0])
            & (ij[:, 1] < floor.shape[1])
        ]
        barrier[tuple(ij.T)] = True
    if door_cells is not None and len(door_cells):
        ij = door_cells[:, :2]
        ok = (
            (ij[:, 0] >= 0)
            & (ij[:, 1] >= 0)
            & (ij[:, 0] < floor.shape[0])
            & (ij[:, 1] < floor.shape[1])
        )
        barrier[tuple(ij[ok].T)] = True
    barrier = ndimage.binary_dilation(barrier, np.ones((3, 3)))
    labels, n = ndimage.label(floor & ~barrier)
    # grow rooms back over the barrier band so boundaries meet
    idx = ndimage.distance_transform_edt(
        labels == 0, return_indices=True, return_distances=False
    )
    labels = labels[tuple(idx)]
    labels[~floor] = 0
    return labels, n


def _trace_boundary(mask: np.ndarray) -> np.ndarray:
    from skimage import measure

    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.empty((0, 2))
    return max(contours, key=len)[:-1]


def _vectorize_room(mask: np.ndarray, lines: list[Line], vox: float):
    """Boundary -> maximal same-line runs -> intersected polygon."""
    contour = _trace_boundary(mask)
    if len(contour) < 8:
        return []
    pts = contour + 0.5
    # nearest line per boundary vertex (None when too far from any)
    assign = np.full(len(pts), -1)
    for li, ln in enumerate(lines):
        d = ln.dist(pts)
        take = (d < ASSIGN_TOL_CELLS) & (
            (assign == -1)
            | (
                d
                < np.array(
                    [
                        lines[a].dist(p[None])[0] if a >= 0 else 1e9
                        for a, p in zip(assign, pts)
                    ]
                )
            )
        )
        assign[take] = li
    # compress into runs (circular)
    n = len(pts)
    start = 0
    while start < n and assign[start] == assign[start - 1]:
        start += 1  # begin at a run boundary
    if start == n:
        start = 0
    runs = []
    i = start
    for _ in range(n):
        j = i
        cur = assign[i]
        length = 0
        while assign[j % n] == cur and length < n:
            j += 1
            length += 1
        runs.append((cur, i % n, length))
        i = j % n
        if i == start:
            break
    # keep runs in boundary order; merge only CONSECUTIVE same-line
    # runs (a line visited twice on opposite sides of a room must stay
    # two edges, or the polygon bowties)
    ordered = []
    for cur, i0, length in runs:
        if cur < 0 or length < MIN_RUN:
            continue
        seg = pts[[k % n for k in range(i0, i0 + length)]]
        ln = lines[cur]
        u = (seg - ln.point) @ ln.direction
        if ordered and ordered[-1]["line"] == cur:
            ordered[-1]["u0"] = min(ordered[-1]["u0"], float(u.min()))
            ordered[-1]["u1"] = max(ordered[-1]["u1"], float(u.max()))
            continue
        ordered.append(
            {"line": cur, "u0": float(u.min()), "u1": float(u.max())}
        )
    if len(ordered) > 1 and ordered[0]["line"] == ordered[-1]["line"]:
        ordered[0]["u0"] = min(ordered[0]["u0"], ordered[-1]["u0"])
        ordered[0]["u1"] = max(ordered[0]["u1"], ordered[-1]["u1"])
        ordered.pop()
    poly = []
    for r in ordered:
        ln = lines[r["line"]]
        p0 = ln.point + ln.direction * r["u0"]
        p1 = ln.point + ln.direction * r["u1"]
        poly.append({"line": r["line"], "p0": p0, "p1": p1})
    # exact corners: consecutive edges intersect
    m = len(poly)
    for k in range(m):
        a, b = poly[k], poly[(k + 1) % m]
        la, lb = lines[a["line"]], lines[b["line"]]
        A = np.stack([la.direction, -lb.direction], 1)
        if abs(np.linalg.det(A)) < 0.15:
            continue
        ta, _tb = np.linalg.solve(A, lb.point - la.point)
        corner = la.point + la.direction * ta
        if (
            np.linalg.norm(corner - a["p1"]) < 6.0
            and np.linalg.norm(corner - b["p0"]) < 6.0
        ):
            a["p1"] = corner
            b["p0"] = corner
    return poly


# ---------------------------------------------------------------- stage ----


def refine_scene(scene_dir: str) -> dict:
    with open(os.path.join(scene_dir, "scene.json")) as fh:
        doc = SceneDocument(**json.load(fh))
    vox = doc.shell.vox_m
    d = np.load(os.path.join(scene_dir, "patchgraph.npz"))
    node = d["node_probs"] > doc.shell.node_threshold
    door_cells = (
        np.argwhere(d["openings"][0] > 0.69) if "openings" in d.files else None
    )
    cloud = trimesh.load(os.path.join(scene_dir, "cloud.ply"))
    cloud_xy = np.asarray(cloud.vertices, np.float64)[:, :2]
    tree = cKDTree(cloud_xy)

    rng = np.random.default_rng(7)
    foot = np.argwhere(node[0].any(2))
    lines = _ransac_lines(foot, rng)
    for ln in lines:
        # cells are in cell units; snap in meters
        ln.point = ln.point * vox
        ln.cells_m = ln.cells * vox
        ln.snap(cloud_xy, tree, vox)
        ln.point = ln.point / vox  # back to cell units for labeling
    floor = ndimage.binary_fill_holes(node[1].any(2))
    labels, n = _room_labels(floor, lines, door_cells, vox)

    rooms = []
    for lab in range(1, n + 1):
        mask = labels == lab
        if mask.sum() * vox * vox < MIN_ROOM_M2:
            continue
        poly = _vectorize_room(mask, lines, vox)
        if len(poly) >= 3:
            rooms.append(poly)

    refined = [ln for ln in lines if ln.refined]
    report = {
        "lines": len(lines),
        "snapped": len(refined),
        "rooms": len(rooms),
        "mean_rms_mm": round(
            1e3 * float(np.mean([ln.rms_after for ln in refined])), 1
        )
        if refined
        else None,
    }
    out = {
        "schema": "roomform.refined-cells.v0",
        "report": report,
        "rooms": [
            {
                "loop": [
                    {
                        "p0": (np.asarray(e["p0"]) * vox).tolist(),
                        "p1": (np.asarray(e["p1"]) * vox).tolist(),
                        "refined": bool(lines[e["line"]].refined),
                    }
                    for e in poly
                ]
            }
            for poly in rooms
        ],
    }
    with open(os.path.join(scene_dir, "refined-cells.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    args = ap.parse_args()
    print(json.dumps(refine_scene(args.scene_dir), indent=2))


if __name__ == "__main__":
    main()
