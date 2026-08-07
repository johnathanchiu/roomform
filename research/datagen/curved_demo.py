"""Synthetic curved-building scans for the demo set.

The training distribution is polygonal (floorplan-derived), so curved
shells are a genuine out-of-distribution probe. Each generator emits a
registered-scan-style colored PLY: surface-sampled walls with door and
window cutouts, floor and ceiling, mild sensor noise, patchy angular
dropout so coverage looks scanned rather than CAD-perfect, and a few
box/cylinder furniture blobs for the lifter.

    uv run python research/datagen/curved_demo.py --out data/scans
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh

RNG = np.random.default_rng(1337)
DENSITY = 5200  # pts per m^2 before dropout
NOISE = 0.005
WALL_TINT = np.array([206, 199, 188])
FLOOR_TINT = np.array([148, 128, 108])
CEIL_TINT = np.array([225, 222, 214])


def _noisy(pts: np.ndarray) -> np.ndarray:
    return pts + RNG.normal(0, NOISE, pts.shape)


def _dropout(pts: np.ndarray, cols: np.ndarray, keep: float = 0.82):
    """Patchy angular dropout: some sectors scan thin, like occlusion."""
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    n_holes = RNG.integers(2, 5)
    mask = RNG.random(len(pts)) < keep
    for _ in range(n_holes):
        a0 = RNG.uniform(-np.pi, np.pi)
        width = RNG.uniform(0.1, 0.35)
        in_hole = np.abs((ang - a0 + np.pi) % (2 * np.pi) - np.pi) < width
        mask &= ~(in_hole & (RNG.random(len(pts)) < 0.85))
    return pts[mask], cols[mask]


def _tinted(n: int, tint: np.ndarray) -> np.ndarray:
    return np.clip(tint + RNG.normal(0, 7, (n, 3)), 0, 255).astype(np.uint8)


def arc_wall(radius, a0, a1, height, center=(0.0, 0.0), doors=(), windows=()):
    """Points on a vertical cylindrical wall segment with cutouts."""
    area = radius * (a1 - a0) * height
    n = int(area * DENSITY)
    ang = RNG.uniform(a0, a1, n)
    z = RNG.uniform(0, height, n)
    keep = np.ones(n, bool)
    for d0, d1 in doors:  # full-height gaps
        keep &= ~((ang > d0) & (ang < d1) & (z < 2.05))
    for w0, w1 in windows:  # sill 0.9 to 1.9
        keep &= ~((ang > w0) & (ang < w1) & (z > 0.9) & (z < 1.9))
    ang, z = ang[keep], z[keep]
    pts = np.stack(
        [
            center[0] + radius * np.cos(ang),
            center[1] + radius * np.sin(ang),
            z,
        ],
        1,
    )
    return _noisy(pts), _tinted(len(pts), WALL_TINT)


def disc(radius, z, tint, center=(0.0, 0.0), r_inner=0.0):
    n = int(np.pi * (radius**2 - r_inner**2) * DENSITY)
    r = np.sqrt(RNG.uniform(r_inner**2, radius**2, n))
    ang = RNG.uniform(-np.pi, np.pi, n)
    pts = np.stack(
        [
            center[0] + r * np.cos(ang),
            center[1] + r * np.sin(ang),
            np.full(n, float(z)),
        ],
        1,
    )
    return _noisy(pts), _tinted(n, tint)


def box_blob(center, size, tint):
    """Surface-sampled axis-aligned box (furniture stand-in)."""
    b = trimesh.creation.box(extents=size)
    b.apply_translation(center)
    n = int(b.area * DENSITY * 0.7)
    pts = trimesh.sample.sample_surface(b, n)[0]
    return _noisy(pts), _tinted(len(pts), np.asarray(tint))


def cyl_blob(center, radius, height, tint):
    c = trimesh.creation.cylinder(radius=radius, height=height)
    c.apply_translation(center)
    n = int(c.area * DENSITY * 0.7)
    pts = trimesh.sample.sample_surface(c, n)[0]
    return _noisy(pts), _tinted(len(pts), np.asarray(tint))


def prop(name, pos, yaw=0.0, upsample=5):
    """A real scanned furniture chunk (data/props/<name>.ply), jitter-
    upsampled to scan density, yaw-rotated, grounded at pos (z=floor)."""
    p = trimesh.load(f"data/props/{name}.ply")
    pts = np.asarray(p.vertices)
    cols = np.asarray(p.colors)[:, :3]
    pts = np.vstack(
        [pts + RNG.normal(0, 0.008, pts.shape) for _ in range(upsample)]
    )
    cols = np.vstack([cols] * upsample)
    c, sn = np.cos(yaw), np.sin(yaw)
    pts = pts @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]]).T
    pts[:, 2] -= pts[:, 2].min()
    pts = pts + np.asarray(pos, np.float64)
    return pts, np.clip(cols, 0, 255).astype(np.uint8)


def _assemble(parts, out_path):
    pts = np.vstack([p for p, _ in parts])
    cols = np.vstack([c for _, c in parts])
    pts, cols = _dropout(pts, cols)
    cloud = trimesh.PointCloud(
        pts, colors=np.hstack([cols, np.full((len(cols), 1), 255, np.uint8)])
    )
    cloud.export(out_path)
    print(f"{out_path}: {len(pts):,} pts")


def rotunda(out):
    """Single round room, one door, three windows, some furniture."""
    h, r = 2.75, 4.2
    parts = [
        arc_wall(
            r,
            -np.pi,
            np.pi,
            h,
            doors=[(0.1, 0.35)],
            windows=[(1.2, 1.7), (2.4, 2.9), (-2.0, -1.5)],
        ),
        disc(r, 0.0, FLOOR_TINT),
        disc(r, h, CEIL_TINT),
        box_blob([1.6, -1.2, 0.38], [1.9, 0.9, 0.76], [120, 90, 70]),
        box_blob([-1.8, 1.4, 0.23], [1.1, 0.6, 0.46], [90, 100, 120]),
        cyl_blob([-1.2, -2.0, 0.55], 0.35, 1.1, [70, 110, 80]),
    ]
    _assemble(parts, out)


def ellipse_hall(out):
    """Elliptical hall: circle scaled after assembly, two doors."""
    h, r = 2.9, 4.0
    parts = [
        arc_wall(
            r,
            -np.pi,
            np.pi,
            h,
            doors=[(-0.15, 0.12), (np.pi - 0.15, np.pi + 0.1)],
            windows=[(0.8, 1.25), (-1.4, -0.9)],
        ),
        disc(r, 0.0, FLOOR_TINT),
        disc(r, h, CEIL_TINT),
        box_blob([0.0, 0.0, 0.37], [2.6, 1.1, 0.74], [110, 85, 70]),
        box_blob([2.0, 1.3, 0.25], [0.8, 0.8, 0.5], [95, 105, 125]),
    ]
    pts = np.vstack([p for p, _ in parts])
    cols = np.vstack([c for _, c in parts])
    pts[:, 0] *= 1.55  # stretch to an ellipse
    pts, cols = _dropout(pts, cols)
    trimesh.PointCloud(
        pts, colors=np.hstack([cols, np.full((len(cols), 1), 255, np.uint8)])
    ).export(out)
    print(f"{out}: {len(pts):,} pts")


def semicircle_annex(out):
    """Semicircular bay joined to a rectangular room."""
    h, r = 2.7, 3.2
    parts = [
        arc_wall(
            r, -np.pi / 2, np.pi / 2, h, windows=[(-0.5, -0.1), (0.2, 0.6)]
        ),
        disc(r, 0.0, FLOOR_TINT),
        disc(r, h, CEIL_TINT),
    ]
    # rectangular annex on the flat side (x<0)
    w, dpt = 2 * r, 4.5
    for x in (0.0, -dpt):
        n = int(w * h * DENSITY)
        y = RNG.uniform(-r, r, n)
        z = RNG.uniform(0, h, n)
        keep = (
            np.ones(n, bool) if x == 0.0 else ~((np.abs(y) < 0.5) & (z < 2.05))
        )
        if x == 0.0:  # opening between bay and annex
            keep = ~((np.abs(y) < 1.3) & (z < 2.2))
        pts = np.stack([np.full(n, x), y, z], 1)[keep]
        parts.append((_noisy(pts), _tinted(len(pts), WALL_TINT)))
    for y in (-r, r):
        n = int(dpt * h * DENSITY)
        x = RNG.uniform(-dpt, 0, n)
        z = RNG.uniform(0, h, n)
        pts = np.stack([x, np.full(n, y), z], 1)
        parts.append((_noisy(pts), _tinted(n, WALL_TINT)))
    n = int(dpt * w * DENSITY)
    xy = np.stack([RNG.uniform(-dpt, 0, n), RNG.uniform(-r, r, n)], 1)
    parts.append(
        (_noisy(np.column_stack([xy, np.zeros(n)])), _tinted(n, FLOOR_TINT))
    )
    parts.append(
        (_noisy(np.column_stack([xy, np.full(n, h)])), _tinted(n, CEIL_TINT))
    )
    parts.append(box_blob([-2.2, 0.9, 0.37], [1.8, 0.9, 0.74], [115, 88, 72]))
    parts.append(box_blob([1.6, -0.8, 0.3], [1.0, 1.0, 0.6], [90, 100, 120]))
    _assemble(parts, out)


def double_lobe(out):
    """Two overlapping round rooms, figure-eight footprint."""
    h = 2.8
    for cx, r in ((-2.4, 3.4), (2.6, 2.8)):
        other_cx, other_r = (2.6, 2.8) if cx < 0 else (-2.4, 3.4)
        pts_w, cols_w = arc_wall(
            r,
            -np.pi,
            np.pi,
            h,
            center=(cx, 0),
            windows=[(0.6, 1.0)] if cx > 0 else [],
        )
        inside_other = ((pts_w[:, 0] - other_cx) ** 2 + pts_w[:, 1] ** 2) < (
            other_r - 0.05
        ) ** 2
        yield_part = (pts_w[~inside_other], cols_w[~inside_other])
        if cx < 0:
            parts = [yield_part]
        else:
            parts.append(yield_part)
        f_pts, f_cols = disc(r, 0.0, FLOOR_TINT, center=(cx, 0))
        c_pts, c_cols = disc(r, h, CEIL_TINT, center=(cx, 0))
        parts += [(f_pts, f_cols), (c_pts, c_cols)]
    parts.append(
        box_blob([-2.6, -1.0, 0.37], [2.0, 0.95, 0.74], [118, 90, 70])
    )
    parts.append(cyl_blob([2.8, 0.6, 0.5], 0.45, 1.0, [80, 105, 90]))
    _assemble(parts, out)


def curved_corridor(out):
    """Quarter-arc corridor between two wall radii."""
    h = 2.6
    r_in, r_out = 3.0, 5.2
    parts = [
        arc_wall(r_in, 0.1, np.pi / 2 - 0.1, h),
        arc_wall(
            r_out,
            0.0,
            np.pi / 2,
            h,
            doors=[(0.02, 0.16)],
            windows=[(0.7, 0.95)],
        ),
        disc(r_out, 0.0, FLOOR_TINT, r_inner=r_in),
        disc(r_out, h, CEIL_TINT, r_inner=r_in),
        box_blob([3.6, 1.6, 0.25], [0.9, 0.5, 0.5], [100, 95, 115]),
    ]
    _assemble(parts, out)


def rotunda_annex(out):
    """Round hall conjoined with a rectangular wing: one building,
    curved and straight walls meeting — the robustness statement."""
    h, r = 2.75, 3.6
    open_half = 0.62  # angular half-width of the junction opening
    parts = [
        arc_wall(
            r,
            open_half,
            2 * np.pi - open_half,
            h,
            doors=[(np.pi - 0.14, np.pi + 0.14)],
            windows=[(np.pi / 2 - 0.25, np.pi / 2 + 0.2), (4.2, 4.65)],
        ),
        disc(r, 0.0, FLOOR_TINT),
        disc(r, h, CEIL_TINT),
        prop("sofa", [-1.2, 1.0, 0.0], yaw=0.6),
        prop("table", [0.5, -1.5, 0.0], yaw=0.2),
        prop("chair", [1.3, -1.9, 0.0], yaw=2.4),
    ]
    # rectangular wing on +x, sharing the junction opening
    x0 = r * np.cos(open_half) - 0.05
    x1, yw = x0 + 4.6, 2.1
    for y in (-yw, yw):
        n = int((x1 - x0) * h * DENSITY)
        x = RNG.uniform(x0, x1, n)
        z = RNG.uniform(0, h, n)
        pts = np.stack([x, np.full(n, y), z], 1)
        parts.append((_noisy(pts), _tinted(n, WALL_TINT)))
    n = int(2 * yw * h * DENSITY)  # end wall with a window
    y = RNG.uniform(-yw, yw, n)
    z = RNG.uniform(0, h, n)
    keep = ~((np.abs(y) < 0.7) & (z > 0.9) & (z < 1.9))
    pts = np.stack([np.full(n, x1), y, z], 1)[keep]
    parts.append((_noisy(pts), _tinted(len(pts), WALL_TINT)))
    n = int((x1 - x0) * 2 * yw * DENSITY)
    xy = np.stack([RNG.uniform(x0, x1, n), RNG.uniform(-yw, yw, n)], 1)
    parts.append(
        (_noisy(np.column_stack([xy, np.zeros(n)])), _tinted(n, FLOOR_TINT))
    )
    parts.append(
        (_noisy(np.column_stack([xy, np.full(n, h)])), _tinted(n, CEIL_TINT))
    )
    parts.append(prop("bed", [5.1, -0.7, 0.0], yaw=0.1))
    parts.append(prop("chair", [4.2, 1.2, 0.0], yaw=-1.2))
    _assemble(parts, out)


GENERATORS = {
    "synthetic-rotunda": rotunda,
    "synthetic-rotunda-annex": rotunda_annex,
    "synthetic-ellipse-hall": ellipse_hall,
    "synthetic-semicircle-annex": semicircle_annex,
    "synthetic-double-lobe": double_lobe,
    "synthetic-curved-corridor": curved_corridor,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scans")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, gen in GENERATORS.items():
        if args.only and args.only not in name:
            continue
        gen(os.path.join(args.out, f"{name}.ply"))


if __name__ == "__main__":
    main()
