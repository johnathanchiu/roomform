"""Boundary graph -> planar shell mesh.

Walls, floors, and ceilings are planes; rendering the 8 cm lattice
directly (voxel boxes, marching cubes) makes fist-sized lumps out of
flat architecture. This module fits a plane per boundary segment and
re-renders the segment's cells as perfectly coplanar quads — crisp
surfaces with 8 cm-stepped outlines, holes where the opening head
fires, and sub-voxel placement from the offset head where available.

v1 scope (ponytail: planar sheets, not CSG solids): horizontal
segments get one plane per connected component; wall voxels are
bucketed by local in-plane direction before region growing so an
L-corner splits into two sheets. Roof-slopes and curved walls fall
back to their dominant plane.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

CLASS_COLORS = {
    "wall": [188, 195, 204, 255],
    "floor": [168, 190, 172, 255],
    "ceiling": [205, 200, 180, 255],
}
MIN_SEGMENT_CELLS = 24


def _horizontal_sheets(mask, offsets, vox, color):
    """Floors/ceilings: one flat sheet per connected component."""
    meshes = []
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    for lab in range(1, n + 1):
        cells = np.argwhere(labels == lab)
        if len(cells) < MIN_SEGMENT_CELLS:
            continue
        z = cells[:, 2].astype(np.float64) + 0.5
        if offsets is not None:
            z = z + offsets[2][tuple(cells.T)]
        plane_z = float(np.median(z)) * vox
        quads = {(c[0], c[1]) for c in cells}
        meshes.append(_quad_sheet_xy(quads, plane_z, vox, color))
    return meshes


def _quad_sheet_xy(cells, z, vox, color):
    verts, faces, index = [], [], {}

    def vid(i, j):
        key = (i, j)
        if key not in index:
            index[key] = len(verts)
            verts.append([i * vox, j * vox, z])
        return index[key]

    for i, j in cells:
        a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
        faces += [[a, b, c], [a, c, d]]
    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, np.float64),
        faces=np.asarray(faces),
        process=False,
    )
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    return mesh


def _wall_sheets(mask, offsets, opening_cells, vox, color):
    """Walls: bucket voxels by local in-plane direction, region-grow,
    fit a vertical plane per segment, emit coplanar quads."""
    cells = np.argwhere(mask)
    if not len(cells):
        return []
    # local wall direction from the 2D footprint: for each voxel, PCA
    # of nearby footprint cells gives the in-plane horizontal axis
    foot = mask.any(2)
    fp = np.argwhere(foot)
    from scipy.spatial import cKDTree

    tree = cKDTree(fp)
    nbrs = tree.query_ball_point(fp, r=3.0)
    theta = np.zeros(len(fp))
    for k, nb in enumerate(nbrs):
        q = fp[nb] - fp[nb].mean(0)
        cov = q.T @ q
        v = np.linalg.eigh(cov)[1][:, -1]  # dominant footprint direction
        theta[k] = np.arctan2(v[1], v[0]) % np.pi
    theta_grid = np.zeros(foot.shape)
    theta_grid[tuple(fp.T)] = theta
    nbins = 12
    bins = np.floor(theta_grid / np.pi * nbins).astype(int) % nbins

    meshes = []
    done = np.zeros(mask.shape, bool)
    for b in range(nbins):
        # voxels whose footprint direction falls in this bin (or wraps)
        sel2d = foot & (
            (bins == b) | (bins == (b + 1) % nbins) | (bins == (b - 1) % nbins)
        )
        sel = mask & sel2d[:, :, None] & ~done
        labels, n = ndimage.label(sel, structure=np.ones((3, 3, 3)))
        for lab in range(1, n + 1):
            seg = np.argwhere(labels == lab)
            if len(seg) < MIN_SEGMENT_CELLS:
                continue
            done[tuple(seg.T)] = True
            pts = seg.astype(np.float64) + 0.5
            if offsets is not None:
                pts = pts + np.stack(
                    [offsets[a][tuple(seg.T)] for a in range(3)], 1
                )
            pts = pts * vox
            centroid = pts.mean(0)
            q = pts - centroid
            # vertical plane: normal is the smallest horizontal axis
            cov = q[:, :2].T @ q[:, :2]
            normal2 = np.linalg.eigh(cov)[1][:, 0]
            u_dir = np.array([-normal2[1], normal2[0], 0.0])
            u = q[:, :2] @ np.array([u_dir[0], u_dir[1]])
            v = pts[:, 2]
            quads = {
                (int(np.floor(a / vox)), int(np.floor(c / vox)))
                for a, c in zip(u, v)
            }
            if opening_cells is not None and len(opening_cells):
                op = opening_cells.astype(np.float64) + 0.5
                opw = op * vox
                rel = opw - centroid
                dist = np.abs(rel[:, :2] @ normal2)
                near = dist < 2.5 * vox
                ou = rel[near][:, :2] @ np.array([u_dir[0], u_dir[1]])
                ov = opw[near][:, 2]
                holes = {
                    (int(np.floor(a / vox)), int(np.floor(c / vox)))
                    for a, c in zip(ou, ov)
                }
                quads -= holes
            if len(quads) < MIN_SEGMENT_CELLS:
                continue
            meshes.append(
                _quad_sheet_plane(quads, centroid, u_dir, vox, color)
            )
    return meshes


def _quad_sheet_plane(quads, centroid, u_dir, vox, color):
    verts, faces, index = [], [], {}

    def vid(iu, iv):
        key = (iu, iv)
        if key not in index:
            index[key] = len(verts)
            verts.append(
                centroid
                + u_dir * (iu * vox)
                + np.array([0.0, 0.0, iv * vox - centroid[2]])
            )
        return index[key]

    for iu, iv in quads:
        a0, b0 = vid(iu, iv), vid(iu + 1, iv)
        c0, d0 = vid(iu + 1, iv + 1), vid(iu, iv + 1)
        faces += [[a0, b0, c0], [a0, c0, d0]]
    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts), faces=np.asarray(faces), process=False
    )
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    return mesh


def planar_shell(
    node_probs: np.ndarray,
    vox: float,
    offsets: np.ndarray | None = None,
    openings: np.ndarray | None = None,
    node_threshold: float = 0.5,
    opening_threshold: float = 0.69,
) -> trimesh.Trimesh | None:
    """Fit planes to the boundary prediction and return one crisp mesh."""
    node = node_probs > node_threshold
    opening_cells = (
        np.argwhere(openings.max(0) > opening_threshold)
        if openings is not None
        else None
    )
    parts = []
    parts += _wall_sheets(
        node[0], offsets, opening_cells, vox, CLASS_COLORS["wall"]
    )
    parts += _horizontal_sheets(node[1], offsets, vox, CLASS_COLORS["floor"])
    parts += _horizontal_sheets(node[2], offsets, vox, CLASS_COLORS["ceiling"])
    if not parts:
        return None
    return trimesh.util.concatenate(parts)
