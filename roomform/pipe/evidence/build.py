"""ScanInput -> EvidenceGrid: voxelize a point cloud into the
observable channels.

v0 limitations, by design: without stations (or a ported carve),
``sf`` is empty and everything non-occupied is unknown —
``visibility_source="synthesized"`` marks the degradation. Normals are
used when present in the cloud, else zero-filled (channel-dropout
territory).
"""

from __future__ import annotations

import numpy as np
import trimesh

from roomform.contracts import EvidenceGrid


def build_evidence(
    scan_path: str, out_npz: str, vox_m: float = 0.08
) -> EvidenceGrid:
    mesh = trimesh.load(scan_path)
    pts = np.asarray(mesh.vertices, dtype=np.float32)
    origin = pts.min(0) - vox_m
    idx = np.floor((pts - origin) / vox_m).astype(np.int64)
    shape = tuple(int(v) for v in idx.max(0) + 2)

    occ = np.zeros(shape, np.uint8)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    flat = np.ravel_multi_index(idx.T, shape)
    counts = np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape)
    log_density = np.log1p(counts).astype(np.float16)

    gray = np.zeros(shape, np.uint8)
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is not None and len(colors) == len(pts):
        lum = np.asarray(colors)[:, :3].mean(1)
        acc = np.bincount(flat, weights=lum, minlength=counts.size)
        with np.errstate(invalid="ignore"):
            mean = np.where(
                counts.reshape(-1) > 0, acc / counts.reshape(-1).clip(1), 0
            )
        gray = mean.reshape(shape).astype(np.uint8)

    nrm_abs = np.zeros((3, *shape), np.float16)
    normals = getattr(mesh, "vertex_normals", None)
    if normals is not None and len(normals) == len(pts):
        for a in range(3):
            acc = np.bincount(
                flat,
                weights=np.abs(np.asarray(normals)[:, a]),
                minlength=counts.size,
            )
            nrm_abs[a] = (
                (acc / counts.reshape(-1).clip(1))
                .reshape(shape)
                .astype(np.float16)
            )

    sf = np.zeros(shape, np.uint8)  # v0: no carve — all non-occ unknown
    np.savez_compressed(
        out_npz,
        occ=occ,
        sf=sf,
        gray=gray,
        nrm_abs=nrm_abs,
        log_density=log_density,
    )
    return EvidenceGrid(
        npz_path=out_npz,
        vox_m=vox_m,
        origin=tuple(float(v) for v in origin),
        shape=shape,
        visibility_source="synthesized",
    )
