"""Fit reconstructed object meshes to their measured boxes.

Generated meshes (SAM 3D or any image->3D backend) arrive with
approximate scale and pose at best. This stage treats each mesh as a
proposal that must agree with the measurement: a geometry QC gate
rejects degenerate outputs (flat cards, absurd aspect), and survivors
are affinely fitted into the detected box frame -- per-axis scale onto
the box extents, box heading sets facing, floor-standing objects get
their bottom snapped to the box bottom. The fitted pose is baked into
the GLB vertices (scene-frame minus box center), so every consumer
renders the corrected mesh with no transform logic of its own.

    uv run python -m roomform.pipe.objects.reconstruction.fit artifacts/<scene>
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import trimesh

from roomform.contracts import SceneDocument

# QC thresholds: a usable furniture mesh occupies its box in all three
# axes; a photo-billboard card does not.
MIN_THIN_RATIO = 0.30  # mesh extent / box extent on the thinnest box axis
MAX_ASPECT = 7.5  # mesh bbox longest / shortest
MAX_AXIS_DISTORTION = 3.0  # per-axis fit scale max/min; cards need more
MIN_WITHIN_20CM = 0.7  # evidence fraction within 20cm of the fitted mesh
FLOOR_SNAP_Z = 0.25  # box bottoms below this are floor-standing


def _mesh_scene(path: str) -> trimesh.Scene:
    loaded = trimesh.load(path)
    return (
        loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    )


def _extents_in_frame(
    scene: trimesh.Scene, heading: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh bounds expressed in the box's yaw frame (z stays up)."""
    v = np.vstack(
        [
            trimesh.transformations.transform_points(
                g.vertices, scene.graph.get(name)[0]
            )
            for name, g in _named_geometry(scene)
        ]
    )
    c, s = math.cos(heading), math.sin(heading)
    rot = np.stack(
        [v[:, 0] * c + v[:, 1] * s, -v[:, 0] * s + v[:, 1] * c, v[:, 2]], 1
    )
    return rot.min(0), rot.max(0)


def _named_geometry(scene: trimesh.Scene):
    for name in scene.graph.nodes_geometry:
        _, geom_name = scene.graph[name]
        yield name, scene.geometry[geom_name]


def qc(
    scene: trimesh.Scene, box_size: np.ndarray, heading: float
) -> str | None:
    """Return a rejection reason, or None for a usable mesh."""
    lo, hi = _extents_in_frame(scene, heading)
    ext = np.maximum(hi - lo, 1e-6)
    aspect = float(ext.max() / ext.min())
    if aspect > MAX_ASPECT:
        return f"aspect {aspect:.1f}"
    thin_axis = int(np.argmin(box_size))
    ratio = float(ext[thin_axis] / max(box_size[thin_axis], 1e-6))
    if ratio < MIN_THIN_RATIO:
        return f"flat card ({ratio:.2f} of box on axis {thin_axis})"
    return None


def fit_to_box(
    scene: trimesh.Scene, center: np.ndarray, size: np.ndarray, heading: float
) -> trimesh.Scene:
    """Affine-fit the mesh into the box frame and bake, box-local.

    Per-axis scale maps the mesh's yaw-frame extents onto the box
    extents; the box heading sets facing. Floor-standing boxes pin the
    mesh bottom to the box bottom (scan boxes routinely clip legs, so
    center-matching floats furniture). Output vertices are scene-frame
    minus box center: place at ``center`` and it is exactly in place.
    """
    lo, hi = _extents_in_frame(scene, heading)
    ext = np.maximum(hi - lo, 1e-6)
    mid = (lo + hi) / 2
    c, s = math.cos(heading), math.sin(heading)
    to_frame = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    axis_scale = size / ext
    if float(axis_scale.max() / axis_scale.min()) > MAX_AXIS_DISTORTION:
        raise ValueError(
            f"axis distortion {axis_scale.max() / axis_scale.min():.1f}"
        )
    scale = np.diag(axis_scale)
    m = np.eye(4)
    m[:3, :3] = to_frame.T @ scale @ to_frame
    # in the yaw frame: mesh mid -> box origin (then bottom-snap z)
    offset = -scale @ mid
    if center[2] - size[2] / 2 < FLOOR_SNAP_Z:
        offset[2] = -size[2] / 2 - scale[2, 2] * lo[2]
    m[:3, 3] = to_frame.T @ offset
    fitted = scene.copy()
    fitted.apply_transform(m)
    return fitted


def _evidence_points(
    scene_dir: str, doc: SceneDocument
) -> tuple[np.ndarray, np.ndarray | None]:
    """Labeled scan points in the grid frame (+ class names when known)."""
    labels_path = os.path.join(scene_dir, "labels.npz")
    if not os.path.exists(labels_path):
        return np.empty((0, 3), np.float32), None
    lab = np.load(labels_path)
    pts = lab["pts"].astype(np.float32) - np.asarray(
        doc.frame_shift, np.float32
    )
    cls = np.asarray([str(c) for c in lab["classes"][lab["label"]]])
    return pts, cls


def observed_alignment(
    fitted: trimesh.Scene,
    center: np.ndarray,
    evidence: np.ndarray,
) -> float:
    """Fraction of the object's evidence within 20cm of the fitted mesh."""
    from scipy.spatial import cKDTree

    mesh = trimesh.util.concatenate(
        [
            geom.copy().apply_transform(fitted.graph.get(name)[0])
            for name, geom in _named_geometry(fitted)
        ]
    )
    surface = trimesh.sample.sample_surface(mesh, 20_000, seed=27)[0]
    d = cKDTree(surface).query(evidence - center, k=1, workers=-1)[0]
    return float(np.mean(d <= 0.20))


def solve_pose(
    scene: trimesh.Scene,
    center: np.ndarray,
    size: np.ndarray,
    heading: float,
    evidence: np.ndarray,
) -> tuple[trimesh.Scene, float]:
    """Fit a generator-frame mesh: z-up prerotation + yaw search.

    Fresh image->3D outputs live in glTF y-up camera space with no
    scene pose. Rotate to z-up, then try yaw hypotheses; each is
    box-fitted and scored by evidence alignment, best survivor wins.
    Raises ValueError when every hypothesis fails the distortion gate.
    """
    zup = scene.copy()
    zup.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    )
    best: tuple[trimesh.Scene, float] | None = None
    for yaw_deg in range(0, 360, 45):
        cand = zup.copy()
        cand.apply_transform(
            trimesh.transformations.rotation_matrix(
                math.radians(yaw_deg), [0, 0, 1], point=cand.centroid
            )
        )
        try:
            fitted = fit_to_box(cand, center, size, heading)
        except ValueError:
            continue
        score = observed_alignment(fitted, center, evidence)
        if best is None or score > best[1]:
            best = (fitted, score)
    if best is None:
        raise ValueError("no yaw hypothesis fits the box")
    return best


def fit_scene_meshes(scene_dir: str) -> dict[str, str]:
    """QC + fit every mesh_path object in a scene dir, in place.

    Rejected meshes lose their mesh_path (and the GLB is removed);
    accepted ones are overwritten with the fitted, baked copy.
    Returns {object_index: outcome} and rewrites scene.json.
    """
    path = os.path.join(scene_dir, "scene.json")
    with open(path) as fh:
        doc = SceneDocument(**json.load(fh))
    ev_pts, ev_cls = _evidence_points(scene_dir, doc)
    report: dict[str, str] = {}
    for i, o in enumerate(doc.objects):
        if not o.mesh_path:
            continue
        glb = os.path.join(scene_dir, o.mesh_path)
        center, size = np.asarray(o.center), np.asarray(o.size)

        def reject(reason: str, o=o, glb=glb, i=i) -> None:
            os.remove(glb)
            o.mesh_path = None
            report[f"{i}:{o.cls}"] = f"rejected: {reason}"

        mesh = _mesh_scene(glb)
        reason = qc(mesh, size, o.heading)
        if reason is not None:
            reject(reason)
            continue
        try:
            fitted = fit_to_box(mesh, center, size, o.heading)
        except ValueError as exc:
            reject(str(exc))
            continue
        evidence = _object_evidence(ev_pts, ev_cls, o)
        if len(evidence) >= 100:
            within = observed_alignment(fitted, center, evidence)
            if within < MIN_WITHIN_20CM:
                reject(f"evidence within 20cm only {within:.2f}")
                continue
        fitted.export(glb)
        report[f"{i}:{o.cls}"] = "fitted"
    with open(path, "w") as fh:
        fh.write(doc.model_dump_json(indent=2))
    return report


def _object_evidence(pts: np.ndarray, cls: np.ndarray | None, o) -> np.ndarray:
    """Class-matched scan points inside the object's (slightly padded) box."""
    if not len(pts):
        return pts
    c, s = np.asarray(o.center), np.asarray(o.size)
    h = math.cos(o.heading), math.sin(o.heading)
    rel = pts - c
    rot = np.stack(
        [
            rel[:, 0] * h[0] + rel[:, 1] * h[1],
            -rel[:, 0] * h[1] + rel[:, 1] * h[0],
            rel[:, 2],
        ],
        1,
    )
    inside = np.all(np.abs(rot) <= s / 2 + 0.06, axis=1)
    if cls is not None:
        inside &= cls == o.cls
    return pts[inside]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir")
    args = parser.parse_args()
    for key, outcome in fit_scene_meshes(args.scene_dir).items():
        print(f"{key}: {outcome}")
