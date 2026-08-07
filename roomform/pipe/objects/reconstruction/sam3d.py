"""Stage 3 — object_reconstruction: boxes -> aligned object meshes.

Backend: FAL SAM 3D Objects (ported adapter pattern from the internal
pipeline). Given a SceneObject and its cropped evidence, produce a GLB
mesh and attach it via ``mesh_path``. Network calls are isolated here;
everything downstream consumes only the contract.

v0 also supports "adopt": attaching already-reconstructed aligned
meshes (existing SAM3D outputs) to lifted objects by nearest-center
match — so archived reconstructions plug into fresh scenes.
"""

from __future__ import annotations

import glob
import os
import shutil
import urllib.request

import numpy as np
import trimesh

from roomform.contracts import SceneObject

SAM3D_APP = "fal-ai/sam-3/3d-objects"


def _mesh_url(value: object) -> str:
    urls: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif (
            isinstance(item, str)
            and item.startswith("http")
            and item.split("?")[0].endswith(".glb")
        ):
            urls.append(item)

    visit(value)
    if not urls:
        raise RuntimeError("SAM 3D response contains no GLB URL")
    return urls[0]


def adopt_meshes(
    objects: list[SceneObject],
    aligned_dir: str,
    out_dir: str,
    frame_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_center_dist: float = 0.75,
) -> int:
    """Attach existing aligned GLBs to objects by nearest box center.

    Aligned meshes live in the raw scan frame; ``frame_shift`` maps
    their centroids into the grid frame (grid = raw - frame_shift).
    Matched GLBs are copied into ``out_dir`` (inside the scene's
    artifact dir) and ``mesh_path`` is stored relative to the scene
    dir so a viewer can serve them. Returns number of attachments.
    """
    shift = np.asarray(frame_shift, float)
    candidates = [
        (np.asarray(trimesh.load(fp).centroid, float) - shift, fp)
        for fp in sorted(glob.glob(os.path.join(aligned_dir, "*.glb")))
    ]
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for i, o in enumerate(objects):
        if not candidates:
            break
        d = [
            float(np.linalg.norm(c - np.asarray(o.center)))
            for c, _ in candidates
        ]
        k = int(np.argmin(d))
        if d[k] <= max_center_dist:
            dst = os.path.join(out_dir, f"object-{i}.glb")
            shutil.copyfile(candidates[k][1], dst)
            o.mesh_path = os.path.join(
                os.path.basename(out_dir), os.path.basename(dst)
            )
            candidates.pop(k)
            n += 1
    return n


def reconstruct_live(
    obj: SceneObject, crop_image_path: str, out_glb: str
) -> str:
    """Live SAM3D call (requires FAL_KEY). Kept import-lazy so the
    pipeline works offline with adopt_meshes."""
    import fal_client  # optional service dep — lazy by design

    # full-frame box prompt: crops are already object-centered, and
    # auto-segmentation finds no masks on sparse rendered views
    from PIL import Image

    w, h = Image.open(crop_image_path).size
    handle = fal_client.submit(
        SAM3D_APP,
        arguments={
            "image_url": fal_client.upload_file(crop_image_path),
            "box_prompts": [
                {"x_min": 16, "y_min": 16, "x_max": w - 16, "y_max": h - 16}
            ],
        },
    )
    url = _mesh_url(handle.get())
    urllib.request.urlretrieve(url, out_glb)
    obj.mesh_path = out_glb
    return out_glb


# ------------------------------------------------------- cloud crops ----

# classes that are flat by nature or fused into the shell; image->3D
# reconstruction has nothing to add for them
SKIP_CLASSES = {"door", "window", "picture", "curtain"}


def render_crop(
    pts: np.ndarray,
    colors: np.ndarray,
    obj: SceneObject,
    out_png: str,
    pad: float = 0.06,
) -> int:
    """Render a three-quarter view of the object's points to an image.

    White background, medium splats: dense enough that the object reads
    as a surface, sparse enough that the segmenter still finds edges.
    Returns the number of points inside the (padded) box.
    """
    import math as _math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, s_ = np.asarray(obj.center), np.asarray(obj.size)
    h = _math.cos(obj.heading), _math.sin(obj.heading)
    rel = pts - c
    rot = np.stack(
        [
            rel[:, 0] * h[0] + rel[:, 1] * h[1],
            -rel[:, 0] * h[1] + rel[:, 1] * h[0],
            rel[:, 2],
        ],
        1,
    )
    inside = np.all(np.abs(rot) <= s_ / 2 + pad, axis=1)
    if inside.sum() < 200:
        return int(inside.sum())
    p, cc = rot[inside], colors[inside][:, :3] / 255.0
    az, el = _math.radians(35), _math.radians(35)
    fwd = np.array(
        [
            _math.cos(el) * _math.cos(az),
            _math.cos(el) * _math.sin(az),
            _math.sin(el),
        ]
    )
    eye = fwd * float(np.linalg.norm(s_)) * 1.5
    z = -fwd
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam = (p - eye) @ np.stack([x, y, z], 1)
    u, v = -cam[:, 0] / cam[:, 2], -cam[:, 1] / cam[:, 2]
    fig = plt.figure(figsize=(8.2, 8.2), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor("white")
    order = np.argsort(-cam[:, 2])
    size = max(2.0, min(9.0, 6e5 / inside.sum()))
    ax.scatter(u[order], v[order], c=cc[order], s=size, linewidths=0)
    span = max(u.max() - u.min(), v.max() - v.min()) / 2 + 0.03
    mu, mv = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    ax.set_xlim(mu - span, mu + span)
    ax.set_ylim(mv - span, mv + span)
    ax.axis("off")
    fig.savefig(out_png)
    plt.close(fig)
    return int(inside.sum())


def reconstruct_scene(
    scene_dir: str, cloud_path: str, workers: int = 4
) -> None:
    """Reconstruct meshes for every eligible object from a dense cloud.

    Renders a crop per object without a mesh, runs SAM 3D on each
    (parallel network calls), then pose-solves and gates every result;
    only meshes that fit the measured box AND the scan evidence keep
    their mesh_path.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor

    from roomform.contracts import SceneDocument
    from roomform.pipe.objects.reconstruction import fit as fitmod

    with open(os.path.join(scene_dir, "scene.json")) as fh:
        doc = SceneDocument(**json.load(fh))
    cloud = trimesh.load(cloud_path)
    pts = np.asarray(cloud.vertices, np.float32) - np.asarray(
        doc.frame_shift, np.float32
    )
    colors = np.asarray(cloud.colors)
    ev_pts, ev_cls = fitmod._evidence_points(scene_dir, doc)
    out_dir = os.path.join(scene_dir, "sam3d")
    os.makedirs(out_dir, exist_ok=True)

    todo = []
    for i, o in enumerate(doc.objects):
        if o.mesh_path or o.cls in SKIP_CLASSES:
            continue
        png = os.path.join(out_dir, f"crop-{i}.png")
        n = render_crop(pts, colors, o, png)
        if n < 200:
            print(f"{i}:{o.cls}: skipped ({n} pts in box)")
            continue
        todo.append((i, o, png))

    def recon(item):
        i, o, png = item
        glb = os.path.join(out_dir, f"object-{i}.glb")
        try:
            reconstruct_live(o, png, glb)
            return i, o, glb, None
        except Exception as exc:  # noqa: BLE001 — per-object isolation
            return i, o, glb, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(recon, todo))

    kept = 0
    for i, o, glb, err in results:
        o.mesh_path = None
        if err is not None:
            print(
                f"{i}:{o.cls}: generation failed ({err.splitlines()[0][:80]})"
            )
            continue
        evidence = fitmod._object_evidence(ev_pts, ev_cls, o)
        try:
            fitted, score = fitmod.solve_pose(
                fitmod._mesh_scene(glb),
                np.asarray(o.center),
                np.asarray(o.size),
                o.heading,
                evidence,
            )
        except ValueError as exc:
            print(f"{i}:{o.cls}: rejected ({exc})")
            os.remove(glb)
            continue
        if score < fitmod.MIN_WITHIN_20CM:
            print(f"{i}:{o.cls}: rejected (evidence within 20cm {score:.2f})")
            os.remove(glb)
            continue
        fitted.export(glb)
        o.mesh_path = os.path.join("sam3d", f"object-{i}.glb")
        kept += 1
        print(f"{i}:{o.cls}: kept (score {score:.2f})")
    with open(os.path.join(scene_dir, "scene.json"), "w") as fh:
        fh.write(doc.model_dump_json(indent=2))
    print(f"kept {kept}/{len(results)} generated meshes")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--cloud", required=True, help="dense colored source ply")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    reconstruct_scene(args.scene_dir, args.cloud, args.workers)
