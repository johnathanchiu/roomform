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
