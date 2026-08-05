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

import numpy as np

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
    objects: list[SceneObject], aligned_dir: str, max_center_dist: float = 0.75
) -> int:
    """Attach existing aligned GLBs to objects by nearest box center.

    Expects <aligned_dir>/object-*.glb with a sibling object-*-meta.json
    carrying a grid-frame center; falls back to mesh centroid.
    Returns number of attachments.
    """
    import json

    import trimesh

    candidates = []
    for fp in sorted(glob.glob(os.path.join(aligned_dir, "*.glb"))):
        meta = fp.replace(".glb", "-meta.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                c = json.load(fh).get("center")
        else:
            c = trimesh.load(fp).centroid.tolist()
        candidates.append((np.asarray(c, float), fp))
    n = 0
    for o in objects:
        if not candidates:
            break
        d = [
            float(np.linalg.norm(c - np.asarray(o.center)))
            for c, _ in candidates
        ]
        k = int(np.argmin(d))
        if d[k] <= max_center_dist:
            o.mesh_path = candidates[k][1]
            candidates.pop(k)
            n += 1
    return n


def reconstruct_live(
    obj: SceneObject, crop_image_path: str, out_glb: str
) -> str:
    """Live SAM3D call (requires FAL_KEY). Kept import-lazy so the
    pipeline works offline with adopt_meshes."""
    import urllib.request

    import fal_client

    handle = fal_client.submit(
        SAM3D_APP,
        arguments={"image_url": fal_client.upload_file(crop_image_path)},
    )
    url = _mesh_url(handle.get())
    urllib.request.urlretrieve(url, out_glb)
    obj.mesh_path = out_glb
    return out_glb
