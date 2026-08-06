"""Stage 2 alternative backend — UniDet3D object lifting on Modal.

UniDet3D (filaPro/unidet3d, AAAI 2025) is a multi-dataset indoor 3D
detector built on mmdetection3d. We run its ScanNet head (18 classes),
whose boxes are AXIS-ALIGNED — heading is always 0.0. Only the
ARKitScenes head predicts yaw, but it needs superpoint_transformer
partitions we cannot reproduce; the ScanNet head takes mesh-graph
superpoints we compute in-container with `segmentator`.

LICENSE NOTE: UniDet3D code + released weights are CC BY-NC 4.0
(NonCommercial). Fine for research/eval here; do NOT ship in a
commercial product without relicensing.

Detections come back as JSON lines ({cls, score, center, size,
heading}) in the RAW scan frame; ``lift_from_detections`` re-aligns
into the grid frame (grid = raw - frame_shift) like the SpatialLM
parser does. ``up="y"`` rotates a y-up scan to z-up for detection and
rotates boxes back, so outputs always stay in the input frame.

  modal run -m roomform.pipe.lifting.unidet3d \
      --point-cloud scan.ply --out detections.json --up y
"""

from __future__ import annotations

import json
import pathlib

from roomform.contracts import SceneObject
from roomform.inference.modal_adapter import StageApp, torch_image

UNIDET3D_COMMIT = "940a730a09711b0bf266fd972504da29a83b91f6"
CKPT_URL = (
    "https://github.com/filapro/unidet3d/releases/download/v1.0/unidet3d.pth"
)
CONFIG = (
    "/opt/unidet3d/configs/"
    "unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py"
)

stage = StageApp("lifting-unidet3d")
app = stage.app  # `modal run` discovers this name

# Version pins mirror the upstream Dockerfile (torch 2.1.2 + cu121);
# everything --no-deps so the resolver can't move torch or numpy.
_OPENMMLAB = (
    "pip install --no-deps mmengine==0.9.0 mmdet==3.3.0 "
    "mmsegmentation==1.2.0 mmdet3d==1.4.0 mmpretrain==1.2.0"
)
_MMCV = (
    "pip install --no-deps mmcv==2.1.0 "
    "-f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"
)
_TORCH_SCATTER = (
    "pip install --no-deps torch-scatter==2.1.2 "
    "-f https://data.pyg.org/whl/torch-2.1.0+cu121.html"
)
_PINNED = (
    "pip install --no-deps spconv-cu120==2.3.6 cumm-cu120==0.5.1 "
    "pccm==0.4.7 ccimport==0.4.2 pybind11==2.10.4 ninja==1.11.1 "
    "lark==1.1.5 addict==2.4.0 yapf==0.33.0 termcolor==2.3.0 "
    "packaging==23.1 rich==13.3.5 opencv-python==4.7.0.72 "
    "pycocotools==2.0.6 Shapely==1.8.5 scipy==1.10.1 "
    "terminaltables==3.1.10 numba==0.57.0 llvmlite==0.40.0 "
    "pyquaternion==0.9.9 lyft-dataset-sdk==0.0.8 pandas==2.0.1 "
    "python-dateutil==2.8.2 matplotlib==3.5.2 pyparsing==3.0.9 "
    "cycler==0.11.0 kiwisolver==1.4.4 scikit-learn==1.2.2 "
    "joblib==1.2.0 threadpoolctl==3.1.0 cachetools==5.3.0 "
    "nuscenes-devkit==1.1.10 trimesh==3.21.6 plyfile==1.0.2 "
    "natsort==8.4.0 timm==0.9.16 imageio==2.34.0 portalocker==2.8.2 "
    "ftfy==6.2.0 regex==2024.4.16"
)
# pip >=24 dropped --global-option, so build ME via setup.py directly
_MINKOWSKI = (
    "git clone https://github.com/daizhirui/MinkowskiEngine.git /opt/ME"
    " && cd /opt/ME"
    " && git checkout ce930eeb403a8e3f99693662ec5ce329a0ab3528"
    ' && TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6 8.9"'
    " python setup.py install --blas=openblas --force_cuda"
)
_SEGMENTATOR = (
    "git clone https://github.com/Karbo123/segmentator.git /opt/segmentator"
    " && cd /opt/segmentator/csrc"
    " && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696"
    " && sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 17)/g'"
    " CMakeLists.txt && mkdir build && cd build"
    " && cmake .. -DCMAKE_PREFIX_PATH=`python -c"
    " 'import torch;print(torch.utils.cmake_prefix_path)'`"
    ' -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import'
    ' get_python_inc; print(get_python_inc())")'
    ' -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as s;'
    " print(s.get_config_var('LIBDIR'))\")"
    " -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import"
    " get_python_lib; print(get_python_lib())'`"
    " && make && make install"
)

image = torch_image(
    base="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel",
    apt=(
        "git",
        "build-essential",
        "cmake",
        "ninja-build",
        "wget",
        "libgl1",
        "libglib2.0-0",
        "libopenblas-dev",
    ),
    env={"MAX_JOBS": "4", "PYTHONPATH": "/opt/unidet3d"},
    run_commands=(
        _OPENMMLAB,
        _MMCV,
        _TORCH_SCATTER,
        _PINNED,
        _MINKOWSKI,
        _SEGMENTATOR,
        (
            "git clone https://github.com/filaPro/unidet3d.git /opt/unidet3d"
            f" && cd /opt/unidet3d && git checkout {UNIDET3D_COMMIT}"
        ),
        f"wget -q {CKPT_URL} -O /opt/unidet3d/unidet3d.pth",
        # this module imports roomform.contracts (pydantic) remotely
        "pip install pydantic==2.7.4",
        # numba 0.57 needs numpy<1.25; enforce last so nothing bumps it
        "pip install --no-deps --force-reinstall numpy==1.24.1",
    ),
)

CLASSES_SCANNET = [
    "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "showercurtrain", "toilet", "sink", "bathtub",
    "otherfurniture",
]  # fmt: skip


@stage.gpu(
    image=image, timeout=1800, retries=0, max_containers=1, scaledown_window=2
)
def detect(
    point_cloud_bytes: bytes,
    up: str = "z",
    score_thr: float = 0.3,
    max_points: int = 200_000,
) -> str:
    """ply bytes -> UniDet3D detections as JSON lines (raw frame)."""
    if up not in {"y", "z"}:
        raise ValueError(f"unsupported up axis: {up}")

    import tempfile

    import numpy as np
    import segmentator
    import torch
    import trimesh
    import unidet3d  # noqa: F401 — registers custom mmdet3d modules
    from mmdet3d.registry import MODELS
    from mmdet3d.structures import Det3DDataSample, PointData
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint

    with tempfile.NamedTemporaryFile(suffix=".ply") as f:
        f.write(point_cloud_bytes)
        f.flush()
        mesh = trimesh.load(f.name, process=False)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32)
    faces = (
        np.asarray(mesh.faces)
        if hasattr(mesh, "faces") and len(mesh.faces)
        else None
    )
    if up == "y":  # y-up scan -> z-up for the detector
        verts = verts[:, [0, 2, 1]] * np.float32([1, -1, 1])

    # Superpoints pool voxel features into the detector's queries.
    # ScanNet's own superpoints are Felzenszwalb mesh segments, which
    # `segmentator` reproduces; coarsen until the query count is sane.
    if faces is not None:
        vt = torch.from_numpy(verts).float()
        ft = torch.from_numpy(faces).long()
        min_verts = max(20, len(verts) // 10_000)
        for _ in range(5):
            sp = segmentator.segment_mesh(vt, ft, segMinVerts=min_verts)
            sp = sp.numpy()
            if len(np.unique(sp)) <= 4000:
                break
            min_verts *= 2
    else:
        # ponytail: point-cloud fallback = 0.25 m voxel superpoints;
        # swap in a graph partition if quality matters for raw clouds
        vox = np.floor(verts / 0.25).astype(np.int64)
        _, sp = np.unique(vox, axis=0, return_inverse=True)

    if len(verts) > max_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(verts), max_points, replace=False)
        verts, colors, sp = verts[keep], colors[keep], sp[keep]
    _, sp = np.unique(sp, return_inverse=True)

    init_default_scope("mmdet3d")
    cfg = Config.fromfile(CONFIG)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, "/opt/unidet3d/unidet3d.pth", map_location="cpu")
    model = model.cuda().eval()

    points = (
        torch.from_numpy(np.hstack([verts, (colors - 127.5) / 127.5]))
        .float()
        .cuda()
    )
    sample = Det3DDataSample()
    # path selects the ScanNet head: get_dataset() matches "scannet"
    sample.set_metainfo({"lidar_path": "data/scannet/points/scene.bin"})
    sample.gt_pts_seg = PointData(
        sp_pts_mask=torch.from_numpy(sp).long().cuda()
    )

    with torch.no_grad():
        result = model.predict({"points": [points]}, [sample])[0]

    pred = result.pred_instances_3d
    boxes = pred.bboxes_3d
    tensor = boxes.tensor.cpu().numpy()
    centers = boxes.gravity_center.cpu().numpy()
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()

    lines = []
    for i in np.argsort(-scores):
        if scores[i] < score_thr:
            continue
        center, size = centers[i], tensor[i, 3:6]
        heading = float(tensor[i, 6]) if tensor.shape[1] == 7 else 0.0
        if up == "y":  # back to the raw (y-up) input frame
            center = center[[0, 2, 1]] * np.float32([1, 1, -1])
            size = size[[0, 2, 1]]
        lines.append(
            json.dumps(
                {
                    "cls": CLASSES_SCANNET[int(labels[i])],
                    "score": round(float(scores[i]), 4),
                    "center": [round(float(v), 4) for v in center],
                    "size": [round(float(v), 4) for v in size],
                    "heading": heading,
                }
            )
        )
    return "\n".join(lines) + "\n"


def lift_from_detections(
    path: str,
    frame_shift: tuple[float, float, float],
    min_score: float = 0.3,
) -> list[SceneObject]:
    """Parse UniDet3D JSON-line detections into grid-frame objects.

    ScanNet-head boxes are axis-aligned, so heading is 0.0 by
    construction (see module docstring).
    """
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["score"] < min_score:
            continue
        out.append(
            SceneObject(
                cls=d["cls"],
                center=tuple(c - s for c, s in zip(d["center"], frame_shift)),
                size=tuple(d["size"]),
                heading=d["heading"],
                source="unidet3d",
                qa={"score": d["score"]},
            )
        )
    return out


@stage.entrypoint()
def main(
    point_cloud: str, out: str, up: str = "z", score_thr: float = 0.3
) -> None:
    stage.run_file(detect, point_cloud, out, up=up, score_thr=score_thr)


if __name__ == "__main__":
    import sys

    objs = lift_from_detections(sys.argv[1], (0.0, 0.0, 0.0))
    assert objs and all(o.size[0] > 0 for o in objs)
    print(
        f"unidet3d lifting self-check OK: {len(objs)} objects, "
        f"classes {sorted({o.cls for o in objs})[:5]}"
    )
