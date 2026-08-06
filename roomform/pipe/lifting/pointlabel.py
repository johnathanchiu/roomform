"""Stage 2 backend — pointlabel: per-point labels -> oriented boxes.

Point Transformer V3 (Pointcept) with the released ScanNet-20
semantic-segmentation checkpoint labels every point on Modal; instances
are then derived locally — per-class radius connected components,
footprint-PCA yaw, extents from the rotated bounds. Positions come
straight from the scan, so boxes are exact where the labels are.

License: Pointcept code and the released PTv3 checkpoints are MIT
(github.com/Pointcept/Pointcept, hf.co/Pointcept/PointTransformerV3);
the checkpoint was trained on ScanNet v2, whose *data* terms of use are
non-commercial — flag before shipping this backend in a paid product.

  modal run -m roomform.pipe.lifting.pointlabel \
      --point-cloud scan.ply --out labels.npz
  python -m roomform.pipe.lifting.pointlabel labels.npz
"""

from __future__ import annotations

import numpy as np

from roomform.contracts import SceneObject
from roomform.inference.modal_adapter import StageApp, torch_image

POINTCEPT_TAG = "v1.5.1"
CKPT_REPO = "Pointcept/PointTransformerV3"
CKPT_FILE = "scannet-semseg-pt-v3m1-0-base/model/model_best.pth"

GRID = 0.02  # meters; the checkpoint's training voxel size
# ScanNet-20 label order (Pointcept spelling, spaces -> underscores).
CLASSES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower_curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)
EXCLUDE = {"wall", "floor"}  # the shell model owns structure
MIN_CLUSTER_PTS = 200  # at 2 cm sampling
CLUSTER_RADIUS = 0.05  # meters; bridges adjacent 2 cm voxels

stage = StageApp("pointlabel")
app = stage.app  # `modal run` discovers this name

image = torch_image(
    "numpy>=1.26,<2",
    "addict>=2.4",
    "timm",
    "spconv-cu120",
    "plyfile>=1.0",
    "huggingface_hub>=0.25",
    env={"PYTHONPATH": "/opt/Pointcept"},
    run_commands=(
        (
            "pip install torch-scatter "
            "-f https://data.pyg.org/whl/torch-2.4.0+cu124.html"
        ),
        (
            f"git clone --depth 1 --branch {POINTCEPT_TAG} "
            "https://github.com/Pointcept/Pointcept.git /opt/Pointcept"
        ),
        # Trim the model registry to what PT-v3 needs so importing it
        # does not drag in pointops/torch_geometric-only backbones.
        (
            "printf 'from .builder import build_model\\n"
            "from .default import *\\n"
            "from .point_transformer_v3 import *\\n' "
            "> /opt/Pointcept/pointcept/models/__init__.py"
        ),
        (
            'python -c "from huggingface_hub import hf_hub_download; '
            f"hf_hub_download('{CKPT_REPO}', '{CKPT_FILE}')\""
        ),
    ),
    # remote container imports this module -> roomform.contracts needs
    # pydantic in the image
    with_roomform=True,
)

# Model dict from the checkpoint's shipped config.py, with flash
# attention swapped for upcast plain attention (identical math, no
# flash-attn build in the image).
_MODEL = {
    "type": "DefaultSegmentorV2",
    "num_classes": 20,
    "backbone_out_channels": 64,
    "backbone": {
        "type": "PT-v3m1",
        "in_channels": 6,
        "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        "stride": (2, 2, 2, 2),
        "enc_depths": (2, 2, 2, 6, 2),
        "enc_channels": (32, 64, 128, 256, 512),
        "enc_num_head": (2, 4, 8, 16, 32),
        "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
        "dec_depths": (2, 2, 2, 2),
        "dec_channels": (64, 64, 128, 256),
        "dec_num_head": (4, 4, 8, 16),
        "dec_patch_size": (1024, 1024, 1024, 1024),
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "drop_path": 0.3,
        "shuffle_orders": True,
        "pre_norm": True,
        "enable_rpe": False,
        "enable_flash": False,
        "upcast_attention": True,
        "upcast_softmax": True,
        "cls_mode": False,
        "pdnorm_bn": False,
        "pdnorm_ln": False,
        "pdnorm_decouple": True,
        "pdnorm_adaptive": False,
        "pdnorm_affine": True,
        "pdnorm_conditions": ("ScanNet", "S3DIS", "Structured3D"),
    },
    "criteria": [{"type": "CrossEntropyLoss", "loss_weight": 1.0}],
}


@stage.gpu(image=image, retries=0, max_containers=1, scaledown_window=2)
def segment(point_cloud_bytes: bytes) -> bytes:
    """Point cloud bytes (ply, with normals+colors) -> labels npz
    bytes: pts [N,3] f32 raw-frame (2 cm-deduped), label [N] u8."""
    import io

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from plyfile import PlyData
    from pointcept.models import build_model

    vertex = PlyData.read(io.BytesIO(point_cloud_bytes))["vertex"]
    names = set(vertex.data.dtype.names)
    if not {"nx", "red"} <= names:
        # ponytail: estimate normals / gray colors if a source without
        # them ever matters; every current scan ships both.
        raise ValueError("ply must carry normals (nx..) and colors")
    pts = np.stack([vertex["x"], vertex["y"], vertex["z"]], 1)
    pts = pts.astype(np.float32)
    normal = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], 1)
    color = np.stack([vertex["red"], vertex["green"], vertex["blue"]], 1)

    # Training-time transforms: CenterShift(apply_z=True) + one point
    # per 2 cm voxel (GridSample test mode) + NormalizeColor.
    lo, hi = pts.min(0), pts.max(0)
    coord = pts - [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    grid = np.floor((coord - coord.min(0)) / GRID).astype(np.int64)
    key = grid[:, 0]
    for axis in (1, 2):
        key = key * (grid[:, axis].max() + 1) + grid[:, axis]
    _, keep = np.unique(key, return_index=True)

    feat = np.concatenate(
        [color[keep] / 127.5 - 1, normal[keep]], 1, dtype=np.float32
    )
    data = {
        "coord": torch.from_numpy(coord[keep]).float().cuda(),
        "grid_coord": torch.from_numpy(grid[keep]).long().cuda(),
        "feat": torch.from_numpy(feat).cuda(),
        "offset": torch.tensor([len(keep)], device="cuda"),
    }

    ckpt = torch.load(
        hf_hub_download(CKPT_REPO, CKPT_FILE), map_location="cpu"
    )
    model = build_model(_MODEL).cuda().eval()
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}
    )
    with torch.inference_mode():
        label = model(data)["seg_logits"].argmax(1)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        pts=pts[keep],
        label=label.cpu().numpy().astype(np.uint8),
        classes=np.array(CLASSES),
    )
    return buf.getvalue()


def _components(pts: np.ndarray):
    """Radius connected components over one class's points."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(pts).query_pairs(CLUSTER_RADIUS, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(len(pts), len(pts)),
    )
    _, comp = connected_components(graph, directed=False)
    for c in np.unique(comp):
        mask = comp == c
        if mask.sum() >= MIN_CLUSTER_PTS:
            yield pts[mask]


def _box(cls: str, pts: np.ndarray, shift) -> SceneObject:
    """Oriented box: footprint-PCA yaw, extents from rotated bounds."""
    xy = pts[:, :2]
    mean = xy.mean(0)
    centered = xy - mean
    axis = np.linalg.eigh(np.cov(centered.T))[1][:, -1]
    perp = np.array([-axis[1], axis[0]])
    u, t = centered @ axis, centered @ perp
    center_xy = mean + axis * (u.min() + u.max()) / 2
    center_xy += perp * (t.min() + t.max()) / 2
    z0, z1 = pts[:, 2].min(), pts[:, 2].max()
    return SceneObject(
        cls=cls,
        center=(
            float(center_xy[0] - shift[0]),
            float(center_xy[1] - shift[1]),
            float((z0 + z1) / 2 - shift[2]),
        ),
        size=(
            max(float(np.ptp(u)), GRID),
            max(float(np.ptp(t)), GRID),
            max(float(z1 - z0), GRID),
        ),
        heading=float(np.arctan2(axis[1], axis[0])),
        source="pointlabel",
    )


def lift_from_labels(
    path: str, frame_shift: tuple[float, float, float]
) -> list[SceneObject]:
    """Parse a labels npz into grid-frame SceneObjects."""
    data = np.load(path)
    classes = [str(c) for c in data["classes"]]
    out = []
    for cls_id, cls in enumerate(classes):
        if cls in EXCLUDE:
            continue
        pts = data["pts"][data["label"] == cls_id]
        if len(pts) < MIN_CLUSTER_PTS:
            continue
        out.extend(_box(cls, c, frame_shift) for c in _components(pts))
    return out


@stage.entrypoint()
def main(point_cloud: str, out: str) -> None:
    stage.run_file(segment, point_cloud, out)


if __name__ == "__main__":
    import collections
    import sys

    objs = lift_from_labels(sys.argv[1], (0.0, 0.0, 0.0))
    assert all(min(o.size) > 0 for o in objs)
    counts = collections.Counter(o.cls for o in objs)
    print(f"pointlabel self-check OK: {len(objs)} objects, {dict(counts)}")
