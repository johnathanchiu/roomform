"""Stage 2 remote runner: SpatialLM object lifting on Modal.

Runs SpatialLM 1.1 (Qwen-0.5B, ARKitScenes-SFT) on a scene point cloud
and returns the structured-language layout ("Bbox(...)" lines), which
``roomform.pipe.lifting.spatiallm.lift_from_proposals`` parses into
SceneObjects. Only the image layers and the function body live here —
app/GPU/file conventions come from ``roomform.inference.modal_adapter``.

  modal run -m roomform.pipe.lifting.modal_app \
      --point-cloud scan.ply --out proposals.txt
"""

from __future__ import annotations

from pathlib import Path

from roomform.inference.modal_adapter import StageApp, torch_image

SPATIALLM_COMMIT = "8913c44d84a450c53e9340b13317f8cf7144a738"
MODEL_ID = "ysmao/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT"

stage = StageApp("lifting")
app = stage.app  # `modal run` discovers this name

image = torch_image(
    "transformers>=4.41.2,<=4.46.1",
    "safetensors>=0.4.5",
    "pandas>=2.2.3",
    "einops>=0.8.1",
    "numpy>=1.26,<2",
    "scipy>=1.15.2",
    "scikit-learn>=1.6.1",
    "toml>=0.10.2",
    "tokenizers>=0.19.0,<0.20.4",
    "huggingface_hub>=0.25.0",
    "shapely>=2.0.7",
    "bbox>=0.9.4",
    "terminaltables>=3.1.10",
    "open3d>=0.18.0",
    "trimesh>=4.0,<5",
    "pydantic>=2.0",
    "addict>=2.4.0",
    "timm",
    "spconv-cu120",
    "tqdm",
    apt=("git", "build-essential", "libgl1", "libglib2.0-0", "ninja-build"),
    env={"MAX_JOBS": "4", "PYTHONPATH": "/opt/SpatialLM"},
    run_commands=(
        (
            "pip install torch-scatter "
            "-f https://data.pyg.org/whl/torch-2.4.0+cu124.html"
        ),
        "pip install flash-attn --no-build-isolation",
        (
            "git clone https://github.com/manycore-research/SpatialLM.git "
            "/opt/SpatialLM"
        ),
        f"cd /opt/SpatialLM && git checkout {SPATIALLM_COMMIT}",
        (
            'python -c "from huggingface_hub import snapshot_download; '
            f"snapshot_download('{MODEL_ID}')\""
        ),
    ),
)


@stage.gpu(image=image, retries=0, max_containers=1, scaledown_window=2)
def lift(point_cloud_bytes: bytes, detect_type: str = "object") -> str:
    """Point cloud bytes (ply) -> SpatialLM structured-language layout."""
    if detect_type not in {"all", "arch", "object"}:
        raise ValueError(f"unsupported detection type: {detect_type}")

    import tempfile

    import numpy as np
    import torch
    from inference import generate_layout, preprocess_point_cloud
    from spatiallm import Layout
    from spatiallm.pcd import cleanup_pcd, get_points_and_colors, load_o3d_pcd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory(prefix="roomform-lifting-") as directory:
        point_cloud_path = Path(directory) / "scene.ply"
        point_cloud_path.write_bytes(point_cloud_bytes)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        )
        model.to("cuda")
        model.set_point_backbone_dtype(torch.float32)
        model.eval()

        num_bins = model.config.point_config["num_bins"]
        grid_size = Layout.get_grid_size(num_bins)
        point_cloud = cleanup_pcd(
            load_o3d_pcd(point_cloud_path), voxel_size=grid_size
        )
        points, colors = get_points_and_colors(point_cloud)
        minimum = np.min(points, axis=0)
        input_point_cloud = preprocess_point_cloud(
            points, colors, grid_size, num_bins
        )

        with torch.inference_mode():
            layout = generate_layout(
                model,
                input_point_cloud,
                tokenizer,
                "/opt/SpatialLM/code_template.txt",
                seed=42,
                temperature=0.1,
                top_k=1,
                top_p=1.0,
                max_new_tokens=2048,
                detect_type=detect_type,
            )
        layout.translate(minimum)
        return layout.to_language_string()


@stage.entrypoint()
def main(point_cloud: str, out: str, detect_type: str = "object") -> None:
    stage.run_file(lift, point_cloud, out, detect_type=detect_type)
