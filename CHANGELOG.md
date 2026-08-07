# Changelog

Pipeline-facing changes. Versions follow the package version in
`pyproject.toml`; the model checkpoints on
[HF](https://huggingface.co/jchiu/roomform) version independently.

## 0.1.0 — 2026-08-07

Initial release.

### Model
- Patch-graph ConvFormer (UNet encoder/decoder, global-attention
  bottleneck): 3 boundary node classes + 13 forward-edge connectivity
  channels at 8 cm, with sub-voxel offset and door/window opening
  heads. Two checkpoints: 55M RGB (default) and 27M grayscale.
- Checkpoints carry their config; `ModelConfig(**ckpt["config"]).build()`
  reconstructs the network (enforced by `tests/test_parity.py`).

### Pipeline
- Evidence: 11-channel RGB observable construction (training-parity),
  PCA normals for plys without them, robust percentile grounding,
  gravity-axis detection + floor-plane leveling, dominant-region crop
  against tripod beam spill, oversized-scan thinning for remote
  payloads.
- Object lifting: PTv3 point-labeling default (per-class clustering,
  footprint-PCA yaw, robust percentile boxes, architectural-scale and
  wall-band rejection); SpatialLM and UniDet3D as optional backends.
- Fusion: per-object QA (wall leaks, floor support), scene document
  with provenance; per-object mesh carving for the editor.
- Optional SAM 3D object reconstruction (fal.ai) and an
  envelope-completion agent (profile cards + single-shot VLM judgment;
  anthropic / openai / claude-cli providers).

### Apps
- `viewer/debug/` — single-file inspection viewer (:8790).
- `viewer/editor/` — the production editor as an in-repo app reading
  `scene.json` natively, with persistent object editing (:8792).

### Infra
- Modal StageApp adapter (`roomform-<stage>[-<backend>]` apps),
  weight auto-download from the HF hub, GitHub Actions CI
  (ruff format/lint + pytest), uv-native packaging.
