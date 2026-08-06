# roomform

Parses point cloud indoor scans into structured, editable
scenes: room boundaries (walls, floors, ceilings — inferred
through occlusion) + objects out.

Pipeline spine (see docs/data-contracts.md — the contracts ARE the
architecture; code transfers in behind them):

    ScanInput -> EvidenceGrid -> PatchGraph -> SceneDocument

The boundary model is a patch-graph ConvFormer: raw 8 cm observable
evidence (occupancy, grayscale, |normal|, density) in, three surface
node classes (wall/floor/ceiling) plus 13 forward-edge connectivity
channels out. No semantic inputs, no deterministic compiler. Objects
come from SpatialLM box lifting, with optional SAM 3D mesh
reconstruction.

Measured (held-out synthetic rooms): boundary F1 0.98, occluded-
boundary F1 0.94, connectivity F1 0.97 at 8 cm. On real scans
(Redwood, ARKitScenes, SceneNN) roughly a third of the predicted
boundary lands on cells the scanner never observed — the model infers
structure behind furniture and between scan stations. A scene runs
end-to-end in ~10 s on a laptop CPU once object lifting returns.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Install with the extras you
need:

    uv sync --extra model --extra modal --extra dev

| extra   | pulls in            | needed for                          |
|---------|---------------------|-------------------------------------|
| `model` | torch, scipy        | shell completion (local inference)  |
| `modal` | modal               | SpatialLM lifting on Modal          |
| `fal`   | fal-client, pillow  | live SAM 3D object reconstruction   |
| `agent` | anthropic, openai…  | the geometry-fixing agent harness   |
| `dev`   | ruff, pytest, pyyaml| lint + tests                        |

Credentials, only for the stages that use them:

- Modal (lifting): `uv run modal setup` once.
- FAL (SAM 3D): put `FAL_KEY=...` in `.env` (gitignored).

## Checkpoint

Trained weights are not in git. Drop a checkpoint at
`checkpoints/patch-graph-400-offset-isolated-r1.pt` and the pipeline
picks it up automatically (`--ckpt` overrides the path). Checkpoints
are `{"model": state_dict, "config": {...}}`; the config dict is what
`roomform.model.config.ModelConfig` reads, so any compatible training
run loads directly. Without a checkpoint the pipeline falls back to
random weights and marks the shell `RANDOM-INIT` (excluded from
exports).

## Run the pipeline

    uv run python -m roomform.pipe.e2e SCAN.ply

- `SCAN.ply` — a registered point cloud (colors optional; normals are
  PCA-estimated when missing). An `.npz` with `pts` / `pts_normal` /
  `pts_color` works too.
- Output goes to `artifacts/<scan-stem>/`: `evidence.npz`,
  `patchgraph.npz`, `proposals.txt`, `scene.json`, `scene.glb`.
- Progress streams as NDJSON events on stdout; a shell-only
  `scene.partial.json` is emitted seconds in, while SpatialLM lifting
  finishes on Modal (spawned first, runs concurrently).
- `--proposals FILE` reuses existing SpatialLM output and skips Modal
  entirely (fully local, ~10 s per scene on CPU).
- `--sequential` disables the parallel DAG for debugging.

Attach reconstructed object meshes (optional):

```python
from roomform.pipe.reconstruction.sam3d import adopt_meshes, reconstruct_live
```

`adopt_meshes` matches existing aligned GLBs to objects by center and
copies them into the scene's artifact dir; `reconstruct_live` calls
FAL SAM 3D on an object crop image (needs `FAL_KEY`).

## View and edit results

    uv run python viewer/serve.py          # http://127.0.0.1:8790

Single-file three.js scene editor over `artifacts/`: layer toggles
(evidence / shell classes / boxes / meshes), gravity-aligned object
editing — drag slides on the floor plane, shift-drag raises, Q/E
rotate, WASD nudge — and Save writes `scene.json` back.

`scene.glb` is also self-contained: drop it into any glTF viewer.

### Demo editor bridge

Scenes export into the product editor's fixture format:

    uv run python -m roomform.export fixtures artifacts/<scene>
    uv run python viewer/serve_demo.py PATH/TO/editor/dist   # :8792

The editor discovers exported scenes from `fixtures/index.json` at
runtime — no editor changes. `--fixtures DIR` overrides the fixtures
directory (the default currently points at a local editor build).

Standalone GLB export: `uv run python -m roomform.export glb
artifacts/<scene>/scene.json out.glb --evidence
artifacts/<scene>/evidence.npz`.

## Tests and lint

    uv run pytest tests/
    uv run ruff format . && uv run ruff check .

`tests/test_parity.py` is the contract that matters: every checkpoint
written in the documented format must round-trip through
`roomform.inference.local`.

## Layout

    roomform/
    ├── roomform/            the pip package
    │   ├── contracts/       pydantic documents (the spine)
    │   ├── pipe/            evidence -> lifting -> reconstruction -> fuse, e2e driver
    │   ├── model/           patch-graph ConvFormer (the ONE architecture)
    │   ├── inference/       local runner + Modal adapter (StageApp)
    │   ├── agent/           multi-provider LLM harness (Claude/OpenAI/…)
    │   ├── export.py        scene.glb + demo-editor fixture exports
    │   └── eval.py          shell/connectivity F1 metrics
    ├── viewer/              web viewers + dev servers (not in the wheel)
    ├── research/            training loop, experiments, datagen (never
    │                        defines architectures or metrics — parity
    │                        with the package is enforced by tests)
    ├── docs/                data contracts + migration log
    └── artifacts/           pipeline outputs (gitignored)

## Citation

If you use roomform in your research, please cite:

```bibtex
@software{roomform2026,
  title  = {roomform: parsing point cloud indoor scans into
            structured, editable scenes},
  author = {Chiu, Johnathan and Zhou, Matthew and Bourne, Preston},
  year   = {2026},
}
```

Object lifting uses [SpatialLM](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B)
(Manycore Research); object mesh reconstruction uses
[SAM 3D Objects](https://ai.meta.com/sam3d/) (Meta) via fal.ai.
Benchmark scenes in the docs come from Redwood, ARKitScenes, SceneNN,
CV4AEC, and HouseLayout3D.
