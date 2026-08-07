# Roomform

Parses indoor point cloud scans into structured, editable scenes.

![roomform boundary prediction on a real apartment scan](assets/media/pipeline.gif)

## Try it (no setup)

A pre-baked result ships in [samples/](samples/README.md):

    uv sync
    uv run python viewer/debug/serve.py --artifacts samples   # :8790

## Setup

Requires [uv](https://docs.astral.sh/uv/):

    uv sync --extra model --extra modal --extra dev

| extra   | needed for                                    |
|---------|-----------------------------------------------|
| `model` | boundary inference (torch)                    |
| `modal` | remote object lifting ([credentials](https://modal.com/): `uv run modal setup`) |
| `fal`   | SAM 3D object reconstruction ([`FAL_KEY`](https://fal.ai/) in `.env`) |
| `agent` | envelope completion — VLM-judged fills (WIP)  |
| `dev`   | `ruff format . && ruff check .` · `pytest tests/` |

## Run

    uv run python -m roomform.pipe.e2e SCAN.ply

Takes a registered point cloud (`.ply`, colors optional, normals
estimated when missing; `.npz` with `pts`/`pts_normal`/`pts_color`
also works). Writes `artifacts/<scan-stem>/` — `scene.json`,
`scene.glb`, per-object meshes, intermediate grids — streaming NDJSON
progress with a boundary-only partial scene seconds in. Weights pull
automatically from
[jchiu/roomform](https://huggingface.co/jchiu/roomform) on first run
(`--ckpt` overrides).

Lifting backends (`--lifter`):

| backend | model | upstream license |
|---|---|---|
| `pointlabel` (default) | PTv3 semseg + local clustering | MIT (ScanNet data ToU applies) |
| `spatiallm` | SpatialLM 1.1 Qwen-0.5B | CC BY-NC 4.0 |
| `unidet3d` (eval) | UniDet3D detector | CC BY-NC 4.0 |

## View and edit

**Debug viewer** (:8790) — read-only inspection of everything the
pipeline produced: RGB cloud, boundary layers, evidence, boxes.

    uv run python viewer/debug/serve.py

**Editor** (:8792) — move objects, save back; reads
`artifacts/<scene>/scene.json` natively, new runs appear on reload:

    cd viewer/editor && bun install && bun run build
    uv run python viewer/editor/serve.py

`scene.glb` also opens in any glTF viewer.

## Known limitations

Broken scans in, broken scenes out: bad registration, mirror/glass
ghosting, and tripod beam spill degrade results despite the built-in
guards (orientation + leveling, spill cropping, wall-band rejection).
A more robust boundary model is the next training iteration.
Building-scale floors exceed local CPU attention; use GPU inference.

## Layout

    roomform/
    ├── roomform/        the pip package
    │   ├── contracts/   pydantic documents (the spine)
    │   ├── pipe/        evidence -> objects -> fuse, e2e driver
    │   ├── model/       patch-graph ConvFormer
    │   ├── inference/   local runner + Modal adapter
    │   ├── agent/       envelope completion (VLM judge)
    │   └── export.py    GLB + per-object mesh exports
    ├── viewer/          debug/ + editor/ apps
    ├── research/        training + experiments (parity-ruled)
    ├── docs/            documentation
    └── artifacts/       pipeline outputs (gitignored)

## Docs

- [Data contracts](docs/data-contracts.md)
- [Model](docs/model.md)
- [Research](docs/research.md)
- [Changelog](CHANGELOG.md)

## Citation

```bibtex
@software{roomform2026,
  title   = {roomform: parsing point cloud indoor scans into structured, editable scenes},
  author  = {Chiu, Johnathan and Zhou, Matthew and Bourne, Preston},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/johnathanchiu/roomform},
}
```

Built on [Point Transformer V3](https://github.com/Pointcept/Pointcept),
[SpatialLM](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B),
[UniDet3D](https://github.com/filaPro/unidet3d), and
[SAM 3D Objects](https://ai.meta.com/sam3d/) via
[fal.ai](https://fal.ai/models/fal-ai/sam-3/3d-objects). Benchmark
scenes from [Redwood](http://redwood-data.org/indoor_lidar_rgbd/),
[ARKitScenes](https://github.com/apple/ARKitScenes), and
[SceneNN](https://hkust-vgd.github.io/scenenn/).

## Supported by

<p align="center">
  <a href="https://modal.com"><img src="https://avatars.githubusercontent.com/u/88658467?v=4" height="64" alt="Modal"></a>
</p>
<p align="center">
  This work is generously supported by <a href="https://modal.com">Modal</a>
  and <a href="https://github.com/aksh-at">Akshat Bubna</a>
  (<a href="https://x.com/akshat_b">@akshat_b</a>).
</p>

## License

- Code: [Apache 2.0](LICENSE)
- Trained weights + synthetic datagen (when published):
  [CC BY-NC 4.0](LICENSE-WEIGHTS) — free for research; commercial use
  needs a separate license.
