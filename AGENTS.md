# Repository guidelines

## Project purpose

roomform parses point cloud indoor scans into structured,
editable scenes: room boundaries (walls, floors, ceilings —
inferred through occlusion) + objects out.

The pipeline spine is four contracts (docs/data-contracts.md):

```text
ScanInput -> EvidenceGrid -> PatchGraph -> SceneDocument
```

The contracts ARE the architecture. Code changes happen behind them;
contract changes get an entry in the migration log in the same change.

Keep this repo simple and readable. It is the public face of a larger
internal research effort — bring over the smallest working idea, not
the ceremony around it.

## Project shape

The package lives in `roomform/roomform/` (no `src/` layout).

- `contracts/` owns the pydantic documents. Everything downstream
  consumes only these.
- `pipe/` owns the stages: `evidence/` (voxelize + channels),
  `lifting/` (SpatialLM boxes), `reconstruction/` (SAM 3D meshes),
  `fuse.py` (scene assembly + QA), `e2e.py` (the parallel driver).
- `model/` owns the ONE architecture (patch-graph ConvFormer). There
  is never a second copy of it anywhere in the repo.
- `inference/` owns running the model: `local.py` is canonical;
  `modal_app.py` is a thin remote shell over exactly the same code.
  All Modal stages go through `modal_adapter.StageApp` — never write a
  bespoke Modal app per stage.
- `agent/` owns the multi-provider LLM harness.
- `export.py` owns artifacts -> display geometry (GLB + demo-editor
  fixtures). `eval.py` owns the metrics.
- `viewer/` (top level, outside the wheel) owns the web viewers and
  their dev servers.
- `research/` owns training, datagen, and experiments. It NEVER
  defines architectures or metrics — it imports them from the package.
  Checkpoints carry their config dict; `tests/test_parity.py` enforces
  that any checkpoint in the documented format loads through
  `roomform.inference.local`. Graduation or death: experiments either
  promote into the package or get deleted.

Outputs: pipeline results go to `artifacts/` (gitignored), weights to
`checkpoints/` (gitignored), secrets to `.env` (gitignored).

## Hard rules

- The evidence channel construction in `pipe/evidence/raw.py` is the
  training-time construction, ported verbatim. Do not "improve" its
  math (dedup, normalization, channel order) — any change silently
  breaks every existing checkpoint.
- `EDGE_OFFSETS` order in `model/convformer.py` is a serialized
  contract (lexicographic forward half of the 26-neighborhood). Never
  reorder it.
- Model outputs are a tuple; consume `outputs[:2]` for (node, edge)
  unless you know the checkpoint's optional heads.
- Inline imports only for truly optional paid-service deps
  (`fal_client`). Everything else imports at module top.
- Never orphan a spawned Modal app: keep the client context alive
  until `.get()` returns.

## Development commands

Use `uv` for everything.

```bash
uv sync --extra model --extra modal --extra dev   # extras are additive per
                                                  # invocation — resyncing with
                                                  # fewer extras REMOVES the rest
uv run pytest tests/ -q
uv run ruff format . && uv run ruff check .
uv run python -m roomform.pipe.e2e SCAN.ply       # full pipeline
uv run python viewer/serve.py                     # editor on :8790
```

Credentials: `uv run modal setup` for lifting; `FAL_KEY` in `.env` for
live SAM 3D.

## Coding style

Target Python 3.11. Ruff enforces line length 79, import sorting, and
the configured lint set — run format + check before handing off.

Use `from __future__ import annotations`. Keep pydantic models and
helpers small; prefer plain functions and obvious control flow over
frameworks, registries, and fallback machinery. Module docstrings
state what the module owns and its usage line; comments state
constraints the code can't show, not narration.

Commit messages: imperative subject, short body explaining the why.

## Testing

Add or update tests when behavior changes; keep them fast and free of
paid calls (no live Modal/FAL in tests). `tests/test_parity.py` is the
gate that matters — if you touch the model, config, or checkpoint
format, it must still pass unmodified checkpoints.

Run before handing off:

```bash
uv run pytest tests/ -q && uv run ruff check .
```
