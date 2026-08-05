# research/

Experiments and the synthetic data supply chain. **Not part of the pip
package** — nothing in `roomform/` may import from here.

## Layout

- `datagen/` — synthetic building generation → `ScanInput` /
  `EvidenceGrid` documents (the training supply chain; fully seeded and
  deterministic so any dataset is regenerable from the repo).
- `exp_NNNN_<slug>/` — one directory per experiment:
  - `config.yaml` — overlays on `TrainConfig` / `ModelConfig` defaults
  - `run.py` — a thin launcher importing `roomform.{model,train}`
  - `README.md` — hypothesis → result → verdict (write the verdict!)

## The parity rules

1. **No architectures here.** An experiment is a config plus a script.
   Architecture changes go into `roomform/model` behind a config field
   — never a forked copy.
2. **No metrics here.** All metrics come from `roomform.train.eval`,
   so every experiment's numbers are comparable by construction.
3. **Prototype escape valve:** a genuinely new architecture may live in
   its experiment dir while being evaluated, but it graduates
   (merged into `roomform/model`, config-gated) or dies with its
   verdict. It must never be imported by anything outside its dir, and
   release checkpoints are never trained from research code.
4. **Data and checkpoints never enter git** — volumes / releases only.

The enforcement is `tests/test_parity.py`: every released checkpoint
must round-trip through `roomform.inference.local`.
