# research/

Experiments and the synthetic data supply chain. **Not part of the pip
package** — nothing in `roomform/` may import from here.

## Layout

- `train/` — TrainConfig, losses, the training loop and
  checkpointing (launch machinery lives with the experiments).
- `datagen/` — synthetic building generation → `ScanInput` /
  `EvidenceGrid` documents (the training supply chain; fully seeded and
  deterministic so any dataset is regenerable from the repo).
- `exp_NNNN_<slug>/` — one directory per experiment:
  - `config.yaml` — overlays on `research/train` TrainConfig / `ModelConfig` defaults
  - `run.py` — a thin launcher importing `roomform.model` + `research.train`
  - `README.md` — hypothesis → result → verdict (write the verdict!)

Working rules for this directory (the parity rules) live in the
repo-root [AGENTS.md](../AGENTS.md); `tests/test_parity.py` enforces
them — every released checkpoint must round-trip through
`roomform.inference.local`.
