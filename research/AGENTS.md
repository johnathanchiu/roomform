# research/ guidelines

Training and experiments. Not part of the pip package — nothing in
`roomform/` may import from here.

## The parity rules

1. **No architectures here.** An experiment is a config plus a
   script; architecture changes go into `roomform/model` behind a
   config field — never a forked copy.
2. **No metrics here.** All numbers come from `roomform.eval`, so
   experiments are comparable by construction.
3. **Prototype escape valve:** a genuinely new architecture may live
   in its experiment dir while being evaluated, but it graduates
   (merged into `roomform/model`, config-gated) or dies with its
   verdict. It must never be imported outside its dir, and release
   checkpoints are never trained from research code.
4. **Data and checkpoints never enter git** — volumes / releases
   only.

Enforcement: `tests/test_parity.py` — every released checkpoint must
round-trip through `roomform.inference.local`.

## Experiments

One directory per experiment (`exp_NNNN_<slug>/`): `config.yaml`
(overlays on TrainConfig/ModelConfig defaults), `run.py` (thin
launcher), `README.md` (hypothesis → result → verdict — write the
verdict!).
