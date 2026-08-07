# The patch-graph ConvFormer

One model, one job: raw 8 cm observable evidence in, a connected
surface graph of the building shell out — including the parts the
scanner never saw.

## The representation

The output is deliberately not a dense semantic volume and not a
parametric floor plan. It is a **patch graph** over the voxel grid:

- `node logits [3, X, Y, Z]` — is this cell a wall / floor / ceiling
  surface patch? Three independent sigmoids, not a softmax: a corner
  cell can be wall *and* floor, so classes never compete for
  probability mass.
- `edge logits [13, X, Y, Z]` — is this cell connected to its
  neighbor? The 13 channels are the lexicographic forward half of the
  26-neighborhood (`EDGE_OFFSETS` in `convformer.py`, generated
  dx → dy → dz; reverse edges are implied). Edges make the shell a
  *surface* rather than a soup of independent voxels — connectivity is
  supervised, not post-processed.

Sparse and easy to learn: a typical room is ~4–6% surface cells, and
the network only has to make local yes/no calls that global attention
has already coordinated. There is no deterministic compiler or polygon
fill downstream — what the heads predict is the shell.

## Inputs

Six observable channels at 8 cm (`pipe/evidence/raw.py`, ported
verbatim from training — do not change its math):

    occupancy · grayscale · |normal| xyz · log-density

plus, for `in_ch=9` checkpoints, the mean sub-voxel offset xyz of the
points in each cell (fuel for the offset head, below). Nothing
semantic goes in: no labels, no detections, no ray carving. Grayscale
appearance is kept but heavily augmented at training time so it can
never become a semantic shortcut.

## Architecture

Four explicit stages (~27 M parameters at the default
`base=48, depth=6, heads=8`):

    input [B, in_ch, X, Y, Z]
      stem 3×3×3 conv ─ enc0 (48) ──────────────┐ skip
        down ×2 ─ enc1 (96) ─────────────┐ skip │
          down ×2 ─ enc2 (192) ────┐ skip│      │
            down ×2 (384, 64 cm tokens)  │      │
              attention × depth/2        │      │
              ConvRefine (3×3×3)         │      │
              attention × depth/2        │      │
            up ×2 + skip ─ dec2 (192) ───┘      │
          up ×2 + skip ─ dec1 (96) ─────────────┘
        up ×2 + skip ─ dec0 (48) ───────────────┘
      node head 1×1×1 → 3   ·   edge head 1×1×1 → 13

The implementation mirrors that diagram: `FeatureEncoder` produces the
skip pyramid, `GlobalLocalRefiner` performs the room-scale reasoning,
`FeatureDecoder` restores native resolution, and `PatchGraphHeads`
implement the output contract. `PatchGraphConvFormer.forward` only wires
those stages together.

The reasoning happens at the bottleneck: after three stride-2 convs
each token summarizes a 64 cm block, and full global attention over
those tokens (a ~10 m room is only ~15×16×6 ≈ 1.4k tokens) lets the
model close walls it cannot see by looking at the whole building at
once. A sinusoidal *metric* positional encoding (`_metric_pos`, in
meters, wavelengths 0.32–40 m) means distances stay physical: the
model knows how far apart two walls are, not just their token indices.
The ConvRefine block between the two attention stacks snaps globally
propagated structure back onto the local voxel lattice.

The convolutional encoder/decoder with skip connections keeps the
measured evidence sharp — observed edges and openings come from the
skips, completion comes from the bottleneck.

Everything is padded to multiples of 8 internally (`_pad`); masking
(`valid_voxels`) exists so padded batch cells can never leak context
(ChannelNorm3d normalizes channels only, so padding stays invariant).

## Optional heads

Checkpoints enable these via their config; `forward` returns a tuple
that always starts `(node, edge)` and appends what is enabled:

- **offsets** (`predict_offsets`) — 3 channels, `0.5 * tanh`, in
  voxel widths: a sub-voxel displacement of each surface patch from
  its cell center. Zero-initialized so an untrained head reproduces
  the voxel-center baseline exactly. With `isolated_refinement` the
  raw offset channels feed a separate zero-initialized stem that
  cannot perturb node/edge logits — it only straightens geometry.
- **openings** (`predict_openings`) — 2 channels (door, window).
  Openings are evidence that *subtracts* from latent wall support,
  not a fourth node class; bias starts at −4 so they must be earned.

## Checkpoints

    {"model": state_dict, "epoch": int, "config": {in_ch, base, depth,
     heads, predict_offsets, ...}}

`ModelConfig` (config.py) mirrors the config dict and ignores
training-only extras, so `ModelConfig(**ckpt["config"]).build()`
reconstructs the exact network. `tests/test_parity.py` enforces the
round-trip through `roomform.inference.local`.

## Why this shape wins

Measured on held-out synthetic rooms (400-room training run): shell
F1 0.98, occluded-shell F1 0.94, connectivity F1 0.97 — where dense
semantic-volume and seq2seq/parametric formulations of the same
problem plateaued far lower. On real scans roughly a third of the
predicted shell lands on cells the scanner never observed, which is
the point: the observable channels are enough for the model to learn
what buildings do behind furniture.
