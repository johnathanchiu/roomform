# Roomform data contracts (v0.1)

The pipeline is four documents; every stage reads one and writes the
next. Schemas live in `roomform/contracts/core.py` (pydantic); this
file is the human spec and the migration log.

```
ScanInput ──voxelize+carve──▶ EvidenceGrid ──model──▶ PatchGraph ──fuse──▶ SceneDocument
 (capture)                    (tri-state)             (shell graph)        (+ objects)
```

## Global conventions

- Units: meters. Axes: z-up, gravity-aligned. Arrays: C-order (i,j,k).
- Grid origin = min corner of scan AABB; `frame_shift` carries it so
  raw-frame artifacts (SpatialLM boxes, SAM3D meshes) re-align exactly:
  `grid = raw - frame_shift`.
- Voxel (i,j,k) spans `[origin + idx*vox, origin + (idx+1)*vox)`.

## 1. ScanInput

npz with `pts [N,3] f32` required; optional `pts_color u8`,
`pts_normal f32`, `stations [S,3] f32`. If `stations` is absent,
downstream visibility is synthesized from free-space viewpoints and
tagged `visibility_source="synthesized"` — consumers must treat
seen-free as approximate in that case.

## 2. EvidenceGrid

Tri-state observability, the model-facing input:

| key | dtype | meaning |
|---|---|---|
| `occ` | u8 | scan hit (evidence) |
| `sf` | u8 | ray provably traversed — structure impossible |
| *(derived)* unknown | — | `~occ & ~sf` — never stored |

Observable channels are stored pre-stacked as
`features [9,X,Y,Z] f16` in the exact training construction
(`roomform/pipe/evidence/raw.py`): occupancy, grayscale, |normal|
xyz, log-density / 7 clipped to [0,1], mean sub-voxel offset xyz.
Points are deduped to max 3 per 2 cm cell before voxelization.
Grid extents are ceiled to multiples of 8 (UNet stride).

Invariants: `occ & sf` disjoint. `vox_m ∈ {0.02, 0.04, 0.08}`.

Rationale (measured, 2026-08): tri-state beats distance-to-evidence
proxies (+0.09 hidden recall at gate scale); color adds nothing over
occupancy; observable channels keep real scans in-contract without
trajectories.

## 3. PatchGraph

The shell as a connected surface graph at grid resolution:

- `node_probs [3,X,Y,Z] f16` — wall / floor / ceiling.
- `edge_probs [13,X,Y,Z] f16` — forward half of the 26-neighborhood.
  `edge_probs[k]` at cell `c` scores the connection `c -> c+OFFSETS[k]`.

`OFFSETS` (fixed order; reverse edges implied). The order is the
lexicographic-forward half of the 26-neighborhood, generated dx -> dy
-> dz — the authoritative definition is `EDGE_OFFSETS` in
`roomform/model/convformer.py`:

```
k : (di,dj,dk)
0 : ( 0, 0, 1)   1 : ( 0, 1,-1)   2 : ( 0, 1, 0)   3 : ( 0, 1, 1)
4 : ( 1,-1,-1)   5 : ( 1,-1, 0)   6 : ( 1,-1, 1)   7 : ( 1, 0,-1)
8 : ( 1, 0, 0)   9 : ( 1, 0, 1)  10 : ( 1, 1,-1)  11 : ( 1, 1, 0)
12 : ( 1, 1, 1)
```

Openings: v0.1 represents doors/windows as *absence* of nodes/edges.
Planned v0.2: explicit opening head (positive prediction) + sub-voxel
surface offset/normal for geometry straightening. Both are additive
keys — no breaking change.

## 4. SceneDocument

`scene.json` — shell reference + object list + provenance:

- objects carry `cls, center, size, heading (z-yaw), source
  (spatiallm|sam3d), mesh_path?`, and a `qa` dict
  (`wall_leak_pts`: predicted-wall points inside the box, interior
  margin 4cm; `floor_support_m`: box bottom minus shell floor_z).
- everything in the grid frame; `frame_shift` restores raw frame.

## Migration log

- **0.1.0** — initial: four-document spine, tri-state evidence,
  patch-graph 13-edge convention, scene doc with QA.
- **0.1.1** — OFFSETS table corrected to the trained model's
  lexicographic order; evidence stores stacked `features` (the
  training channel construction) instead of separate raw arrays;
  PatchGraph npz gains optional `offsets [3,X,Y,Z] f16` (sub-voxel
  surface refinement, voxel-width units) when the checkpoint
  predicts them.
