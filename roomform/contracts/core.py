"""Roomform data contracts — the interchange spine of the pipeline.

Every stage boundary in the system is one of these documents. Code may
evolve freely; these schemas change only deliberately (bump VERSION and
document the migration in docs/data-contracts.md).

    capture (ScanInput)
      -> EvidenceGrid          voxelized tri-state evidence
      -> PatchGraph            predicted shell: nodes + connectivity
      -> SceneDocument         shell + objects + per-object QA

Conventions (apply everywhere):
  - meters, z-up, gravity-aligned; grid origin = min corner of the
    scan AABB, stored so raw-frame data (detections, meshes) can be
    re-aligned exactly.
  - voxel index (i, j, k) covers [origin + idx*vox, origin+(idx+1)*vox)
  - arrays are C-order (i, j, k); npz keys are lowercase snake_case.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

VERSION = "0.1.0"


class ScanInput(BaseModel):
    """A capture: point cloud + optional per-point/pose extras.

    Stored as npz. Required: pts [N,3] f32 (meters, z-up). Optional:
    pts_color [N,3] u8, pts_normal [N,3] f32 (unit), stations [S,3]
    f32 (scanner positions — enables exact visibility carving; absent
    for trajectory-less scans, in which case visibility falls back to
    synthesized viewpoints and MUST be marked approximate).
    """

    npz_path: str
    n_points: int
    has_color: bool = False
    has_normals: bool = False
    has_stations: bool = False


class EvidenceGrid(BaseModel):
    """Voxelized tri-state evidence — the model-facing input contract.

    npz keys (all [X,Y,Z], C-order):
      occ  u8  observed occupancy (scan hit)
      sf   u8  seen-free: a ray provably traversed (wall impossible)
               (unknown = ~occ & ~sf is derived, never stored)
    Optional observable channels (raw-recoverable, no semantics):
      gray u8, nrm_abs [3,X,Y,Z] f16, log_density f16
    Invariants: occ & sf disjoint; vox_m in {0.02, 0.04, 0.08};
    origin recorded; visibility_source in {"stations", "synthesized"}.
    """

    npz_path: str
    vox_m: float
    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    visibility_source: str = Field(pattern="^(stations|synthesized)$")


class PatchGraphNode(BaseModel):
    """One predicted structural cell at grid resolution."""

    idx: tuple[int, int, int]
    p_wall: float
    p_floor: float
    p_ceiling: float


class PatchGraph(BaseModel):
    """Predicted shell as a connected surface graph.

    Dense storage (npz) referenced here; this model is the header.
      node_probs [3,X,Y,Z] f16   wall/floor/ceiling probabilities
      agent_fill [3,X,Y,Z] u8    optional VLM-approved additive fills
      edge_probs [13,X,Y,Z] f16  forward-edge connectivity: the 13
        non-duplicate offsets of the 26-neighborhood, fixed order
        documented in docs/data-contracts.md (edge k at cell c connects
        c -> c + OFFSETS[k]; reverse edges are implied)
    Openings are represented by ABSENCE of nodes/edges (v0); an
    explicit opening head is a planned v0.2 extension.
    """

    npz_path: str
    vox_m: float
    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    node_threshold: float = 0.5
    edge_threshold: float = 0.5
    model_id: str = ""


class ObjectQA(BaseModel):
    """Fusion-time quality checks for one object."""

    wall_leak_pts: int | None = None  # predicted-wall pts inside the box
    floor_support_m: float | None = None  # box bottom minus shell floor_z

    @property
    def leaking(self) -> bool:
        return (self.wall_leak_pts or 0) > 50


class SceneObject(BaseModel):
    """One detected/reconstructed object in the scene."""

    cls: str
    center: tuple[float, float, float]  # grid frame (origin-shifted)
    size: tuple[float, float, float]
    heading: float  # z-yaw, radians
    source: str = "spatiallm"  # spatiallm | sam3d | ...
    mesh_path: str | None = None  # aligned mesh if reconstructed
    qa: ObjectQA = Field(default_factory=lambda: ObjectQA())


class SceneDocument(BaseModel):
    """The end-to-end product: shell + objects + provenance."""

    version: str = VERSION
    source_scan: str
    frame_shift: tuple[float, float, float]
    shell: PatchGraph
    objects: list[SceneObject]
    floor_z: float | None = None
