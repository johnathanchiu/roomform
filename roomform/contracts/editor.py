"""Typed mirror of the production editor's fixture schemas.

These shapes belong to the editor (its zod schemas are the source of
truth); this module keeps the export adapter honest — a field rename
here fails loudly at construction instead of silently emitting JSON
the editor drops. The ``Splat*`` vocabulary is legacy from the
editor's gaussian-splat era — its schemas kept the names when it moved
to clouds/meshes, and this mirror keeps them so fields map 1:1 to the
zod source of truth. Frames: `position` is the editor's y-up frame
[x, height, -y]; `bounds` is [dx, height, dy]; `rotation_xyzw` is a
quaternion about +Y.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SPLAT_ANALYSIS_SCHEMA = "roomform.splat-analysis.result.v1"
FIXTURES_INDEX_SCHEMA = "roomform.fixtures-index.v1"
SCENE_PROGRAM_SCHEMA = "roomform.scene-program.v1"


class SplatObjectProposal(BaseModel):
    id: str
    label: str
    category: str
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    bounds: tuple[float, float, float]
    confidence: float = 0.7
    method: str = "spatiallm"
    source_views: list = Field(default_factory=list)
    measured_mesh_uri: str | None = None
    measured_mesh_z_up: bool | None = None


class SplatAnalysis(BaseModel):
    schema_: str = Field(SPLAT_ANALYSIS_SCHEMA, alias="schema")
    scene_id: str
    splat_uri: str
    detector: str = "spatiallm/qwen"
    coordinate_system: str = "source-splat"
    objects: list[SplatObjectProposal] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class SceneProgram(BaseModel):
    schema_: str = Field(SCENE_PROGRAM_SCHEMA, alias="schema")
    dataset_id: str
    statements: list = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class FixtureAssets(BaseModel):
    input: str
    mesh: str
    shell: str
    editable: str
    program: str
    objectsAnalysis: str | None = None


class FixtureStats(BaseModel):
    objects: int | None = None
    walls: int | None = None


class FixtureEntry(BaseModel):
    id: str
    title: str
    stats: FixtureStats = FixtureStats()
    assets: FixtureAssets
    updated_at: str


class FixturesIndex(BaseModel):
    schema_: str = Field(FIXTURES_INDEX_SCHEMA, alias="schema")
    scenes: list[FixtureEntry] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    def upsert(self, entry: FixtureEntry) -> None:
        self.scenes = [s for s in self.scenes if s.id != entry.id] + [entry]
