# roomform

Indoor structure completion from real-world scans: point cloud in,
watertight architectural shell + objects out.

Pipeline spine (see docs/data-contracts.md — the contracts ARE the
architecture; code transfers in behind them):

    ScanInput -> EvidenceGrid -> PatchGraph -> SceneDocument

Status: contracts-first scaffold. Model/datagen/eval code migrates
from the internal monorepo behind these interfaces.
