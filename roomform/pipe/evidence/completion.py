"""Stage 1 — evidence completion: EvidenceGrid -> PatchGraph.

The patch-graph ConvFormer owns this stage. Until its trainer/weights
migrate from the internal repo, this module defines the interface and
a loader for exported predictions, so e2e runs today on archived
outputs and swaps to live inference with no downstream change.
"""

from __future__ import annotations

from roomform.contracts import EvidenceGrid, PatchGraph


def complete_shell(evidence: EvidenceGrid, model_id: str = "") -> PatchGraph:
    """Live inference entrypoint (model migrates in behind this)."""
    raise NotImplementedError(
        "patch-graph model not yet migrated; use load_patchgraph() on "
        "exported predictions"
    )


def load_patchgraph(
    npz_path: str,
    vox_m: float,
    origin: tuple[float, float, float],
    shape: tuple[int, int, int],
    model_id: str = "archived",
) -> PatchGraph:
    """Wrap an exported prediction npz (node_probs [3,...],
    edge_probs [13,...]) as the contract document."""
    return PatchGraph(
        npz_path=npz_path,
        vox_m=vox_m,
        origin=origin,
        shape=shape,
        model_id=model_id,
    )
