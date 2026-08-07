"""propose_gaps on synthetic grids + provenance of apply()."""

from __future__ import annotations

import numpy as np

from roomform.agent.envelope import Verdict, apply, propose_gaps

VOX = 0.08


def _grid(gap_cells: int):
    """Two coplanar wall sheets at x=5 separated along y by a gap."""
    node = np.zeros((3, 40, 40, 12), dtype=np.float16)
    node[0, 5, 2:10, 1:9] = 1.0
    node[0, 5, 10 + gap_cells : 30, 1:9] = 1.0
    pg = {"node_probs": node}
    ev = {"occ": np.zeros((40, 40, 12), dtype=np.uint8)}
    return pg, ev


def test_two_sheets_close_gap_is_candidate():
    pg, _ = _grid(gap_cells=4)  # 4 * 0.08 = 0.32 m
    cands = propose_gaps(pg, vox_m=VOX)
    walls = [c for c in cands if c.kind == "wall_gap"]
    assert len(walls) == 1
    c = walls[0]
    assert c.span_m <= 1.2
    assert len(c.fill) > 0
    assert (c.fill >= 0).all()
    assert (c.fill < np.array([40, 40, 12])).all()


def test_far_sheets_are_not_candidates():
    pg, _ = _grid(gap_cells=18)  # 18 * 0.08 = 1.44 m > 1.2 m
    cands = propose_gaps(pg, vox_m=VOX)
    assert [c for c in cands if c.kind == "wall_gap"] == []


def test_apply_is_additive_and_preserves_node_probs(tmp_path):
    pg, _ = _grid(gap_cells=4)
    np.savez_compressed(tmp_path / "patchgraph.npz", **pg)
    cands = propose_gaps(pg, vox_m=VOX)
    decisions = [
        Verdict(verdict="approve", reason="approved for test") for _ in cands
    ]
    record = apply(tmp_path, cands, decisions, approve_all=True)
    assert record.approved_voxels > 0
    with np.load(tmp_path / "patchgraph.npz") as d:
        assert d["agent_fill"].shape == pg["node_probs"].shape
        assert d["agent_fill"][0].sum() == record.approved_voxels
        assert d["agent_fill"][1:].sum() == 0
        assert np.array_equal(d["node_probs"], pg["node_probs"])
    assert (tmp_path / "agent-completion.json").exists()


def test_apply_preserves_existing_fills(tmp_path):
    pg, _ = _grid(gap_cells=4)
    previous = np.zeros_like(pg["node_probs"], dtype=np.uint8)
    previous[2, 1, 1, 1] = 1
    np.savez_compressed(tmp_path / "patchgraph.npz", **pg, agent_fill=previous)
    cands = propose_gaps(pg, vox_m=VOX)
    decisions = [
        Verdict(verdict="approve", reason="approved for test") for _ in cands
    ]

    apply(tmp_path, cands, decisions)

    with np.load(tmp_path / "patchgraph.npz") as d:
        assert d["agent_fill"][2, 1, 1, 1] == 1
        assert d["agent_fill"][0].sum() > 0
