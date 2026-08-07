"""Agent envelope completion: propose -> render card -> judge -> apply.

The model leaves residual boundary defects: disconnected wall sheets
and floor/ceiling holes. This module proposes candidate fills
deterministically, renders one self-contained "profile card" (top-down
footprint + vertical section) per candidate, gets a SINGLE-SHOT VLM
    approve/reject per candidate (no agent loops), and writes approved
    fills as an ADDITIVE `agent_fill [3,X,Y,Z]` key in patchgraph.npz — measured
`node_probs` are never modified (provenance separation).

Usage:
    uv run python -m roomform.agent.envelope artifacts/<scene> \
        [--approve-all] [--provider anthropic|openai|claude-cli]

Providers: `anthropic` (default, claude-opus-5, ANTHROPIC_API_KEY),
`openai` (OPENAI_API_KEY), `claude-cli` (headless Claude Code
subprocess using the machine's subscription login — no API key).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import label

from roomform.agent.provider import create_client
from roomform.agent.settings import Settings, get_settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_SPAN_M = 1.2  # widest wall gap we propose closing
MIN_COMPONENT = 30  # wall sheets smaller than this are speck noise
MAX_COMPONENTS = 12  # ponytail: O(K^2) pair scan; KD-forest if scenes grow
MAX_CANDIDATES = 20  # bound on VLM calls per scene
MIN_HOLE_CELLS = 4
COL = {
    "evidence": "#9a9a9a",
    "boundary": "#3b6fd4",
    "fill": "#f5a942",
    "opening": "#c542a4",
}


@dataclass
class Candidate:
    id: str
    kind: str  # wall_gap | floor_hole | ceiling_hole
    span_m: float
    a: tuple[int, int, int]  # closest cell on sheet A (wall gaps)
    b: tuple[int, int, int]
    fill: np.ndarray = field(repr=False)  # [N,3] int voxel indices
    card: str = ""


# ---------------------------------------------------------------- propose


def _line_fill(a: np.ndarray, b: np.ndarray, shape) -> np.ndarray:
    """Voxels along a->b, thickened by one voxel."""
    n = int(np.max(np.abs(b - a))) * 2 + 1
    pts = np.unique(np.rint(np.linspace(a, b, n)).astype(int), axis=0)
    mask = np.zeros(shape, dtype=bool)
    mask[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    mask = ndimage.binary_dilation(mask, iterations=1)
    return np.argwhere(mask)


def _wall_gaps(wall: np.ndarray, vox: float) -> list[Candidate]:
    labels = label(wall, connectivity=3)
    sizes = np.bincount(labels.ravel())
    keep = [
        i
        for i in np.argsort(sizes)[::-1]
        if i != 0 and sizes[i] >= MIN_COMPONENT
    ][:MAX_COMPONENTS]
    comps = {i: np.argwhere(labels == i) for i in keep}
    trees = {i: cKDTree(comps[i]) for i in keep}
    out = []
    for ai in range(len(keep)):
        for bi in range(ai + 1, len(keep)):
            ca, cb = comps[keep[ai]], comps[keep[bi]]
            d, j = trees[keep[bi]].query(ca)
            k = int(np.argmin(d))
            span = float(d[k]) * vox
            if span > MAX_SPAN_M:
                continue
            a, b = ca[k], cb[j[k]]
            out.append(
                Candidate(
                    id="",
                    kind="wall_gap",
                    span_m=round(span, 3),
                    a=tuple(int(v) for v in a),
                    b=tuple(int(v) for v in b),
                    fill=_line_fill(a, b, wall.shape),
                )
            )
    return out


def _slab_holes(
    plane: np.ndarray, wall: np.ndarray, vox: float, kind: str
) -> list[Candidate]:
    """Holes in the floor/ceiling sheet that touch a wall."""
    counts = plane.sum(axis=(0, 1))
    if not counts.any():
        return []
    z = int(np.argmax(counts))  # dominant slab height
    sheet2d = plane.any(axis=2)
    holes = ndimage.binary_fill_holes(sheet2d) & ~sheet2d
    wall2d = wall.any(axis=2)
    out = []
    lab, n = ndimage.label(holes)
    for i in range(1, n + 1):
        cells = np.argwhere(lab == i)
        if len(cells) < MIN_HOLE_CELLS:
            continue
        near_wall = ndimage.binary_dilation(lab == i, iterations=2)
        if not (near_wall & wall2d).any():
            continue
        fill = np.column_stack([cells, np.full(len(cells), z, dtype=int)])
        c = cells[len(cells) // 2]
        out.append(
            Candidate(
                id="",
                kind=kind,
                span_m=round(float(np.sqrt(len(cells))) * vox, 3),
                a=(int(c[0]), int(c[1]), z),
                b=(int(c[0]), int(c[1]), z),
                fill=fill,
            )
        )
    return out


def propose_gaps(
    pg: dict[str, np.ndarray],
    *,
    vox_m: float,
    node_threshold: float = 0.5,
) -> list[Candidate]:
    """Deterministic candidate defects; no ML.

    Wall gaps: pairs of nearby-but-disconnected wall components whose
    closing span is <= MAX_SPAN_M. Floor/ceiling holes: 2D holes in
    the dominant slab that sit adjacent to walls.
    """
    node = pg["node_probs"].astype(np.float32) > node_threshold
    wall, floor, ceil = node[0], node[1], node[2]
    cands = _wall_gaps(wall, vox_m)
    cands += _slab_holes(floor, wall, vox_m, "floor_hole")
    cands += _slab_holes(ceil, wall, vox_m, "ceiling_hole")
    cands.sort(key=lambda c: c.span_m)
    cands = cands[:MAX_CANDIDATES]
    for i, c in enumerate(cands):
        c.id = f"{c.kind}-{i:03d}"
    return cands


# ----------------------------------------------------------------- render


def _scatter(ax, pts, color, size, label_):
    if len(pts):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=size,
            c=color,
            linewidths=0,
            label=label_,
        )


def render_profile_card(
    cand: Candidate,
    pg: dict[str, np.ndarray],
    evidence: dict[str, np.ndarray],
    *,
    vox_m: float,
    node_threshold: float,
    out_path: Path,
) -> Path:
    """Self-contained PNG a VLM can judge: top-down crop + section."""
    node = pg["node_probs"].astype(np.float32) > node_threshold
    boundary = node.any(axis=0)
    occ = evidence["occ"].astype(bool)
    openings = pg.get("openings")
    open_mask = (
        (openings.astype(np.float32) > 0.5).any(axis=0)
        if openings is not None
        else np.zeros_like(occ)
    )
    fill = np.zeros_like(occ)
    fill[cand.fill[:, 0], cand.fill[:, 1], cand.fill[:, 2]] = True

    a = np.asarray(cand.a, float)
    b = np.asarray(cand.b, float)
    center = (a + b) / 2
    pad = max(1.5 / vox_m, np.abs(b - a).max() + 6)

    def crop(mask):
        idx = np.argwhere(mask)
        if not len(idx):
            return idx
        keep = (np.abs(idx[:, :2] - center[:2]) <= pad).all(axis=1)
        return idx[keep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
    # top-down footprint crop (grid x, y)
    _scatter(ax1, crop(occ), COL["evidence"], 2, "evidence")
    _scatter(ax1, crop(boundary), COL["boundary"], 4, "predicted")
    _scatter(ax1, crop(open_mask), COL["opening"], 6, "opening")
    _scatter(ax1, crop(fill), COL["fill"], 10, "proposed fill")
    ax1.set_title("top-down (crop)")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", fontsize=7)

    # vertical section through the candidate
    d = b[:2] - a[:2]
    if np.linalg.norm(d) < 1e-6:
        d = np.array([1.0, 0.0])
    d = d / np.linalg.norm(d)
    perp = np.array([-d[1], d[0]])

    def section(mask, halfwidth=2.0):
        idx = np.argwhere(mask)
        if not len(idx):
            return idx
        rel = idx[:, :2] - center[:2]
        u = rel @ d
        v = rel @ perp
        keep = (np.abs(v) <= halfwidth) & (np.abs(u) <= pad)
        return np.column_stack([u[keep], idx[keep, 2]])

    _scatter(ax2, section(occ), COL["evidence"], 3, None)
    _scatter(ax2, section(boundary), COL["boundary"], 6, None)
    _scatter(ax2, section(open_mask), COL["opening"], 8, None)
    _scatter(ax2, section(fill), COL["fill"], 14, None)
    ax2.set_title("vertical section along candidate")
    ax2.set_aspect("equal")
    ax2.set_xlabel("u (voxels)")
    ax2.set_ylabel("z (voxels)")

    fig.suptitle(
        f"{cand.id} | {cand.kind} | span {cand.span_m} m | "
        f"vox {vox_m} m | gray=evidence blue=predicted "
        f"amber=proposed-fill magenta=opening"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    cand.card = str(out_path)
    return out_path


# ------------------------------------------------------------------ judge

_PROMPT = """You are auditing a room-boundary reconstruction.

The image is a profile card for ONE candidate fill patch:
left = top-down crop, right = vertical section through the candidate.
Colors: gray = scan evidence points, blue = predicted boundary,
amber = the proposed fill patch under review, magenta = predicted
door/window openings.

Candidate: kind={kind}, closing span={span_m} m.

Approve the fill only if it closes a genuinely missing structural
surface (a wall/floor/ceiling continuation, e.g. occluded by
furniture or unscanned). Reject if the amber region is plausibly a
real opening (door, window, stair, room boundary end) or if nearby
evidence/openings contradict a solid surface there.

Respond with JSON only: {{"verdict": "approve"|"reject",
"reason": "<one sentence>"}}"""


def judge(
    candidates: list[Candidate],
    *,
    settings: Settings | None = None,
    approve_all: bool = False,
) -> list[dict[str, str]]:
    """One VLM call per candidate, or deterministic approval for inspection."""
    if approve_all:
        return [
            {"verdict": "approve", "reason": "approved (--approve-all)"}
            for _ in candidates
        ]
    client = create_client(settings or get_settings())
    decisions = []
    for c in candidates:
        prompt = _PROMPT.format(kind=c.kind, span_m=c.span_m)
        raw = client.judge(prompt, Path(c.card))
        verdict = str(raw.get("verdict", "reject"))
        if verdict not in ("approve", "reject"):
            verdict = "reject"
        decisions.append(
            {"verdict": verdict, "reason": str(raw.get("reason", ""))}
        )
    return decisions


# ------------------------------------------------------------------ apply


def apply(
    scene_dir: Path,
    candidates: list[Candidate],
    decisions: list[dict[str, str]],
    *,
    approve_all: bool = False,
    provider: str = "anthropic",
) -> dict:
    """Write additive agent_fill into patchgraph.npz + append the log.

    node_probs are never modified: agent fills stay a separate key so
    measured geometry and inferred patches remain distinguishable.
    """
    pg_path = scene_dir / "patchgraph.npz"
    with np.load(pg_path) as d:
        data = {k: d[k] for k in d.files}
    if len(candidates) != len(decisions):
        raise ValueError("each candidate requires exactly one decision")

    shape = data["node_probs"].shape
    existing = data.get("agent_fill")
    agent_fill = (
        existing.astype(np.uint8, copy=True)
        if existing is not None
        else np.zeros(shape, dtype=np.uint8)
    )
    if agent_fill.shape != shape:
        raise ValueError(
            f"agent_fill shape {agent_fill.shape} does not match {shape}"
        )
    class_index = {"wall_gap": 0, "floor_hole": 1, "ceiling_hole": 2}
    for c, dec in zip(candidates, decisions):
        if dec["verdict"] == "approve":
            channel = class_index[c.kind]
            agent_fill[channel, c.fill[:, 0], c.fill[:, 1], c.fill[:, 2]] = 1
    data["agent_fill"] = agent_fill
    np.savez_compressed(pg_path, **data)

    record = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": provider,
        "approve_all": approve_all,
        "approved_voxels": int(agent_fill.sum()),
        "decisions": [
            {
                "candidate": c.id,
                "kind": c.kind,
                "span_m": c.span_m,
                "verdict": dec["verdict"],
                "reason": dec["reason"],
                "card": c.card,
            }
            for c, dec in zip(candidates, decisions)
        ],
    }
    log_path = scene_dir / "agent-completion.json"
    runs = json.loads(log_path.read_text()) if log_path.exists() else []
    runs.append(record)
    log_path.write_text(json.dumps(runs, indent=2) + "\n")
    return record


# -------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="skip VLM calls and approve every proposal (inspection only)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "claude-cli"],
        default=None,
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"provider": args.provider})

    scene = json.loads((args.scene_dir / "scene.json").read_text())
    vox = float(scene["shell"]["vox_m"])
    thr = float(scene["shell"].get("node_threshold", 0.5))
    with np.load(args.scene_dir / "patchgraph.npz") as d:
        pg = {k: d[k] for k in d.files}
    with np.load(args.scene_dir / "evidence.npz") as d:
        ev = {k: d[k] for k in d.files}

    cands = propose_gaps(pg, vox_m=vox, node_threshold=thr)
    for c in cands:
        render_profile_card(
            c,
            pg,
            ev,
            vox_m=vox,
            node_threshold=thr,
            out_path=args.scene_dir / "agent-cards" / f"{c.id}.png",
        )
    decisions = judge(cands, settings=settings, approve_all=args.approve_all)
    record = apply(
        args.scene_dir,
        cands,
        decisions,
        approve_all=args.approve_all,
        provider=settings.provider,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
