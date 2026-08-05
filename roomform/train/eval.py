"""THE eval harness — every experiment reports these metrics and no
others. Definitions match the project's standing decomposition:

- shell_f1: F1 over all structure cells (any of wall/floor/ceiling).
- occluded_shell_f1: same, restricted to GT-structure cells with no
  observed evidence (the completion metric).
- connectivity_f1: F1 over forward-edge channels, evaluated where
  either endpoint region is GT structure.
- per-class F1s for wall/floor/ceiling.

All inputs are numpy arrays; thresholds are explicit arguments so
operating points are part of the reported result, never implicit.
"""

from __future__ import annotations

import numpy as np

CLASSES = ("wall", "floor", "ceiling")


def _f1(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = float((pred & gt).sum())
    p = tp / max(float(pred.sum()), 1.0)
    r = tp / max(float(gt.sum()), 1.0)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


def evaluate(
    node_probs: np.ndarray,  # [3,X,Y,Z]
    edge_probs: np.ndarray,  # [13,X,Y,Z]
    node_gt: np.ndarray,  # [3,X,Y,Z] bool
    edge_gt: np.ndarray,  # [13,X,Y,Z] bool
    observed: np.ndarray,  # [X,Y,Z] bool — evidence cells
    node_threshold: float = 0.5,
    edge_threshold: float = 0.5,
) -> dict:
    node_pred = node_probs > node_threshold
    any_pred = node_pred.any(axis=0)
    any_gt = node_gt.any(axis=0)

    out: dict = {"shell": _f1(any_pred, any_gt)}
    occluded = any_gt & ~observed
    out["occluded_shell"] = _f1(any_pred & ~observed, occluded)
    for k, name in enumerate(CLASSES):
        out[name] = _f1(node_pred[k], node_gt[k])

    edge_pred = edge_probs > edge_threshold
    edge_mask = any_gt[None]  # edges scored where source is structure
    out["connectivity"] = _f1(edge_pred & edge_mask, edge_gt & edge_mask)
    out["thresholds"] = {"node": node_threshold, "edge": edge_threshold}
    return out
