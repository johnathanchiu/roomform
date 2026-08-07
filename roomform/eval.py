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
from pydantic import BaseModel

CLASSES = ("wall", "floor", "ceiling")


class F1(BaseModel):
    precision: float
    recall: float
    f1: float


class Thresholds(BaseModel):
    node: float
    edge: float


class EvalResult(BaseModel):
    shell: F1
    occluded_shell: F1
    wall: F1
    floor: F1
    ceiling: F1
    connectivity: F1
    thresholds: Thresholds

    def __getitem__(self, key: str) -> F1:  # metric-by-name access
        value = getattr(self, key)
        if not isinstance(value, F1):
            raise KeyError(key)
        return value


def _f1(pred: np.ndarray, gt: np.ndarray) -> F1:
    tp = float((pred & gt).sum())
    p = tp / max(float(pred.sum()), 1.0)
    r = tp / max(float(gt.sum()), 1.0)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return F1(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4))


def evaluate(
    node_probs: np.ndarray,  # [3,X,Y,Z]
    edge_probs: np.ndarray,  # [13,X,Y,Z]
    node_gt: np.ndarray,  # [3,X,Y,Z] bool
    edge_gt: np.ndarray,  # [13,X,Y,Z] bool
    observed: np.ndarray,  # [X,Y,Z] bool — evidence cells
    node_threshold: float = 0.5,
    edge_threshold: float = 0.5,
) -> EvalResult:
    node_pred = node_probs > node_threshold
    any_pred = node_pred.any(axis=0)
    any_gt = node_gt.any(axis=0)

    edge_pred = edge_probs > edge_threshold
    edge_mask = any_gt[None]  # edges scored where source is structure
    per_class = {
        name: _f1(node_pred[k], node_gt[k]) for k, name in enumerate(CLASSES)
    }
    return EvalResult(
        shell=_f1(any_pred, any_gt),
        occluded_shell=_f1(any_pred & ~observed, any_gt & ~observed),
        connectivity=_f1(edge_pred & edge_mask, edge_gt & edge_mask),
        thresholds=Thresholds(node=node_threshold, edge=edge_threshold),
        **per_class,
    )
