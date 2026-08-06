"""Clustering/box derivation for the pointlabel lifting backend."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("modal")

from roomform.pipe.lifting.pointlabel import CLASSES, lift_from_labels


def _block(center, size, yaw, step=0.04):
    """Axis grid at real-scan density, rotated by yaw about z."""
    axes = [np.arange(-s / 2, s / 2 + 1e-9, step) for s in size]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    c, s = np.cos(yaw), np.sin(yaw)
    pts[:, :2] = pts[:, :2] @ np.array([[c, s], [-s, c]])
    return pts + center


def test_lift_from_labels(tmp_path):
    yaw = 0.5
    chairs = [
        _block((1.0, 1.0, 0.4), (0.9, 0.5, 0.8), yaw),
        _block((4.0, 1.0, 0.4), (0.9, 0.5, 0.8), yaw),
    ]
    floor = _block((2.0, 2.0, 0.0), (6.0, 6.0, 0.02), 0.0)
    tiny = _block((0.0, 3.0, 0.4), (0.15, 0.15, 0.15), 0.0)
    pts = np.concatenate(chairs + [floor, tiny]).astype(np.float32)
    label = np.concatenate(
        [np.full(len(c), CLASSES.index("chair")) for c in chairs]
        + [
            np.full(len(floor), CLASSES.index("floor")),
            np.full(len(tiny), CLASSES.index("chair")),
        ]
    ).astype(np.uint8)
    path = tmp_path / "labels.npz"
    np.savez(path, pts=pts, label=label, classes=np.array(CLASSES))

    objs = lift_from_labels(str(path), (1.0, 0.0, 0.0))
    # floor excluded, tiny cluster below MIN_CLUSTER_PTS dropped
    assert [o.cls for o in objs] == ["chair", "chair"]
    a = objs[0]
    assert a.center == pytest.approx((0.0, 1.0, 0.4), abs=0.05)
    assert a.size == pytest.approx((0.9, 0.5, 0.8), abs=0.05)
    d = abs(a.heading - yaw) % np.pi
    assert min(d, np.pi - d) < 0.05
    assert a.source == "pointlabel"
