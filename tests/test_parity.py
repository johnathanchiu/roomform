"""Parity enforcement: the packaged architecture must load and run
every checkpoint written in the documented format (the read side lives
in roomform.inference.local; research/train writes the same shape)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_checkpoint_roundtrip(tmp_path):
    from roomform.model.config import ModelConfig

    cfg = ModelConfig(base=8, depth=2, heads=2)
    model = cfg.build()
    path = tmp_path / "checkpoint.pt"
    # training-only extras (lr, batch, ...) must be tolerated on load
    torch.save(
        {
            "model": model.state_dict(),
            "config": {**cfg.model_dump(), "lr": 3e-4},
        },
        path,
    )

    from roomform.inference.local import load_checkpoint

    loaded, loaded_cfg = load_checkpoint(str(path))
    assert loaded_cfg == cfg
    x = torch.zeros(1, cfg.in_ch, 8, 8, 8)
    nodes, edges = loaded(x)
    assert nodes.shape == (1, 3, 8, 8, 8)
    assert edges.shape == (1, 13, 8, 8, 8)


def test_eval_metrics_shapes():
    import numpy as np

    from roomform.eval import evaluate

    rng = np.random.default_rng(0)
    node = rng.random((3, 8, 8, 8)).astype(np.float32)
    edge = rng.random((13, 8, 8, 8)).astype(np.float32)
    ngt = node > 0.6
    egt = edge > 0.6
    obs = rng.random((8, 8, 8)) > 0.5
    out = evaluate(node, edge, ngt, egt, obs)
    for key in ("shell", "occluded_shell", "connectivity", "wall"):
        assert 0.0 <= out[key].f1 <= 1.0
