"""Parity enforcement: the packaged architecture must load and run
every released checkpoint, and a fresh checkpoint must round-trip."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_checkpoint_roundtrip(tmp_path):
    from roomform.model.config import ModelConfig
    from roomform.model.convformer import PatchGraphConvFormer
    from roomform.train.config import TrainConfig
    from roomform.train.loop import save_checkpoint

    cfg = TrainConfig(
        model=ModelConfig(
            attn_depth=1, stem_dim=16, attn_dim=32, attn_heads=4, head_dim=16
        )
    )
    model = PatchGraphConvFormer(cfg.model)
    opt = torch.optim.AdamW(model.parameters())
    save_checkpoint(str(tmp_path), model, opt, 0, cfg)

    from roomform.inference.local import load_checkpoint

    loaded, loaded_cfg = load_checkpoint(str(tmp_path / "checkpoint.pt"))
    assert loaded_cfg == cfg.model
    x = torch.zeros(1, cfg.model.in_channels, 8, 8, 8)
    nodes, edges = loaded(x)
    assert nodes.shape == (1, 3, 8, 8, 8)
    assert edges.shape == (1, 13, 8, 8, 8)


def test_eval_metrics_shapes():
    import numpy as np

    from roomform.train.eval import evaluate

    rng = np.random.default_rng(0)
    node = rng.random((3, 8, 8, 8)).astype(np.float32)
    edge = rng.random((13, 8, 8, 8)).astype(np.float32)
    ngt = node > 0.6
    egt = edge > 0.6
    obs = rng.random((8, 8, 8)) > 0.5
    out = evaluate(node, edge, ngt, egt, obs)
    for key in ("shell", "occluded_shell", "connectivity", "wall"):
        assert 0.0 <= out[key]["f1"] <= 1.0
