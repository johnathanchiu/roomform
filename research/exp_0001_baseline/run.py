"""Baseline patch-graph training — the worked example every experiment
copies. Config overlays + package imports; no local model code."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from research.train.config import TrainConfig
from research.train.loop import train


def main(data_dir: str, out_dir: str) -> None:
    with open("config.yaml") as f:
        cfg = TrainConfig(**yaml.safe_load(f))
    raise NotImplementedError("wire research/datagen shard loader here")
    train_data = val_data = None
    result = train(cfg, train_data, val_data, out_dir)
    print("final:", result)


if __name__ == "__main__":
    main(*sys.argv[1:3])
