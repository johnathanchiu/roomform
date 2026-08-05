"""Baseline patch-graph training — the worked example every experiment
copies. Config overlays + package imports; no local model code."""

from __future__ import annotations

import sys

import yaml

from roomform.train.config import TrainConfig
from roomform.train.loop import train


def main(data_dir: str, out_dir: str) -> None:
    with open("config.yaml") as f:
        cfg = TrainConfig(**yaml.safe_load(f))
    train_data, val_data = load_shards(data_dir)  # research/datagen loader
    result = train(cfg, train_data, val_data, out_dir)
    print("final:", result)


if __name__ == "__main__":
    main(*sys.argv[1:3])
