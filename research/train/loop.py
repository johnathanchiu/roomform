"""Training loop for the patch-graph model.

Deliberately plain: AdamW + cosine, checkpoint (model + optimizer +
epoch + the full TrainConfig as JSON) every val, atomic writes. The
dataset interface is any iterable yielding dicts with keys
``input`` [C,X,Y,Z] f32, ``node_gt`` [3,X,Y,Z] bool,
``edge_gt`` [13,X,Y,Z] bool, ``observed`` [X,Y,Z] bool —
research/datagen produces these; the loop doesn't care from where.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable

import numpy as np
import torch

from research.train.config import TrainConfig
from research.train.losses import total_loss
from roomform.eval import evaluate


def _log(path: str, rec: dict) -> None:
    rec["t"] = round(time.time(), 1)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def save_checkpoint(
    out_dir: str, model, opt, epoch: int, cfg: TrainConfig
) -> str:
    path = os.path.join(out_dir, "checkpoint.pt")
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "config": cfg.model.model_dump(),
            "train_config": cfg.model_dump_json(),
        },
        tmp,
    )
    os.replace(tmp, path)
    return path


def train(
    cfg: TrainConfig,
    train_data: Iterable[dict],
    val_data: list[dict],
    out_dir: str,
    device: str = "cuda",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    torch.manual_seed(cfg.seed)

    model = cfg.model.build().to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * cfg.lr_min_factor
    )

    last_val: dict = {}
    for epoch in range(cfg.epochs):
        model.train()
        t0, losses = time.time(), []
        for item in train_data:
            x = torch.as_tensor(item["input"])[None].to(device)
            ngt = torch.as_tensor(item["node_gt"])[None].to(device)
            egt = torch.as_tensor(item["edge_gt"])[None].to(device)
            node_logits, edge_logits = model(x)[:2]
            loss, _parts = total_loss(
                node_logits,
                edge_logits,
                ngt,
                egt,
                node_pos_weight=cfg.node_pos_weight,
                edge_pos_weight=cfg.edge_pos_weight,
                edge_weight=cfg.edge_loss_weight,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss))
        sched.step()
        _log(
            log_path,
            {
                "event": "epoch",
                "epoch": epoch,
                "loss": round(float(np.mean(losses)), 4),
                "sec": round(time.time() - t0, 1),
            },
        )

        if epoch % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            reports = []
            with torch.no_grad():
                for item in val_data:
                    x = torch.as_tensor(item["input"])[None].to(device)
                    node_logits, edge_logits = model(x)[:2]
                    reports.append(
                        evaluate(
                            torch.sigmoid(node_logits)[0].cpu().numpy(),
                            torch.sigmoid(edge_logits)[0].cpu().numpy(),
                            np.asarray(item["node_gt"], bool),
                            np.asarray(item["edge_gt"], bool),
                            np.asarray(item["observed"], bool),
                        )
                    )
            last_val = {
                key: round(float(np.mean([r[key]["f1"] for r in reports])), 4)
                for key in ("shell", "occluded_shell", "connectivity")
            }
            _log(log_path, {"event": "val", "epoch": epoch, **last_val})
            save_checkpoint(out_dir, model, opt, epoch, cfg)
    return last_val
