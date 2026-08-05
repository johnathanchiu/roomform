"""Local inference: EvidenceGrid -> PatchGraph, on whatever device is
present. The Modal wrapper (inference/modal_app.py) is a thin remote
shell around exactly this function — one implementation, two homes.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from roomform.contracts import EvidenceGrid, PatchGraph
from roomform.model.config import ModelConfig
from roomform.model.convformer import PatchGraphConvFormer


def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = (
        ModelConfig(**json.loads(ck["config"]))
        if "config" in ck
        else ModelConfig()
    )
    model = PatchGraphConvFormer(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def build_input(evidence: EvidenceGrid) -> np.ndarray:
    """Stack the observable channels the contract defines, in order:
    occ, gray, |nrm| xyz, log_density. Missing optional channels are
    zero-filled (pair with a model trained with channel dropout)."""
    d = np.load(evidence.npz_path)
    x, y, z = evidence.shape
    occ = (d["occ"] > 0).astype(np.float32)
    gray = (
        d["gray"].astype(np.float32) / 255.0
        if "gray" in d.files
        else np.zeros_like(occ)
    )
    nrm = (
        np.abs(d["nrm_abs"].astype(np.float32))
        if "nrm_abs" in d.files
        else np.zeros((3, x, y, z), np.float32)
    )
    dens = (
        d["log_density"].astype(np.float32)
        if "log_density" in d.files
        else np.zeros_like(occ)
    )
    return np.concatenate([occ[None], gray[None], nrm, dens[None]])


def run(
    evidence: EvidenceGrid,
    ckpt_path: str,
    out_npz: str,
    device: str = "cpu",
    node_threshold: float = 0.5,
    edge_threshold: float = 0.5,
) -> PatchGraph:
    model, _cfg = load_checkpoint(ckpt_path, device)
    x = torch.from_numpy(build_input(evidence))[None].to(device)
    pad = [
        (cfg_pool - s % cfg_pool) % cfg_pool
        for s, cfg_pool in zip(x.shape[2:], [model.cfg.attn_pool] * 3)
    ]
    if any(pad):
        x = torch.nn.functional.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
    with torch.no_grad():
        node_logits, edge_logits = model(x)
    sx, sy, sz = evidence.shape
    nodes = torch.sigmoid(node_logits)[0, :, :sx, :sy, :sz]
    edges = torch.sigmoid(edge_logits)[0, :, :sx, :sy, :sz]
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz,
        node_probs=nodes.cpu().numpy().astype(np.float16),
        edge_probs=edges.cpu().numpy().astype(np.float16),
    )
    return PatchGraph(
        npz_path=out_npz,
        vox_m=evidence.vox_m,
        origin=evidence.origin,
        shape=evidence.shape,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        model_id=os.path.basename(ckpt_path),
    )
