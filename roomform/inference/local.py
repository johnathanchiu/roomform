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


def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    raw_cfg = ck.get("config", {})
    if isinstance(raw_cfg, str):
        raw_cfg = json.loads(raw_cfg)
    cfg = ModelConfig(**raw_cfg)
    model = cfg.build()
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def build_input(evidence: EvidenceGrid, cfg: ModelConfig) -> np.ndarray:
    """Slice the stored observable channels to what the checkpoint was
    trained on: first 6 (occ, gray, |nrm| xyz, density), +3 local
    offsets for in_ch=9, +1 zero visibility channel if the model
    expects one (archived scans carry no scanner origins)."""
    d = np.load(evidence.npz_path)
    features = d["features"].astype(np.float32)
    need = cfg.in_ch - (1 if cfg.opening_visibility else 0)
    if features.shape[0] < need:
        raise ValueError(
            f"evidence has {features.shape[0]} channels, model needs {need}"
        )
    x = features[:need]
    if cfg.opening_visibility:
        x = np.concatenate([x, np.zeros_like(x[:1])])
    return x


def run(
    evidence: EvidenceGrid,
    ckpt_path: str,
    out_npz: str,
    device: str = "cpu",
    node_threshold: float = 0.5,
    edge_threshold: float = 0.5,
) -> PatchGraph:
    model, cfg = load_checkpoint(ckpt_path, device)
    x = torch.from_numpy(build_input(evidence, cfg))[None].to(device)
    with torch.no_grad():
        out = model(x)  # model pads to /8 internally
    sx, sy, sz = evidence.shape
    nodes = torch.sigmoid(out[0])[0, :, :sx, :sy, :sz]
    edges = torch.sigmoid(out[1])[0, :, :sx, :sy, :sz]
    arrays = {
        "node_probs": nodes.cpu().numpy().astype(np.float16),
        "edge_probs": edges.cpu().numpy().astype(np.float16),
    }
    if cfg.predict_offsets:
        arrays["offsets"] = (
            out[2][0, :, :sx, :sy, :sz].cpu().numpy().astype(np.float16)
        )
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    return PatchGraph(
        npz_path=out_npz,
        vox_m=evidence.vox_m,
        origin=evidence.origin,
        shape=evidence.shape,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        model_id=os.path.basename(ckpt_path),
    )
