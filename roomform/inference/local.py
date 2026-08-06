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
    """Adapt the stored observable channels to the checkpoint.

    Evidence stores the RGB superset (occ, r, g, b, |nrm| xyz,
    density, offsets = 11ch; older grids may hold the 9ch grayscale
    layout). Grayscale checkpoints get BT.709 luma; a zero visibility
    channel is appended when the model expects one."""
    d = np.load(evidence.npz_path)
    features = d["features"].astype(np.float32)
    stored_rgb = features.shape[0] == 11
    if cfg.color_mode == "rgb":
        if not stored_rgb:
            raise ValueError(
                "rgb checkpoint needs rgb evidence — rebuild evidence.npz"
            )
        x = features
    elif stored_rgb:
        luma = (
            0.2126 * features[1] + 0.7152 * features[2] + 0.0722 * features[3]
        )
        x = np.concatenate([features[:1], luma[None], features[4:]])
    else:
        x = features
    need = cfg.in_ch - (1 if cfg.opening_visibility else 0)
    if x.shape[0] < need:
        raise ValueError(
            f"evidence provides {x.shape[0]} channels, model needs {need}"
        )
    x = x[:need]
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
    if cfg.predict_openings:
        k = 3 if cfg.predict_offsets else 2
        arrays["openings"] = (
            torch.sigmoid(out[k])[0, :, :sx, :sy, :sz]
            .cpu()
            .numpy()
            .astype(np.float16)
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
