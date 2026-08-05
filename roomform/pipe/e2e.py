"""End-to-end driver: scan -> EvidenceGrid -> PatchGraph -> objects ->
SceneDocument, with per-stage wall-clock timings.

  uv run python -m roomform.pipe.e2e SCAN.ply OUT_DIR \
      [--ckpt model.pt] [--proposals proposals.txt]

Without --ckpt, a random-init ConvFormer is used — the shell output is
NOISE (clearly marked in the scene doc's model_id); useful only for
plumbing and timing until trained weights land. Without --proposals,
lifting runs live on Modal (requires the modal extra + account).
"""

from __future__ import annotations

import argparse
import json
import os
import time

from roomform.pipe.evidence.build import build_evidence
from roomform.pipe.fuse import fuse
from roomform.pipe.lifting.spatiallm import lift_from_proposals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("out_dir")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--proposals", default="")
    ap.add_argument("--vox", type=float, default=0.08)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    timings: dict[str, float] = {}

    t0 = time.time()
    evidence = build_evidence(
        args.scan, os.path.join(args.out_dir, "evidence.npz"), args.vox
    )
    timings["evidence_s"] = round(time.time() - t0, 1)

    ckpt = args.ckpt
    if not ckpt:
        import torch

        from roomform.model.config import ModelConfig
        from roomform.model.convformer import PatchGraphConvFormer

        cfg = ModelConfig(vox_m=args.vox)
        ckpt = os.path.join(args.out_dir, "RANDOM-INIT.pt")
        torch.save(
            {
                "model": PatchGraphConvFormer(cfg).state_dict(),
                "config": cfg.model_dump_json(),
            },
            ckpt,
        )

    t0 = time.time()
    from roomform.inference.local import run as run_shell

    shell = run_shell(
        evidence, ckpt, os.path.join(args.out_dir, "patchgraph.npz")
    )
    timings["shell_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    if args.proposals:
        proposals_path = args.proposals
    else:
        from roomform.inference.modal_adapter import StageApp
        from roomform.pipe.lifting.modal_app import lift

        stage = StageApp("lifting")
        proposals_path = os.path.join(args.out_dir, "proposals.txt")
        with __import__("modal").enable_output():
            stage.run_file(lift, args.scan, proposals_path)
    frame_shift = evidence.origin
    objects = lift_from_proposals(proposals_path, frame_shift)
    timings["lifting_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    doc = fuse(
        shell,
        objects,
        args.scan,
        frame_shift,
        os.path.join(args.out_dir, "scene.json"),
    )
    timings["fuse_s"] = round(time.time() - t0, 1)
    timings["total_s"] = round(sum(timings.values()), 1)

    flagged = sum(
        1 for o in doc.objects if (o.qa.get("wall_leak_pts") or 0) > 50
    )
    print(
        json.dumps(
            {
                "timings": timings,
                "grid": evidence.shape,
                "objects": len(doc.objects),
                "wall_leak_flagged": flagged,
                "model": shell.model_id,
                "scene": os.path.join(args.out_dir, "scene.json"),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
