"""End-to-end driver: scan -> SceneDocument, parallel + progressive.

The DAG has two independent branches that run concurrently:

  scan ──┬─ evidence -> shell completion ──────────┐
         └─ lifting (Modal, spawned first) ─────────┼─ fuse -> scene.json
                                                    │
        (shell-only partial scene emitted as soon as the shell lands)

Progress streams as NDJSON events on stdout (`stage`, `elapsed_s`,
`artifact`) so a consumer can render incrementally: shell first —
seconds — objects when lifting returns, QA last.

  uv run python -m roomform.pipe.e2e SCAN.ply OUT_DIR \
      [--ckpt model.pt] [--proposals proposals.txt] [--sequential]
"""

from __future__ import annotations

import argparse
import json
import os
import time

from roomform.pipe.evidence.build import build_evidence
from roomform.pipe.fuse import fuse
from roomform.pipe.lifting.spatiallm import lift_from_proposals

T0 = time.time()


def emit(stage: str, **fields) -> None:
    print(
        json.dumps(
            {"stage": stage, "elapsed_s": round(time.time() - T0, 1), **fields}
        ),
        flush=True,
    )


DEFAULT_CKPT = os.path.join(
    "checkpoints", "patch-graph-joint-rgb-55m-offset-head-r2.pt"
)


def _random_ckpt(out_dir: str, vox_m: float) -> str:
    import torch

    from roomform.model.config import ModelConfig

    cfg = ModelConfig(vox_m=vox_m)
    path = os.path.join(out_dir, "RANDOM-INIT.pt")
    torch.save(
        {"model": cfg.build().state_dict(), "config": cfg.model_dump()}, path
    )
    return path


def _shell_branch(args, out_dir: str):
    evidence = build_evidence(
        args.scan, os.path.join(out_dir, "evidence.npz"), args.vox
    )
    emit("evidence.done", grid=list(evidence.shape))
    ckpt = args.ckpt or (
        DEFAULT_CKPT
        if os.path.exists(DEFAULT_CKPT)
        else _random_ckpt(out_dir, args.vox)
    )

    from roomform.inference.local import run as run_shell

    shell = run_shell(evidence, ckpt, os.path.join(out_dir, "patchgraph.npz"))
    emit("shell.done", model=shell.model_id)

    # progressive: shell-only partial scene, before objects arrive
    partial = fuse(
        shell,
        [],
        args.scan,
        evidence.origin,
        os.path.join(out_dir, "scene.partial.json"),
    )
    emit(
        "scene.partial",
        artifact=os.path.join(out_dir, "scene.partial.json"),
        floor_z=partial.floor_z,
    )
    return evidence, shell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("out_dir", nargs="?", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--proposals", default="")
    ap.add_argument(
        "--lifter", choices=("pointlabel", "spatiallm"), default="pointlabel"
    )
    ap.add_argument("--up", choices=("auto", "x", "y", "z"), default="auto")
    ap.add_argument("--vox", type=float, default=0.08)
    ap.add_argument("--sequential", action="store_true")
    args = ap.parse_args()
    scan_stem = os.path.splitext(os.path.basename(args.scan))[0]
    out_dir = args.out_dir or os.path.join("artifacts", scan_stem)
    os.makedirs(out_dir, exist_ok=True)
    emit("start", scan=os.path.basename(args.scan))

    from roomform.pipe.evidence.orient import ensure_z_up

    scan_path = ensure_z_up(args.scan, out_dir, args.up)
    if scan_path != args.scan:
        emit("orient.rotated", scan=os.path.basename(scan_path))
    args.scan = scan_path

    if args.lifter == "pointlabel":
        default_out = os.path.join(out_dir, "labels.npz")
    else:
        default_out = os.path.join(out_dir, "proposals.txt")
    proposals_path = args.proposals or default_out
    lifting_handle = None
    app_ctx = None
    if not args.proposals:
        # spawn lifting FIRST — it only needs the raw scan and is the
        # long pole; it runs on Modal while the shell runs locally
        import modal

        from roomform.inference.modal_adapter import StageApp

        if args.lifter == "pointlabel":
            from roomform.pipe.lifting.pointlabel import app as lifting_app
            from roomform.pipe.lifting.pointlabel import segment as lift
        else:
            from roomform.pipe.lifting.modal_app import app as lifting_app
            from roomform.pipe.lifting.modal_app import lift

        data = StageApp.read_input(args.scan)
        app_ctx = modal.enable_output(), lifting_app.run()
        app_ctx[0].__enter__()
        app_ctx[1].__enter__()
        lifting_handle = lift.spawn(data)
        emit("lifting.spawned")

    try:
        if args.sequential and lifting_handle is not None:
            StageApp.write_output(proposals_path, lifting_handle.get())
            emit("lifting.done", objects_file=proposals_path)
            lifting_handle = None
        evidence, shell = _shell_branch(args, out_dir)

        if lifting_handle is not None:
            StageApp.write_output(proposals_path, lifting_handle.get())
            emit("lifting.done", objects_file=proposals_path)
    finally:
        if app_ctx is not None:
            app_ctx[1].__exit__(None, None, None)
            app_ctx[0].__exit__(None, None, None)

    if proposals_path.endswith(".npz"):
        from roomform.pipe.lifting.pointlabel import lift_from_labels

        objects = lift_from_labels(proposals_path, evidence.origin)
    else:
        objects = lift_from_proposals(proposals_path, evidence.origin)
    doc = fuse(
        shell,
        objects,
        args.scan,
        evidence.origin,
        os.path.join(out_dir, "scene.json"),
    )
    from roomform.export import export_glb

    export_glb(
        doc,
        os.path.join(out_dir, "scene.glb"),
        os.path.join(out_dir, "evidence.npz"),
    )
    emit("glb.done", artifact=os.path.join(out_dir, "scene.glb"))
    flagged = sum(
        1 for o in doc.objects if (o.qa.get("wall_leak_pts") or 0) > 50
    )
    emit(
        "scene.done",
        artifact=os.path.join(out_dir, "scene.json"),
        objects=len(doc.objects),
        wall_leak_flagged=flagged,
        total_s=round(time.time() - T0, 1),
    )


if __name__ == "__main__":
    main()
