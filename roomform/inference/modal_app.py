"""Modal adapter: remote GPU home for roomform.inference.local.run.

Thin by design — all logic lives in inference/local.py; this file only
declares infrastructure. Requires the `modal` extra and a Modal
account; local inference needs neither.

  modal run -m roomform.inference.modal_app --evidence <npz> --ckpt <pt>
"""

from __future__ import annotations

import modal

app = modal.App("roomform-inference")
volume = modal.Volume.from_name("roomform-inference", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pydantic")
    .add_local_python_source("roomform")
)

VOL = "/vol"


@app.function(image=image, gpu="L4", timeout=1800, volumes={VOL: volume})
def infer(
    evidence_npz: str,
    ckpt: str,
    vox_m: float,
    origin: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> str:
    """Paths are volume-relative; returns the volume-relative output
    npz path. The PatchGraph header is reconstructed client-side."""
    from roomform.contracts import EvidenceGrid
    from roomform.inference.local import run

    evidence = EvidenceGrid(
        npz_path=f"{VOL}/{evidence_npz}",
        vox_m=vox_m,
        origin=origin,
        shape=shape,
        visibility_source="synthesized",
    )
    out = evidence_npz.replace(".npz", ".patchgraph.npz")
    run(evidence, f"{VOL}/{ckpt}", f"{VOL}/{out}", device="cuda")
    volume.commit()
    return out


@app.local_entrypoint()
def main(evidence: str, ckpt: str, vox_m: float = 0.08):
    import numpy as np

    d = np.load(evidence)
    shape = tuple(int(s) for s in d["occ"].shape)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(evidence, f"in/{evidence.split('/')[-1]}")
        batch.put_file(ckpt, f"ckpt/{ckpt.split('/')[-1]}")
    out = infer.remote(
        f"in/{evidence.split('/')[-1]}",
        f"ckpt/{ckpt.split('/')[-1]}",
        vox_m,
        (0.0, 0.0, 0.0),
        shape,
    )
    print(f"patch graph written to volume: {out}")
