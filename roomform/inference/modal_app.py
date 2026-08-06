"""Modal adapter: remote GPU home for roomform.inference.local.run.

Thin by design — all logic lives in inference/local.py; conventions
(app naming, volume, GPU defaults) come from roomform.modal_adapter.
Requires the `modal` extra and a Modal account; local inference needs
neither.

  modal run -m roomform.inference.modal_app --evidence <npz> --ckpt <pt>
"""

from __future__ import annotations

import numpy as np

from roomform.contracts import EvidenceGrid
from roomform.inference.modal_adapter import StageApp, torch_image

stage = StageApp("completion")
app = stage.app  # `modal run` discovers this name

image = torch_image(with_roomform=True)


@stage.gpu(image=image, timeout=1800, with_volume=True)
def infer(
    evidence_npz: str,
    ckpt: str,
    vox_m: float,
    origin: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> str:
    """Paths are volume-relative; returns the volume-relative output
    npz path. The PatchGraph header is reconstructed client-side."""
    from roomform.inference.local import run

    evidence = EvidenceGrid(
        npz_path=f"{stage.VOL}/{evidence_npz}",
        vox_m=vox_m,
        origin=origin,
        shape=shape,
        visibility_source="synthesized",
    )
    out = evidence_npz.replace(".npz", ".patchgraph.npz")
    run(evidence, f"{stage.VOL}/{ckpt}", f"{stage.VOL}/{out}", device="cuda")
    stage.volume.commit()
    return out


@stage.entrypoint()
def main(evidence: str, ckpt: str, vox_m: float = 0.08):
    d = np.load(evidence)
    shape = tuple(int(s) for s in d["occ"].shape)
    with stage.volume.batch_upload(force=True) as batch:
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
