"""Stage 2 — object_lifting: scan -> oriented object boxes.

Backend: SpatialLM detections. v0 consumes an existing proposals file
(``Bbox(cls,cx,cy,cz,heading,dx,dy,dz)`` lines); the live-invocation
backend slots in behind the same function once the runner is ported.
Output boxes are re-aligned into the grid frame via frame_shift.
"""
from __future__ import annotations

import re

from roomform.contracts import SceneObject

BBOX_RE = re.compile(
    r"Bbox\(([a-z_]+),([-\d.e]+),([-\d.e]+),([-\d.e]+),([-\d.e]+),"
    r"([-\d.e]+),([-\d.e]+),([-\d.e]+)\)")


def lift_from_proposals(path: str,
                        frame_shift: tuple[float, float, float]
                        ) -> list[SceneObject]:
    """Parse SpatialLM proposal lines into grid-frame SceneObjects."""
    out = []
    for line in open(path):
        m = BBOX_RE.search(line)
        if not m:
            continue
        v = [float(x) for x in m.groups()[1:]]
        center = tuple(c - s for c, s in zip(v[0:3], frame_shift))
        out.append(SceneObject(cls=m.group(1), center=center,
                               heading=v[3], size=tuple(v[4:7]),
                               source="spatiallm"))
    return out


if __name__ == "__main__":
    import sys
    objs = lift_from_proposals(sys.argv[1], (0.0, 0.0, 0.0))
    assert objs and all(o.size[0] > 0 for o in objs)
    print(f"object_lifting self-check OK: {len(objs)} objects, "
          f"classes {sorted({o.cls for o in objs})[:5]}...")
