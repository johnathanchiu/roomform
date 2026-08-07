import { describe, expect, test } from "bun:test";
import { BufferGeometry, Float32BufferAttribute } from "three";

import type { SplatObjectProposal } from "@roomform/scene/selection";

import { splitGeometry, splitGeometryByProposals } from "./mesh-selection";

const proposal: SplatObjectProposal = {
  id: "chair",
  label: "chair",
  category: "chair",
  position: [0, 0, 0],
  rotation_xyzw: [0, 0, 0, 1],
  bounds: [2, 2, 2],
  confidence: 1,
  method: "depth-heuristic",
  source_views: [],
};

describe("mesh object extraction", () => {
  test("keeps a large background triangle when only its centroid crosses the box", () => {
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new Float32BufferAttribute([
      -4, 0, -4,
      4, 0, -4,
      0, 0, 4,
      -0.25, 0, -0.25,
      0.25, 0, -0.25,
      0, 0.5, 0.25,
    ], 3));

    const split = splitGeometry(geometry, proposal);

    expect(split.background.getIndex()?.count).toBe(3);
    expect(split.object?.getIndex()?.count).toBe(3);
  });
});

test("partitions multiple objects into persistent independent layers", () => {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute([
    -1.1, 0, 0, -0.9, 0, 0, -1, 0.2, 0,
    0.9, 0, 0, 1.1, 0, 0, 1, 0.2, 0,
    4, 0, 0, 5, 0, 0, 4, 1, 0,
  ], 3));
  const proposal = (id: string, x: number) => ({
    id, label: id, category: id, position: [x, 0, 0] as [number, number, number],
    rotation_xyzw: [0, 0, 0, 1] as [number, number, number, number],
    bounds: [1, 1, 1] as [number, number, number], confidence: 1,
    method: "depth-heuristic" as const, source_views: [],
  });

  const split = splitGeometryByProposals(geometry, [proposal("left", -1), proposal("right", 1)]);

  expect(split.objects.get("left")?.getIndex()?.count).toBe(3);
  expect(split.objects.get("right")?.getIndex()?.count).toBe(3);
  expect(split.background.getIndex()?.count).toBe(3);
});
