import { describe, expect, it } from "bun:test";

import { objectIntersectsWall, wallSegmentsFromProgram } from "./wall-contact";

// wall along world y=2, x in [0, 5]; editor frame z = -world y
const WALLS = [{ a: [0, 2] as [number, number], b: [5, 2] as [number, number] }];

describe("objectIntersectsWall", () => {
  it("flags a box straddling the wall", () => {
    expect(objectIntersectsWall([2.5, 0.5, -2], [1, 1, 1], WALLS)).toBe(true);
  });

  it("does not flag flush placement against the wall", () => {
    // box touches the wall exactly (edge at y=2, body on y<2 side)
    expect(objectIntersectsWall([2.5, 0.5, -1.5], [1, 1, 1], WALLS)).toBe(false);
  });

  it("does not flag a distant object", () => {
    expect(objectIntersectsWall([2.5, 0.5, -0.5], [1, 1, 1], WALLS)).toBe(false);
  });

  it("does not flag straddling beyond the segment ends", () => {
    expect(objectIntersectsWall([7, 0.5, -2], [1, 1, 1], WALLS)).toBe(false);
  });

  it("extracts only vertical statements with endpoints", () => {
    const walls = wallSegmentsFromProgram({
      statements: [
        { kind: "ground" },
        { kind: "vertical", from: [0, 0], to: [3, 0] },
        { kind: "vertical" },
      ],
    });
    expect(walls).toEqual([{ a: [0, 0], b: [3, 0] }]);
  });
});
