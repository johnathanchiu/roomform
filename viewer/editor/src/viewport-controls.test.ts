import { describe, expect, test } from "bun:test";

import { cameraRelativeMovement } from "./viewport-controls";

describe("camera-relative object movement", () => {
  test("maps screen directions through the current horizontal camera basis", () => {
    const basis = {
      right: [0, 0, -1] as [number, number, number],
      forward: [-1, 0, 0] as [number, number, number],
    };

    expect(cameraRelativeMovement(basis, [0, 1])).toEqual([-0.1, 0, 0]);
    expect(cameraRelativeMovement(basis, [1, 0])).toEqual([0, 0, -0.1]);
  });
});
