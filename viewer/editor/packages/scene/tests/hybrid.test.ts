import { describe, expect, test } from "bun:test";

import { canMoveEntity, hybridSceneSchema, repairPatchForEntity } from "../src/hybrid";

const artifact = { uri: "https://example.com/asset.ply", mediaType: "application/ply" };

describe("hybrid scene", () => {
  test("requires both a lifted entity and completed repair before movement", () => {
    const scene = hybridSceneSchema.parse({
      schema: "roomform.hybrid-scene.v1",
      sceneId: "room-1",
      coordinateSystem: "source-capture",
      background: {
        appearance: { kind: "splat", artifact },
        collisionMesh: artifact,
        sourceCameras: { ...artifact, mediaType: "application/json" },
      },
      entities: [{
        id: "sofa-1",
        label: "sofa",
        status: "ready",
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        bounds: [2, 1, 1],
        repairPatchId: "repair-sofa-1",
      }],
      repairPatches: [{ id: "repair-sofa-1", status: "running" }],
    });

    expect(canMoveEntity(scene, "sofa-1")).toBe(false);
    const patch = repairPatchForEntity(scene, "sofa-1");
    if (!patch) throw new Error("missing patch");
    patch.status = "ready";
    expect(canMoveEntity(scene, "sofa-1")).toBe(true);
  });
});
