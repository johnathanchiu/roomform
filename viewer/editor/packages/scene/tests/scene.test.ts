import { describe, expect, test } from "bun:test";

import { demoRoom } from "../src/fixture";
import { parseRoomformScene } from "../src/index";
import {
  objectizationRequestSchema,
  splatAnalysisResultSchema,
} from "../src/selection";

describe("roomform scene contract", () => {
  test("accepts the canonical demo room", () => {
    expect(demoRoom.schema).toBe("roomform.scene.v1");
    expect(demoRoom.shell.walls).toHaveLength(4);
    expect(demoRoom.objects).toHaveLength(5);
    expect(demoRoom.objects.filter((object) => object.asset.kind === "glb")).toHaveLength(3);
  });

  test("rejects a shell without a room polygon", () => {
    expect(() =>
      parseRoomformScene({
        ...demoRoom,
        shell: { ...demoRoom.shell, floor: [] },
      }),
    ).toThrow();
  });

  test("represents a click as transient objectization input", () => {
    const request = objectizationRequestSchema.parse({
      schema: "roomform.objectization.request.v1",
      sceneId: demoRoom.id,
      referenceUri: "capture://demo-room",
      seed: {
        referenceKind: "splat",
        worldPoint: [0, 0.8, 1.7],
        screenPoint: [0.48, 0.56],
      },
    });

    expect(request.seed.referenceKind).toBe("splat");
  });

  test("accepts object proposals in the source splat coordinate frame", () => {
    const result = splatAnalysisResultSchema.parse({
      schema: "roomform.splat-analysis.result.v1",
      scene_id: "bedroom",
      splat_uri: "https://example.com/bedroom.spz",
      detector: "splat-analyzer/owlv2",
      coordinate_system: "source-splat",
      objects: [
        {
          id: "proposal-bed-1",
          label: "bed",
          category: "bed",
          position: [0, 0, 1],
          rotation_xyzw: [0, 0, 0, 1],
          bounds: [2, 1, 2.2],
          confidence: 0.82,
          source_views: [],
        },
      ],
    });

    expect(result.objects[0]?.label).toBe("bed");
    expect(result.coordinate_system).toBe("source-splat");
  });

  test("accepts measured object meshes without reducing them to boxes", () => {
    const result = splatAnalysisResultSchema.parse({
      schema: "roomform.splat-analysis.result.v1",
      scene_id: "measured-bedroom",
      splat_uri: "/capture.ply",
      detector: "sam3/fal",
      coordinate_system: "source-splat",
      objects: [{
        id: "bed-0",
        label: "bed",
        category: "bed",
        position: [0, 0, 0],
        rotation_xyzw: [0, 0, 0, 1],
        bounds: [2, 1, 1],
        confidence: 0.9,
        method: "sam3-multiview",
        measured_mesh_uri: "/objects/bed-0.glb",
        source_views: [],
      }],
    });

    expect(result.objects[0]?.measured_mesh_uri).toBe("/objects/bed-0.glb");
  });
});
