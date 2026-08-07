import { describe, expect, test } from "bun:test";

import { sensorRichCaptureSchema } from "../src/capture";
import { interiorCollisionSceneSchema } from "../src/interior";

const artifact = (uri: string, mediaType = "application/octet-stream") => ({
  uri,
  mediaType,
});

const identityTransform = {
  position: [1, 0.4, -0.5],
  rotation: [0, 0, 0, 1],
  scale: [1, 1, 1],
};

describe("sensor-rich capture", () => {
  test("validates frame calibration and depth sidecars", () => {
    const capture = sensorRichCaptureSchema.parse({
      schema: "roomform.capture.v1",
      captureId: "apartment-1",
      units: "meters",
      coordinateSystem: "right-handed-y-up",
      gravity: [0, -1, 0],
      rgbVideo: artifact("rgb.mov", "video/quicktime"),
      calibrations: {
        wide: {
          imageSize: [1920, 1440],
          intrinsics: [[1200, 0, 960], [0, 1200, 720], [0, 0, 1]],
        },
      },
      frames: [{
        id: "frame-0",
        frameIndex: 0,
        timestampSeconds: 0,
        cameraToWorld: [
          [1, 0, 0, 0],
          [0, 1, 0, 1.4],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
        ],
        calibrationId: "wide",
        depth: artifact("depth/0.exr", "image/x-exr"),
        depthTimestampSeconds: 0,
        depthConfidence: artifact("confidence/0.png", "image/png"),
      }],
    });

    expect(capture.frames[0]?.trackingState).toBe("normal");
  });
});

describe("interior collision scene", () => {
  test("keeps a movable entity tracked while it remains in place", () => {
    const scene = interiorCollisionSceneSchema.parse({
      schema: "roomform.interior-collision.v1",
      sceneId: "scene-1",
      captureId: "apartment-1",
      structural: {
        status: "building",
        evidence: {
          occupied: artifact("occupied.vdb"),
          knownFree: artifact("free.vdb"),
          confidence: artifact("confidence.vdb"),
          format: "tsdf",
          voxelSizeMeters: 0.025,
        },
        excludedEntityIds: ["sofa-1"],
        confidence: 0.65,
      },
      entities: [{
        id: "sofa-1",
        label: "sofa",
        mobility: "movable",
        status: "confirmed",
        poseState: "in-place",
        capturedTransform: identityTransform,
        currentTransform: identityTransform,
        confidence: 0.92,
      }],
    });

    expect(scene.entities[0]?.mobility).toBe("movable");
    expect(scene.entities[0]?.poseState).toBe("in-place");
    expect(scene.structural.excludedEntityIds).toEqual(["sofa-1"]);
  });
});
