import { z } from "zod";

import { artifactRefSchema } from "./hybrid";

const matrix3Schema = z.tuple([
  z.tuple([z.number(), z.number(), z.number()]),
  z.tuple([z.number(), z.number(), z.number()]),
  z.tuple([z.number(), z.number(), z.number()]),
]);

const matrix4Schema = z.tuple([
  z.tuple([z.number(), z.number(), z.number(), z.number()]),
  z.tuple([z.number(), z.number(), z.number(), z.number()]),
  z.tuple([z.number(), z.number(), z.number(), z.number()]),
  z.tuple([z.number(), z.number(), z.number(), z.number()]),
]);

export const cameraCalibrationSchema = z.object({
  imageSize: z.tuple([z.number().int().positive(), z.number().int().positive()]),
  intrinsics: matrix3Schema,
  distortion: z.array(z.number()).optional(),
});

export const captureFrameSchema = z.object({
  id: z.string().min(1),
  frameIndex: z.number().int().nonnegative(),
  timestampSeconds: z.number().nonnegative(),
  cameraToWorld: matrix4Schema,
  calibrationId: z.string().min(1),
  depth: artifactRefSchema.optional(),
  depthTimestampSeconds: z.number().nonnegative().optional(),
  depthConfidence: artifactRefSchema.optional(),
  trackingState: z.enum(["normal", "limited", "unavailable"]).default("normal"),
}).superRefine((frame, context) => {
  if (!frame.depth && (frame.depthTimestampSeconds !== undefined || frame.depthConfidence)) {
    context.addIssue({
      code: "custom",
      message: "depth timestamp/confidence require a depth artifact",
    });
  }
});

export const sensorRichCaptureSchema = z.object({
  schema: z.literal("roomform.capture.v1"),
  captureId: z.string().min(1),
  units: z.literal("meters"),
  coordinateSystem: z.literal("right-handed-y-up"),
  gravity: z.tuple([z.number(), z.number(), z.number()]),
  rgbVideo: artifactRefSchema,
  calibrations: z.record(z.string().min(1), cameraCalibrationSchema),
  frames: z.array(captureFrameSchema).min(1),
  arkitMesh: artifactRefSchema.optional(),
  arkitMeshClassification: artifactRefSchema.optional(),
  imu: artifactRefSchema.optional(),
  roomplan: artifactRefSchema.optional(),
}).superRefine((capture, context) => {
  let previousFrameIndex = -1;
  let previousTimestamp = -1;
  const indices = new Set<number>();

  capture.frames.forEach((frame, index) => {
    if (!capture.calibrations[frame.calibrationId]) {
      context.addIssue({
        code: "custom",
        path: ["frames", index, "calibrationId"],
        message: `missing calibration: ${frame.calibrationId}`,
      });
    }
    if (indices.has(frame.frameIndex) || frame.frameIndex <= previousFrameIndex) {
      context.addIssue({
        code: "custom",
        path: ["frames", index, "frameIndex"],
        message: "frame indices must be unique and increasing",
      });
    }
    if (frame.timestampSeconds < previousTimestamp) {
      context.addIssue({
        code: "custom",
        path: ["frames", index, "timestampSeconds"],
        message: "frame timestamps must be increasing",
      });
    }
    indices.add(frame.frameIndex);
    previousFrameIndex = frame.frameIndex;
    previousTimestamp = frame.timestampSeconds;
  });
});

export type SensorRichCapture = z.infer<typeof sensorRichCaptureSchema>;
export type CaptureFrame = z.infer<typeof captureFrameSchema>;
