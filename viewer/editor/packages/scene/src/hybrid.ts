import { z } from "zod";

import { quaternionSchema, vector3Schema } from "./index";

export const artifactRefSchema = z.object({
  uri: z.string().min(1),
  mediaType: z.string().min(1),
  sha256: z.string().optional(),
});

export const appearanceLayerSchema = z.object({
  kind: z.enum(["mesh", "splat"]),
  artifact: artifactRefSchema,
});

export const repairPatchSchema = z.object({
  id: z.string().min(1),
  status: z.enum(["queued", "running", "ready", "failed"]),
  strategy: z.enum(["localized-atlas", "full-shell"]).default("localized-atlas"),
  appearance: appearanceLayerSchema.optional(),
  collisionMesh: artifactRefSchema.optional(),
  surfacePlan: artifactRefSchema.optional(),
  observedFraction: z.number().min(0).max(1).optional(),
  inferredFraction: z.number().min(0).max(1).optional(),
  sourceFrameIds: z.array(z.string()).default([]),
});

export const hybridEntitySchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  status: z.enum(["candidate", "lifting", "ready", "failed"]),
  position: vector3Schema,
  rotation: quaternionSchema,
  bounds: vector3Schema,
  gaussianIds: artifactRefSchema.optional(),
  capturedAppearance: appearanceLayerSchema.optional(),
  visualMesh: artifactRefSchema.optional(),
  collisionMesh: artifactRefSchema.optional(),
  repairPatchId: z.string().optional(),
});

export const hybridSceneSchema = z.object({
  schema: z.literal("roomform.hybrid-scene.v1"),
  sceneId: z.string().min(1),
  coordinateSystem: z.literal("source-capture"),
  background: z.object({
    appearance: appearanceLayerSchema,
    collisionMesh: artifactRefSchema,
    sourceCameras: artifactRefSchema,
  }),
  entities: z.array(hybridEntitySchema),
  repairPatches: z.array(repairPatchSchema),
});

export const editJobSchema = z.object({
  schema: z.literal("roomform.edit-job.v1"),
  id: z.string().min(1),
  sceneId: z.string().min(1),
  prompt: z.object({
    frameId: z.string().min(1),
    point: z.tuple([z.number(), z.number()]),
    label: z.string().optional(),
  }),
  stage: z.enum([
    "mask-preview",
    "lifting",
    "revealing-observed-background",
    "repairing",
    "ready",
    "failed",
  ]),
  progress: z.number().min(0).max(1),
  entityId: z.string().optional(),
  observedBackgroundFraction: z.number().min(0).max(1).optional(),
  message: z.string(),
});

export type HybridScene = z.infer<typeof hybridSceneSchema>;
export type HybridEntity = z.infer<typeof hybridEntitySchema>;
export type RepairPatch = z.infer<typeof repairPatchSchema>;
export type EditJob = z.infer<typeof editJobSchema>;

export function repairPatchForEntity(
  scene: HybridScene,
  entityId: string,
): RepairPatch | undefined {
  const patchId = scene.entities.find((entity) => entity.id === entityId)?.repairPatchId;
  return scene.repairPatches.find((patch) => patch.id === patchId);
}

export function canMoveEntity(scene: HybridScene, entityId: string): boolean {
  const entity = scene.entities.find((candidate) => candidate.id === entityId);
  if (entity?.status !== "ready") return false;
  return repairPatchForEntity(scene, entityId)?.status === "ready";
}
