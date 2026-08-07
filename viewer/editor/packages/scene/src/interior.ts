import { z } from "zod";

import { artifactRefSchema } from "./hybrid";
import { transformSchema } from "./index";

export const entityObservationSchema = z.object({
  frameId: z.string().min(1),
  mask: artifactRefSchema.optional(),
  points: artifactRefSchema.optional(),
  confidence: z.number().min(0).max(1),
});

export const entityTrackSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  mobility: z.enum(["movable", "fixed", "unknown"]),
  status: z.enum(["candidate", "confirmed", "rejected"]),
  poseState: z.enum(["in-place", "moved", "removed"]),
  capturedTransform: transformSchema,
  currentTransform: transformSchema,
  observations: z.array(entityObservationSchema).default([]),
  visualGeometry: artifactRefSchema.optional(),
  collisionGeometry: artifactRefSchema.optional(),
  confidence: z.number().min(0).max(1),
}).superRefine((entity, context) => {
  if (
    entity.poseState === "in-place" &&
    JSON.stringify(entity.currentTransform) !== JSON.stringify(entity.capturedTransform)
  ) {
    context.addIssue({
      code: "custom",
      path: ["currentTransform"],
      message: "in-place entity must retain its captured transform",
    });
  }
});

export const evidenceFieldSchema = z.object({
  occupied: artifactRefSchema,
  knownFree: artifactRefSchema,
  confidence: artifactRefSchema,
  format: z.enum(["tsdf", "voxel-grid", "octree"]),
  voxelSizeMeters: z.number().positive(),
});

export const structuralCollisionModelSchema = z.object({
  status: z.enum(["building", "ready", "needs-review"]),
  collisionMesh: artifactRefSchema.optional(),
  surfaceGraph: artifactRefSchema.optional(),
  evidence: evidenceFieldSchema,
  excludedEntityIds: z.array(z.string().min(1)).default([]),
  confidence: z.number().min(0).max(1),
  revision: z.number().int().nonnegative().default(0),
});

export const interiorCollisionSceneSchema = z.object({
  schema: z.literal("roomform.interior-collision.v1"),
  sceneId: z.string().min(1),
  captureId: z.string().min(1),
  structural: structuralCollisionModelSchema,
  entities: z.array(entityTrackSchema),
}).superRefine((scene, context) => {
  const entityIds = new Set<string>();
  const rejectedIds = new Set<string>();
  scene.entities.forEach((entity, index) => {
    if (entityIds.has(entity.id)) {
      context.addIssue({
        code: "custom",
        path: ["entities", index, "id"],
        message: `duplicate entity track: ${entity.id}`,
      });
    }
    entityIds.add(entity.id);
    if (entity.status === "rejected") rejectedIds.add(entity.id);
  });

  scene.structural.excludedEntityIds.forEach((id, index) => {
    if (!entityIds.has(id)) {
      context.addIssue({
        code: "custom",
        path: ["structural", "excludedEntityIds", index],
        message: `structural model excludes missing entity: ${id}`,
      });
    }
    if (rejectedIds.has(id)) {
      context.addIssue({
        code: "custom",
        path: ["structural", "excludedEntityIds", index],
        message: `structural model excludes rejected entity: ${id}`,
      });
    }
  });
});

export type EntityTrack = z.infer<typeof entityTrackSchema>;
export type InteriorCollisionScene = z.infer<typeof interiorCollisionSceneSchema>;
