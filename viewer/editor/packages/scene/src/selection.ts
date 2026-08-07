import { z } from "zod";

import {
  roomObjectSchema,
  transformSchema,
  vector3Schema,
} from "./index";

export const selectionSeedSchema = z.object({
  referenceKind: z.enum(["mesh", "splat"]),
  worldPoint: vector3Schema,
  worldNormal: vector3Schema.optional(),
  cameraId: z.string().optional(),
  screenPoint: z.tuple([z.number(), z.number()]).optional(),
});

export const objectizationRequestSchema = z.object({
  schema: z.literal("roomform.objectization.request.v1"),
  sceneId: z.string().min(1),
  referenceUri: z.string().min(1),
  seed: selectionSeedSchema,
});

export const objectizationProposalSchema = z.object({
  schema: z.literal("roomform.objectization.proposal.v1"),
  proposalId: z.string().min(1),
  sceneId: z.string().min(1),
  label: z.string().min(1),
  category: z.string().min(1),
  transform: transformSchema,
  bounds: vector3Schema,
  confidence: z.number().min(0).max(1),
  temporaryMaskUri: z.string().min(1),
  sourceViewIds: z.array(z.string()).min(1),
});

export const objectizationCommitSchema = z.object({
  schema: z.literal("roomform.objectization.commit.v1"),
  proposalId: z.string().min(1),
  object: roomObjectSchema,
  background: z.object({
    strategy: z.enum(["shell", "inpaint", "completion"]),
    patchUri: z.string().min(1).optional(),
  }),
});

export const sourceViewDetectionSchema = z.object({
  frame_id: z.string(),
  box_xyxy: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  score: z.number().min(0).max(1),
});

export const splatObjectProposalSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  category: z.string().min(1),
  position: vector3Schema,
  rotation_xyzw: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  bounds: vector3Schema,
  confidence: z.number().min(0).max(1),
  method: z.enum([
    "depth-heuristic",
    "boxernet",
    "spatiallm",
    "sam3-multiview",
    "oracle-gt-labels",
    "mask3d-pointcloud",
    "oracle-gt-labels+poisson-cleanup",
    "mask3d-pointcloud+poisson-cleanup",
  ]).default("depth-heuristic"),
  measured_mesh_uri: z.string().min(1).optional(),
  measured_mesh_z_up: z.boolean().optional(),
  observed_points_uri: z.string().min(1).optional(),
  observed_points_z_up: z.boolean().optional(),
  attachment: z.enum(["floor", "wall", "ceiling", "free"]).optional(),
  source_views: z.array(sourceViewDetectionSchema).default([]),
  repair_patch: z.object({
    kind: z.enum(["mesh", "splat"]),
    uri: z.string().min(1),
    status: z.enum(["queued", "running", "ready", "failed"]),
  }).optional(),
});

export const collisionWallSegmentSchema = z.object({
  id: z.string().min(1),
  start: vector3Schema,
  end: vector3Schema,
  height: z.number().positive(),
  thickness: z.number().positive(),
});

export const collisionOpeningSchema = z.object({
  id: z.string().min(1),
  kind: z.enum(["door", "window"]),
  wall_id: z.string().min(1),
  center: vector3Schema,
  width: z.number().positive(),
  height: z.number().positive(),
});

export const spatialArchitectureSchema = z.object({
  walls: z.array(collisionWallSegmentSchema),
  openings: z.array(collisionOpeningSchema),
});

export const splatAnalysisResultSchema = z.object({
  schema: z.literal("roomform.splat-analysis.result.v1"),
  scene_id: z.string().min(1),
  splat_uri: z.string().min(1),
  detector: z.enum([
    "splat-analyzer/owlv2",
    "boxer/owlv2",
    "spatiallm/qwen",
    "sam3/fal",
    "oracle-gt-labels",
    "mask3d-pointcloud",
  ]),
  coordinate_system: z.literal("source-splat"),
  objects: z.array(splatObjectProposalSchema),
  architecture: spatialArchitectureSchema.optional(),
});

export type SelectionSeed = z.infer<typeof selectionSeedSchema>;
export type ObjectizationRequest = z.infer<typeof objectizationRequestSchema>;
export type ObjectizationProposal = z.infer<typeof objectizationProposalSchema>;
export type ObjectizationCommit = z.infer<typeof objectizationCommitSchema>;
export type SplatObjectProposal = z.infer<typeof splatObjectProposalSchema>;
export type SpatialArchitecture = z.infer<typeof spatialArchitectureSchema>;
export type SplatAnalysisResult = z.infer<typeof splatAnalysisResultSchema>;
