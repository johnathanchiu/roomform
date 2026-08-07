import { z } from "zod";

export const vector2Schema = z.tuple([z.number(), z.number()]);
export const vector3Schema = z.tuple([z.number(), z.number(), z.number()]);
export const quaternionSchema = z.tuple([
  z.number(),
  z.number(),
  z.number(),
  z.number(),
]);

export const transformSchema = z.object({
  position: vector3Schema,
  rotation: quaternionSchema.default([0, 0, 0, 1]),
  scale: vector3Schema.default([1, 1, 1]),
});

export const openingSchema = z.object({
  id: z.string().min(1),
  kind: z.enum(["door", "window", "opening"]),
  offset: z.number().nonnegative(),
  width: z.number().positive(),
  height: z.number().positive(),
  sillHeight: z.number().nonnegative().default(0),
});

export const wallSchema = z.object({
  id: z.string().min(1),
  start: vector2Schema,
  end: vector2Schema,
  height: z.number().positive(),
  thickness: z.number().positive().default(0.12),
  openings: z.array(openingSchema).default([]),
});

export const shellSchema = z.object({
  floor: z.array(vector2Schema).min(3),
  floorElevation: z.number().default(0),
  ceilingHeight: z.number().positive(),
  walls: z.array(wallSchema).min(3),
});

export const assetSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("box") }),
  z.object({ kind: z.literal("glb"), uri: z.string().min(1) }),
]);

export const roomObjectSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  category: z.string().min(1),
  transform: transformSchema,
  originalTransform: transformSchema,
  bounds: vector3Schema,
  asset: assetSchema.default({ kind: "box" }),
  anchor: z.enum(["floor", "wall", "surface", "free"]).default("floor"),
  visible: z.boolean().default(true),
});

export const referenceSchema = z.object({
  kind: z.enum(["mesh", "splat"]),
  uri: z.string().min(1),
  transform: transformSchema,
});

export const roomformSceneSchema = z.object({
  schema: z.literal("roomform.scene.v1"),
  id: z.string().min(1),
  name: z.string().min(1),
  units: z.literal("meters"),
  coordinateSystem: z.literal("right-handed-y-up"),
  shell: shellSchema,
  objects: z.array(roomObjectSchema),
  reference: referenceSchema.optional(),
  revision: z.number().int().nonnegative().default(0),
});

export type RoomformScene = z.infer<typeof roomformSceneSchema>;
export type RoomObject = z.infer<typeof roomObjectSchema>;
export type RoomWall = z.infer<typeof wallSchema>;
export type RoomOpening = z.infer<typeof openingSchema>;
export type Transform = z.infer<typeof transformSchema>;

export function parseRoomformScene(input: unknown): RoomformScene {
  return roomformSceneSchema.parse(input);
}
