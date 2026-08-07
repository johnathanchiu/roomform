// Native project loading from the roomform pipeline's own contracts:
// GET /api/scenes lists artifacts/<id> directories, each holding a
// scene.json (roomform SceneDocument, z-up grid frame) plus cloud.ply
// evidence and optional per-object meshes under objects/.
//
// This module owns the frame conversion (mirrored from the deleted
// roomform/export.py fixture math): the editor works y-up, centered —
// position [x - cx, z, -(y - cy)], yaw becomes a quaternion about +Y,
// bounds [dx, dz, dy], with (cx, cy) = half the grid xy extent and the
// floor left at 0. Per-object GLBs are object-local (center-subtracted,
// heading-unrotated, z-up), so meshes follow saved poses across reloads.

import { z } from "zod";
import type { SplatAnalysisResult, SplatObjectProposal } from "@roomform/scene/selection";
import type { WallSegment } from "./wall-contact";

export type SceneView = {
  key: string;
  button: string;
  uri: string;
  desc: string;
  tone: string;
  zUp?: boolean;
  yDown?: boolean;
  normalColors?: boolean;
  entities?: boolean;
  reference?: boolean;
  objectEvidence?: boolean;
  analysis?: SplatAnalysisResult;
  dollhouse?: boolean;
  overlayUri?: string;
  /** y-up world offset applied to reference geometry (centers raw
   * grid-frame artifacts the way fixture exports used to pre-center). */
  offset?: [number, number, number];
};

export type BenchmarkScene = {
  title: string;
  viewTarget: [number, number, number];
  analysis: SplatAnalysisResult;
  views: SceneView[];
  walls?: WallSegment[];
};

export const sceneObjectSchema = z.object({
  cls: z.string().min(1),
  center: z.tuple([z.number(), z.number(), z.number()]),
  size: z.tuple([z.number(), z.number(), z.number()]),
  heading: z.number(),
  source: z.string().default("spatiallm"),
  mesh_path: z.string().nullish(),
  qa: z.record(z.string(), z.unknown()).optional(),
});

// The subset of roomform.contracts.SceneDocument the editor consumes;
// passthrough keeps unknown fields intact for lossless save round-trips.
export const sceneDocumentSchema = z.looseObject({
  version: z.string(),
  source_scan: z.string(),
  shell: z.looseObject({
    vox_m: z.number().positive(),
    shape: z.tuple([z.number(), z.number(), z.number()]),
  }),
  objects: z.array(sceneObjectSchema),
  floor_z: z.number().nullish(),
});

export type SceneDocument = z.infer<typeof sceneDocumentSchema>;

export type NativeProject = BenchmarkScene & {
  id: string;
  document: SceneDocument;
  /** grid-frame xy half-extents subtracted by the loader */
  center: [number, number];
};

function gridCenter(document: SceneDocument): [number, number] {
  const { shape, vox_m } = document.shell;
  return [(shape[0] * vox_m) / 2, (shape[1] * vox_m) / 2];
}

export function proposalFromObject(
  object: z.infer<typeof sceneObjectSchema>,
  index: number,
  center: [number, number],
  meshUri?: string,
): SplatObjectProposal {
  const [cx, cy, cz] = object.center;
  const [sx, sy, sz] = object.size;
  return {
    id: `object-${index}`,
    label: object.cls,
    category: object.cls,
    position: [cx - center[0], cz, -(cy - center[1])],
    rotation_xyzw: [
      0,
      Math.sin(object.heading / 2),
      0,
      Math.cos(object.heading / 2),
    ],
    bounds: [sx, sz, sy],
    confidence: 1,
    method: "spatiallm",
    source_views: [],
    measured_mesh_uri: meshUri,
    measured_mesh_z_up: meshUri ? true : undefined,
  };
}

/** Inverse of proposalFromObject: fold edited transforms (position and
 * heading only — sizes untouched) back into z-up grid-frame objects. */
export function documentWithEdits(
  project: NativeProject,
  proposals: SplatObjectProposal[],
): SceneDocument {
  const byId = new Map(proposals.map((proposal) => [proposal.id, proposal]));
  return {
    ...project.document,
    objects: project.document.objects.map((object, index) => {
      const proposal = byId.get(`object-${index}`);
      if (!proposal) return object;
      const [px, py, pz] = proposal.position;
      const [, qy, , qw] = proposal.rotation_xyzw;
      return {
        ...object,
        center: [
          px + project.center[0],
          project.center[1] - pz,
          py,
        ] as [number, number, number],
        heading: 2 * Math.atan2(qy, qw),
      };
    }),
  };
}

// Vite's SPA fallback answers 200 + index.html for missing paths, so
// presence needs a content-type sniff, not just an ok check.
async function assetAvailable(uri: string): Promise<boolean> {
  try {
    const response = await fetch(uri, { method: "HEAD" });
    const type = response.headers.get("content-type") ?? "";
    return response.ok && !type.includes("text/html");
  } catch {
    return false;
  }
}

async function fetchJson(uri: string): Promise<unknown> {
  const response = await fetch(uri);
  if (!response.ok) throw new Error(`${uri}: ${response.status}`);
  return response.json();
}

async function loadProject(id: string): Promise<NativeProject> {
  const document = sceneDocumentSchema.parse(
    await fetchJson(`/artifacts/${id}/scene.json`),
  );
  const center = gridCenter(document);
  const cloudUri = `/artifacts/${id}/cloud.ply`;
  // objects/ meshes exist when `roomform.export objects` ran for the
  // scene; only present ones become draggable.
  const meshUris = await Promise.all(
    document.objects.map(async (_object, index) => {
      const uri = `/artifacts/${id}/objects/object-${index}.glb`;
      return (await assetAvailable(uri)) ? uri : undefined;
    }),
  );
  const analysis: SplatAnalysisResult = {
    schema: "roomform.splat-analysis.result.v1",
    scene_id: id,
    splat_uri: cloudUri,
    detector: "spatiallm/qwen",
    coordinate_system: "source-splat",
    objects: document.objects.map((object, index) =>
      proposalFromObject(object, index, center, meshUris[index])),
  };
  // cloud.ply is raw grid frame; shift it into the centered editor frame
  const offset: [number, number, number] = [-center[0], 0, center[1]];
  return {
    id,
    document,
    center,
    title: id,
    viewTarget: [0, 1.4, 0],
    analysis,
    views: [
      {
        key: "evidence",
        button: "Evidence",
        uri: cloudUri,
        desc: "Observed scan evidence",
        tone: "Raw registered geometry and color",
        zUp: true,
        offset,
      },
      {
        key: "reconstruction",
        button: "Reconstruction",
        uri: cloudUri,
        desc: "Structured reconstruction with independently editable scene objects",
        tone: "Reconstructed structure and objects",
        // TODO(native): amber inferred-structure points (patchgraph
        // cells without evidence) are not rendered yet — the fixture
        // path baked them into editable.ply.
        zUp: true,
        offset,
        entities: true,
        analysis,
      },
    ],
  };
}

export async function loadNativeProjects(): Promise<Record<string, NativeProject>> {
  const ids = z.array(z.string()).parse(await fetchJson("/api/scenes"));
  const projects: Record<string, NativeProject> = {};
  for (const id of ids) {
    try {
      if (!(await assetAvailable(`/artifacts/${id}/cloud.ply`))) continue;
      projects[id] = await loadProject(id);
    } catch (error) {
      console.warn(`native scenes: skipping ${id}`, error);
    }
  }
  return projects;
}

export async function saveScene(
  project: NativeProject,
  proposals: SplatObjectProposal[],
): Promise<SceneDocument> {
  const document = documentWithEdits(project, proposals);
  const response = await fetch(`/api/scenes/${project.id}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(document),
  });
  if (!response.ok) {
    throw new Error(`save failed: ${response.status} ${await response.text()}`);
  }
  return document;
}
