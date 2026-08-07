import type { BenchmarkScene } from "./native-scenes";

export const PRODUCT_VIEW_KEYS = ["evidence", "reconstruction"] as const;

export type ProductLayers = {
  evidence: boolean;
  reconstruction: boolean;
};

/** Reduce any benchmark/debug scene to the two modes exposed by the product. */
export function productViewerContract(project: BenchmarkScene): BenchmarkScene {
  const evidence = project.views.find((view) => view.key === "evidence" || view.key === "input")
    ?? project.views[0];
  const reconstruction = project.views.find(
    (view) => view.key === "reconstruction" || view.key === "agent" || view.key === "editor",
  ) ?? project.views[project.views.length - 1];
  return {
    ...project,
    views: [
      {
        ...evidence,
        key: "evidence",
        button: "Evidence",
        desc: "Observed scan evidence",
        tone: "Raw registered geometry and color",
        entities: false,
        analysis: undefined,
        overlayUri: undefined,
      },
      {
        ...reconstruction,
        key: "reconstruction",
        button: "Reconstruction",
        desc: "Structured reconstruction with independently editable scene objects",
        tone: "Reconstructed structure and objects",
        overlayUri: undefined,
      },
    ],
  };
}

export function productViewIndex(search: string): 0 | 1 {
  return new URLSearchParams(search).get("view") === "reconstruction" ? 1 : 0;
}

export function productLayersFromSearch(search: string): ProductLayers {
  const params = new URLSearchParams(search);
  if (params.has("layers")) {
    const layers = new Set(params.get("layers")?.split(",").filter(Boolean));
    return {
      evidence: layers.has("evidence"),
      reconstruction: layers.has("reconstruction"),
    };
  }
  return params.get("view") === "reconstruction"
    ? { evidence: false, reconstruction: true }
    : { evidence: true, reconstruction: false };
}

export function productLayersParam(layers: ProductLayers): string {
  return [
    layers.evidence && "evidence",
    layers.reconstruction && "reconstruction",
  ].filter(Boolean).join(",");
}
