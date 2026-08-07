import { describe, expect, test } from "bun:test";

import type { BenchmarkScene } from "./native-scenes";
import {
  productLayersFromSearch,
  productLayersParam,
  productViewerContract,
  productViewIndex,
} from "./product-viewer-contract";

const project: BenchmarkScene = {
  title: "Test room",
  viewTarget: [0, 1, 0],
  analysis: {
    schema: "roomform.splat-analysis.result.v1",
    scene_id: "test",
    splat_uri: "/test.ply",
    detector: "spatiallm/qwen",
    coordinate_system: "source-splat",
    objects: [],
  },
  views: [
    { key: "input", button: "Stage one", uri: "/input.ply", desc: "input", tone: "raw", entities: true },
    { key: "mesh", button: "Stage two", uri: "/mesh.glb", desc: "mesh", tone: "debug" },
    { key: "editor", button: "Stage three", uri: "/scene.glb", desc: "editor", tone: "editable", entities: true },
  ],
};

describe("product viewer contract", () => {
  test("always exposes only Evidence and Reconstruction", () => {
    const result = productViewerContract(project);
    expect(result.views.map((view) => [view.key, view.button])).toEqual([
      ["evidence", "Evidence"],
      ["reconstruction", "Reconstruction"],
    ]);
    expect(result.views[0].uri).toBe("/input.ply");
    expect(result.views[0].entities).toBe(false);
    expect(result.views[1].uri).toBe("/scene.glb");
    expect(result.views[1].entities).toBe(true);
  });

  test("uses a stable URL vocabulary", () => {
    expect(productViewIndex("?view=evidence")).toBe(0);
    expect(productViewIndex("?view=reconstruction")).toBe(1);
    expect(productViewIndex("?view=editor")).toBe(0);
  });

  test("supports independent evidence and reconstruction layers", () => {
    expect(productLayersFromSearch("?layers=evidence,reconstruction")).toEqual({
      evidence: true,
      reconstruction: true,
    });
    expect(productLayersFromSearch("?layers=")).toEqual({
      evidence: false,
      reconstruction: false,
    });
    expect(productLayersFromSearch("?view=reconstruction")).toEqual({
      evidence: false,
      reconstruction: true,
    });
    expect(productLayersParam({ evidence: true, reconstruction: true }))
      .toBe("evidence,reconstruction");
  });
});
