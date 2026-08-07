import type { SplatAnalysisResult, SplatObjectProposal } from "@roomform/scene/selection";
import { splatAnalysisResultSchema } from "@roomform/scene/selection";
import {
  type ChangeEvent,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Quaternion } from "three";

import { HierarchyPanel } from "@/components/editor/hierarchy-panel";
import type { HierarchyLayer } from "@/components/editor/hierarchy-panel";
import { TopBar } from "@/components/editor/top-bar";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cameraRelativeMovement } from "./viewport-controls";
import type { ViewBasis } from "./viewport-controls";
import { loadNativeProjects, saveScene } from "./native-scenes";
import type { NativeProject } from "./native-scenes";
import {
  productLayersFromSearch,
  productLayersParam,
} from "./product-viewer-contract";

const SplatPreview = lazy(() => import("./splat-preview"));

function projectFromUrl(): string {
  return new URLSearchParams(window.location.search).get("project") ?? "";
}

// Rendered while projects are probed or when no scene data exists locally.
const PLACEHOLDER_SCENE: NativeProject = {
  id: "placeholder",
  document: {
    version: "",
    source_scan: "",
    shell: { vox_m: 0.08, shape: [1, 1, 1] },
    objects: [],
  },
  center: [0, 0],
  title: "No project data",
  viewTarget: [0, 1.4, 0],
  analysis: {
    schema: "roomform.splat-analysis.result.v1",
    scene_id: "placeholder",
    splat_uri: "",
    detector: "mask3d-pointcloud",
    coordinate_system: "source-splat",
    objects: [],
  },
  views: [
    {
      key: "placeholder",
      button: "No data",
      uri: "",
      desc: "No scene artifacts found on this machine",
      tone: "Run the pipeline to populate artifacts/",
    },
  ],
};

async function readAnalysisFile(file: File): Promise<SplatAnalysisResult> {
  return splatAnalysisResultSchema.parse(JSON.parse(await file.text()));
}

function transformsMatch(
  objects: SplatObjectProposal[],
  reference: SplatObjectProposal[],
): boolean {
  return objects.every((object, index) => {
    const other = reference[index];
    return other !== undefined
      && object.position.every((value, axis) => Math.abs(value - other.position[axis]) < 1e-7)
      && object.rotation_xyzw.every((value, axis) => Math.abs(value - other.rotation_xyzw[axis]) < 1e-7);
  });
}

function SceneApp() {
  const [analysis, setAnalysis] = useState<SplatAnalysisResult>(
    PLACEHOLDER_SCENE.analysis,
  );
  const [referenceKind, setReferenceKind] = useState<"mesh" | "splat">("mesh");
  const [selectedId, setSelectedId] = useState<string>();
  const [productLayers, setProductLayers] = useState(
    () => productLayersFromSearch(window.location.search),
  );
  const [projectId, setProjectId] = useState(projectFromUrl);
  const [hiddenIds, setHiddenIds] = useState<ReadonlySet<string>>(new Set());
  const [wallsVisible, setWallsVisible] = useState(true);
  const [focusTarget, setFocusTarget] = useState<[number, number, number]>();
  // objects as last persisted to scene.json — the dirty/save baseline
  const [savedObjects, setSavedObjects] = useState<SplatObjectProposal[]>([]);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "error">("idle");
  // Native projects from /api/scenes: every artifacts/<id> directory
  // with a scene.json and its evidence cloud. undefined = still probing.
  const [projects, setProjects] = useState<Record<string, NativeProject>>();
  useEffect(() => {
    let cancelled = false;
    loadNativeProjects().then((available) => {
      if (!cancelled) setProjects(available);
    });
    return () => {
      cancelled = true;
    };
  }, []);
  const loadedProjects = projects ?? {};
  const activeProjectId = projectId in loadedProjects
    ? projectId
    : Object.keys(loadedProjects)[0];
  const scene = activeProjectId !== undefined
    ? loadedProjects[activeProjectId]
    : PLACEHOLDER_SCENE;
  const evidenceView = scene.views.find((candidate) => candidate.key === "evidence")
    ?? scene.views[0];
  const reconstructionView = scene.views.find(
    (candidate) => candidate.key === "reconstruction",
  ) ?? scene.views[scene.views.length - 1];
  const view = productLayers.reconstruction ? reconstructionView : evidenceView;
  const primaryView = productLayers.evidence ? evidenceView : view;
  const showPrimaryReference = productLayers.evidence
    ? evidenceView.reference !== false
    : productLayers.reconstruction && reconstructionView.reference !== false;
  const secondaryReferenceView = productLayers.evidence
    && productLayers.reconstruction
    && reconstructionView.reference !== false
    ? reconstructionView
    : undefined;

  const resetSceneLocal = useCallback(() => {
    setSelectedId(undefined);
    setHiddenIds(new Set());
    setWallsVisible(true);
    setFocusTarget(undefined);
  }, []);

  useEffect(() => {
    setAnalysis(reconstructionView.analysis ?? scene.analysis);
    setSavedObjects((reconstructionView.analysis ?? scene.analysis).objects);
    setSaveState("idle");
    resetSceneLocal();
  }, [scene, reconstructionView, resetSceneLocal]);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (activeProjectId !== undefined) url.searchParams.set("project", activeProjectId);
    url.searchParams.set("layers", productLayersParam(productLayers));
    url.searchParams.set(
      "view",
      productLayers.reconstruction ? "reconstruction" : "evidence",
    );
    window.history.replaceState(null, "", url);
  }, [activeProjectId, productLayers]);
  const [importError, setImportError] = useState<string>();
  const input = useRef<HTMLInputElement>(null);
  const viewBasis = useRef<ViewBasis>({
    right: [1, 0, 0],
    forward: [0, 0, -1],
  });
  const proposals = analysis?.objects ?? [];
  const dirty = !transformsMatch(proposals, savedObjects);
  const visibleProposals = useMemo(
    () => proposals.filter((proposal) => !hiddenIds.has(proposal.id)),
    [proposals, hiddenIds],
  );
  const visibleOriginals = useMemo(
    () => (referenceKind === "mesh" ? scene.analysis.objects : proposals)
      .filter((proposal) => !hiddenIds.has(proposal.id)),
    [referenceKind, scene, proposals, hiddenIds],
  );
  const entitiesActive = productLayers.reconstruction
    && Boolean(reconstructionView.entities);

  const handleSave = useCallback(async () => {
    if (scene.id === "placeholder" || saveState === "saving") return;
    setSaveState("saving");
    try {
      await saveScene(scene, proposals);
      setSavedObjects(proposals);
      setSaveState("idle");
    } catch (error) {
      console.error(error);
      setSaveState("error");
    }
  }, [scene, proposals, saveState]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void handleSave();
        return;
      }
      if (event.key === "Escape") setSelectedId(undefined);
      const editingField = event.target instanceof HTMLElement
        && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
      if (editingField) return;
      if (event.key.toLowerCase() === "r" && referenceKind === "mesh") {
        setAnalysis(scene.analysis);
        setSelectedId(undefined);
        return;
      }
      if (!selectedId || event.metaKey || event.ctrlKey || event.altKey) return;
      const movement: Record<string, [number, number]> = {
        ArrowLeft: [-1, 0], a: [-1, 0],
        ArrowRight: [1, 0], d: [1, 0],
        ArrowUp: [0, 1], w: [0, 1],
        ArrowDown: [0, -1], s: [0, -1],
      };
      const screenDelta = movement[event.key];
      if (screenDelta) {
        event.preventDefault();
        const { right, forward } = viewBasis.current;
        const delta = cameraRelativeMovement({ right, forward }, screenDelta);
        setAnalysis((current) => ({
          ...current,
          objects: current.objects.map((object) => object.id === selectedId
            ? { ...object, position: object.position.map((value, index) => value + delta[index]) as [number, number, number] }
            : object),
        }));
      }
      if (event.key === "q" || event.key === "e") {
        const angle = (event.key === "q" ? -1 : 1) * Math.PI / 18;
        setAnalysis((current) => ({
          ...current,
          objects: current.objects.map((object) => {
            if (object.id !== selectedId) return object;
            const next = new Quaternion(...object.rotation_xyzw)
              .premultiply(new Quaternion().setFromAxisAngle({ x: 0, y: 1, z: 0 }, angle));
            return { ...object, rotation_xyzw: [next.x, next.y, next.z, next.w] };
          }),
        }));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleSave, referenceKind, scene, selectedId]);

  const handleViewBasisChange = useCallback((basis: ViewBasis) => {
    viewBasis.current = basis;
  }, []);

  const importAnalysis = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const result = await readAnalysisFile(file);
      setAnalysis(result);
      setReferenceKind("splat");
      setSelectedId(result.objects[0]?.id);
      setImportError(undefined);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Invalid analysis result");
    } finally {
      event.target.value = "";
    }
  };

  const moveObject = useCallback((id: string, position: [number, number, number]) => {
    setAnalysis((current) => ({
      ...current,
      objects: current.objects.map((object) => object.id === id
        ? { ...object, position }
        : object),
    }));
  }, []);

  const toggleObjectHidden = useCallback((id: string) => {
    setHiddenIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else {
        next.add(id);
        setSelectedId((selected) => (selected === id ? undefined : selected));
      }
      return next;
    });
  }, []);

  const focusObject = useCallback((object: SplatObjectProposal) => {
    setFocusTarget([...object.position]);
    setSelectedId(object.id);
  }, []);

  const hierarchyLayers: HierarchyLayer[] = scene.views
    .filter((candidate) => candidate.key !== "placeholder")
    .map((candidate) => ({
      key: candidate.key,
      label: candidate.button,
      visible: candidate.key === "evidence"
        ? productLayers.evidence
        : productLayers.reconstruction,
      onToggle: () => setProductLayers((current) => (
        candidate.key === "evidence"
          ? { ...current, evidence: !current.evidence }
          : { ...current, reconstruction: !current.reconstruction }
      )),
    }));

  return (
    <TooltipProvider>
      <div className="flex h-svh flex-col overflow-hidden bg-background text-foreground">
        <TopBar
          project={activeProjectId !== undefined
            ? {
              value: activeProjectId,
              options: Object.entries(loadedProjects).map(([id, project]) => ({
                value: id,
                label: project.title.split(" · ")[0],
              })),
              onChange: (id) => {
                setProjectId(id);
                setProductLayers({ evidence: true, reconstruction: false });
              },
            }
            : undefined}
          save={{
            dirty,
            state: saveState,
            onSave: () => void handleSave(),
          }}
          importInput={input}
          onImport={importAnalysis}
        />

        <div className="min-h-0 flex-1">
          <ResizablePanelGroup orientation="horizontal">
            <ResizablePanel defaultSize={280} minSize={200} maxSize={460}>
              <HierarchyPanel
                sceneTitle={scene.title.split(" · ")[0]}
                layers={hierarchyLayers}
                walls={scene.walls && scene.walls.length > 0
                  ? {
                    count: scene.walls.length,
                    visible: wallsVisible,
                    onToggle: () => setWallsVisible((current) => !current),
                  }
                  : undefined}
                objects={proposals}
                entitiesActive={entitiesActive}
                hiddenIds={hiddenIds}
                selectedId={selectedId}
                onSelect={setSelectedId}
                onToggleObject={toggleObjectHidden}
                onFocusObject={focusObject}
              />
            </ResizablePanel>
            <ResizableHandle />
            <ResizablePanel minSize={360}>
              <div className="relative h-full min-w-0 overflow-hidden bg-[#0c0a09]">
                <div className="viewport-canvas absolute inset-0">
                  {projects === undefined ? (
                    <div className="absolute left-1/2 top-4 z-10 -translate-x-1/2 rounded-lg border bg-card/90 px-3 py-2 text-xs text-muted-foreground backdrop-blur">
                      Loading scenes…
                    </div>
                  ) : activeProjectId === undefined ? (
                    <div className="absolute inset-0 flex items-center justify-center p-6">
                      <p className="max-w-sm text-center text-sm leading-relaxed text-muted-foreground">
                        No scene has artifacts on this machine. Run
                        <span className="font-mono"> roomform.pipe.e2e </span>
                        to produce one, then reload.
                      </p>
                    </div>
                  ) : (
                  <Suspense
                    fallback={
                      <div className="absolute left-1/2 top-4 z-10 -translate-x-1/2 rounded-lg border bg-card/90 px-3 py-2 text-xs text-muted-foreground backdrop-blur">
                        Loading renderer…
                      </div>
                    }
                  >
                    <SplatPreview
                      referenceKind={referenceKind}
                      referenceUri={referenceKind === "mesh" ? primaryView.uri : analysis.splat_uri}
                      proposals={visibleProposals}
                      originalProposals={visibleOriginals}
                      selectedId={selectedId}
                      collisionDebug={false}
                      showEntities={entitiesActive}
                      showObjectEvidence={reconstructionView.objectEvidence !== false}
                      showShell={false}
                      showReference={showPrimaryReference}
                      referenceZUp={Boolean(primaryView.zUp)}
                      referenceYDown={Boolean(primaryView.yDown)}
                      referenceCull={Boolean(primaryView.dollhouse)}
                      referenceOffset={primaryView.offset}
                      secondaryReferenceUri={secondaryReferenceView?.uri}
                      secondaryReferenceZUp={Boolean(secondaryReferenceView?.zUp)}
                      secondaryReferenceYDown={Boolean(secondaryReferenceView?.yDown)}
                      secondaryReferenceCull={Boolean(secondaryReferenceView?.dollhouse)}
                      secondaryReferenceOffset={secondaryReferenceView?.offset}
                      overlayUri={undefined}
                      normalColors={Boolean(primaryView.normalColors)}
                      comparisonZUp
                      benchmarkView={false}
                      viewTarget={focusTarget ?? scene.viewTarget}
                      walls={wallsVisible ? scene.walls : undefined}
                      onSelect={setSelectedId}
                      onMoveObject={moveObject}
                      onViewBasisChange={handleViewBasisChange}
                    />
                  </Suspense>
                  )}
                  {importError && (
                    <p className="absolute right-3 top-3 z-20 max-w-sm rounded-lg border border-destructive/40 bg-destructive/10 p-2 text-xs leading-relaxed text-destructive backdrop-blur">
                      {importError}
                    </p>
                  )}
                </div>
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>

        <footer className="flex h-7 shrink-0 items-center gap-2 border-t bg-background px-3 text-xs text-muted-foreground">
          <span className="truncate">{view.desc}</span>
          <span className="ml-auto hidden shrink-0 lg:inline">
            {entitiesActive && proposals.length > 0
              ? "⌘-drag move · Q/E rotate · WASD nudge · R reset · ⌘S save · Esc deselect"
              : "Left-drag orbit · Right-drag pan · Scroll zoom"}
          </span>
        </footer>
      </div>
    </TooltipProvider>
  );
}

export function App() {
  return <SceneApp />;
}
