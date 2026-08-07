import { Html, OrbitControls, useCursor } from "@react-three/drei";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import type {
  SpatialArchitecture,
  SplatObjectProposal,
} from "@roomform/scene/selection";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DoubleSide,
  FrontSide,
  MOUSE,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshNormalMaterial,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
} from "three";
import type { WebGLRenderer } from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

import { splitGeometryByProposals } from "./mesh-selection";
import type { ViewBasis } from "./viewport-controls";
import { objectIntersectsWall } from "./wall-contact";
import type { WallSegment } from "./wall-contact";

const roomTarget = new Vector3(2.1, 0.1, -0.7);

function InitialCamera({ target = roomTarget }: { target?: Vector3 }) {
  const { camera, invalidate } = useThree();
  useEffect(() => {
    camera.position.copy(target).add(new Vector3(5.5, 2.5, 5.5));
    camera.lookAt(target);
    camera.updateProjectionMatrix();
    invalidate();
  }, [camera, invalidate]);
  return null;
}

function ViewBasisReporter({ onChange }: { onChange: (basis: ViewBasis) => void }) {
  const { camera } = useThree();
  const forward = useMemo(() => new Vector3(), []);
  const right = useMemo(() => new Vector3(), []);
  useFrame(() => {
    camera.getWorldDirection(forward);
    forward.y = 0;
    if (forward.lengthSq() < 1e-8) return;
    forward.normalize();
    right.copy(forward).cross(camera.up).normalize();
    onChange({
      right: [right.x, right.y, right.z],
      forward: [forward.x, forward.y, forward.z],
    });
  });
  return null;
}

function RoomSplat({ uri, onLoad }: { uri: string; onLoad: () => void }) {
  const { gl, invalidate, scene } = useThree();
  const { spark, splat } = useMemo(() => {
    const renderer = new SparkRenderer({
      renderer: gl as WebGLRenderer,
      onDirty: invalidate,
      minPixelRadius: 0.35,
      minSortIntervalMs: 32,
    });
    const mesh = new SplatMesh({
      url: uri,
      onLoad: (loaded) => {
        loaded.rotation.x = Math.PI;
        onLoad();
        invalidate();
      },
    });
    return { spark: renderer, splat: mesh };
  }, [gl, invalidate, onLoad, uri]);
  useEffect(() => {
    scene.add(spark, splat);
    return () => {
      scene.remove(spark, splat);
      splat.dispose();
      spark.dispose();
    };
  }, [scene, spark, splat]);
  return null;
}

function RepairMesh({ uri }: { uri: string }) {
  const loaded = useLoader(PLYLoader, uri);
  const geometry = useMemo(() => {
    const next = loaded.clone();
    next.applyMatrix4(new Matrix4().makeRotationX(-Math.PI / 2));
    if (!next.getAttribute("normal")) next.computeVertexNormals();
    return next;
  }, [loaded]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        vertexColors={Boolean(geometry.getAttribute("color"))}
        roughness={0.9}
      />
    </mesh>
  );
}

function PointCloud({ uri, sourceZUp, sourceYDown, interactive = true, onLoad }: {
  uri: string;
  sourceZUp: boolean;
  sourceYDown: boolean;
  interactive?: boolean;
  onLoad: () => void;
}) {
  const loaded = useLoader(PLYLoader, uri);
  const geometry = useMemo(() => {
    const next = loaded.clone();
    if (sourceZUp) next.applyMatrix4(new Matrix4().makeRotationX(-Math.PI / 2));
    if (sourceYDown) next.applyMatrix4(new Matrix4().makeRotationX(Math.PI));
    return next;
  }, [loaded, sourceYDown, sourceZUp]);
  useEffect(() => onLoad(), [onLoad]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <points
      geometry={geometry}
      raycast={interactive ? undefined : () => undefined}
    >
      <pointsMaterial
        color={geometry.getAttribute("color") ? "#ffffff" : "#e7e5e4"}
        vertexColors={Boolean(geometry.getAttribute("color"))}
        size={0.025}
        sizeAttenuation
      />
    </points>
  );
}

function CleanShell({ uri, collisionDebug, onLoad }: {
  uri: string;
  collisionDebug: boolean;
  onLoad: () => void;
}) {
  const loaded = useLoader(GLTFLoader, uri);
  const object = useMemo(() => {
    const clone = loaded.scene.clone(true);
    clone.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      child.material = new MeshStandardMaterial({
        color: "#a5f3fc",
        emissive: collisionDebug ? "#155e75" : "#000000",
        emissiveIntensity: collisionDebug ? 0.3 : 0,
        opacity: collisionDebug ? 0.28 : 0.2,
        transparent: true,
        depthWrite: false,
        roughness: 1,
        side: DoubleSide,
      });
      child.raycast = () => undefined;
    });
    return clone;
  }, [collisionDebug, loaded.scene]);
  useEffect(() => onLoad(), [onLoad]);
  useEffect(() => () => {
    object.traverse((child) => {
      if (child instanceof Mesh && child.material instanceof MeshStandardMaterial) {
        child.material.dispose();
      }
    });
  }, [object]);
  return <primitive object={object} />;
}

function CollisionArchitecture({
  architecture,
  onLoad,
}: {
  architecture: SpatialArchitecture;
  onLoad: () => void;
}) {
  const fragments = useMemo(() => architecture.walls.flatMap((wall) => {
    const dx = wall.end[0] - wall.start[0];
    const dz = wall.end[2] - wall.start[2];
    const length = Math.hypot(dx, dz);
    if (length < 1e-4) return [];
    const ux = dx / length;
    const uz = dz / length;
    const floor = wall.start[1];
    const ceiling = floor + wall.height;
    const openings = architecture.openings
      .filter((opening) => opening.wall_id === wall.id)
      .map((opening) => ({
        ...opening,
        along: (opening.center[0] - wall.start[0]) * ux
          + (opening.center[2] - wall.start[2]) * uz,
      }));
    const xs = [0, length];
    const ys = [floor, ceiling];
    openings.forEach((opening) => {
      xs.push(
        Math.max(0, opening.along - opening.width / 2),
        Math.min(length, opening.along + opening.width / 2),
      );
      ys.push(
        Math.max(floor, opening.center[1] - opening.height / 2),
        Math.min(ceiling, opening.center[1] + opening.height / 2),
      );
    });
    const xCuts = [...new Set(xs)].sort((a, b) => a - b);
    const yCuts = [...new Set(ys)].sort((a, b) => a - b);
    const cells = [];
    for (let xi = 0; xi < xCuts.length - 1; xi += 1) {
      for (let yi = 0; yi < yCuts.length - 1; yi += 1) {
        const x0 = xCuts[xi];
        const x1 = xCuts[xi + 1];
        const y0 = yCuts[yi];
        const y1 = yCuts[yi + 1];
        const x = (x0 + x1) / 2;
        const y = (y0 + y1) / 2;
        const insideOpening = openings.some((opening) => (
          Math.abs(x - opening.along) < opening.width / 2 - 1e-4
          && Math.abs(y - opening.center[1]) < opening.height / 2 - 1e-4
        ));
        if (insideOpening || x1 - x0 < 1e-4 || y1 - y0 < 1e-4) continue;
        cells.push({
          key: `${wall.id}-${xi}-${yi}`,
          position: [
            wall.start[0] + ux * x,
            y,
            wall.start[2] + uz * x,
          ] as [number, number, number],
          rotation: [0, -Math.atan2(uz, ux), 0] as [number, number, number],
          size: [x1 - x0, y1 - y0, wall.thickness] as [number, number, number],
        });
      }
    }
    return cells;
  }), [architecture]);
  useEffect(() => onLoad(), [onLoad]);
  return fragments.map((fragment) => (
    <mesh key={fragment.key} position={fragment.position} rotation={fragment.rotation}>
      <boxGeometry args={fragment.size} />
      <meshBasicMaterial
        color="#67e8f9"
        transparent
        opacity={0.28}
        depthWrite={false}
      />
    </mesh>
  ));
}

function RepairAppearance({
  patch,
}: {
  patch: NonNullable<SplatObjectProposal["repair_patch"]>;
}) {
  if (patch.kind === "mesh") return <RepairMesh uri={patch.uri} />;
  return <RoomSplat uri={patch.uri} onLoad={() => undefined} />;
}

function transformChanged(
  proposal: SplatObjectProposal,
  original: SplatObjectProposal,
) {
  const positionChanged = proposal.position.some(
    (value, index) => Math.abs(value - original.position[index]) > 1e-5,
  );
  const rotationChanged = proposal.rotation_xyzw.some(
    (value, index) => Math.abs(value - original.rotation_xyzw[index]) > 1e-5,
  );
  return positionChanged || rotationChanged;
}

function MeasuredObject({
  proposal,
  active,
  dragging = false,
  showObservedPoints = true,
  collisionDebug,
  walls,
  onHover,
  onSelect,
  onDragStart,
}: {
  proposal: SplatObjectProposal;
  active: boolean;
  dragging?: boolean;
  showObservedPoints?: boolean;
  collisionDebug: boolean;
  walls?: WallSegment[];
  onHover: (id?: string) => void;
  onSelect: (id?: string) => void;
  onDragStart?: (id: string, point: Vector3) => void;
}) {
  const loaded = useLoader(GLTFLoader, proposal.measured_mesh_uri!);
  const interferes = useMemo(
    () => Boolean(walls?.length) && objectIntersectsWall(
      proposal.position as [number, number, number],
      proposal.bounds as [number, number, number],
      walls!,
    ),
    [proposal.bounds, proposal.position, walls],
  );
  const object = useMemo(() => {
    const clone = loaded.scene.clone(true);
    if (proposal.measured_mesh_z_up !== false) {
      clone.applyMatrix4(new Matrix4().makeRotationX(-Math.PI / 2));
    }
    clone.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      if (!child.geometry.getAttribute("normal")) child.geometry.computeVertexNormals();
      const hasColors = Boolean(child.geometry.getAttribute("color"));
      child.material = new MeshStandardMaterial({
        color: collisionDebug ? "#f59e0b" : "#ffffff",
        emissive: interferes ? "#dc2626" : "#f97316",
        emissiveIntensity: interferes ? 0.55 : active ? 0.65 : collisionDebug ? 0.16 : 0,
        opacity: collisionDebug ? 0.72 : 1,
        transparent: collisionDebug,
        depthWrite: !collisionDebug,
        roughness: 0.88,
        side: DoubleSide,
        vertexColors: !collisionDebug && hasColors,
      });
    });
    return clone;
  }, [active, collisionDebug, interferes, loaded.scene]);
  useEffect(() => {
    // while dragging, the object must not intercept the pointer: raycasts
    // alternating between object and drag plane cause jumps + flicker
    object.traverse((child) => {
      if (child instanceof Mesh) {
        child.raycast = dragging ? () => undefined : Mesh.prototype.raycast;
      }
    });
  }, [dragging, object]);
  // Native object meshes are object-local (center-subtracted, heading-
  // unrotated), so the proposal transform IS the group transform and
  // saved poses stay consistent across reloads.
  const rotation = useMemo(
    () => new Quaternion(...proposal.rotation_xyzw),
    [proposal.rotation_xyzw],
  );
  const position = useMemo(
    () => new Vector3(...proposal.position),
    [proposal.position],
  );
  useEffect(() => () => {
    object.traverse((child) => {
      if (child instanceof Mesh && child.material instanceof MeshStandardMaterial) {
        child.material.dispose();
      }
    });
  }, [object]);
  return (
    <group
      position={position}
      quaternion={rotation}
      onClick={(event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        onSelect(proposal.id);
      }}
      onPointerDown={(event: ThreeEvent<PointerEvent>) => {
        // plain left-drag stays camera orbit; hold cmd/ctrl to grab the object
        const modifier = event.nativeEvent.metaKey || event.nativeEvent.ctrlKey;
        if (event.button !== 0 || !modifier || !onDragStart) return;
        event.stopPropagation();
        onSelect(proposal.id);
        onDragStart(proposal.id, event.point.clone());
      }}
      onPointerOver={(event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();
        onHover(proposal.id);
      }}
      onPointerOut={() => onHover(undefined)}
    >
      <primitive object={object} />
      {showObservedPoints && proposal.observed_points_uri && (
        <PointCloud
          uri={proposal.observed_points_uri}
          sourceZUp={proposal.observed_points_z_up === true}
          sourceYDown={false}
          interactive={!dragging}
          onLoad={() => undefined}
        />
      )}
      {active && (
        <Html
          center
          position={[0, proposal.bounds[1] / 2 + 0.12, 0]}
        >
          <div className="pointer-events-none whitespace-nowrap rounded-md border border-amber-300/20 bg-stone-950/90 px-2 py-1 text-[11px] font-medium text-amber-100 shadow-lg">
            {proposal.label} · measured surface{interferes ? " · ⚠ intersects wall" : ""}
          </div>
        </Html>
      )}
    </group>
  );
}

function MeasuredObjectMeshes({
  proposals,
  activeId,
  draggingId,
  showObservedPoints,
  collisionDebug,
  walls,
  onHover,
  onSelect,
  onDragStart,
}: {
  proposals: SplatObjectProposal[];
  activeId?: string;
  draggingId?: string;
  showObservedPoints: boolean;
  collisionDebug: boolean;
  walls?: WallSegment[];
  onHover: (id?: string) => void;
  onSelect: (id?: string) => void;
  onDragStart?: (id: string, point: Vector3) => void;
}) {
  return proposals.flatMap((proposal) => {
    if (!proposal.measured_mesh_uri) return [];
    return [(
      <MeasuredObject
        key={proposal.id}
        proposal={proposal}
        active={proposal.id === activeId}
        dragging={proposal.id === draggingId}
        showObservedPoints={showObservedPoints}
        collisionDebug={collisionDebug}
        walls={walls}
        onHover={onHover}
        onSelect={onSelect}
        onDragStart={onDragStart}
      />
    )];
  });
}

function DragPlane({
  position,
  quaternion,
  onDragMove,
  onDragEnd,
}: {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  onDragMove: (point: Vector3) => void;
  onDragEnd: () => void;
}) {
  return (
    <mesh
      position={position}
      quaternion={quaternion}
      onPointerMove={(event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();
        onDragMove(event.point);
      }}
      onPointerUp={(event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();
        onDragEnd();
      }}
    >
      <planeGeometry args={[400, 400]} />
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
    </mesh>
  );
}

function RoomMesh({
  uri,
  proposals,
  originals,
  activeId,
  onLoad,
}: {
  uri: string;
  proposals: SplatObjectProposal[];
  originals: SplatObjectProposal[];
  activeId?: string;
  onLoad: () => void;
}) {
  const loaded = useLoader(PLYLoader, uri);
  const geometry = useMemo(() => {
    const next = loaded.clone();
    next.applyMatrix4(new Matrix4().makeRotationX(-Math.PI / 2));
    if (!next.getAttribute("normal")) next.computeVertexNormals();
    return next;
  }, [loaded]);
  const split = useMemo(
    () => splitGeometryByProposals(
      geometry,
      originals.filter((proposal) => !proposal.measured_mesh_uri),
    ),
    [geometry, originals],
  );
  const layers = useMemo(() => proposals.flatMap((proposal) => {
    const original = originals.find((candidate) => candidate.id === proposal.id);
    const objectGeometry = split.objects.get(proposal.id);
    if (!original || !objectGeometry) return [];
    const rotation = new Quaternion(...proposal.rotation_xyzw)
      .multiply(new Quaternion(...original.rotation_xyzw).invert());
    const position = new Vector3(...proposal.position)
      .sub(new Vector3(...original.position).applyQuaternion(rotation));
    return [{ proposal, geometry: objectGeometry, position, rotation }];
  }), [originals, proposals, split.objects]);

  useEffect(() => onLoad(), [onLoad]);
  useEffect(() => () => {
    if (split.background !== geometry) split.background.dispose();
    split.objects.forEach((object) => object.dispose());
  }, [geometry, split]);

  const vertexColors = Boolean(geometry.getAttribute("color"));
  return (
    <group>
      <mesh geometry={split.background}>
        <meshStandardMaterial vertexColors={vertexColors} roughness={0.9} />
      </mesh>
      {layers.map((layer) => (
        <mesh
          key={layer.proposal.id}
          geometry={layer.geometry}
          position={layer.position}
          quaternion={layer.rotation}
        >
          <meshStandardMaterial
            vertexColors={vertexColors}
            roughness={0.9}
            emissive="#fb923c"
            emissiveIntensity={layer.proposal.id === activeId ? 0.35 : 0}
          />
        </mesh>
      ))}
    </group>
  );
}

function RoomGlb({
  uri,
  sourceZUp,
  sourceYDown = false,
  normalColors = false,
  cull = false,
  color = "#8dd3dc",
  opacity = 1,
  onLoad,
}: {
  uri: string;
  sourceZUp: boolean;
  sourceYDown?: boolean;
  normalColors?: boolean;
  cull?: boolean;
  color?: string;
  opacity?: number;
  onLoad: () => void;
}) {
  const loaded = useLoader(GLTFLoader, uri);
  const object = useMemo(() => {
    const clone = loaded.scene.clone(true);
    // Raw ARKitScenes mesh evidence is Z-up. Roomform collision GLBs have
    // already been normalized to the editor's Y-up coordinate system.
    if (sourceZUp) clone.applyMatrix4(new Matrix4().makeRotationX(-Math.PI / 2));
    if (sourceYDown) clone.applyMatrix4(new Matrix4().makeRotationX(Math.PI));
    clone.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      if (!child.geometry.getAttribute("normal")) child.geometry.computeVertexNormals();
      const hasColors = Boolean(child.geometry.getAttribute("color"));
      const side = cull ? FrontSide : DoubleSide;
      child.material = normalColors && hasColors
        ? new MeshBasicMaterial({ vertexColors: true, side })
        : normalColors
        ? new MeshNormalMaterial({ side, flatShading: false })
        : new MeshStandardMaterial({
            color: hasColors ? "#ffffff" : color,
            vertexColors: hasColors,
            roughness: 0.9,
            side,
            transparent: opacity < 1,
            opacity,
            depthWrite: opacity >= 1,
          });
    });
    return clone;
  }, [color, cull, loaded.scene, normalColors, opacity, sourceYDown, sourceZUp]);
  useEffect(() => onLoad(), [onLoad]);
  useEffect(() => () => {
    object.traverse((child) => {
      if (child instanceof Mesh && child.material) {
        child.material.dispose();
      }
    });
  }, [object]);
  return <primitive object={object} />;
}

export default function SplatPreview({
  referenceKind,
  referenceUri,
  proposals,
  originalProposals,
  selectedId,
  shellUri,
  architecture,
  collisionDebug = false,
  showReference,
  showEntities = true,
  showObjectEvidence = true,
  showShell,
  referenceZUp = false,
  referenceYDown = false,
  referenceCull = false,
  referenceOffset,
  secondaryReferenceUri,
  secondaryReferenceZUp = false,
  secondaryReferenceYDown = false,
  secondaryReferenceCull = false,
  secondaryReferenceOffset,
  overlayUri,
  comparisonUri,
  comparisonZUp = false,
  benchmarkView = false,
  viewTarget,
  normalColors = false,
  walls,
  onSelect,
  onMoveObject,
  onViewBasisChange,
}: {
  referenceKind: "mesh" | "splat";
  referenceUri: string;
  proposals: SplatObjectProposal[];
  originalProposals: SplatObjectProposal[];
  selectedId?: string;
  shellUri?: string;
  architecture?: SpatialArchitecture;
  collisionDebug?: boolean;
  showReference?: boolean;
  showEntities?: boolean;
  showObjectEvidence?: boolean;
  showShell?: boolean;
  referenceZUp?: boolean;
  referenceYDown?: boolean;
  referenceCull?: boolean;
  referenceOffset?: [number, number, number];
  secondaryReferenceUri?: string;
  secondaryReferenceZUp?: boolean;
  secondaryReferenceYDown?: boolean;
  secondaryReferenceCull?: boolean;
  secondaryReferenceOffset?: [number, number, number];
  overlayUri?: string;
  comparisonUri?: string;
  comparisonZUp?: boolean;
  benchmarkView?: boolean;
  viewTarget?: [number, number, number];
  normalColors?: boolean;
  walls?: WallSegment[];
  onSelect: (id?: string) => void;
  onMoveObject?: (id: string, position: [number, number, number]) => void;
  onViewBasisChange: (basis: ViewBasis) => void;
}) {
  const [ready, setReady] = useState(false);
  const [isNavigating, setIsNavigating] = useState(false);
  const [hoveredId, setHoveredId] = useState<string>();
  useCursor(Boolean(hoveredId));
  const handleLoad = useCallback(() => setReady(true), []);
  const [dragId, setDragId] = useState<string>();
  const dragState = useRef<{
    id: string;
    attachment: "floor" | "wall" | "ceiling" | "free";
    offset: Vector3;
    planePosition: [number, number, number];
    planeQuaternion: [number, number, number, number];
    fixedY: number;
    last: [number, number, number];
  } | undefined>(undefined);
  const handleDragStart = useCallback((id: string, point: Vector3) => {
    const proposal = proposals.find((candidate) => candidate.id === id);
    if (!proposal || !onMoveObject) return;
    const attachment = proposal.attachment ?? "floor";
    const position = new Vector3(...proposal.position);
    const planeQuaternion = attachment === "wall"
      ? new Quaternion(...proposal.rotation_xyzw)
      : new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), -Math.PI / 2);
    dragState.current = {
      id,
      attachment,
      offset: position.clone().sub(point),
      planePosition: attachment === "wall"
        ? [...proposal.position]
        : [0, point.y, 0],
      planeQuaternion: [planeQuaternion.x, planeQuaternion.y, planeQuaternion.z, planeQuaternion.w],
      fixedY: proposal.position[1],
      last: [...proposal.position],
    };
    setDragId(id);
  }, [onMoveObject, proposals]);
  const handleDragMove = useCallback((point: Vector3) => {
    const state = dragState.current;
    if (!state || !onMoveObject) return;
    const moved = point.clone().add(state.offset);
    const next: [number, number, number] = state.attachment === "wall"
      ? [moved.x, moved.y, moved.z]
      : [moved.x, state.fixedY, moved.z];
    // jump filter: shallow-angle plane hits can teleport the object
    const jump = Math.hypot(
      next[0] - state.last[0],
      next[1] - state.last[1],
      next[2] - state.last[2],
    );
    if (jump > 1.2) return;
    state.last = next;
    onMoveObject(state.id, next);
  }, [onMoveObject]);
  const handleDragEnd = useCallback(() => {
    dragState.current = undefined;
    setDragId(undefined);
  }, []);
  useEffect(() => {
    // Install these permanently rather than waiting for DragPlane to mount:
    // a quick press/release can otherwise miss pointerup and strand drag mode.
    const handleVisibilityChange = () => {
      if (document.hidden) handleDragEnd();
    };
    window.addEventListener("pointerup", handleDragEnd);
    window.addEventListener("pointercancel", handleDragEnd);
    window.addEventListener("blur", handleDragEnd);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("pointerup", handleDragEnd);
      window.removeEventListener("pointercancel", handleDragEnd);
      window.removeEventListener("blur", handleDragEnd);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [handleDragEnd]);
  const activeId = selectedId ?? hoveredId;
  const benchmarkTarget = useMemo(() => new Vector3(1.5, 1.5, -11), []);
  const requestedTarget = useMemo(
    () => viewTarget ? new Vector3(...viewTarget) : roomTarget,
    [viewTarget],
  );
  const activeTarget = benchmarkView ? benchmarkTarget : requestedTarget;
  const orbitTarget = [activeTarget.x, activeTarget.y, activeTarget.z] as const;
  const visibleRepairs = originalProposals.filter((original) => {
    const current = proposals.find((proposal) => proposal.id === original.id);
    return current
      && transformChanged(current, original)
      && original.repair_patch?.status === "ready";
  });

  // reset the loading banner only when the reference actually changes —
  // native views share one cloud.ply, so layer toggles keep the same uri
  // and no loader onLoad would ever clear a spurious reset
  const loadedUri = useRef<string>(undefined);
  useEffect(() => {
    if (loadedUri.current === referenceUri) return;
    loadedUri.current = referenceUri;
    setReady(showReference === false && !secondaryReferenceUri);
  }, [referenceUri, secondaryReferenceUri, showReference]);
  return (
    <>
      <Canvas
        camera={{ fov: 48, near: 0.05, far: 150, position: [5.5, 2.5, 5.5] }}
        dpr={isNavigating ? 0.75 : 1}
        frameloop="demand"
        gl={{ antialias: false }}
        onPointerMissed={() => onSelect(undefined)}
      >
        <color attach="background" args={["#0c0a09"]} />
        <ambientLight intensity={1.7} />
        <directionalLight position={[3, 6, 4]} intensity={2.2} />
        <Suspense fallback={null}>
          {collisionDebug && architecture && (
            <CollisionArchitecture architecture={architecture} onLoad={handleLoad} />
          )}
          {showShell && shellUri && (
            <CleanShell
              uri={shellUri}
              collisionDebug={collisionDebug}
              onLoad={handleLoad}
            />
          )}
          {visibleRepairs.map((proposal) => (
            <RepairAppearance key={proposal.id} patch={proposal.repair_patch!} />
          ))}
          {showEntities && (
            <MeasuredObjectMeshes
              proposals={proposals}
              activeId={activeId}
              showObservedPoints={showObjectEvidence}
              collisionDebug={collisionDebug}
              walls={walls}
              onHover={setHoveredId}
              onSelect={onSelect}
              draggingId={dragId}
              onDragStart={onMoveObject ? handleDragStart : undefined}
            />
          )}
          {dragId && dragState.current && (
            <DragPlane
              position={dragState.current.planePosition}
              quaternion={dragState.current.planeQuaternion}
              onDragMove={handleDragMove}
              onDragEnd={handleDragEnd}
            />
          )}
          {showReference !== false && (
          <group position={referenceOffset ?? [0, 0, 0]}>
          {referenceKind === "mesh" ? (
            referenceUri.endsWith(".glb") ? (
              <RoomGlb
                uri={referenceUri}
                sourceZUp={referenceZUp}
                sourceYDown={referenceYDown}
                normalColors={normalColors}
                cull={referenceCull}
                color={benchmarkView ? "#f59e0b" : undefined}
                onLoad={handleLoad}
              />
            ) : referenceUri.endsWith(".ply") ? (
              <PointCloud
                uri={referenceUri}
                sourceZUp={referenceZUp}
                sourceYDown={referenceYDown}
                onLoad={handleLoad}
              />
            ) : (
              <RoomMesh
                uri={referenceUri}
                proposals={proposals}
                originals={originalProposals}
                activeId={activeId}
                onLoad={handleLoad}
              />
            )
          ) : (
            <RoomSplat uri={referenceUri} onLoad={handleLoad} />
          )}
          </group>
          )}
          {secondaryReferenceUri && (
          <group position={secondaryReferenceOffset ?? [0, 0, 0]}>
          {secondaryReferenceUri.endsWith(".glb") ? (
              <RoomGlb
                uri={secondaryReferenceUri}
                sourceZUp={secondaryReferenceZUp}
                sourceYDown={secondaryReferenceYDown}
                normalColors={false}
                cull={secondaryReferenceCull}
                onLoad={handleLoad}
              />
            ) : secondaryReferenceUri.endsWith(".ply") ? (
              <PointCloud
                uri={secondaryReferenceUri}
                sourceZUp={secondaryReferenceZUp}
                sourceYDown={secondaryReferenceYDown}
                onLoad={handleLoad}
              />
            ) : (
              <RoomMesh
                uri={secondaryReferenceUri}
                proposals={proposals}
                originals={originalProposals}
                activeId={activeId}
                onLoad={handleLoad}
              />
            )}
          </group>
          )}
          {overlayUri && (
            <PointCloud
              uri={overlayUri}
              sourceZUp
              sourceYDown={false}
              onLoad={handleLoad}
            />
          )}
          {comparisonUri && (
            <RoomGlb
              uri={comparisonUri}
              sourceZUp={comparisonZUp}
              sourceYDown={false}
              color="#67e8f9"
              opacity={0.24}
              onLoad={handleLoad}
            />
          )}
        </Suspense>
        <InitialCamera target={activeTarget} />
        <ViewBasisReporter onChange={onViewBasisChange} />
        <OrbitControls
          makeDefault
          enabled={!dragId}
          target={orbitTarget}
          enablePan
          mouseButtons={{ LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN }}
          minDistance={0.5}
          maxDistance={50}
          minPolarAngle={0.2}
          maxPolarAngle={2.6}
          onStart={() => setIsNavigating(true)}
          onEnd={() => setIsNavigating(false)}
        />
      </Canvas>
      {!ready && (
        <div className="absolute left-1/2 top-4 z-10 -translate-x-1/2 rounded-lg border border-white/10 bg-stone-900/90 px-3 py-2 text-xs text-stone-400 shadow-sm backdrop-blur">
          Loading real room…
        </div>
      )}
    </>
  );
}
