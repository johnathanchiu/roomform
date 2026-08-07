import {
  BufferGeometry,
  Quaternion,
  Uint32BufferAttribute,
  Vector3,
} from "three";

import type { SplatObjectProposal } from "@roomform/scene/selection";

function geometryWithIndices(source: BufferGeometry, indices: number[]) {
  const geometry = new BufferGeometry();
  for (const name of Object.keys(source.attributes)) {
    geometry.setAttribute(name, source.getAttribute(name));
  }
  geometry.setIndex(new Uint32BufferAttribute(indices, 1));
  return geometry;
}

export function splitGeometry(source: BufferGeometry, proposal?: SplatObjectProposal) {
  if (!proposal) return { background: source, object: undefined };
  const positions = source.getAttribute("position");
  const index = source.getIndex();
  const triangleCount = index ? index.count / 3 : positions.count / 3;
  const objectIndices: number[] = [];
  const backgroundIndices: number[] = [];
  const center = new Vector3(...proposal.position);
  const inverseRotation = new Quaternion(...proposal.rotation_xyzw).invert();
  const half = new Vector3(...proposal.bounds).multiplyScalar(0.54);
  const vertices = [new Vector3(), new Vector3(), new Vector3()];

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const indices = [
      index ? index.getX(triangle * 3) : triangle * 3,
      index ? index.getX(triangle * 3 + 1) : triangle * 3 + 1,
      index ? index.getX(triangle * 3 + 2) : triangle * 3 + 2,
    ];
    const belongsToObject = vertices.every((vertex, vertexIndex) => {
      const positionIndex = indices[vertexIndex];
      vertex
        .set(
          positions.getX(positionIndex),
          positions.getY(positionIndex),
          positions.getZ(positionIndex),
        )
        .sub(center)
        .applyQuaternion(inverseRotation);
      return Math.abs(vertex.x) <= half.x
        && Math.abs(vertex.y) <= half.y
        && Math.abs(vertex.z) <= half.z;
    });
    const target = belongsToObject ? objectIndices : backgroundIndices;
    target.push(...indices);
  }
  return {
    background: geometryWithIndices(source, backgroundIndices),
    object: geometryWithIndices(source, objectIndices),
  };
}

export function splitGeometryByProposals(
  source: BufferGeometry,
  proposals: SplatObjectProposal[],
) {
  const positions = source.getAttribute("position");
  const index = source.getIndex();
  const triangleCount = index ? index.count / 3 : positions.count / 3;
  const backgroundIndices: number[] = [];
  const objectIndices = new Map(proposals.map((proposal) => [proposal.id, [] as number[]]));
  const volumes = proposals.map((proposal) => ({
    proposal,
    center: new Vector3(...proposal.position),
    inverseRotation: new Quaternion(...proposal.rotation_xyzw).invert(),
    half: new Vector3(...proposal.bounds).multiplyScalar(0.54),
  }));
  const vertices = [new Vector3(), new Vector3(), new Vector3()];

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const indices = [
      index ? index.getX(triangle * 3) : triangle * 3,
      index ? index.getX(triangle * 3 + 1) : triangle * 3 + 1,
      index ? index.getX(triangle * 3 + 2) : triangle * 3 + 2,
    ];
    const owner = volumes.find(({ center, inverseRotation, half }) =>
      vertices.every((vertex, vertexIndex) => {
        const positionIndex = indices[vertexIndex];
        vertex
          .set(
            positions.getX(positionIndex),
            positions.getY(positionIndex),
            positions.getZ(positionIndex),
          )
          .sub(center)
          .applyQuaternion(inverseRotation);
        return Math.abs(vertex.x) <= half.x
          && Math.abs(vertex.y) <= half.y
          && Math.abs(vertex.z) <= half.z;
      }),
    );
    (owner ? objectIndices.get(owner.proposal.id) : backgroundIndices)?.push(...indices);
  }

  return {
    background: geometryWithIndices(source, backgroundIndices),
    objects: new Map(
      [...objectIndices].map(([id, indices]) => [id, geometryWithIndices(source, indices)]),
    ),
  };
}
