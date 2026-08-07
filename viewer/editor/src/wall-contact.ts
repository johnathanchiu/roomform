// Object-vs-wall interference from the scene program's wall statements.
// A wall is a vertical plane over a 2D segment; an object interferes when its
// footprint STRADDLES that line by more than a tolerance. Flush placement
// (bed against a wall) keeps every corner on one side and never flags.

export type WallSegment = { a: [number, number]; b: [number, number] };

export function wallSegmentsFromProgram(program: {
  statements: Array<{ kind: string; from?: number[]; to?: number[] }>;
}): WallSegment[] {
  return program.statements
    .filter((s) => s.kind === "vertical" && s.from && s.to)
    .map((s) => ({
      a: [s.from![0], s.from![1]] as [number, number],
      b: [s.to![0], s.to![1]] as [number, number],
    }));
}

// position is editor-frame [x, height, z] where world xy = (x, -z);
// bounds are Z-up extents [dx, dy, dz] (dz = height).
export function objectIntersectsWall(
  position: [number, number, number],
  bounds: [number, number, number],
  walls: WallSegment[],
  // 8cm: measured on bedroom fixtures — flush wall-adjacent furniture has
  // ≤6cm of AABB slop past the snapped shell line; real clip-throughs
  // (curtain, over-deep desk) start at 10cm+
  tolerance = 0.08,
): boolean {
  const cx = position[0];
  const cy = -position[2];
  const hx = bounds[0] / 2;
  const hy = bounds[1] / 2;
  const corners: Array<[number, number]> = [
    [cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy],
  ];
  for (const wall of walls) {
    const dx = wall.b[0] - wall.a[0];
    const dy = wall.b[1] - wall.a[1];
    const length = Math.hypot(dx, dy);
    if (length < 1e-6) continue;
    const ux = dx / length;
    const uy = dy / length;
    let minSide = Infinity;
    let maxSide = -Infinity;
    let minU = Infinity;
    let maxU = -Infinity;
    for (const [px, py] of corners) {
      const rx = px - wall.a[0];
      const ry = py - wall.a[1];
      const side = rx * -uy + ry * ux;
      const u = rx * ux + ry * uy;
      minSide = Math.min(minSide, side);
      maxSide = Math.max(maxSide, side);
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
    }
    // straddles the line beyond tolerance AND overlaps the segment's span
    if (minSide < -tolerance && maxSide > tolerance && maxU > 0 && minU < length) {
      return true;
    }
  }
  return false;
}
