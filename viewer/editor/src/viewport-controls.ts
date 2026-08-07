export type ViewBasis = {
  right: [number, number, number];
  forward: [number, number, number];
};

export function cameraRelativeMovement(
  basis: ViewBasis,
  screenDelta: [number, number],
  step = 0.1,
): [number, number, number] {
  return basis.right.map((value, index) => (
    (value * screenDelta[0] + basis.forward[index] * screenDelta[1]) * step
  )) as [number, number, number];
}
