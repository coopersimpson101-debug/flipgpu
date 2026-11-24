/**
 * Adaptive substepping from the CFL condition  dt <= C * dx / |u|_max.
 * The GPU reports |u|_max one frame late (async readback), which is why a safety factor is applied.
 */
export function substepsForFrame(frameDt: number, maxSpeed: number, dx: number, cfl: number,
  minSteps = 1, maxSteps = 8, safety = 1.25): number {
  if (!(maxSpeed > 0) || !(dx > 0) || !(cfl > 0)) return minSteps;
  const need = Math.ceil((frameDt * maxSpeed * safety) / (cfl * dx));
  return Math.min(maxSteps, Math.max(minSteps, need));
}
