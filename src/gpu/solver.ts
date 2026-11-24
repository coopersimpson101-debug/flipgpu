import type { SceneName, Vec3 } from '../sim/scenes';

export type SolverKind = 'flip' | 'pbf';

/** Live-tunable parameters (written into uniform buffers once per frame). */
export interface SimSettings {
  flipRatio: number;
  /** FLIP: kinematic viscosity nu [m^2/s];  PBF: XSPH coefficient c (both 0..1 slider) */
  viscosity: number;
  gravity: number;
  /** FLIP: Jacobi / RB-SOR iterations;  PBF: constraint iterations (scaled down) */
  pressureIters: number;
  pressureSolver: 'jacobi' | 'rbgs';
  sorOmega: number;
  densityCorrection: number;
  pushOutIters: number;
  /** extra body acceleration from shaking the tank (fictitious force -a_tank) */
  tankAccel: Vec3;
  mouse: { pos: Vec3; vel: Vec3; radius: number; strength: number };
}

export interface Solver {
  readonly kind: SolverKind;
  readonly scene: SceneName;
  readonly numParticles: number;
  readonly box: Vec3;
  /** length scale for the CFL condition (FLIP: cell size; PBF: particle spacing) */
  readonly cflLength: number;
  readonly renderRadius: number;
  readonly pos: GPUBuffer;
  readonly vel: GPUBuffer;
  /** stats[0] = max particle speed as f32 bits (atomicMax) */
  readonly stats: GPUBuffer;
  writeUniforms(dt: number, s: SimSettings): void;
  encodeSubstep(pass: GPUComputePassEncoder, s: SimSettings): void;
  describe(): string;
  destroy(): void;
}
