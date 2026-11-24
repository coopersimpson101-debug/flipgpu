/** Scene definitions and particle seeding shared by both GPU solvers (pure TS, unit-tested). */
import { mulberry32 } from '../cpu/flip2d';

export type SceneName = 'dam' | 'double' | 'drop';
export type Vec3 = [number, number, number];

/** Nominal tank interior in metres. */
export const TANK: Vec3 = [3.0, 1.6, 1.2];

interface Region { inside(p: Vec3): boolean; min: Vec3; max: Vec3; volume: number }

function boxRegion(min: Vec3, max: Vec3): Region {
  return {
    min, max,
    volume: (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]),
    inside: (p) => p[0] >= min[0] && p[0] < max[0] && p[1] >= min[1] && p[1] < max[1] && p[2] >= min[2] && p[2] < max[2],
  };
}
function sphereRegion(c: Vec3, r: number): Region {
  return {
    min: [c[0] - r, c[1] - r, c[2] - r], max: [c[0] + r, c[1] + r, c[2] + r],
    volume: (4 / 3) * Math.PI * r ** 3,
    inside: (p) => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2 < r * r,
  };
}

export function sceneRegions(scene: SceneName, box: Vec3): Region[] {
  const [X, Y, Z] = box;
  switch (scene) {
    case 'dam':
      return [boxRegion([0, 0, 0], [0.38 * X, 0.8 * Y, Z])];
    case 'double':
      return [boxRegion([0, 0, 0], [0.24 * X, 0.75 * Y, Z]), boxRegion([0.76 * X, 0, 0], [X, 0.75 * Y, Z])];
    case 'drop': {
      const r = 0.2 * Math.min(X, Y, Z) * 1.4;
      return [boxRegion([0, 0, 0], [X, 0.22 * Y, Z]), sphereRegion([0.5 * X, 0.68 * Y, 0.5 * Z], r)];
    }
  }
}

export function sceneVolume(scene: SceneName, box: Vec3) {
  return sceneRegions(scene, box).reduce((s, r) => s + r.volume, 0);
}

/** Lattice spacing that yields approximately `target` particles for the scene. */
export function spacingForCount(scene: SceneName, box: Vec3, target: number) {
  return Math.cbrt(sceneVolume(scene, box) / target);
}

/**
 * Fills the scene regions with a simple-cubic lattice of spacing `s` (optionally jittered by
 * `jitter * s`). Returns xyzw-packed positions ready for upload (w = 1).
 */
export function seedScene(scene: SceneName, box: Vec3, s: number, jitter = 0, seed = 7, margin = 0): Float32Array<ArrayBuffer> {
  const rng = mulberry32(seed);
  const regions = sceneRegions(scene, box);
  const out: number[] = [];
  const nx = Math.floor(box[0] / s), ny = Math.floor(box[1] / s), nz = Math.floor(box[2] / s);
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const p: Vec3 = [(i + 0.5) * s, (j + 0.5) * s, (k + 0.5) * s];
        if (!regions.some((r) => r.inside(p))) continue;
        for (let a = 0; a < 3; a++) {
          const q = p[a] + jitter * s * (rng() - 0.5);
          p[a] = Math.min(box[a] - margin, Math.max(margin, q));
        }
        out.push(p[0], p[1], p[2], 1);
      }
  return new Float32Array(out);
}

export interface FlipGrid { nx: number; ny: number; nz: number; h: number; box: Vec3 }

/**
 * MAC grid for the FLIP solver: interior of the tank is resolved with cell size ~h, plus one ring of
 * solid cells. The returned `box` is the exactly-gridded interior (within h/2 of the nominal tank).
 */
export function makeFlipGrid(tank: Vec3, targetH: number): FlipGrid {
  const ix = Math.max(4, Math.round(tank[0] / targetH));
  const h = tank[0] / ix;
  const iy = Math.max(4, Math.round(tank[1] / h));
  const iz = Math.max(4, Math.round(tank[2] / h));
  return { nx: ix + 2, ny: iy + 2, nz: iz + 2, h, box: [ix * h, iy * h, iz * h] };
}
