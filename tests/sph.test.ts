import { describe, expect, it } from 'vitest';
import {
  PBF3D, cubicSpline2, cubicSpline3, integrate2D, integrate3D, latticeRestDensity, poly6, poly6_2d,
  spiky, spiky2d, spikyDeriv, viscosityLaplacian,
} from '../src/cpu/sph';

describe('SPH kernel normalisation', () => {
  for (const h of [0.05, 0.3, 1.7]) {
    it(`3D kernels integrate to 1 (h=${h})`, () => {
      expect(integrate3D((r) => poly6(r, h), h)).toBeCloseTo(1, 6);
      expect(integrate3D((r) => spiky(r, h), h)).toBeCloseTo(1, 6);
      expect(integrate3D((r) => cubicSpline3(r, h), h)).toBeCloseTo(1, 6);
    });
    it(`2D kernels integrate to 1 (h=${h})`, () => {
      expect(integrate2D((r) => poly6_2d(r, h), h)).toBeCloseTo(1, 6);
      expect(integrate2D((r) => spiky2d(r, h), h)).toBeCloseTo(1, 6);
      expect(integrate2D((r) => cubicSpline2(r, h), h)).toBeCloseTo(1, 6);
    });
  }

  it('spiky gradient is the derivative of spiky (finite differences) and vanishes at the support', () => {
    const h = 0.4;
    for (const r of [0.05, 0.13, 0.27, 0.39]) {
      const fd = (spiky(r + 1e-6, h) - spiky(r - 1e-6, h)) / 2e-6;
      expect(spikyDeriv(r, h) / fd).toBeCloseTo(1, 5);
    }
    expect(spiky(h, h)).toBe(0);
    expect(spikyDeriv(h, h)).toBe(0);
    expect(viscosityLaplacian(h, h)).toBe(0);
    expect(viscosityLaplacian(0.1, h)).toBeGreaterThan(0);
  });

  it('lattice rest density approaches the continuum number density 1/s^3 as h/s grows', () => {
    const s = 0.1;
    const rel = (k: number) => latticeRestDensity(s, k * s) * s ** 3;
    expect(Math.abs(rel(4) - 1)).toBeLessThan(0.01);
    expect(Math.abs(rel(2) - 1)).toBeLessThan(0.1);
  });
});

describe('Position Based Fluids (CPU reference)', () => {
  function block(n: number, s: number, offset = 0.0) {
    const pos: number[] = [];
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++)
      pos.push((i + 0.5) * s + offset, (j + 0.5) * s, (k + 0.5) * s + offset);
    return pos;
  }
  const s = 0.05, h = 2 * s;
  const params = {
    h, restDensity: latticeRestDensity(s, h), iterations: 4, relaxation: 0.5,
    sCorrK: 0.01, sCorrDeltaQ: 0.2, sCorrN: 4, xsph: 0.01,
    gravity: [0, -9.81, 0] as [number, number, number], box: [0.6, 0.8, 0.6] as [number, number, number],
  };

  it('constraint iterations reduce the compressive density error', () => {
    const pos = block(8, s * 0.85, 0.05); // 15% compressed in each axis
    const sim = new PBF3D({ ...params, gravity: [0, 0, 0] }, pos);
    const { errBefore, errAfter } = sim.step(1 / 120);
    expect(errBefore).toBeGreaterThan(0.2);
    expect(errAfter).toBeLessThan(errBefore * 0.5);
  });

  it('a falling block settles inside the box without blowing up and conserves particles', () => {
    const pos = block(8, s, 0.1);
    const sim = new PBF3D(params, pos);
    const n0 = sim.n;
    for (let i = 0; i < 90; i++) sim.step(1 / 120);
    expect(sim.n).toBe(n0);
    let maxV = 0, minY = Infinity;
    for (let i = 0; i < sim.n; i++) {
      for (let a = 0; a < 3; a++) {
        expect(Number.isFinite(sim.x[3 * i + a])).toBe(true);
        expect(sim.x[3 * i + a]).toBeGreaterThanOrEqual(0);
        expect(sim.x[3 * i + a]).toBeLessThanOrEqual(params.box[a]);
      }
      maxV = Math.max(maxV, Math.hypot(sim.v[3 * i], sim.v[3 * i + 1], sim.v[3 * i + 2]));
      minY = Math.min(minY, sim.x[3 * i + 1]);
    }
    expect(maxV).toBeLessThan(5);
    expect(minY).toBeLessThan(0.01);
    sim.computeDensities();
    // after settling the fluid should be close to its rest density (weakly compressible at most ~10%)
    let mean = 0;
    for (let i = 0; i < sim.n; i++) mean += sim.density[i];
    mean /= sim.n;
    expect(mean / params.restDensity).toBeLessThan(1.1);
  });

  it('dam break: no particle exceeds the free-fall speed bound sqrt(2 g H) (no artificial energy gain)', { timeout: 120000 }, () => {
    const pos: number[] = [];
    for (let k = 0; k < 6; k++) for (let j = 0; j < 16; j++) for (let i = 0; i < 8; i++) pos.push((i + 0.5) * s, (j + 0.5) * s, (k + 0.5) * s);
    const H = 16 * s;
    const sim = new PBF3D({ ...params, box: [1.6, 1.0, 0.3] }, pos);
    let vmax = 0;
    for (let t = 0; t < 1.0; t += 1 / 120) {
      sim.step(1 / 120);
      for (let p = 0; p < sim.n; p++) vmax = Math.max(vmax, Math.hypot(sim.v[3 * p], sim.v[3 * p + 1], sim.v[3 * p + 2]));
    }
    expect(vmax).toBeLessThan(1.1 * Math.sqrt(2 * 9.81 * H));
  });
});
