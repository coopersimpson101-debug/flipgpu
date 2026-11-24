import { describe, expect, it } from 'vitest';
import { substepsForFrame } from '../src/sim/timestep';

describe('CFL substepping', () => {
  it('uses the minimum when the fluid is at rest', () => {
    expect(substepsForFrame(1 / 60, 0, 0.05, 1)).toBe(1);
    expect(substepsForFrame(1 / 60, 0, 0.05, 1, 2)).toBe(2);
  });
  it('keeps per-substep displacement under C*dx (with safety factor) until the cap', () => {
    const dx = 0.04, cfl = 1, fdt = 1 / 60;
    for (const v of [0.5, 2, 5, 9]) {
      const n = substepsForFrame(fdt, v, dx, cfl, 1, 100);
      expect((v * fdt) / n).toBeLessThanOrEqual((cfl * dx) / 1.25 + 1e-12);
      expect((v * fdt) / (n - 1 || 1e-9)).toBeGreaterThan((cfl * dx) / 1.25 - 1e-12);
    }
    expect(substepsForFrame(fdt, 1000, dx, cfl, 1, 8)).toBe(8);
  });
});
