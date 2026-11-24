import { describe, expect, it } from 'vitest';
import { blockedExclusiveScan, countingSortGrid, exclusiveScanInPlace, scanLevels } from '../src/cpu/spatial';
import { mulberry32 } from '../src/cpu/flip2d';
import { TANK, makeFlipGrid, sceneVolume, seedScene, spacingForCount, type SceneName } from '../src/sim/scenes';
import { invert, lookAt, multiply, perspective, rayPlane, screenRay, transformPoint } from '../src/math';

describe('prefix scan', () => {
  it('blocked (GPU-structured) scan equals sequential scan for many sizes', () => {
    const rng = mulberry32(1);
    for (const n of [1, 7, 255, 256, 257, 1000, 65536, 65537, 70001]) {
      const a = new Uint32Array(n);
      for (let i = 0; i < n; i++) a[i] = Math.floor(rng() * 9);
      const ref = Uint32Array.from(a);
      exclusiveScanInPlace(ref);
      expect(Array.from(blockedExclusiveScan(a, 256))).toEqual(Array.from(ref));
    }
  });
  it('scan level sizes', () => {
    expect(scanLevels(100)).toEqual([100]);
    expect(scanLevels(70000)).toEqual([70000, 274, 2]);
  });
});

describe('counting-sort uniform grid', () => {
  it('finds exactly the brute-force neighbour set', () => {
    const rng = mulberry32(4);
    const n = 800, L = 1, r = 0.08;
    const pts = Array.from({ length: n }, () => [rng() * L, rng() * L, rng() * L]);
    const g = Math.ceil(L / r);
    const cc = (p: number) => pts[p].map((x) => Math.min(g - 1, Math.floor(x / r))) as [number, number, number];
    const grid = countingSortGrid(n, g, g, g, cc);
    expect(grid.cellStart[g * g * g]).toBe(n);
    expect(new Set(grid.sorted).size).toBe(n);
    for (let i = 0; i < n; i += 37) {
      const brute = new Set<number>();
      for (let j = 0; j < n; j++) if (Math.hypot(...pts[i].map((x, a) => x - pts[j][a])) < r) brute.add(j);
      const found = new Set<number>();
      const [ci, cj, ck] = cc(i);
      for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const a = ci + di, b = cj + dj, c = ck + dk;
        if (a < 0 || b < 0 || c < 0 || a >= g || b >= g || c >= g) continue;
        const cell = a + g * (b + g * c);
        for (let s = grid.cellStart[cell]; s < grid.cellStart[cell + 1]; s++) {
          const j = grid.sorted[s];
          if (Math.hypot(...pts[i].map((x, a2) => x - pts[j][a2])) < r) found.add(j);
        }
      }
      expect([...found].sort()).toEqual([...brute].sort());
    }
  });
});

describe('scenes', () => {
  const scenes: SceneName[] = ['dam', 'double', 'drop'];
  for (const sc of scenes)
    it(`${sc}: seeds ~target particles inside the tank`, () => {
      const target = 40000;
      const s = spacingForCount(sc, TANK, target);
      const p = seedScene(sc, TANK, s, 0.5, 3);
      const n = p.length / 4;
      expect(Math.abs(n - target) / target).toBeLessThan(0.15);
      let outside = 0;
      for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) if (p[4 * i + a] < 0 || p[4 * i + a] > TANK[a]) outside++;
      expect(outside).toBe(0);
      expect(sceneVolume(sc, TANK)).toBeGreaterThan(0);
    });
  it('FLIP grid has a solid ring and near-nominal interior', () => {
    const g = makeFlipGrid(TANK, 0.05);
    expect(g.nx).toBe(62);
    for (let a = 0; a < 3; a++) expect(Math.abs(g.box[a] - TANK[a])).toBeLessThanOrEqual(g.h / 2 + 1e-12);
  });
});

describe('camera math', () => {
  it('perspective maps near/far planes to depth 0/1', () => {
    const P = perspective(1, 1.5, 0.1, 50);
    const n = transformPoint(P, [0, 0, -0.1]); const f = transformPoint(P, [0, 0, -50]);
    expect(n[2] / n[3]).toBeCloseTo(0, 6);
    expect(f[2] / f[3]).toBeCloseTo(1, 6);
  });
  it('lookAt puts the eye at the origin and the target on -z', () => {
    const V = lookAt([3, 2, 5], [1, 0.5, 0.2], [0, 1, 0]);
    const e = transformPoint(V, [3, 2, 5]);
    const t = transformPoint(V, [1, 0.5, 0.2]);
    expect(Math.hypot(e[0], e[1], e[2])).toBeLessThan(1e-6);
    expect(Math.abs(t[0]) + Math.abs(t[1])).toBeLessThan(1e-5);
    expect(t[2]).toBeLessThan(0);
  });
  it('inverse and screen rays round-trip', () => {
    const VP = multiply(perspective(0.8, 1.3, 0.05, 60), lookAt([4, 2, 3], [1, 0.5, 0.5], [0, 1, 0]));
    const inv = invert(VP)!;
    const I = multiply(VP, inv);
    for (let i = 0; i < 16; i++) expect(I[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 4);
    const ray = screenRay(inv, 0, 0);
    const t = rayPlane(ray.origin, ray.dir, [0, 1, 0], 0.5)!;
    const hit = ray.dir.map((d, a) => ray.origin[a] + t * d);
    expect(hit[0]).toBeCloseTo(1, 3);
    expect(hit[2]).toBeCloseTo(0.5, 3);
  });
});
