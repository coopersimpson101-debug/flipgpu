/**
 * SPH smoothing kernels and a CPU reference of Position Based Fluids (Macklin & Mueller 2013).
 * `src/gpu/shaders/pbf.wgsl` implements the same algorithm on the GPU.
 *
 * All kernels have compact support r < h and satisfy  integral W(r) dV = 1.
 */
import { countingSortGrid } from './spatial';

const PI = Math.PI;

// ---------- 3D kernels ----------
export const poly6Coef3 = (h: number) => 315 / (64 * PI * h ** 9);
export const spikyCoef3 = (h: number) => 15 / (PI * h ** 6);
export const spikyGradCoef3 = (h: number) => -45 / (PI * h ** 6);

export function poly6(r: number, h: number) {
  if (r >= h) return 0;
  const d = h * h - r * r;
  return poly6Coef3(h) * d * d * d;
}
export function spiky(r: number, h: number) {
  if (r >= h) return 0;
  const d = h - r;
  return spikyCoef3(h) * d * d * d;
}
/** dW_spiky/dr (scalar, <= 0); the gradient vector is this times r_hat. */
export function spikyDeriv(r: number, h: number) {
  if (r >= h) return 0;
  const d = h - r;
  return spikyGradCoef3(h) * d * d;
}
/** Mueller et al. 2003 viscosity kernel and its Laplacian. */
export function viscosityKernel(r: number, h: number) {
  if (r >= h || r === 0) return 0;
  return (15 / (2 * PI * h ** 3)) * (-(r ** 3) / (2 * h ** 3) + (r * r) / (h * h) + h / (2 * r) - 1);
}
export function viscosityLaplacian(r: number, h: number) {
  if (r >= h) return 0;
  return (45 / (PI * h ** 6)) * (h - r);
}
/** Monaghan cubic B-spline with support radius h (q = r/h in [0,1]). */
export function cubicSpline3(r: number, h: number) {
  const q = r / h;
  const s = 8 / (PI * h ** 3);
  if (q <= 0.5) return s * (6 * (q ** 3 - q ** 2) + 1);
  if (q <= 1) return s * 2 * (1 - q) ** 3;
  return 0;
}

// ---------- 2D kernels ----------
export function poly6_2d(r: number, h: number) {
  if (r >= h) return 0;
  return (4 / (PI * h ** 8)) * (h * h - r * r) ** 3;
}
export function spiky2d(r: number, h: number) {
  if (r >= h) return 0;
  return (10 / (PI * h ** 5)) * (h - r) ** 3;
}
export function cubicSpline2(r: number, h: number) {
  const q = r / h;
  const s = 40 / (7 * PI * h * h);
  if (q <= 0.5) return s * (6 * (q ** 3 - q ** 2) + 1);
  if (q <= 1) return s * 2 * (1 - q) ** 3;
  return 0;
}

/** Composite Simpson integration of f over [a,b] with n (even) intervals. */
export function simpson(f: (x: number) => number, a: number, b: number, n = 2000) {
  if (n % 2) n++;
  const dx = (b - a) / n;
  let s = f(a) + f(b);
  for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * f(a + i * dx);
  return (s * dx) / 3;
}
/** integral over R^3 of a radial function = int_0^h 4 pi r^2 W(r) dr */
export const integrate3D = (W: (r: number) => number, h: number) => simpson((r) => 4 * PI * r * r * W(r), 0, h);
/** integral over R^2 of a radial function = int_0^h 2 pi r W(r) dr */
export const integrate2D = (W: (r: number) => number, h: number) => simpson((r) => 2 * PI * r * W(r), 0, h);

/** Rest density (unit particle mass) of an infinite simple-cubic lattice of spacing s, poly6 kernel radius h. */
export function latticeRestDensity(s: number, h: number) {
  const m = Math.ceil(h / s);
  let rho = 0;
  for (let k = -m; k <= m; k++)
    for (let j = -m; j <= m; j++)
      for (let i = -m; i <= m; i++) rho += poly6(s * Math.hypot(i, j, k), h);
  return rho;
}

// ---------- Position Based Fluids, CPU reference ----------
export interface PBFParams {
  h: number; // kernel radius
  restDensity: number; // for unit mass
  iterations: number;
  /** constraint-force-mixing relaxation, dimensionless (scaled by 1/h^2 internally) */
  relaxation: number;
  /** artificial-pressure strength; multiplied by h^2 so it is commensurate with lambda (m^2) */
  sCorrK: number;
  sCorrDeltaQ: number; // as fraction of h
  sCorrN: number;
  xsph: number;
  gravity: [number, number, number];
  box: [number, number, number];
}

export class PBF3D {
  n: number;
  x: Float64Array; v: Float64Array; xp: Float64Array;
  lambda: Float64Array; density: Float64Array;
  constructor(public p: PBFParams, positions: ArrayLike<number>) {
    this.n = positions.length / 3;
    this.x = Float64Array.from(positions);
    this.xp = Float64Array.from(positions);
    this.v = new Float64Array(positions.length);
    this.lambda = new Float64Array(this.n);
    this.density = new Float64Array(this.n);
  }

  private neighbours() {
    const { h, box } = this.p;
    const g = box.map((b) => Math.max(1, Math.ceil(b / h))) as [number, number, number];
    const cc = (k: number) => [0, 1, 2].map((a) => Math.min(g[a] - 1, Math.max(0, Math.floor(this.xp[3 * k + a] / h)))) as [number, number, number];
    const grid = countingSortGrid(this.n, g[0], g[1], g[2], cc);
    return (i: number, fn: (j: number, dx: number, dy: number, dz: number, r: number) => void) => {
      const [ci, cj, ck] = cc(i);
      for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const a = ci + di, b = cj + dj, c = ck + dk;
        if (a < 0 || b < 0 || c < 0 || a >= g[0] || b >= g[1] || c >= g[2]) continue;
        const cell = a + g[0] * (b + g[1] * c);
        for (let s = grid.cellStart[cell]; s < grid.cellStart[cell + 1]; s++) {
          const j = grid.sorted[s];
          const dx = this.xp[3 * i] - this.xp[3 * j], dy = this.xp[3 * i + 1] - this.xp[3 * j + 1], dz = this.xp[3 * i + 2] - this.xp[3 * j + 2];
          const r = Math.hypot(dx, dy, dz);
          if (r < h) fn(j, dx, dy, dz, r);
        }
      }
    };
  }

  private clampBox(k: number) {
    for (let a = 0; a < 3; a++) {
      const lo = 1e-4, hi = this.p.box[a] - 1e-4;
      if (this.xp[3 * k + a] < lo) this.xp[3 * k + a] = lo;
      if (this.xp[3 * k + a] > hi) this.xp[3 * k + a] = hi;
    }
  }

  /** mean of max(0, rho_i/rho0 - 1) -- the compressive density error */
  densityError() {
    let e = 0;
    for (let i = 0; i < this.n; i++) e += Math.max(0, this.density[i] / this.p.restDensity - 1);
    return e / this.n;
  }

  computeDensities(each = this.neighbours()) {
    const { h } = this.p;
    for (let i = 0; i < this.n; i++) {
      let rho = 0;
      each(i, (_j, _dx, _dy, _dz, r) => { rho += poly6(r, h); });
      this.density[i] = rho;
    }
  }

  step(dt: number): { errBefore: number; errAfter: number } {
    const { h, restDensity: rho0, gravity } = this.p;
    const n = this.n;
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {
      this.v[3 * i + a] += gravity[a] * dt;
      this.xp[3 * i + a] = this.x[3 * i + a] + this.v[3 * i + a] * dt;
    }
    for (let i = 0; i < n; i++) this.clampBox(i);
    const each = this.neighbours();
    const eps = this.p.relaxation / (h * h);
    const wdq = poly6(this.p.sCorrDeltaQ * h, h);
    const dP = new Float64Array(3 * n);
    let errBefore = 0;
    for (let it = 0; it < this.p.iterations; it++) {
      // lambda_i = -C_i / (sum_k |grad_k C_i|^2 + eps)
      for (let i = 0; i < n; i++) {
        let rho = 0, gx = 0, gy = 0, gz = 0, sum2 = 0;
        each(i, (j, dx, dy, dz, r) => {
          rho += poly6(r, h);
          if (j !== i && r > 1e-9) {
            const s = spikyDeriv(r, h) / (r * rho0);
            const ax = s * dx, ay = s * dy, az = s * dz;
            gx += ax; gy += ay; gz += az;
            sum2 += ax * ax + ay * ay + az * az;
          }
        });
        this.density[i] = rho;
        // unilateral constraint: only compression is corrected (avoids artificial surface attraction)
        const C = Math.max(rho / rho0 - 1, 0);
        this.lambda[i] = -C / (sum2 + gx * gx + gy * gy + gz * gz + eps);
      }
      if (it === 0) errBefore = this.densityError();
      // delta p_i = 1/rho0 sum_j (lambda_i + lambda_j + s_corr) grad W(p_i - p_j)
      for (let i = 0; i < n; i++) {
        let ex = 0, ey = 0, ez = 0;
        each(i, (j, dx, dy, dz, r) => {
          if (j === i || r < 1e-9) return;
          const sc = -this.p.sCorrK * h * h * (poly6(r, h) / wdq) ** this.p.sCorrN; // lambda has units m^2
          const s = ((this.lambda[i] + this.lambda[j] + sc) * spikyDeriv(r, h)) / (r * rho0);
          ex += s * dx; ey += s * dy; ez += s * dz;
        });
        dP[3 * i] = ex; dP[3 * i + 1] = ey; dP[3 * i + 2] = ez;
      }
      for (let k = 0; k < 3 * n; k++) this.xp[k] += dP[k];
      for (let i = 0; i < n; i++) this.clampBox(i);
    }
    this.computeDensities(this.neighbours());
    const errAfter = this.densityError();
    // velocity update + XSPH viscosity
    for (let k = 0; k < 3 * n; k++) this.v[k] = (this.xp[k] - this.x[k]) / dt;
    if (this.p.xsph > 0) {
      const each2 = this.neighbours();
      const vNew = Float64Array.from(this.v);
      for (let i = 0; i < n; i++)
        each2(i, (j, _dx, _dy, _dz, r) => {
          if (j === i) return;
          const w = (this.p.xsph * poly6(r, h)) / rho0;
          for (let a = 0; a < 3; a++) vNew[3 * i + a] += w * (this.v[3 * j + a] - this.v[3 * i + a]);
        });
      this.v = vNew;
    }
    this.x.set(this.xp);
    return { errBefore, errAfter };
  }
}
