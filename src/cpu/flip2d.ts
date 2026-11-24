/**
 * CPU reference implementation of a 2D PIC/FLIP fluid on a MAC staggered grid.
 *
 * This file mirrors, in 2D and in double precision, the algorithm that the WebGPU
 * compute shaders in `src/gpu/shaders/flip.wgsl` run in 3D. It exists so that the
 * numerics (P2G/G2P transfers, cell marking, boundary conditions, pressure
 * projection) can be unit-tested on any machine without a GPU.
 *
 * Grid layout (cell size h, cells nx * ny, origin at (0,0)):
 *   - pressure / cell type at cell centres     ((i+1/2)h, (j+1/2)h)
 *   - u (x-velocity) on vertical faces          (i h,      (j+1/2)h)   (nx+1) * ny
 *   - v (y-velocity) on horizontal faces        ((i+1/2)h, j h)        nx * (ny+1)
 * The outermost ring of cells is SOLID (static walls).
 */
import { countingSortGrid } from './spatial';

export const FLUID = 0;
export const AIR = 1;
export const SOLID = 2;

export type PressureSolver = 'jacobi' | 'rbgs' | 'pcg';

export interface Flip2DParams {
  nx: number;
  ny: number;
  h: number;
  /** fluid density rho [kg/m^3] */
  density: number;
  /** gravitational acceleration (negative = down) [m/s^2] */
  gravityY: number;
  /** 0 = pure PIC, 1 = pure FLIP */
  flipRatio: number;
  solver: PressureSolver;
  pressureIters: number;
  /** relative residual tolerance for PCG */
  pcgTol: number;
  /** over-relaxation factor for red-black Gauss-Seidel (1 = plain GS) */
  sorOmega: number;
  /** particle radius used by the push-apart pass */
  particleRadius: number;
  pushApartIters: number;
  /** fraction of the volume error removed per step by the density correction (0 = off) */
  densityCorrection: number;
}

export function defaultParams(nx: number, ny: number, h: number): Flip2DParams {
  return {
    nx, ny, h,
    density: 1000,
    gravityY: -9.81,
    flipRatio: 0.95,
    solver: 'pcg',
    pressureIters: 200,
    pcgTol: 1e-10,
    sorOmega: 1.5,
    // 2 x 2 particles per cell => rest spacing h/2; push-apart diameter 2r must not exceed it
    particleRadius: 0.24 * h,
    pushApartIters: 2,
    densityCorrection: 0,
  };
}

export interface SolveInfo { iterations: number; residual: number }

/** CFL-limited timestep: a particle may not cross more than `cfl` cells per step. */
export function cflTimestep(maxSpeed: number, h: number, cfl: number, dtMax: number): number {
  if (!(maxSpeed > 1e-9)) return dtMax;
  return Math.min(dtMax, (cfl * h) / maxSpeed);
}

export class FlipSim2D {
  readonly p: Flip2DParams;
  readonly nx: number;
  readonly ny: number;
  readonly h: number;
  // face velocities and their weights / previous values (for FLIP)
  u: Float64Array; v: Float64Array;
  uPrev: Float64Array; vPrev: Float64Array;
  uW: Float64Array; vW: Float64Array;
  // cell quantities
  pressure: Float64Array; rhs: Float64Array;
  cellType: Uint8Array; particleDensity: Float64Array;
  restDensity = 0;
  // particles
  n = 0;
  px = new Float64Array(0); py = new Float64Array(0);
  vx = new Float64Array(0); vy = new Float64Array(0);
  lastSolve: SolveInfo = { iterations: 0, residual: 0 };

  constructor(params: Flip2DParams) {
    this.p = params;
    this.nx = params.nx; this.ny = params.ny; this.h = params.h;
    const nu = (this.nx + 1) * this.ny, nv = this.nx * (this.ny + 1), nc = this.nx * this.ny;
    this.u = new Float64Array(nu); this.uPrev = new Float64Array(nu); this.uW = new Float64Array(nu);
    this.v = new Float64Array(nv); this.vPrev = new Float64Array(nv); this.vW = new Float64Array(nv);
    this.pressure = new Float64Array(nc); this.rhs = new Float64Array(nc);
    this.cellType = new Uint8Array(nc); this.particleDensity = new Float64Array(nc);
    this.resetCellTypes();
  }

  // ---- indexing ----
  uIdx(i: number, j: number) { return i + (this.nx + 1) * j; }
  vIdx(i: number, j: number) { return i + this.nx * j; }
  cIdx(i: number, j: number) { return i + this.nx * j; }
  isBorder(i: number, j: number) { return i <= 0 || j <= 0 || i >= this.nx - 1 || j >= this.ny - 1; }
  typeAt(i: number, j: number) {
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return SOLID;
    return this.cellType[this.cIdx(i, j)];
  }

  resetCellTypes() {
    for (let j = 0; j < this.ny; j++)
      for (let i = 0; i < this.nx; i++)
        this.cellType[this.cIdx(i, j)] = this.isBorder(i, j) ? SOLID : AIR;
  }

  setParticles(x: ArrayLike<number>, y: ArrayLike<number>, vx?: ArrayLike<number>, vy?: ArrayLike<number>) {
    this.n = x.length;
    this.px = Float64Array.from(x); this.py = Float64Array.from(y);
    this.vx = vx ? Float64Array.from(vx) : new Float64Array(this.n);
    this.vy = vy ? Float64Array.from(vy) : new Float64Array(this.n);
  }

  // ---- particle -> grid ----
  /**
   * Bilinear splat of particle velocities onto the staggered faces.
   * For component c the sample lattice is offset by half a cell in the other axis.
   */
  transferToGrid() {
    const { nx, ny, h } = this;
    const inv = 1 / h;
    this.u.fill(0); this.v.fill(0); this.uW.fill(0); this.vW.fill(0);
    this.particleDensity.fill(0);
    const splat = (arr: Float64Array, w: Float64Array, gx: number, gy: number, maxI: number, maxJ: number,
      stride: number, val: number) => {
      const i0 = clampInt(Math.floor(gx), 0, maxI - 1), j0 = clampInt(Math.floor(gy), 0, maxJ - 1);
      const fx = clamp01(gx - i0), fy = clamp01(gy - j0);
      const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
      const a = i0 + stride * j0;
      arr[a] += val * w00; w[a] += w00;
      arr[a + 1] += val * w10; w[a + 1] += w10;
      arr[a + stride] += val * w01; w[a + stride] += w01;
      arr[a + stride + 1] += val * w11; w[a + stride + 1] += w11;
    };
    for (let p = 0; p < this.n; p++) {
      const x = this.px[p] * inv, y = this.py[p] * inv;
      splat(this.u, this.uW, x, y - 0.5, nx, ny - 1, nx + 1, this.vx[p]);
      splat(this.v, this.vW, x - 0.5, y, nx - 1, ny, nx, this.vy[p]);
      // particle number density sampled at cell centres
      const dummy = this.particleDensity;
      const gx = x - 0.5, gy = y - 0.5;
      const i0 = clampInt(Math.floor(gx), 0, nx - 2), j0 = clampInt(Math.floor(gy), 0, ny - 2);
      const fx = clamp01(gx - i0), fy = clamp01(gy - j0);
      const a = i0 + nx * j0;
      dummy[a] += (1 - fx) * (1 - fy); dummy[a + 1] += fx * (1 - fy);
      dummy[a + nx] += (1 - fx) * fy; dummy[a + nx + 1] += fx * fy;
    }
    for (let k = 0; k < this.u.length; k++) if (this.uW[k] > 0) this.u[k] /= this.uW[k];
    for (let k = 0; k < this.v.length; k++) if (this.vW[k] > 0) this.v[k] /= this.vW[k];
    this.uPrev.set(this.u); this.vPrev.set(this.v);
  }

  /** A non-solid cell containing at least one particle is FLUID, otherwise AIR. */
  markCells() {
    this.resetCellTypes();
    const inv = 1 / this.h;
    for (let p = 0; p < this.n; p++) {
      const i = clampInt(Math.floor(this.px[p] * inv), 0, this.nx - 1);
      const j = clampInt(Math.floor(this.py[p] * inv), 0, this.ny - 1);
      const c = this.cIdx(i, j);
      if (this.cellType[c] === AIR) this.cellType[c] = FLUID;
    }
  }

  /** Mean particle density of interior fluid cells; used as the rest density. */
  computeRestDensity() {
    let sum = 0, cnt = 0;
    for (let c = 0; c < this.cellType.length; c++)
      if (this.cellType[c] === FLUID) { sum += this.particleDensity[c]; cnt++; }
    this.restDensity = cnt > 0 ? sum / cnt : 0;
    return this.restDensity;
  }

  applyGravity(dt: number) {
    const g = this.p.gravityY * dt;
    for (let k = 0; k < this.v.length; k++) this.v[k] += g;
  }

  /** Free-slip static walls: the normal velocity on any face touching a SOLID cell is zero. */
  enforceBoundaries() {
    const { nx, ny } = this;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i <= nx; i++)
        if (this.typeAt(i - 1, j) === SOLID || this.typeAt(i, j) === SOLID) this.u[this.uIdx(i, j)] = 0;
    for (let j = 0; j <= ny; j++)
      for (let i = 0; i < nx; i++)
        if (this.typeAt(i, j - 1) === SOLID || this.typeAt(i, j) === SOLID) this.v[this.vIdx(i, j)] = 0;
  }

  /** Discrete divergence (1/s) of the face velocity field in cell (i,j). */
  divergenceAt(i: number, j: number) {
    return (this.u[this.uIdx(i + 1, j)] - this.u[this.uIdx(i, j)]
      + this.v[this.vIdx(i, j + 1)] - this.v[this.vIdx(i, j)]) / this.h;
  }

  maxAbsDivergence() {
    let m = 0;
    for (let j = 0; j < this.ny; j++)
      for (let i = 0; i < this.nx; i++)
        if (this.cellType[this.cIdx(i, j)] === FLUID) m = Math.max(m, Math.abs(this.divergenceAt(i, j)));
    return m;
  }

  /**
   * Builds the right-hand side of the pressure Poisson equation
   *   sum_{non-solid n} (p_i - p_n) = -(rho h^2 / dt) div_i     (air p_n = 0)
   * with an optional drift correction for particle over-density.
   */
  buildRhs(dt: number) {
    const { nx, ny, h } = this;
    const scale = (this.p.density * h * h) / dt;
    const k = this.p.densityCorrection;
    this.rhs.fill(0);
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const c = this.cIdx(i, j);
        if (this.cellType[c] !== FLUID) continue;
        let div = this.divergenceAt(i, j);
        if (k > 0 && this.restDensity > 0) {
          const compression = (this.particleDensity[c] - this.restDensity) / this.restDensity;
          if (compression > 0) div -= (k * compression) / dt;
        }
        this.rhs[c] = -scale * div;
      }
  }

  /** Applies A to x (A = negative discrete Laplacian restricted to fluid cells). */
  applyA(x: Float64Array, out: Float64Array) {
    const { nx, ny } = this;
    out.fill(0);
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const c = this.cIdx(i, j);
        if (this.cellType[c] !== FLUID) continue;
        let diag = 0, off = 0;
        const nb = [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]];
        for (const [a, b] of nb) {
          const t = this.cellType[this.cIdx(a, b)];
          if (t !== SOLID) diag++;
          if (t === FLUID) off += x[this.cIdx(a, b)];
        }
        out[c] = diag * x[c] - off;
      }
  }

  residualMax() {
    const Ap = new Float64Array(this.pressure.length);
    this.applyA(this.pressure, Ap);
    let m = 0;
    for (let c = 0; c < Ap.length; c++) if (this.cellType[c] === FLUID) m = Math.max(m, Math.abs(this.rhs[c] - Ap[c]));
    return m;
  }

  private neighbourSums(i: number, j: number, p: Float64Array) {
    let diag = 0, off = 0;
    const t0 = this.typeAt(i - 1, j), t1 = this.typeAt(i + 1, j), t2 = this.typeAt(i, j - 1), t3 = this.typeAt(i, j + 1);
    if (t0 !== SOLID) diag++; if (t0 === FLUID) off += p[this.cIdx(i - 1, j)];
    if (t1 !== SOLID) diag++; if (t1 === FLUID) off += p[this.cIdx(i + 1, j)];
    if (t2 !== SOLID) diag++; if (t2 === FLUID) off += p[this.cIdx(i, j - 1)];
    if (t3 !== SOLID) diag++; if (t3 === FLUID) off += p[this.cIdx(i, j + 1)];
    return { diag, off };
  }

  solveJacobi(iters: number): SolveInfo {
    let p: Float64Array = this.pressure, q: Float64Array = new Float64Array(this.pressure.length);
    for (let it = 0; it < iters; it++) {
      q.fill(0);
      for (let j = 1; j < this.ny - 1; j++)
        for (let i = 1; i < this.nx - 1; i++) {
          const c = this.cIdx(i, j);
          if (this.cellType[c] !== FLUID) continue;
          const { diag, off } = this.neighbourSums(i, j, p);
          q[c] = diag > 0 ? (this.rhs[c] + off) / diag : 0;
        }
      const t = p; p = q; q = t;
    }
    this.pressure.set(p);
    return { iterations: iters, residual: this.residualMax() };
  }

  /** Red-black (checkerboard) Gauss-Seidel with optional successive over-relaxation. */
  solveRedBlack(iters: number, omega = this.p.sorOmega): SolveInfo {
    const p = this.pressure;
    for (let it = 0; it < iters; it++)
      for (let colour = 0; colour < 2; colour++)
        for (let j = 1; j < this.ny - 1; j++)
          for (let i = 1 + ((j + colour) & 1); i < this.nx - 1; i += 2) {
            const c = this.cIdx(i, j);
            if (this.cellType[c] !== FLUID) continue;
            const { diag, off } = this.neighbourSums(i, j, p);
            if (diag === 0) continue;
            const gs = (this.rhs[c] + off) / diag;
            p[c] = (1 - omega) * p[c] + omega * gs;
          }
    return { iterations: iters, residual: this.residualMax() };
  }

  /**
   * Conjugate gradient preconditioned with Modified Incomplete Cholesky MIC(0)
   * (Bridson, "Fluid Simulation for Computer Graphics", 2nd ed., ch. 5).
   */
  solvePCG(maxIters: number, tol: number): SolveInfo {
    const { nx, ny } = this;
    const N = nx * ny;
    const fluid = (c: number) => this.cellType[c] === FLUID;
    // matrix in compressed stencil form
    const Adiag = new Float64Array(N), Ax = new Float64Array(N), Ay = new Float64Array(N);
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const c = this.cIdx(i, j);
        if (!fluid(c)) continue;
        Adiag[c] = this.neighbourSums(i, j, this.pressure).diag;
        if (fluid(c + 1)) Ax[c] = -1;
        if (fluid(c + nx)) Ay[c] = -1;
      }
    // MIC(0) preconditioner
    const tau = 0.97, sigma = 0.25;
    const precon = new Float64Array(N);
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const c = this.cIdx(i, j);
        if (!fluid(c)) continue;
        const l = c - 1, b = c - nx;
        let e = Adiag[c]
          - (Ax[l] * precon[l]) ** 2 - (Ay[b] * precon[b]) ** 2
          - tau * (Ax[l] * Ay[l] * precon[l] ** 2 + Ay[b] * Ax[b] * precon[b] ** 2);
        if (e < sigma * Adiag[c]) e = Adiag[c];
        precon[c] = 1 / Math.sqrt(e);
      }
    const q = new Float64Array(N);
    const applyPrecon = (r: Float64Array, z: Float64Array) => {
      for (let j = 1; j < ny - 1; j++)
        for (let i = 1; i < nx - 1; i++) {
          const c = this.cIdx(i, j);
          if (!fluid(c)) { q[c] = 0; continue; }
          const l = c - 1, b = c - nx;
          const t = r[c] - Ax[l] * precon[l] * q[l] - Ay[b] * precon[b] * q[b];
          q[c] = t * precon[c];
        }
      for (let j = ny - 2; j >= 1; j--)
        for (let i = nx - 2; i >= 1; i--) {
          const c = this.cIdx(i, j);
          if (!fluid(c)) { z[c] = 0; continue; }
          const t = q[c] - Ax[c] * precon[c] * z[c + 1] - Ay[c] * precon[c] * z[c + nx];
          z[c] = t * precon[c];
        }
    };
    const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let k = 0; k < N; k++) s += a[k] * b[k]; return s; };
    const maxAbs = (a: Float64Array) => { let m = 0; for (let k = 0; k < N; k++) m = Math.max(m, Math.abs(a[k])); return m; };

    const p = this.pressure; p.fill(0);
    const r = Float64Array.from(this.rhs);
    const r0 = maxAbs(r);
    if (r0 === 0) return { iterations: 0, residual: 0 };
    const z = new Float64Array(N), s = new Float64Array(N), As = new Float64Array(N);
    applyPrecon(r, z); s.set(z);
    let sigmaRz = dot(z, r);
    let it = 0;
    for (; it < maxIters; it++) {
      this.applyA(s, As);
      const alpha = sigmaRz / dot(s, As);
      for (let k = 0; k < N; k++) { p[k] += alpha * s[k]; r[k] -= alpha * As[k]; }
      if (maxAbs(r) <= tol * r0) { it++; break; }
      applyPrecon(r, z);
      const sigmaNew = dot(z, r);
      const beta = sigmaNew / sigmaRz;
      for (let k = 0; k < N; k++) s[k] = z[k] + beta * s[k];
      sigmaRz = sigmaNew;
    }
    return { iterations: it, residual: this.residualMax() };
  }

  solvePressure(dt: number): SolveInfo {
    this.buildRhs(dt);
    const { solver, pressureIters, pcgTol } = this.p;
    if (solver === 'jacobi') this.pressure.fill(0);
    if (solver === 'rbgs') this.pressure.fill(0);
    const info = solver === 'jacobi' ? this.solveJacobi(pressureIters)
      : solver === 'rbgs' ? this.solveRedBlack(pressureIters)
        : this.solvePCG(pressureIters, pcgTol);
    this.lastSolve = info;
    return info;
  }

  /** u <- u - dt/(rho h) * grad p on faces adjacent to fluid; solid faces stay zero. */
  project(dt: number) {
    const { nx, ny, h } = this;
    const k = dt / (this.p.density * h);
    const pAt = (i: number, j: number) => (this.typeAt(i, j) === FLUID ? this.pressure[this.cIdx(i, j)] : 0);
    for (let j = 0; j < ny; j++)
      for (let i = 0; i <= nx; i++) {
        const a = this.typeAt(i - 1, j), b = this.typeAt(i, j);
        const f = this.uIdx(i, j);
        if (a === SOLID || b === SOLID) this.u[f] = 0;
        else if (a === FLUID || b === FLUID) this.u[f] -= k * (pAt(i, j) - pAt(i - 1, j));
      }
    for (let j = 0; j <= ny; j++)
      for (let i = 0; i < nx; i++) {
        const a = this.typeAt(i, j - 1), b = this.typeAt(i, j);
        const f = this.vIdx(i, j);
        if (a === SOLID || b === SOLID) this.v[f] = 0;
        else if (a === FLUID || b === FLUID) this.v[f] -= k * (pAt(i, j) - pAt(i, j - 1));
      }
  }

  // ---- grid -> particle ----
  /** Face is "valid" (carries a meaningful velocity) if it touches at least one FLUID cell. */
  private uValid(i: number, j: number) { return this.typeAt(i - 1, j) === FLUID || this.typeAt(i, j) === FLUID; }
  private vValid(i: number, j: number) { return this.typeAt(i, j - 1) === FLUID || this.typeAt(i, j) === FLUID; }

  /** Bilinear interpolation restricted to valid faces (weights renormalised). Returns NaN if none valid. */
  sampleU(x: number, y: number, arr = this.u) {
    const gx = x / this.h, gy = y / this.h - 0.5;
    const i0 = clampInt(Math.floor(gx), 0, this.nx - 1), j0 = clampInt(Math.floor(gy), 0, this.ny - 2);
    const fx = clamp01(gx - i0), fy = clamp01(gy - j0);
    return this.sample4(arr, i0, j0, fx, fy, this.nx + 1, (i, j) => this.uValid(i, j));
  }
  sampleV(x: number, y: number, arr = this.v) {
    const gx = x / this.h - 0.5, gy = y / this.h;
    const i0 = clampInt(Math.floor(gx), 0, this.nx - 2), j0 = clampInt(Math.floor(gy), 0, this.ny - 1);
    const fx = clamp01(gx - i0), fy = clamp01(gy - j0);
    return this.sample4(arr, i0, j0, fx, fy, this.nx, (i, j) => this.vValid(i, j));
  }
  private sample4(arr: Float64Array, i0: number, j0: number, fx: number, fy: number, stride: number,
    valid: (i: number, j: number) => boolean) {
    let s = 0, w = 0;
    const ws = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
    const off = [[0, 0], [1, 0], [0, 1], [1, 1]];
    for (let k = 0; k < 4; k++) {
      const i = i0 + off[k][0], j = j0 + off[k][1];
      if (!valid(i, j) || ws[k] === 0) continue;
      s += ws[k] * arr[i + stride * j]; w += ws[k];
    }
    return w > 0 ? s / w : NaN;
  }

  /** v_p <- (1-a) * u_grid(x_p) + a * (v_p + u_grid(x_p) - u_grid_old(x_p)),  a = flipRatio */
  transferToParticles(flipRatio = this.p.flipRatio) {
    for (let p = 0; p < this.n; p++) {
      const x = this.px[p], y = this.py[p];
      const un = this.sampleU(x, y), uo = this.sampleU(x, y, this.uPrev);
      const vn = this.sampleV(x, y), vo = this.sampleV(x, y, this.vPrev);
      if (!Number.isNaN(un)) {
        const pic = un, flip = this.vx[p] + (un - uo);
        this.vx[p] = (1 - flipRatio) * pic + flipRatio * flip;
      }
      if (!Number.isNaN(vn)) {
        const pic = vn, flip = this.vy[p] + (vn - vo);
        this.vy[p] = (1 - flipRatio) * pic + flipRatio * flip;
      }
    }
  }

  gridVelocity(x: number, y: number, fallbackX: number, fallbackY: number): [number, number] {
    const u = this.sampleU(x, y), v = this.sampleV(x, y);
    return [Number.isNaN(u) ? fallbackX : u, Number.isNaN(v) ? fallbackY : v];
  }

  /** Midpoint (RK2) advection through the divergence-free grid velocity field. */
  advect(dt: number) {
    const lo = this.h + this.p.particleRadius;
    const hiX = (this.nx - 1) * this.h - this.p.particleRadius;
    const hiY = (this.ny - 1) * this.h - this.p.particleRadius;
    for (let p = 0; p < this.n; p++) {
      const x = this.px[p], y = this.py[p];
      const [u1, v1] = this.gridVelocity(x, y, this.vx[p], this.vy[p]);
      const xm = x + 0.5 * dt * u1, ym = y + 0.5 * dt * v1;
      const [u2, v2] = this.gridVelocity(xm, ym, this.vx[p], this.vy[p]);
      let nxp = x + dt * u2, nyp = y + dt * v2;
      if (nxp < lo) { nxp = lo; this.vx[p] = Math.max(0, this.vx[p]); }
      if (nxp > hiX) { nxp = hiX; this.vx[p] = Math.min(0, this.vx[p]); }
      if (nyp < lo) { nyp = lo; this.vy[p] = Math.max(0, this.vy[p]); }
      if (nyp > hiY) { nyp = hiY; this.vy[p] = Math.min(0, this.vy[p]); }
      this.px[p] = nxp; this.py[p] = nyp;
    }
  }

  /**
   * Particle separation: overlapping pairs (distance < 2r) are pushed apart symmetrically.
   * Neighbours are found with a uniform grid built by counting sort (same scheme as the GPU).
   */
  pushApart(iters = this.p.pushApartIters) {
    const r = this.p.particleRadius, minD = 2 * r;
    const cell = minD;
    const gx = Math.ceil((this.nx * this.h) / cell), gy = Math.ceil((this.ny * this.h) / cell);
    for (let it = 0; it < iters; it++) {
      const grid = countingSortGrid(this.n, gx, gy, 1, (k) => [
        clampInt(Math.floor(this.px[k] / cell), 0, gx - 1), clampInt(Math.floor(this.py[k] / cell), 0, gy - 1), 0]);
      for (let a = 0; a < this.n; a++) {
        const ci = clampInt(Math.floor(this.px[a] / cell), 0, gx - 1), cj = clampInt(Math.floor(this.py[a] / cell), 0, gy - 1);
        for (let dj = -1; dj <= 1; dj++)
          for (let di = -1; di <= 1; di++) {
            const i = ci + di, j = cj + dj;
            if (i < 0 || j < 0 || i >= gx || j >= gy) continue;
            const c = i + gx * j;
            for (let s = grid.cellStart[c]; s < grid.cellStart[c + 1]; s++) {
              const b = grid.sorted[s];
              if (b <= a) continue;
              const dx = this.px[b] - this.px[a], dy = this.py[b] - this.py[a];
              const d2 = dx * dx + dy * dy;
              if (d2 >= minD * minD || d2 === 0) continue;
              const d = Math.sqrt(d2), corr = (0.5 * (minD - d)) / d;
              this.px[a] -= dx * corr; this.py[a] -= dy * corr;
              this.px[b] += dx * corr; this.py[b] += dy * corr;
            }
          }
      }
    }
    const lo = this.h + r, hiX = (this.nx - 1) * this.h - r, hiY = (this.ny - 1) * this.h - r;
    for (let p = 0; p < this.n; p++) {
      this.px[p] = Math.min(hiX, Math.max(lo, this.px[p]));
      this.py[p] = Math.min(hiY, Math.max(lo, this.py[p]));
    }
  }

  maxParticleSpeed() {
    let m = 0;
    for (let p = 0; p < this.n; p++) m = Math.max(m, Math.hypot(this.vx[p], this.vy[p]));
    return m;
  }

  kineticEnergy() {
    let e = 0;
    for (let p = 0; p < this.n; p++) e += 0.5 * (this.vx[p] ** 2 + this.vy[p] ** 2);
    return e;
  }

  fluidCellCount() {
    let c = 0;
    for (let k = 0; k < this.cellType.length; k++) if (this.cellType[k] === FLUID) c++;
    return c;
  }

  /** One full PIC/FLIP step. */
  step(dt: number) {
    this.transferToGrid();
    this.markCells();
    if (this.restDensity === 0) this.computeRestDensity();
    this.applyGravity(dt);
    this.enforceBoundaries();
    this.solvePressure(dt);
    this.project(dt);
    this.transferToParticles();
    this.advect(dt);
    if (this.p.pushApartIters > 0) this.pushApart();
  }
}

function clampInt(x: number, lo: number, hi: number) { return x < lo ? lo : x > hi ? hi : x; }
function clamp01(x: number) { return x < 0 ? 0 : x > 1 ? 1 : x; }

/** Seeds `perCell x perCell` jittered particles per cell in the cell rectangle [i0,i1) x [j0,j1). */
export function seedBlock(h: number, i0: number, i1: number, j0: number, j1: number, perCell = 2,
  jitter = 0, seed = 1): { x: number[]; y: number[] } {
  const rng = mulberry32(seed);
  const x: number[] = [], y: number[] = [];
  const s = h / perCell;
  for (let j = j0; j < j1; j++)
    for (let i = i0; i < i1; i++)
      for (let b = 0; b < perCell; b++)
        for (let a = 0; a < perCell; a++) {
          x.push(i * h + (a + 0.5) * s + jitter * s * (rng() - 0.5));
          y.push(j * h + (b + 0.5) * s + jitter * s * (rng() - 0.5));
        }
  return { x, y };
}

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
