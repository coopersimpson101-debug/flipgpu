import { describe, expect, it } from 'vitest';
import { AIR, FLUID, FlipSim2D, SOLID, cflTimestep, defaultParams, mulberry32, seedBlock } from '../src/cpu/flip2d';

/** Tank of nx*ny cells (including the solid ring) filled with fluid in cell rows [1, fillRows]. */
function poolSim(nx = 24, ny = 24, fillRows = 12, h = 0.05, perCell = 2) {
  const sim = new FlipSim2D(defaultParams(nx, ny, h));
  const { x, y } = seedBlock(h, 1, nx - 1, 1, 1 + fillRows, perCell, 0);
  sim.setParticles(x, y);
  return sim;
}

describe('MAC grid setup', () => {
  it('marks the outer ring solid and particle cells fluid', () => {
    const sim = poolSim(10, 10, 4);
    sim.transferToGrid();
    sim.markCells();
    expect(sim.typeAt(0, 5)).toBe(SOLID);
    expect(sim.typeAt(9, 5)).toBe(SOLID);
    expect(sim.typeAt(5, 0)).toBe(SOLID);
    expect(sim.typeAt(5, 2)).toBe(FLUID);
    expect(sim.typeAt(5, 4)).toBe(FLUID);
    expect(sim.typeAt(5, 5)).toBe(AIR);
    expect(sim.fluidCellCount()).toBe(8 * 4);
  });

  it('P2G of a uniform velocity reproduces it exactly on covered faces (partition of unity)', () => {
    const sim = poolSim(12, 12, 6);
    sim.vx.fill(0.7); sim.vy.fill(-0.3);
    sim.transferToGrid();
    // interior face well inside the block
    expect(sim.u[sim.uIdx(5, 3)]).toBeCloseTo(0.7, 12);
    expect(sim.v[sim.vIdx(5, 3)]).toBeCloseTo(-0.3, 12);
  });
});

describe('pressure projection', () => {
  function randomDivergentField(sim: FlipSim2D, seed = 3) {
    const rng = mulberry32(seed);
    sim.transferToGrid();
    sim.markCells();
    for (let k = 0; k < sim.u.length; k++) sim.u[k] = rng() * 2 - 1;
    for (let k = 0; k < sim.v.length; k++) sim.v[k] = rng() * 2 - 1;
    sim.enforceBoundaries();
  }

  it('MIC(0)-PCG projection makes the field discretely divergence-free', () => {
    const sim = poolSim(32, 32, 20);
    randomDivergentField(sim);
    const before = sim.maxAbsDivergence();
    sim.p.solver = 'pcg'; sim.p.pressureIters = 500; sim.p.pcgTol = 1e-12;
    const info = sim.solvePressure(1 / 60);
    sim.project(1 / 60);
    const after = sim.maxAbsDivergence();
    expect(before).toBeGreaterThan(10);
    expect(after).toBeLessThan(1e-8);
    expect(info.iterations).toBeLessThan(200);
  });

  it('Jacobi and red-black SOR reduce divergence; RB-SOR converges faster than Jacobi', () => {
    const make = () => { const s = poolSim(24, 24, 14); randomDivergentField(s, 11); return s; };
    const a = make(), b = make();
    const before = a.maxAbsDivergence();
    a.p.solver = 'jacobi'; a.p.pressureIters = 400;
    b.p.solver = 'rbgs'; b.p.pressureIters = 400; b.p.sorOmega = 1.7;
    const ra = a.solvePressure(1 / 60).residual; a.project(1 / 60);
    const rb = b.solvePressure(1 / 60).residual; b.project(1 / 60);
    expect(a.maxAbsDivergence()).toBeLessThan(before / 20);
    expect(b.maxAbsDivergence()).toBeLessThan(before / 1e4);
    expect(rb).toBeLessThan(ra / 100);
  });

  it('a fluid at rest develops a hydrostatic pressure p = rho g depth', () => {
    const nx = 16, ny = 28, rows = 20, h = 0.05;
    const sim = poolSim(nx, ny, rows, h);
    const dt = 1 / 120;
    sim.p.solver = 'pcg'; sim.p.pressureIters = 1000; sim.p.pcgTol = 1e-13;
    sim.transferToGrid(); sim.markCells(); sim.applyGravity(dt); sim.enforceBoundaries();
    sim.solvePressure(dt); sim.project(dt);
    const rho = 1000, g = 9.81;
    // Free surface sits on the boundary of the top fluid cell (y = (rows+1) h). With p = 0 imposed at
    // the first air cell centre (no ghost-fluid fraction) the discrete surface is half a cell higher.
    const ySurf = (rows + 1) * h + 0.5 * h;
    for (let j = 1; j <= rows; j++) {
      const depth = ySurf - (j + 0.5) * h;
      const p = sim.pressure[sim.cIdx(nx / 2, j)];
      expect(Math.abs(p - rho * g * depth) / (rho * g * depth)).toBeLessThan(1e-6);
    }
    // pressure gradient balances gravity: the fluid stays at rest
    let maxV = 0;
    for (let j = 2; j < rows; j++) maxV = Math.max(maxV, Math.abs(sim.v[sim.vIdx(nx / 2, j)]));
    expect(maxV).toBeLessThan(1e-9);
    // and against the continuum profile with the physical surface at (rows+1)h, error <= rho g h/2
    for (let j = 1; j <= rows; j++) {
      const depth = (rows + 1) * h - (j + 0.5) * h;
      expect(Math.abs(sim.pressure[sim.cIdx(5, j)] - rho * g * depth)).toBeLessThanOrEqual(0.5 * rho * g * h + 1e-6);
    }
  });
});

describe('FLIP vs PIC transfers', () => {
  it('PIC dissipates sub-grid velocity detail, FLIP preserves it when the grid does not change', () => {
    const run = (ratio: number) => {
      const sim = poolSim(20, 20, 18, 0.05, 3);
      const rng = mulberry32(5);
      for (let p = 0; p < sim.n; p++) { sim.vx[p] = rng() - 0.5; sim.vy[p] = rng() - 0.5; }
      const e0 = sim.kineticEnergy();
      for (let k = 0; k < 10; k++) { sim.transferToGrid(); sim.markCells(); sim.transferToParticles(ratio); }
      return sim.kineticEnergy() / e0;
    };
    const pic = run(0), flip = run(1), blend = run(0.95);
    expect(flip).toBeCloseTo(1, 10);
    expect(pic).toBeLessThan(0.5);
    expect(blend).toBeGreaterThan(pic);
    expect(blend).toBeLessThan(flip);
  });
});

describe('full simulation', () => {
  it('dam break conserves particles, stays bounded and converts potential energy', () => {
    const nx = 40, ny = 24, h = 0.05;
    const params = defaultParams(nx, ny, h);
    params.solver = 'pcg'; params.pressureIters = 300; params.densityCorrection = 0.5;
    const sim = new FlipSim2D(params);
    const { x, y } = seedBlock(h, 1, 14, 1, 17, 2, 0.5, 9);
    sim.setParticles(x, y);
    const n0 = sim.n;
    const com = () => {
      let cx = 0, cy = 0;
      for (let p = 0; p < sim.n; p++) { cx += sim.px[p]; cy += sim.py[p]; }
      return [cx / sim.n, cy / sim.n];
    };
    const [cx0, cy0] = com();
    sim.transferToGrid(); sim.markCells();
    const rest = sim.computeRestDensity();
    // liquid volume (in cells) = sum of fractional occupancies min(1, density/rest); ignores the
    // bias of counting partially filled surface cells as full
    const denseCells = () => { let c = 0; for (let k = 0; k < sim.cellType.length; k++) if (sim.cellType[k] === FLUID) c += Math.min(1, sim.particleDensity[k] / rest); return c; };
    const fluid0 = denseCells();
    let t = 0, cxMid = 0;
    while (t < 4.0) {
      const dt = cflTimestep(sim.maxParticleSpeed(), h, 1.0, 1 / 60);
      sim.step(dt);
      t += dt;
      if (cxMid === 0 && t > 0.6) cxMid = com()[0];
      expect(sim.lastSolve.residual).toBeLessThan(1e-3 * (1000 * h * h / dt));
    }
    const [cx1, cy1] = com();
    expect(sim.n).toBe(n0);
    for (let p = 0; p < sim.n; p++) {
      expect(Number.isFinite(sim.px[p]) && Number.isFinite(sim.vx[p])).toBe(true);
      expect(sim.px[p]).toBeGreaterThanOrEqual(h);
      expect(sim.px[p]).toBeLessThanOrEqual((nx - 1) * h);
      expect(sim.py[p]).toBeGreaterThanOrEqual(h);
      expect(sim.py[p]).toBeLessThanOrEqual((ny - 1) * h);
    }
    expect(cxMid).toBeGreaterThan(cx0 + 0.1); // column collapsed to the right
    expect(cy1).toBeLessThan(cy0 - 0.1); // centre of mass fell
    sim.transferToGrid(); sim.markCells();
    const ratio = denseCells() / fluid0; // liquid volume after settling vs initial
    console.log(`dam break: settled volume ratio ${ratio.toFixed(3)}, com x ${cx0.toFixed(3)} -> ${cx1.toFixed(3)}, y ${cy0.toFixed(3)} -> ${cy1.toFixed(3)}`);
    expect(ratio).toBeGreaterThan(0.85);
    expect(ratio).toBeLessThan(1.15);
  });

  it('CFL timestep limits the per-step displacement to cfl * h', () => {
    expect(cflTimestep(0, 0.1, 1, 0.02)).toBe(0.02);
    expect(cflTimestep(10, 0.1, 1, 0.02)).toBeCloseTo(0.01, 12);
    expect(cflTimestep(10, 0.1, 0.5, 0.02) * 10).toBeCloseTo(0.05, 12);
  });
});
