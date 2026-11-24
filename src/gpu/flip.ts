import flipSrc from './shaders/flip.wgsl?raw';
import { HASH_DECLS, HashGrid } from './hashgrid';
import { Kernel, createModule, storageBuffer, uniformBuffer } from './kernel';
import type { SimSettings, Solver } from './solver';
import { TANK, makeFlipGrid, seedScene, spacingForCount, type FlipGrid, type SceneName, type Vec3 } from '../sim/scenes';

const MAX_SPEED = 40;
/** particles per cell per axis => 8 per cell */
const PPC_AXIS = 2;

export class FlipSolver implements Solver {
  readonly kind = 'flip' as const;
  readonly numParticles: number;
  readonly box: Vec3;
  readonly cflLength: number;
  readonly renderRadius: number;
  readonly pos: GPUBuffer; readonly vel: GPUBuffer; readonly stats: GPUBuffer;
  private posTmp: GPUBuffer; private accum: GPUBuffer;
  private grid: GPUBuffer; private gridPrev: GPUBuffer; private gridTmp: GPUBuffer;
  private cellType: GPUBuffer; private pA: GPUBuffer; private pB: GPUBuffer; private rhs: GPUBuffer;
  private params: GPUBuffer;
  private nodes: number;
  private pushRadius: number;
  private hash!: HashGrid;
  private k!: Record<string, Kernel>;
  private g!: Record<string, GPUBindGroup>;
  private ubuf = new ArrayBuffer(112);

  private constructor(private device: GPUDevice, readonly scene: SceneName, readonly g3: FlipGrid, positions: Float32Array<ArrayBuffer>) {
    this.numParticles = positions.length / 4;
    this.box = g3.box;
    this.cflLength = g3.h;
    const s = g3.h / PPC_AXIS;
    this.pushRadius = 0.45 * s;
    this.renderRadius = 0.62 * s;
    this.nodes = (g3.nx + 1) * (g3.ny + 1) * (g3.nz + 1);
    const n = this.numParticles, nodes = this.nodes;
    this.pos = storageBuffer(device, 16 * n, 'flip.pos');
    this.vel = storageBuffer(device, 16 * n, 'flip.vel');
    this.posTmp = storageBuffer(device, 16 * n, 'flip.posTmp');
    this.accum = storageBuffer(device, 4 * 8 * nodes, 'flip.accum');
    this.grid = storageBuffer(device, 16 * nodes, 'flip.grid');
    this.gridPrev = storageBuffer(device, 16 * nodes, 'flip.gridPrev');
    this.gridTmp = storageBuffer(device, 16 * nodes, 'flip.gridTmp');
    this.cellType = storageBuffer(device, 4 * nodes, 'flip.cellType');
    this.pA = storageBuffer(device, 4 * nodes, 'flip.pressureA');
    this.pB = storageBuffer(device, 4 * nodes, 'flip.pressureB');
    this.rhs = storageBuffer(device, 4 * nodes, 'flip.rhs');
    this.stats = storageBuffer(device, 16, 'flip.stats');
    this.params = uniformBuffer(device, 112, 'flip.params');
    device.queue.writeBuffer(this.pos, 0, positions);
  }

  static async create(device: GPUDevice, scene: SceneName, targetParticles: number): Promise<FlipSolver> {
    const s = spacingForCount(scene, TANK, targetParticles);
    const g3 = makeFlipGrid(TANK, PPC_AXIS * s);
    const positions = seedScene(scene, g3.box, g3.h / PPC_AXIS, 0.6, 11, 0.5 * g3.h / PPC_AXIS);
    const solver = new FlipSolver(device, scene, g3, positions);
    await solver.init();
    return solver;
  }

  private async init() {
    const d = this.device;
    const module = await createModule(d, HASH_DECLS + flipSrc, 'flip.wgsl');
    const U = 'uniform' as const, S = 'storage' as const;
    const kd = (entry: string, b: number[]) => {
      const bind: Record<number, 'uniform' | 'storage'> = { 0: U };
      for (const x of b) bind[x] = x === 20 ? U : S;
      return new Kernel(d, module, entry, bind);
    };
    this.k = {
      clearGrid: kd('clearGrid', [3]),
      p2g: kd('p2g', [1, 2, 3]),
      gridNormalize: kd('gridNormalize', [3, 4, 5, 6]),
      applyForces: kd('applyForces', [4, 6]),
      diffuse: kd('diffuse', [4, 6, 10]),
      copyTmpToGrid: kd('copyTmpToGrid', [4, 10]),
      divergence: kd('divergence', [4, 6, 7, 8, 9]),
      jacobi: kd('jacobi', [6, 7, 8, 9]),
      rbRed: kd('rbRed', [6, 7, 9]),
      rbBlack: kd('rbBlack', [6, 7, 9]),
      project: kd('project', [4, 6, 7]),
      g2p: kd('g2p', [1, 2, 4, 5, 6, 11]),
      pushOut: kd('pushOut', [1, 12, 20, 21, 22]),
      copyPos: kd('copyPos', [1, 12]),
    };
    // hash cell = FLIP cell (push-out diameter 0.9 * h/2 < h, so a 27-cell search suffices)
    this.hash = await HashGrid.create(d, this.pos, this.numParticles, [0, 0, 0], this.g3.h, this.box);
    const res: Record<number, GPUBuffer> = {
      0: this.params, 1: this.pos, 2: this.vel, 3: this.accum, 4: this.grid, 5: this.gridPrev, 6: this.cellType,
      7: this.pA, 8: this.pB, 9: this.rhs, 10: this.gridTmp, 11: this.stats, 12: this.posTmp,
      20: this.hash.params, 21: this.hash.start, 22: this.hash.sorted,
    };
    this.g = {};
    for (const [name, k] of Object.entries(this.k)) this.g[name] = k.bind(res);
    // swapped ping-pong group for Jacobi (reads B, writes A)
    this.g.jacobiBA = this.k.jacobi.bind({ ...res, 7: this.pB, 8: this.pA });
  }

  writeUniforms(dt: number, s: SimSettings) {
    const { nx, ny, nz, h } = this.g3;
    const u = new Uint32Array(this.ubuf, 0, 4);
    u.set([nx, ny, nz, this.numParticles]);
    const f = new Float32Array(this.ubuf, 16, 24);
    f.set([h, 1 / h, dt, s.flipRatio], 0);
    f.set([s.tankAccel[0], -s.gravity + s.tankAccel[1], s.tankAccel[2], 0], 4);
    f.set([...s.mouse.pos, s.mouse.radius], 8);
    f.set([...s.mouse.vel, s.mouse.strength], 12);
    const nu = 0.02 * s.viscosity;
    const restDensity = PPC_AXIS ** 3;
    f.set([nu, s.sorOmega, restDensity, s.densityCorrection], 16);
    f.set([this.pushRadius, 0, MAX_SPEED, 0], 20);
    this.device.queue.writeBuffer(this.params, 0, this.ubuf);
  }

  encodeSubstep(pass: GPUComputePassEncoder, s: SimSettings) {
    const { k, g } = this;
    const N = this.numParticles, M = this.nodes;
    k.clearGrid.dispatch(pass, g.clearGrid, M);
    k.p2g.dispatch(pass, g.p2g, N);
    k.gridNormalize.dispatch(pass, g.gridNormalize, M);
    k.applyForces.dispatch(pass, g.applyForces, M);
    if (s.viscosity > 0) {
      k.diffuse.dispatch(pass, g.diffuse, M);
      k.copyTmpToGrid.dispatch(pass, g.copyTmpToGrid, M);
    }
    k.divergence.dispatch(pass, g.divergence, M);
    const iters = Math.max(1, Math.round(s.pressureIters));
    if (s.pressureSolver === 'jacobi') {
      const even = iters + (iters & 1); // result must land back in pressure A
      for (let i = 0; i < even; i++) k.jacobi.dispatch(pass, i & 1 ? g.jacobiBA : g.jacobi, M);
    } else {
      for (let i = 0; i < iters; i++) {
        k.rbRed.dispatch(pass, g.rbRed, M);
        k.rbBlack.dispatch(pass, g.rbBlack, M);
      }
    }
    k.project.dispatch(pass, g.project, M);
    k.g2p.dispatch(pass, g.g2p, N);
    if (s.pushOutIters > 0) {
      this.hash.encode(pass);
      for (let i = 0; i < s.pushOutIters; i++) {
        k.pushOut.dispatch(pass, g.pushOut, N);
        k.copyPos.dispatch(pass, g.copyPos, N);
      }
    }
  }

  describe() {
    const { nx, ny, nz } = this.g3;
    return `PIC/FLIP · MAC grid ${nx}×${ny}×${nz} (h=${(this.g3.h * 100).toFixed(1)} cm)`;
  }

  destroy() {
    for (const b of [this.pos, this.vel, this.posTmp, this.accum, this.grid, this.gridPrev, this.gridTmp,
      this.cellType, this.pA, this.pB, this.rhs, this.stats, this.params]) b.destroy();
    this.hash?.destroy();
  }
}
