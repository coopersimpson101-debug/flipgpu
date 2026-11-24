import pbfSrc from './shaders/pbf.wgsl?raw';
import { HASH_DECLS, HashGrid } from './hashgrid';
import { Kernel, createModule, storageBuffer, uniformBuffer } from './kernel';
import type { SimSettings, Solver } from './solver';
import { TANK, seedScene, spacingForCount, type SceneName, type Vec3 } from '../sim/scenes';
import { latticeRestDensity, poly6, poly6Coef3, spikyGradCoef3 } from '../cpu/sph';

const MAX_SPEED = 40;
/** kernel radius in units of the initial particle spacing */
const H_OVER_S = 2.0;

export class PbfSolver implements Solver {
  readonly kind = 'pbf' as const;
  readonly numParticles: number;
  readonly box: Vec3 = TANK;
  readonly cflLength: number;
  readonly renderRadius: number;
  readonly pos: GPUBuffer; readonly vel: GPUBuffer; readonly stats: GPUBuffer;
  private posPred: GPUBuffer; private posTmp: GPUBuffer; private velTmp: GPUBuffer; private lambda: GPUBuffer;
  private params: GPUBuffer;
  private h: number; private rho0: number;
  private hash!: HashGrid;
  private k!: Record<string, Kernel>;
  private g!: Record<string, GPUBindGroup>;
  private ubuf = new ArrayBuffer(128);

  private constructor(private device: GPUDevice, readonly scene: SceneName, readonly spacing: number, positions: Float32Array<ArrayBuffer>) {
    this.numParticles = positions.length / 4;
    this.h = H_OVER_S * spacing;
    this.rho0 = latticeRestDensity(spacing, this.h);
    this.cflLength = spacing;
    this.renderRadius = 0.62 * spacing;
    const n = this.numParticles;
    this.pos = storageBuffer(device, 16 * n, 'pbf.pos');
    this.posPred = storageBuffer(device, 16 * n, 'pbf.posPred');
    this.posTmp = storageBuffer(device, 16 * n, 'pbf.posTmp');
    this.vel = storageBuffer(device, 16 * n, 'pbf.vel');
    this.velTmp = storageBuffer(device, 16 * n, 'pbf.velTmp');
    this.lambda = storageBuffer(device, 4 * n, 'pbf.lambda');
    this.stats = storageBuffer(device, 16, 'pbf.stats');
    this.params = uniformBuffer(device, 128, 'pbf.params');
    device.queue.writeBuffer(this.pos, 0, positions);
    device.queue.writeBuffer(this.posPred, 0, positions);
  }

  static async create(device: GPUDevice, scene: SceneName, targetParticles: number): Promise<PbfSolver> {
    const s = spacingForCount(scene, TANK, targetParticles);
    const positions = seedScene(scene, TANK, s, 0.02, 5, 0.3 * s);
    const solver = new PbfSolver(device, scene, s, positions);
    await solver.init();
    return solver;
  }

  private async init() {
    const d = this.device;
    const module = await createModule(d, HASH_DECLS + pbfSrc, 'pbf.wgsl');
    const kd = (entry: string, b: number[]) => {
      const bind: Record<number, 'uniform' | 'storage'> = { 0: 'uniform' };
      for (const x of b) bind[x] = x === 20 ? 'uniform' : 'storage';
      return new Kernel(d, module, entry, bind);
    };
    const H = [20, 21, 22];
    this.k = {
      predict: kd('predict', [1, 2, 4]),
      computeLambda: kd('computeLambda', [2, 6, ...H]),
      computeDelta: kd('computeDelta', [2, 3, 6, ...H]),
      applyDelta: kd('applyDelta', [2, 3]),
      updateVelocity: kd('updateVelocity', [1, 2, 4]),
      xsph: kd('xsph', [2, 4, 5, ...H]),
      finalize: kd('finalize', [1, 2, 4, 5, 7]),
    };
    this.hash = await HashGrid.create(d, this.posPred, this.numParticles, [0, 0, 0], this.h, this.box);
    const res: Record<number, GPUBuffer> = {
      0: this.params, 1: this.pos, 2: this.posPred, 3: this.posTmp, 4: this.vel, 5: this.velTmp, 6: this.lambda,
      7: this.stats, 20: this.hash.params, 21: this.hash.start, 22: this.hash.sorted,
    };
    this.g = {};
    for (const [name, k] of Object.entries(this.k)) this.g[name] = k.bind(res);
  }

  writeUniforms(dt: number, s: SimSettings) {
    const h = this.h;
    new Uint32Array(this.ubuf, 0, 4).set([0, 0, 0, this.numParticles]);
    const f = new Float32Array(this.ubuf, 16, 28);
    const relaxation = 0.5; // dimensionless CFM, see PBF3D
    f.set([h, dt, this.rho0, relaxation / (h * h)], 0);
    const sCorrK = 0.01, dq = 0.2 * h; // same convention as PBF3D: k * h^2 (lambda has units m^2)
    f.set([sCorrK * h * h, 1 / poly6(dq, h), 0.3 * s.viscosity, 0.3 * this.cflLength], 4);
    f.set([poly6Coef3(h), spikyGradCoef3(h), MAX_SPEED, 0], 8);
    f.set([s.tankAccel[0], -s.gravity + s.tankAccel[1], s.tankAccel[2], 0], 12);
    f.set([...this.box, 0], 16);
    f.set([...s.mouse.pos, s.mouse.radius], 20);
    f.set([...s.mouse.vel, s.mouse.strength], 24);
    this.device.queue.writeBuffer(this.params, 0, this.ubuf);
  }

  /** constraint iterations derived from the shared "pressure iterations" slider */
  static iterations(pressureIters: number) { return Math.max(1, Math.min(10, Math.round(pressureIters / 10))); }

  encodeSubstep(pass: GPUComputePassEncoder, s: SimSettings) {
    const { k, g } = this;
    const N = this.numParticles;
    k.predict.dispatch(pass, g.predict, N);
    this.hash.encode(pass);
    const iters = PbfSolver.iterations(s.pressureIters);
    for (let i = 0; i < iters; i++) {
      k.computeLambda.dispatch(pass, g.computeLambda, N);
      k.computeDelta.dispatch(pass, g.computeDelta, N);
      k.applyDelta.dispatch(pass, g.applyDelta, N);
    }
    k.updateVelocity.dispatch(pass, g.updateVelocity, N);
    k.xsph.dispatch(pass, g.xsph, N);
    k.finalize.dispatch(pass, g.finalize, N);
  }

  describe() {
    return `PBF · h=${(this.h * 100).toFixed(1)} cm, ρ₀=${this.rho0.toExponential(2)} m⁻³`;
  }

  destroy() {
    for (const b of [this.pos, this.posPred, this.posTmp, this.vel, this.velTmp, this.lambda, this.stats, this.params]) b.destroy();
    this.hash?.destroy();
  }
}
