import hashSrc from './shaders/hashgrid.wgsl?raw';
import scanSrc from './shaders/scan.wgsl?raw';
import { Kernel, createModule, storageBuffer, uniformBuffer } from './kernel';
import { scanLevels } from '../cpu/spatial';

/** GPU uniform-grid binning (counting sort). After `encode`, `start` and `sorted` are valid. */
export class HashGrid {
  readonly cells: number;
  readonly count: GPUBuffer; readonly slot: GPUBuffer; readonly cell: GPUBuffer;
  readonly start: GPUBuffer; readonly sorted: GPUBuffer; readonly params: GPUBuffer;
  private levelBufs: GPUBuffer[] = [];
  private levelSizes: number[] = [];
  private scanGroups: GPUBindGroup[] = [];
  private addGroups: GPUBindGroup[] = [];
  private k!: { clear: Kernel; count: Kernel; copy: Kernel; scatter: Kernel; scan: Kernel; add: Kernel };
  private g!: { clear: GPUBindGroup; count: GPUBindGroup; copy: GPUBindGroup; scatter: GPUBindGroup };

  private constructor(device: GPUDevice, readonly dims: [number, number, number], readonly n: number) {
    this.cells = dims[0] * dims[1] * dims[2];
    this.count = storageBuffer(device, 4 * (this.cells + 1), 'hash.count');
    this.start = storageBuffer(device, 4 * (this.cells + 1), 'hash.start');
    this.slot = storageBuffer(device, 4 * n, 'hash.slot');
    this.cell = storageBuffer(device, 4 * n, 'hash.cell');
    this.sorted = storageBuffer(device, 4 * n, 'hash.sorted');
    this.params = uniformBuffer(device, 32, 'hash.params');
  }

  static async create(device: GPUDevice, pos: GPUBuffer, n: number, origin: [number, number, number],
    cellSize: number, extent: [number, number, number]): Promise<HashGrid> {
    const dims = extent.map((e) => Math.max(1, Math.ceil(e / cellSize))) as [number, number, number];
    const hg = new HashGrid(device, dims, n);
    const p = new ArrayBuffer(32);
    new Uint32Array(p, 0, 4).set([dims[0], dims[1], dims[2], n]);
    new Float32Array(p, 16, 4).set([origin[0], origin[1], origin[2], 1 / cellSize]);
    device.queue.writeBuffer(hg.params, 0, p);

    const hm = await createModule(device, hashSrc, 'hashgrid.wgsl');
    const sm = await createModule(device, scanSrc, 'scan.wgsl');
    const S = 'storage' as const, U = 'uniform' as const;
    hg.k = {
      clear: new Kernel(device, hm, 'clearCounts', { 0: U, 2: S }, 256),
      count: new Kernel(device, hm, 'countParticles', { 0: U, 1: S, 2: S, 3: S, 4: S }),
      copy: new Kernel(device, hm, 'copyCounts', { 0: U, 5: S, 7: S }, 256),
      scatter: new Kernel(device, hm, 'scatter', { 0: U, 3: S, 4: S, 5: S, 6: S }),
      scan: new Kernel(device, sm, 'scanBlocks', { 0: U, 1: S, 2: S }, 256),
      add: new Kernel(device, sm, 'addBlockSums', { 0: U, 1: S, 2: S }, 256),
    };
    const r = { 0: hg.params, 1: pos, 2: hg.count, 3: hg.slot, 4: hg.cell, 5: hg.start, 6: hg.sorted, 7: hg.count };
    hg.g = { clear: hg.k.clear.bind(r), count: hg.k.count.bind(r), copy: hg.k.copy.bind(r), scatter: hg.k.scatter.bind(r) };

    hg.levelSizes = scanLevels(hg.cells + 1, 256);
    hg.levelBufs = [hg.start];
    for (let l = 1; l < hg.levelSizes.length; l++) hg.levelBufs.push(storageBuffer(device, 4 * hg.levelSizes[l], `scan.l${l}`));
    const dummy = storageBuffer(device, 16, 'scan.dummy');
    for (let l = 0; l < hg.levelSizes.length; l++) {
      const ub = uniformBuffer(device, 16, `scan.params${l}`);
      device.queue.writeBuffer(ub, 0, new Uint32Array([hg.levelSizes[l], 0, 0, 0]));
      const sums = l + 1 < hg.levelBufs.length ? hg.levelBufs[l + 1] : dummy;
      hg.scanGroups.push(hg.k.scan.bind({ 0: ub, 1: hg.levelBufs[l], 2: sums }));
      hg.addGroups.push(hg.k.add.bind({ 0: ub, 1: hg.levelBufs[l], 2: sums }));
    }
    return hg;
  }

  encode(pass: GPUComputePassEncoder) {
    const { k, g } = this;
    k.clear.dispatch(pass, g.clear, this.cells + 1);
    k.count.dispatch(pass, g.count, this.n);
    k.copy.dispatch(pass, g.copy, this.cells + 1);
    const L = this.levelSizes.length;
    for (let l = 0; l < L; l++) k.scan.dispatch(pass, this.scanGroups[l], this.levelSizes[l]);
    for (let l = L - 2; l >= 0; l--) k.add.dispatch(pass, this.addGroups[l], this.levelSizes[l]);
    k.scatter.dispatch(pass, g.scatter, this.n);
  }

  destroy() {
    for (const b of [this.count, this.start, this.slot, this.cell, this.sorted, this.params, ...this.levelBufs.slice(1)]) b.destroy();
  }
}

/** WGSL snippet for iterating neighbours, shared by the FLIP push-out and PBF kernels. */
export const HASH_DECLS = /* wgsl */ `
struct HashParams { dims: vec4<u32>, geo: vec4<f32> };
fn hashCoord(p: vec3<f32>) -> vec3<i32> {
  let g = vec3<i32>(floor((p - HP.geo.xyz) * HP.geo.w));
  return clamp(g, vec3<i32>(0), vec3<i32>(HP.dims.xyz) - vec3<i32>(1));
}
fn hashIndex(c: vec3<i32>) -> u32 { return u32(c.x) + HP.dims.x * (u32(c.y) + HP.dims.y * u32(c.z)); }
fn hashInside(c: vec3<i32>) -> bool { return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(HP.dims.xyz)); }
`;
