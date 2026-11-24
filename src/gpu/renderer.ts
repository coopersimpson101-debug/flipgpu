import renderSrc from './shaders/render.wgsl?raw';
import { createModule, uniformBuffer } from './kernel';
import { invert, type Mat4 } from '../math';
import type { Vec3 } from '../sim/scenes';

export type RenderMode = 'surface' | 'particles';

const DEPTH_FMT: GPUTextureFormat = 'depth24plus';

export class Renderer {
  private camBuf: GPUBuffer;
  private camBGL: GPUBindGroupLayout; private texBGL: GPUBindGroupLayout;
  private pipes!: Record<'sphere' | 'depth' | 'thick' | 'bg' | 'blurX' | 'blurY' | 'composite' | 'lines', GPURenderPipeline>;
  private depthTex?: GPUTexture; private fluidA?: GPUTexture; private fluidB?: GPUTexture; private thick?: GPUTexture;
  private texGroupA?: GPUBindGroup; private texGroupB?: GPUBindGroup;
  private camGroup?: GPUBindGroup; private camGroupKey?: GPUBuffer;
  private size = [0, 0];
  smoothingIterations = 2;

  private constructor(private device: GPUDevice, private format: GPUTextureFormat) {
    this.camBuf = uniformBuffer(device, 320, 'camera');
    this.camBGL = device.createBindGroupLayout({
      label: 'camera', entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.texBGL = device.createBindGroupLayout({
      label: 'textures', entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
  }

  static async create(device: GPUDevice, format: GPUTextureFormat) {
    const r = new Renderer(device, format);
    await r.init();
    return r;
  }

  private async init() {
    const d = this.device;
    const module = await createModule(d, renderSrc, 'render.wgsl');
    const lay1 = d.createPipelineLayout({ bindGroupLayouts: [this.camBGL] });
    const lay2 = d.createPipelineLayout({ bindGroupLayouts: [this.camBGL, this.texBGL] });
    const depth = (write: boolean, compare: GPUCompareFunction): GPUDepthStencilState =>
      ({ format: DEPTH_FMT, depthWriteEnabled: write, depthCompare: compare });
    const mk = (label: string, layout: GPUPipelineLayout, vs: string, fs: string, target: GPUColorTargetState,
      ds?: GPUDepthStencilState, topology: GPUPrimitiveTopology = 'triangle-list') =>
      d.createRenderPipeline({
        label, layout,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [target] },
        primitive: { topology },
        depthStencil: ds,
      });
    const add: GPUBlendState = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
    this.pipes = {
      sphere: mk('sphere', lay1, 'vsSphere', 'fsSphereColor', { format: this.format }, depth(true, 'less')),
      depth: mk('fluidDepth', lay1, 'vsSphere', 'fsSphereDepth', { format: 'r32float' }, depth(true, 'less')),
      thick: mk('thickness', lay1, 'vsSphere', 'fsThickness', { format: 'r16float', blend: add }),
      bg: mk('background', lay2, 'vsFull', 'fsBackground', { format: this.format }, depth(false, 'always')),
      blurX: mk('blurX', lay2, 'vsFull', 'fsBlurX', { format: 'r32float' }),
      blurY: mk('blurY', lay2, 'vsFull', 'fsBlurY', { format: 'r32float' }),
      composite: mk('composite', lay2, 'vsFull', 'fsComposite', { format: this.format }, depth(false, 'always')),
      lines: mk('lines', lay1, 'vsLines', 'fsLines', { format: this.format }, depth(false, 'less'), 'line-list'),
    };
  }

  private ensureTargets(w: number, h: number) {
    if (this.size[0] === w && this.size[1] === h && this.depthTex) return;
    for (const t of [this.depthTex, this.fluidA, this.fluidB, this.thick]) t?.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.depthTex = this.device.createTexture({ size: [w, h], format: DEPTH_FMT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.fluidA = this.device.createTexture({ size: [w, h], format: 'r32float', usage, label: 'fluidDepthA' });
    this.fluidB = this.device.createTexture({ size: [w, h], format: 'r32float', usage, label: 'fluidDepthB' });
    this.thick = this.device.createTexture({ size: [w, h], format: 'r16float', usage, label: 'thickness' });
    const tg = (a: GPUTexture) => this.device.createBindGroup({
      layout: this.texBGL, entries: [{ binding: 0, resource: a.createView() }, { binding: 1, resource: this.thick!.createView() }],
    });
    this.texGroupA = tg(this.fluidA);
    this.texGroupB = tg(this.fluidB);
    this.size = [w, h];
  }

  render(encoder: GPUCommandEncoder, target: GPUTextureView, w: number, h: number, view: Mat4, proj: Mat4,
    pos: GPUBuffer, vel: GPUBuffer, n: number, radius: number, box: Vec3, mode: RenderMode) {
    this.ensureTargets(w, h);
    if (this.camGroupKey !== pos || !this.camGroup) {
      this.camGroup = this.device.createBindGroup({
        layout: this.camBGL, entries: [
          { binding: 0, resource: { buffer: this.camBuf } },
          { binding: 1, resource: { buffer: pos } },
          { binding: 2, resource: { buffer: vel } },
        ],
      });
      this.camGroupKey = pos;
    }
    const cam = new Float32Array(80);
    cam.set(view, 0); cam.set(proj, 16); cam.set(invert(proj)!, 32); cam.set(invert(view)!, 48);
    cam.set([w, h, 1 / w, 1 / h], 64);
    cam.set([radius, 3.5, 2.2 * radius, 3.0 * radius], 68);
    // light direction in view space (from a fixed world direction)
    const L = [0.35, 0.9, 0.25];
    const lv = [0, 1, 2].map((r) => view[r] * L[0] + view[4 + r] * L[1] + view[8 + r] * L[2]);
    cam.set([lv[0], lv[1], lv[2], 0], 72);
    cam.set([box[0], box[1], box[2], 0], 76);
    this.device.queue.writeBuffer(this.camBuf, 0, cam);

    const depthView = this.depthTex!.createView();
    if (mode === 'particles') {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        depthStencilAttachment: { view: depthView, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
      });
      pass.setBindGroup(0, this.camGroup);
      pass.setBindGroup(1, this.texGroupA!);
      pass.setPipeline(this.pipes.bg); pass.draw(3);
      pass.setPipeline(this.pipes.sphere); pass.draw(6, n);
      pass.setPipeline(this.pipes.lines); pass.draw(24);
      pass.end();
      return;
    }
    // 1) nearest view depth of the particle spheres
    let pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.fluidA!.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      depthStencilAttachment: { view: depthView, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    pass.setBindGroup(0, this.camGroup);
    pass.setPipeline(this.pipes.depth); pass.draw(6, n);
    pass.end();
    // 2) thickness (additive)
    pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.thick!.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    });
    pass.setBindGroup(0, this.camGroup);
    pass.setPipeline(this.pipes.thick); pass.draw(6, n);
    pass.end();
    // 3) separable bilateral smoothing A -> B -> A
    for (let it = 0; it < this.smoothingIterations; it++) {
      for (const [pipe, src, dst] of [[this.pipes.blurX, this.texGroupA!, this.fluidB!], [this.pipes.blurY, this.texGroupB!, this.fluidA!]] as const) {
        pass = encoder.beginRenderPass({
          colorAttachments: [{ view: dst.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
        });
        pass.setBindGroup(0, this.camGroup);
        pass.setBindGroup(1, src);
        pass.setPipeline(pipe); pass.draw(3);
        pass.end();
      }
    }
    // 4) shading + tank outline (depth-tested against the particle depth buffer)
    pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      depthStencilAttachment: { view: depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    pass.setBindGroup(0, this.camGroup);
    pass.setBindGroup(1, this.texGroupA!);
    pass.setPipeline(this.pipes.composite); pass.draw(3);
    pass.setPipeline(this.pipes.lines); pass.draw(24);
    pass.end();
  }
}
