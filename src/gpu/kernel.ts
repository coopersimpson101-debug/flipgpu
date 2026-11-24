/**
 * Thin helpers over WebGPU compute: every kernel gets an explicit bind-group layout that lists only
 * the bindings it statically uses, so one WGSL module can hold many entry points while each
 * pipeline stays within maxStorageBuffersPerShaderStage.
 */
export type BindKind = 'uniform' | 'storage' | 'read-only-storage';
export const MAX_GROUPS = 65535;

export async function createModule(device: GPUDevice, code: string, label: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  for (const m of info.messages) {
    const line = `[${label}:${m.lineNum}:${m.linePos}] ${m.type}: ${m.message}`;
    if (m.type === 'error') console.error(line); else console.warn(line);
  }
  if (errors.length) throw new Error(`WGSL compile error in ${label}: ${errors[0].message} (line ${errors[0].lineNum})`);
  return module;
}

export class Kernel {
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  constructor(private device: GPUDevice, module: GPUShaderModule, readonly entry: string,
    readonly bindings: Record<number, BindKind>, readonly workgroupSize = 64) {
    this.layout = device.createBindGroupLayout({
      label: entry,
      entries: Object.entries(bindings).map(([b, type]) => ({
        binding: Number(b), visibility: GPUShaderStage.COMPUTE, buffer: { type },
      })),
    });
    this.pipeline = device.createComputePipeline({
      label: entry,
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module, entryPoint: entry },
    });
  }
  bind(resources: Record<number, GPUBuffer>, label = this.entry): GPUBindGroup {
    const entries: GPUBindGroupEntry[] = [];
    for (const b of Object.keys(this.bindings).map(Number)) {
      const buffer = resources[b];
      if (!buffer) throw new Error(`kernel ${this.entry}: missing buffer for binding ${b}`);
      entries.push({ binding: b, resource: { buffer } });
    }
    return this.device.createBindGroup({ label, layout: this.layout, entries });
  }
  dispatch(pass: GPUComputePassEncoder, group: GPUBindGroup, threads: number) {
    const groups = Math.ceil(threads / this.workgroupSize);
    if (groups > MAX_GROUPS) throw new Error(`${this.entry}: ${threads} threads exceeds 1D dispatch limit`);
    if (groups === 0) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups);
  }
}

export function storageBuffer(device: GPUDevice, size: number, label: string, extraUsage = 0): GPUBuffer {
  return device.createBuffer({
    label, size: Math.max(16, Math.ceil(size / 16) * 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
  });
}

export function uniformBuffer(device: GPUDevice, size: number, label: string): GPUBuffer {
  return device.createBuffer({ label, size: Math.ceil(size / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}
