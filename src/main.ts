import { FlipSolver } from './gpu/flip';
import { PbfSolver } from './gpu/pbf';
import { Renderer, type RenderMode } from './gpu/renderer';
import type { SimSettings, Solver, SolverKind } from './gpu/solver';
import { OrbitCamera, invert, multiply, rayPlane, screenRay, sub, normalize, type V3 } from './math';
import { substepsForFrame } from './sim/timestep';
import type { SceneName } from './sim/scenes';
import { bindUI, type UIState } from './ui/panel';

const FRAME_DT = 1 / 60; // simulated seconds per displayed frame

function showNoGPU(reason: string) {
  const el = document.getElementById('nogpu')!;
  el.hidden = false;
  document.getElementById('nogpu-reason')!.textContent = reason;
  document.getElementById('ui')!.hidden = true;
  document.getElementById('stats')!.hidden = true;
}

async function initGPU() {
  if (!('gpu' in navigator) || !navigator.gpu) throw new Error('navigator.gpu is undefined in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No suitable GPU adapter was found (requestAdapter returned null).');
  const features: GPUFeatureName[] = [];
  if (adapter.features.has('timestamp-query')) features.push('timestamp-query');
  const want = (k: keyof GPUSupportedLimits, v: number) => Math.min(v, adapter.limits[k] as number);
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxStorageBufferBindingSize: want('maxStorageBufferBindingSize', 1 << 30),
      maxBufferSize: want('maxBufferSize', 1 << 30),
      maxStorageBuffersPerShaderStage: want('maxStorageBuffersPerShaderStage', 10),
    },
  });
  return { adapter, device, timestamps: features.includes('timestamp-query') };
}

async function main() {
  let gpu: Awaited<ReturnType<typeof initGPU>>;
  try {
    gpu = await initGPU();
  } catch (e) {
    showNoGPU((e as Error).message);
    return;
  }
  const { device } = gpu;
  const statsEl = document.getElementById('stats')!;
  let fatal = '';
  device.addEventListener('uncapturederror', (ev) => {
    const msg = (ev as GPUUncapturedErrorEvent).error.message;
    console.error('WebGPU error:', msg);
    fatal = msg;
  });
  device.lost.then((info) => { if (info.reason !== 'destroyed') showNoGPU(`GPU device lost: ${info.message}`); });

  const canvas = document.getElementById('gfx') as HTMLCanvasElement;
  const ctx = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  const renderer = await Renderer.create(device, format);
  const camera = new OrbitCamera();

  const settings: SimSettings = {
    flipRatio: 0.95, viscosity: 0.03, gravity: 9.81, pressureIters: 40, pressureSolver: 'rbgs', sorOmega: 1.6,
    densityCorrection: 0.3, pushOutIters: 2, tankAccel: [0, 0, 0],
    mouse: { pos: [0, 0, 0], vel: [0, 0, 0], radius: 0.3, strength: 0 },
  };
  let solver: Solver | null = null;
  let building = false;
  let paused = false, stepOnce = false;
  let cfl = 1.5;
  let renderMode: RenderMode = 'surface';

  let queued: [SolverKind, SceneName, number] | null = null;
  const build = async (kind: SolverKind, scene: SceneName, count: number): Promise<void> => {
    if (building) { queued = [kind, scene, count]; return; } // latest request wins
    building = true;
    statsEl.textContent = `building ${kind.toUpperCase()} (${count.toLocaleString()} particles)…`;
    try {
      const next = kind === 'flip' ? await FlipSolver.create(device, scene, count) : await PbfSolver.create(device, scene, count);
      solver?.destroy();
      solver = next;
      vmax = 0;
    } catch (e) {
      console.error(e);
      fatal = (e as Error).message;
    } finally {
      building = false;
    }
    if (queued) { const q = queued; queued = null; return build(...q); }
  };

  const ui: UIState = bindUI({
    onChange: (st) => {
      settings.flipRatio = st.flipRatio; settings.viscosity = st.viscosity; settings.gravity = st.gravity;
      settings.pressureIters = st.pressureIters; settings.pressureSolver = st.pressureSolver;
      settings.densityCorrection = st.densityCorrection;
      cfl = st.cfl; renderMode = st.renderMode;
    },
    onRebuild: (st) => { void build(st.solver, st.scene, st.count); },
    onPause: () => { paused = !paused; return paused; },
    onStep: () => { stepOnce = true; },
  });

  // ---------------------------------------------------------------- interaction
  let drag: 'none' | 'push' | 'shake' | 'orbit' = 'none';
  let lastX = 0, lastY = 0, lastT = 0;
  let prevHit: V3 | null = null;
  let mouseVel: V3 = [0, 0, 0];
  let tankVel: V3 = [0, 0, 0];
  let tankAccel: V3 = [0, 0, 0];

  const worldUnderCursor = (cx: number, cy: number): V3 | null => {
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((cx - rect.left) / rect.width) * 2 - 1, ndcY = 1 - ((cy - rect.top) / rect.height) * 2;
    const vp = multiply(camera.proj(rect.width / rect.height), camera.view());
    const inv = invert(vp);
    if (!inv) return null;
    const ray = screenRay(inv, ndcX, ndcY);
    const fwd = normalize(sub(camera.target, camera.eye()));
    const center: V3 = solver ? [solver.box[0] / 2, solver.box[1] / 2, solver.box[2] / 2] : camera.target;
    const t = rayPlane(ray.origin, ray.dir, fwd, fwd[0] * center[0] + fwd[1] * center[1] + fwd[2] * center[2]);
    if (t === null || t < 0) return null;
    return [ray.origin[0] + t * ray.dir[0], ray.origin[1] + t * ray.dir[1], ray.origin[2] + t * ray.dir[2]];
  };

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    lastX = e.clientX; lastY = e.clientY; lastT = performance.now();
    drag = e.button === 2 || e.altKey ? 'orbit' : e.shiftKey ? 'shake' : 'push';
    prevHit = worldUnderCursor(e.clientX, e.clientY);
    mouseVel = [0, 0, 0]; tankVel = [0, 0, 0];
  });
  canvas.addEventListener('pointermove', (e) => {
    const now = performance.now();
    const dtw = Math.max(1e-3, (now - lastT) / 1000);
    if (drag === 'orbit') camera.orbit(e.clientX - lastX, e.clientY - lastY);
    if (drag === 'push' || drag === 'shake') {
      const hit = worldUnderCursor(e.clientX, e.clientY);
      if (hit && prevHit) {
        const v = sub(hit, prevHit).map((x) => x / dtw) as V3;
        const sp = Math.hypot(...v), cap = drag === 'push' ? 6 : 4;
        const vc = (sp > cap ? v.map((x) => (x * cap) / sp) : v) as V3;
        if (drag === 'push') {
          mouseVel = mouseVel.map((m, a) => 0.5 * m + 0.5 * vc[a]) as V3;
          settings.mouse.pos = hit;
        } else {
          // tank frame is non-inertial: the fluid feels -a_tank
          const a = vc.map((x, k) => (x - tankVel[k]) / Math.max(dtw, 1 / 120)) as V3;
          tankVel = tankVel.map((m, k) => 0.6 * m + 0.4 * vc[k]) as V3;
          tankAccel = tankAccel.map((m, k) => 0.5 * m + 0.5 * a[k]) as V3;
        }
      }
      prevHit = hit;
    }
    lastX = e.clientX; lastY = e.clientY; lastT = now;
  });
  const endDrag = () => { drag = 'none'; prevHit = null; };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); camera.zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); ui.togglePause(); }
    if (e.code === 'KeyR') ui.reset();
  });

  // ---------------------------------------------------------------- GPU timing + CFL readback
  const querySet = gpu.timestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const tsResolve = querySet ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const tsRead = querySet ? device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
  const statsRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let readPending = false;
  let vmax = 0;
  let gpuSimMs = NaN, simMsAvg = NaN;
  let lastSubsteps = 1;
  const zero16 = new Uint32Array(4);

  let frameTimes: number[] = [];
  let prevFrame = performance.now();

  const frame = () => {
    requestAnimationFrame(frame);
    const now = performance.now();
    frameTimes.push(now - prevFrame);
    if (frameTimes.length > 60) frameTimes = frameTimes.slice(-60);
    prevFrame = now;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    if (!solver) return;
    const s = solver;

    // interaction state -> settings
    settings.mouse.strength = drag === 'push' ? 0.6 : 0;
    settings.mouse.vel = mouseVel;
    settings.mouse.radius = 0.1 * s.box[0];
    const decay = drag === 'shake' ? 0.85 : 0.5;
    tankAccel = tankAccel.map((a) => a * decay) as V3;
    const acap = 30, am = Math.hypot(...tankAccel);
    settings.tankAccel = (am > acap ? tankAccel.map((a) => (-a * acap) / am) : tankAccel.map((a) => -a)) as V3;

    const encoder = device.createCommandEncoder();
    const running = !paused || stepOnce;
    stepOnce = false;
    let didSim = false;
    if (running) {
      const substeps = substepsForFrame(FRAME_DT, vmax, s.cflLength, cfl, s.kind === 'pbf' ? 2 : 1, 8);
      lastSubsteps = substeps;
      s.writeUniforms(FRAME_DT / substeps, settings);
      device.queue.writeBuffer(s.stats, 0, zero16);
      const pass = encoder.beginComputePass(querySet && !readPending
        ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
      for (let i = 0; i < substeps; i++) s.encodeSubstep(pass, settings);
      pass.end();
      didSim = true;
    }
    const doRead = didSim && !readPending;
    if (doRead) {
      encoder.copyBufferToBuffer(s.stats, 0, statsRead, 0, 16);
      if (querySet && tsResolve && tsRead) {
        encoder.resolveQuerySet(querySet, 0, 2, tsResolve, 0);
        encoder.copyBufferToBuffer(tsResolve, 0, tsRead, 0, 16);
      }
    }
    renderer.render(encoder, ctx.getCurrentTexture().createView(), w, h, camera.view(), camera.proj(w / h),
      s.pos, s.vel, s.numParticles, s.renderRadius, s.box, renderMode);
    device.queue.submit([encoder.finish()]);

    if (doRead) {
      readPending = true;
      const steps = lastSubsteps;
      const maps = [statsRead.mapAsync(GPUMapMode.READ)];
      if (tsRead) maps.push(tsRead.mapAsync(GPUMapMode.READ));
      Promise.all(maps).then(() => {
        vmax = new Float32Array(statsRead.getMappedRange())[0];
        statsRead.unmap();
        if (tsRead) {
          const t = new BigInt64Array(tsRead.getMappedRange());
          const ns = Number(t[1] - t[0]);
          tsRead.unmap();
          if (ns > 0 && ns < 1e9) {
            gpuSimMs = ns / 1e6 / steps;
            simMsAvg = Number.isNaN(simMsAvg) ? gpuSimMs : 0.9 * simMsAvg + 0.1 * gpuSimMs;
          }
        }
        readPending = false;
      }).catch(() => { readPending = false; });
    }

    const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    const simLine = querySet
      ? (Number.isNaN(simMsAvg) ? 'sim       measuring…' : `sim       ${simMsAvg.toFixed(2)} ms/substep (GPU timestamps)`)
      : 'sim       n/a (timestamp-query unsupported)';
    statsEl.textContent = [
      s.describe(),
      `particles ${s.numParticles.toLocaleString()}`,
      `fps       ${(1000 / avg).toFixed(1)}  (${avg.toFixed(1)} ms/frame)`,
      simLine,
      `substeps  ${lastSubsteps}  dt=${((FRAME_DT / lastSubsteps) * 1000).toFixed(2)} ms  CFL=${cfl.toFixed(1)}`,
      `v_max     ${vmax.toFixed(2)} m/s`,
      paused ? '[paused]' : '',
      fatal ? `GPU error: ${fatal.slice(0, 160)}` : '',
    ].filter(Boolean).join('\n');
  };

  ui.fire();
  requestAnimationFrame(frame);
}

main();
