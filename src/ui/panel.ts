/** Hand-written control panel bindings (no UI library). */
import type { SolverKind } from '../gpu/solver';
import type { RenderMode } from '../gpu/renderer';
import type { SceneName } from '../sim/scenes';

export interface PanelValues {
  solver: SolverKind; scene: SceneName; count: number;
  flipRatio: number; viscosity: number; gravity: number; pressureIters: number;
  pressureSolver: 'jacobi' | 'rbgs'; densityCorrection: number; cfl: number; renderMode: RenderMode;
}

export interface UIState { fire(): void; reset(): void; togglePause(): void; values(): PanelValues }

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export function bindUI(cb: {
  onChange(v: PanelValues): void;
  onRebuild(v: PanelValues): void;
  onPause(): boolean;
  onStep(): void;
}): UIState {
  const sel = (id: string) => $<HTMLSelectElement>(id);
  const rng = (id: string) => $<HTMLInputElement>(id);
  const values = (): PanelValues => ({
    solver: sel('solver').value as SolverKind,
    scene: sel('scene').value as SceneName,
    count: Number(sel('count').value),
    flipRatio: Number(rng('flipRatio').value),
    viscosity: Number(rng('viscosity').value),
    gravity: Number(rng('gravity').value),
    pressureIters: Number(rng('pressureIters').value),
    pressureSolver: sel('pressureSolver').value as 'jacobi' | 'rbgs',
    densityCorrection: Number(rng('densityCorrection').value),
    cfl: Number(rng('cfl').value),
    renderMode: sel('renderMode').value as RenderMode,
  });
  const refreshLabels = () => {
    const v = values();
    const pbf = v.solver === 'pbf';
    $('flipRatioOut').textContent = v.flipRatio.toFixed(2);
    $('viscosityOut').textContent = pbf ? `XSPH c=${(0.3 * v.viscosity).toFixed(3)}` : `ν=${(0.02 * v.viscosity).toExponential(1)} m²/s`;
    $('gravityOut').textContent = `${v.gravity.toFixed(1)} m/s²`;
    $('pressureItersOut').textContent = pbf ? `${Math.max(1, Math.min(10, Math.round(v.pressureIters / 10)))} PBF iters` : String(v.pressureIters);
    $('densityCorrectionOut').textContent = v.densityCorrection.toFixed(2);
    $('cflOut').textContent = v.cfl.toFixed(1);
    document.querySelectorAll('.flip-only').forEach((el) => el.classList.toggle('disabled', pbf));
  };
  const change = () => { refreshLabels(); cb.onChange(values()); };
  const rebuild = () => { change(); cb.onRebuild(values()); };

  for (const id of ['flipRatio', 'viscosity', 'gravity', 'pressureIters', 'densityCorrection', 'cfl']) rng(id).addEventListener('input', change);
  for (const id of ['pressureSolver', 'renderMode']) sel(id).addEventListener('change', change);
  for (const id of ['solver', 'scene', 'count']) sel(id).addEventListener('change', rebuild);
  $('reset').addEventListener('click', rebuild);
  const pauseBtn = $<HTMLButtonElement>('pause');
  const togglePause = () => { pauseBtn.textContent = cb.onPause() ? 'Run' : 'Pause'; };
  pauseBtn.addEventListener('click', togglePause);
  $('step').addEventListener('click', () => cb.onStep());
  $('collapse').addEventListener('click', () => {
    const f = $('ui');
    f.classList.toggle('collapsed');
    $('collapse').textContent = f.classList.contains('collapsed') ? '+' : '–';
  });
  return { fire: rebuild, reset: rebuild, togglePause, values };
}
