/**
 * Generates docs/assets/banner.svg. The motif on the right is real simulation output, not a drawing:
 * the tested CPU reference solver (src/cpu/flip2d.ts, PIC/FLIP with MIC(0)-PCG pressure) runs a 2D
 * dam break, and the particle positions at t = T are written as SVG circles, shaded by speed.
 *
 *   npx vite-node scripts/make_banner.ts [-- --t 0.42]
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { FlipSim2D, defaultParams, seedBlock, cflTimestep } from '../src/cpu/flip2d';

const args = process.argv.slice(2);
const ti = args.indexOf('--t');
const T = ti >= 0 ? parseFloat(args[ti + 1]) : 0.42;

const W = 1280, H = 400;
const nx = 64, ny = 36, h = 0.03125; // 2.0 m x 1.125 m tank
const params = defaultParams(nx, ny, h);
params.solver = 'pcg'; params.pressureIters = 300; params.densityCorrection = 0.5;
const sim = new FlipSim2D(params);
const { x, y } = seedBlock(h, 1, 22, 1, 30, 2, 0.5, 9);
sim.setParticles(x, y);
let t = 0, steps = 0;
while (t < T) {
  const dt = Math.min(cflTimestep(sim.maxParticleSpeed(), h, 1.0, 1 / 120), T - t);
  sim.step(dt);
  t += dt; steps++;
}

// map the tank into the motif area on the right
const X0 = 650, Y0 = 34, S = (H - 2 * Y0) / (ny * h); // px per metre
const TW = nx * h * S, TH = ny * h * S;
const blues = ['#1e3a8a', '#1d4ed8', '#3b82f6', '#60A5FA', '#93c5fd', '#dbeafe'];
let vmax = 0;
for (let p = 0; p < sim.n; p++) vmax = Math.max(vmax, Math.hypot(sim.vx[p], sim.vy[p]));
const dots: string[] = [];
for (let p = 0; p < sim.n; p++) {
  const sp = Math.hypot(sim.vx[p], sim.vy[p]) / (vmax || 1);
  const k = Math.min(blues.length - 1, Math.floor(sp * blues.length));
  const cx = X0 + sim.px[p] * S, cy = Y0 + TH - sim.py[p] * S;
  dots.push(`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="2.6" fill="${blues[k]}"/>`);
}

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="flipgpu: real-time PIC/FLIP and PBF fluids in WebGPU">
<defs>
<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="#21262d" stroke-width="1"/></pattern>
<pattern id="cells" x="${X0}" y="${Y0}" width="${(h * S).toFixed(3)}" height="${(h * S).toFixed(3)}" patternUnits="userSpaceOnUse"><path d="M${(h * S).toFixed(3)} 0H0V${(h * S).toFixed(3)}" fill="none" stroke="#60A5FA" stroke-opacity="0.07" stroke-width="1"/></pattern>
</defs>
<rect width="${W}" height="${H}" fill="#0d1117"/>
<rect width="${W}" height="${H}" fill="url(#grid)"/>
<rect x="${X0}" y="${Y0}" width="${TW.toFixed(1)}" height="${TH.toFixed(1)}" fill="url(#cells)" stroke="#30363d" stroke-width="2"/>
<g>
${dots.join('\n')}
</g>
<text x="${(X0 + TW - 8).toFixed(1)}" y="${(Y0 - 10).toFixed(1)}" text-anchor="end" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="13" fill="#8b949e">PIC/FLIP dam break · ${sim.n} particles · t = ${T.toFixed(2)} s</text>
<g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">
<rect x="72" y="118" width="6" height="92" fill="#60A5FA"/>
<text x="100" y="196" font-size="88" font-weight="700" fill="#e6edf3">flipgpu</text>
<text x="102" y="252" font-size="24" fill="#8b949e">Real-time 3D fluids in WebGPU compute</text>
<text x="102" y="290" font-size="18" fill="#60A5FA">PIC/FLIP on a MAC grid · Position Based Fluids</text>
</g>
</svg>
`;
mkdirSync('docs/assets', { recursive: true });
writeFileSync('docs/assets/banner.svg', svg);
console.log(`wrote docs/assets/banner.svg: ${sim.n} particles at t=${T}s after ${steps} steps, ${(svg.length / 1024).toFixed(0)} KiB`);
