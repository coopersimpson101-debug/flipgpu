// Browser side of scripts/capture_gif.mjs. Served by that script as /recorder.js and imported from
// the demo page's console. It wraps requestAnimationFrame so that, right after the app has encoded
// and submitted a frame (while the WebGPU canvas still holds it), the frame is copied to a 2D
// canvas, downsampled and POSTed as raw RGBA to the capture server.
const cfg = __CONFIG__;
const canvas = document.getElementById('gfx');
const stats = document.getElementById('stats');
const scratch = document.createElement('canvas');
const g = scratch.getContext('2d', { willReadFrequently: true });
const origRAF = window.requestAnimationFrame.bind(window);
let displayed = 0, sent = 0, active = false;
const pending = [];
const t0 = performance.now();

window.requestAnimationFrame = (cb) => origRAF((t) => {
  cb(t);
  if (!active) return;
  displayed++;
  if ((displayed - 1) % cfg.every !== 0 || sent >= cfg.frames) return;
  const w = cfg.width, h = Math.round((canvas.height / canvas.width) * w);
  scratch.width = w; scratch.height = h;
  g.drawImage(canvas, 0, 0, w, h);
  const rgba = g.getImageData(0, 0, w, h).data;
  const i = sent++;
  pending.push(fetch(`${cfg.endpoint}/frame?i=${i}&w=${w}&h=${h}`, { method: 'POST', body: rgba }));
  if (sent === cfg.frames) {
    active = false;
    window.requestAnimationFrame = origRAF;
    Promise.all(pending).then(() => fetch(`${cfg.endpoint}/done`, {
      method: 'POST',
      body: `${sent} frames, every ${cfg.every}th of ${displayed} displayed (${(displayed / 60).toFixed(2)} s simulated), ` +
        `wall ${((performance.now() - t0) / 1000).toFixed(1)} s, canvas ${canvas.width}x${canvas.height}; ` +
        stats.textContent.split('\n').slice(0, 2).join(' | '),
    }));
  }
});

// Pause, rebuild the scene, give the GPU a moment to build it, then resume and record from t = 0.
const pauseBtn = document.getElementById('pause');
const isPaused = () => /^run$/i.test(pauseBtn.textContent);
if (!isPaused()) pauseBtn.click();
document.getElementById('reset').click();
await new Promise((r) => setTimeout(r, 2000));
pauseBtn.click();
active = true;
console.log('[capture] recording', cfg);
