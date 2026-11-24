#!/usr/bin/env node
/**
 * Records a GIF of the running simulation, frame by frame, straight from the WebGPU canvas.
 * Nothing is synthesised: every GIF frame is a canvas frame the real app rendered.
 *
 *   1. npm run dev                                   (or open the live demo)
 *   2. node scripts/capture_gif.mjs --out docs/media/dam-break.gif
 *   3. in the demo page's devtools console:
 *        await import('http://127.0.0.1:8790/recorder.js')
 *
 * The recorder (scripts/capture_recorder.js) presses "Reset scene", then grabs every `every`-th
 * displayed frame right after the app submits it, downsamples it on a 2D canvas and POSTs raw
 * RGBA here. This script quantises each frame (gifenc) and writes the GIF when the recorder
 * says it is done. The simulation advances a fixed 1/60 s per displayed frame, so the GIF's
 * timing is independent of how fast the browser happened to run.
 *
 * Options: --out <file> --port 8790 --frames 80 --every 3 --width 640 --delay 50
 */
import { createServer } from 'node:http';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import gifenc from 'gifenc'; // CommonJS package
const { GIFEncoder, quantize, applyPalette } = gifenc;

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
};
const out = opt('out', 'docs/media/dam-break.gif');
const port = parseInt(opt('port', '8790'), 10);
const cfg = {
  frames: parseInt(opt('frames', '80'), 10),
  every: parseInt(opt('every', '3'), 10),
  width: parseInt(opt('width', '640'), 10),
  endpoint: `http://127.0.0.1:${port}`,
};
const delay = parseInt(opt('delay', '50'), 10);
const here = dirname(fileURLToPath(import.meta.url));
const recorder = readFileSync(join(here, 'capture_recorder.js'), 'utf8').replace('__CONFIG__', JSON.stringify(cfg));

const frames = new Map();
let meta = null;
const cors = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*' };

const server = createServer((req, res) => {
  const url = new URL(req.url, cfg.endpoint);
  if (req.method === 'OPTIONS') { res.writeHead(204, cors); return res.end(); }
  if (req.method === 'GET' && url.pathname === '/recorder.js') {
    res.writeHead(200, { ...cors, 'Content-Type': 'text/javascript' });
    return res.end(recorder);
  }
  if (req.method === 'POST' && url.pathname === '/frame') {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const i = parseInt(url.searchParams.get('i'), 10);
      const w = parseInt(url.searchParams.get('w'), 10), h = parseInt(url.searchParams.get('h'), 10);
      frames.set(i, { w, h, rgba: new Uint8Array(Buffer.concat(chunks)) });
      process.stdout.write(`\rreceived ${frames.size}/${cfg.frames}`);
      res.writeHead(204, cors); res.end();
    });
    return;
  }
  if (req.method === 'POST' && url.pathname === '/done') {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      meta = body;
      res.writeHead(204, cors); res.end();
      encode();
    });
    return;
  }
  res.writeHead(404, cors); res.end();
});

function encode() {
  const gif = GIFEncoder();
  const keys = [...frames.keys()].sort((a, b) => a - b);
  for (const k of keys) {
    const { w, h, rgba } = frames.get(k);
    const palette = quantize(rgba, 256);
    gif.writeFrame(applyPalette(rgba, palette), w, h, { palette, delay, repeat: 0 });
  }
  gif.finish();
  const bytes = gif.bytes();
  mkdirSync(dirname(out) || '.', { recursive: true });
  writeFileSync(out, bytes);
  const f0 = frames.get(keys[0]);
  console.log(`\nwrote ${out}: ${keys.length} frames ${f0.w}x${f0.h}, ${(bytes.length / 1e6).toFixed(2)} MB`);
  if (meta) console.log(`recorder: ${meta}`);
  server.close();
  process.exit(0);
}

server.listen(port, '127.0.0.1', () => {
  console.log(`capture server on ${cfg.endpoint}; in the demo page console run:\n  await import('${cfg.endpoint}/recorder.js')`);
});
