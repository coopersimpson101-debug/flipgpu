# GPU pipeline reference

All simulation work for a frame is encoded into **one compute pass** (WebGPU inserts the necessary
barriers between dispatches). Substeps repeat the dispatch list with the same uniforms.
Workgroup size is 64 for particle/grid kernels and 256 for the scan.

## PIC/FLIP substep (`src/gpu/shaders/flip.wgsl`)

| # | Kernel | Threads | Reads | Writes |
|---|---|---|---|---|
| 1 | `clearGrid` | nodes | – | accum (8 × i32 per node) |
| 2 | `p2g` | particles | pos, vel | accum via `atomicAdd` (16.16 fixed point) |
| 3 | `gridNormalize` | nodes | accum | grid, gridPrev, cellType |
| 4 | `applyForces` | nodes | grid, cellType | grid (gravity, mouse, wall BCs) |
| 5 | `diffuse` + `copyTmpToGrid` | nodes | grid | gridTmp → grid (only if ν > 0) |
| 6 | `divergence` | nodes | grid, cellType | rhs, zeroes p in non-fluid cells |
| 7 | `jacobi` ×2k (ping-pong) or `rbRed`/`rbBlack` ×k | nodes | p, rhs, cellType | p |
| 8 | `project` | nodes | grid, p, cellType | grid |
| 9 | `g2p` | particles | grid, gridPrev, cellType | pos, vel (FLIP/PIC + RK2), stats (atomicMax speed) |
| 10 | hash grid (see below) | – | pos | start, sorted |
| 11 | `pushOut` + `copyPos` ×2 | particles | pos, start, sorted | posTmp → pos |

Accumulator slots per node: `0..2` Σw·v<sub>c</sub>, `3..5` Σw, `6` density weight (cell centre), `7` particle count.

## PBF substep (`src/gpu/shaders/pbf.wgsl`)

`predict` → hash grid on predicted positions → (`computeLambda` → `computeDelta` → `applyDelta`) × iters →
`updateVelocity` → `xsph` → `finalize` (also atomicMax speed).

## Hash grid (`hashgrid.wgsl`, `scan.wgsl`)

`clearCounts` → `countParticles` (atomicAdd returns the slot) → `copyCounts` (cells + 1 elements) →
`scanBlocks` on each level (level sizes from `scanLevels()`) → `addBlockSums` back down → `scatter`.

## Render passes (`render.wgsl`)

- *Particles:* background → `vsSphere/fsSphereColor` (instanced, 6 vertices) → tank lines.
- *Surface:* `fsSphereDepth` (r32float + depth24plus) → `fsThickness` (r16float, additive) →
  `fsBlurX`/`fsBlurY` × 2 → `fsComposite` → tank lines (depth-tested against the particle depth).

## Bind-group convention

Every kernel uses an explicit layout listing only the bindings it statically uses (`src/gpu/kernel.ts`),
so a single WGSL module can declare more storage buffers than `maxStorageBuffersPerShaderStage` (8).
Binding numbers are unique within a module; hash-grid resources are bindings 20–22 in both solvers.
