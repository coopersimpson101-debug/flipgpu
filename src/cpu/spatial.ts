/**
 * Uniform-grid neighbour search via counting sort -- the CPU twin of `src/gpu/shaders/hashgrid.wgsl`.
 *
 *   1. count:   cellCount[c] += 1 for each particle (GPU: atomicAdd, returning a local slot)
 *   2. scan:    cellStart = exclusive_prefix_sum(cellCount), length cells+1
 *   3. scatter: sorted[cellStart[c] + slot] = particle
 *
 * Particles of cell c are then sorted[cellStart[c] .. cellStart[c+1]).
 */
export interface SortedGrid {
  cellStart: Uint32Array; // length cells + 1
  sorted: Uint32Array; // particle indices grouped by cell
  cellOf: Uint32Array;
}

export function countingSortGrid(n: number, gx: number, gy: number, gz: number,
  cellCoord: (p: number) => [number, number, number]): SortedGrid {
  const cells = gx * gy * gz;
  const count = new Uint32Array(cells);
  const slot = new Uint32Array(n);
  const cellOf = new Uint32Array(n);
  for (let p = 0; p < n; p++) {
    const [i, j, k] = cellCoord(p);
    const c = i + gx * (j + gy * k);
    cellOf[p] = c;
    slot[p] = count[c]++;
  }
  const cellStart = new Uint32Array(cells + 1);
  cellStart.set(count);
  exclusiveScanInPlace(cellStart);
  const sorted = new Uint32Array(n);
  for (let p = 0; p < n; p++) sorted[cellStart[cellOf[p]] + slot[p]] = p;
  return { cellStart, sorted, cellOf };
}

export function exclusiveScanInPlace(a: Uint32Array) {
  let acc = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i]; a[i] = acc; acc += x; }
  return acc;
}

/**
 * Emulates the GPU's multi-level blocked scan exactly: each block of `block` elements is
 * scanned with a Hillis-Steele inclusive scan (as in workgroup shared memory), block totals
 * are scanned recursively, then added back. Used to unit-test the GPU algorithm's structure.
 */
export function blockedExclusiveScan(input: Uint32Array, block = 256): Uint32Array {
  const data = Uint32Array.from(input);
  const nBlocks = Math.ceil(data.length / block);
  const sums = new Uint32Array(nBlocks);
  for (let b = 0; b < nBlocks; b++) {
    const tmp = new Uint32Array(block);
    for (let t = 0; t < block; t++) { const i = b * block + t; tmp[t] = i < data.length ? data[i] : 0; }
    const orig = Uint32Array.from(tmp);
    for (let off = 1; off < block; off *= 2) {
      const prev = Uint32Array.from(tmp); // barrier: everyone reads before anyone writes
      for (let t = off; t < block; t++) tmp[t] = prev[t] + prev[t - off];
    }
    for (let t = 0; t < block; t++) { const i = b * block + t; if (i < data.length) data[i] = tmp[t] - orig[t]; }
    sums[b] = tmp[block - 1];
  }
  if (nBlocks > 1) {
    const scannedSums = blockedExclusiveScan(sums, block);
    for (let i = 0; i < data.length; i++) data[i] += scannedSums[Math.floor(i / block)];
  }
  return data;
}

/** Number of levels / sizes the GPU scan needs for `n` elements. */
export function scanLevels(n: number, block = 256): number[] {
  const sizes = [n];
  while (sizes[sizes.length - 1] > block) sizes.push(Math.ceil(sizes[sizes.length - 1] / block));
  return sizes;
}
