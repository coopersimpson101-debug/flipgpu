// Multi-level exclusive prefix sum (in place). Level l scans blocks of 256 elements with a
// Hillis-Steele scan in workgroup memory and writes block totals to the next level's buffer;
// after the top level is scanned, `addBlockSums` propagates offsets back down.
// CPU emulation used in the unit tests: blockedExclusiveScan() in src/cpu/spatial.ts.

struct ScanParams { n: u32, p0: u32, p1: u32, p2: u32 };

@group(0) @binding(0) var<uniform> SP: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;

var<workgroup> tmp: array<u32, 256>;

@compute @workgroup_size(256)
fn scanBlocks(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
  let i = gid.x;
  var x = 0u;
  if (i < SP.n) { x = data[i]; }
  tmp[lid.x] = x;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off * 2u) {
    var t = 0u;
    if (lid.x >= off) { t = tmp[lid.x - off]; }
    workgroupBarrier();
    tmp[lid.x] = tmp[lid.x] + t;
    workgroupBarrier();
  }
  if (i < SP.n) { data[i] = tmp[lid.x] - x; }
  if (lid.x == 255u) { sums[wid.x] = tmp[255]; }
}

@compute @workgroup_size(256)
fn addBlockSums(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(workgroup_id) wid: vec3<u32>) {
  if (gid.x < SP.n) { data[gid.x] = data[gid.x] + sums[wid.x]; }
}
