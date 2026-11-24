// Uniform-grid particle binning by counting sort (count -> exclusive scan -> scatter).
// CPU twin: src/cpu/spatial.ts (countingSortGrid, blockedExclusiveScan).

struct HashParams {
  dims: vec4<u32>,   // gx, gy, gz, numParticles
  geo: vec4<f32>,    // origin.xyz, 1 / cellSize
};

@group(0) @binding(0) var<uniform> HP: HashParams;
@group(0) @binding(1) var<storage, read_write> hPos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> hCount: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> hSlot: array<u32>;
@group(0) @binding(4) var<storage, read_write> hCell: array<u32>;
@group(0) @binding(5) var<storage, read_write> hStart: array<u32>;
@group(0) @binding(6) var<storage, read_write> hSorted: array<u32>;

fn hashCellCoord(p: vec3<f32>) -> vec3<u32> {
  let g = vec3<i32>(floor((p - HP.geo.xyz) * HP.geo.w));
  let m = vec3<i32>(HP.dims.xyz) - vec3<i32>(1);
  return vec3<u32>(clamp(g, vec3<i32>(0), m));
}

@compute @workgroup_size(256)
fn clearCounts(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = HP.dims.x * HP.dims.y * HP.dims.z;
  if (id.x <= n) { atomicStore(&hCount[id.x], 0u); }
}

@compute @workgroup_size(64)
fn countParticles(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= HP.dims.w) { return; }
  let c = hashCellCoord(hPos[i].xyz);
  let cell = c.x + HP.dims.x * (c.y + HP.dims.y * c.z);
  hCell[i] = cell;
  hSlot[i] = atomicAdd(&hCount[cell], 1u);
}

// copies counts into the scan buffer (length cells + 1; the extra element becomes the total)
@group(0) @binding(7) var<storage, read_write> hCountPlain: array<u32>;
@compute @workgroup_size(256)
fn copyCounts(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = HP.dims.x * HP.dims.y * HP.dims.z;
  if (id.x < n) { hStart[id.x] = hCountPlain[id.x]; }
  if (id.x == n) { hStart[id.x] = 0u; }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= HP.dims.w) { return; }
  hSorted[hStart[hCell[i]] + hSlot[i]] = i;
}
