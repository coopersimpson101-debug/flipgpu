// 3D PIC/FLIP on a MAC staggered grid -- WebGPU compute kernels.
// CPU reference (2D, float64, unit-tested): src/cpu/flip2d.ts
//
// Storage layout: every grid array is indexed by node(i,j,k) on a padded (nx+1)(ny+1)(nz+1) lattice.
//   grid[node].x = u on the x-face at (i h, (j+.5)h, (k+.5)h)          (world = grid - h)
//   grid[node].y = v on the y-face at ((i+.5)h, j h, (k+.5)h)
//   grid[node].z = w on the z-face at ((i+.5)h, (j+.5)h, k h)
//   grid[node].w = particle number density at the cell centre ((i+.5)h, (j+.5)h, (k+.5)h)
//   cellType/pressure/rhs[node] = cell (i,j,k) quantities (valid for i<nx, j<ny, k<nz)
// The outer ring of cells is SOLID; world coordinates of the tank interior are [0, box].
// Pressure is stored pre-scaled, p' = p dt / (rho h), so the projection is  u -= p'_i - p'_{i-1}.

struct Params {
  dims: vec4<u32>,     // nx, ny, nz, numParticles
  p0: vec4<f32>,       // h, 1/h, dt, flipRatio
  gravity: vec4<f32>,  // body acceleration xyz (gravity + tank inertial force)
  mousePos: vec4<f32>, // xyz, radius
  mouseVel: vec4<f32>, // xyz, strength in [0,1] (0 = off)
  p1: vec4<f32>,       // kinematic viscosity nu, SOR omega, rest density, density-correction gain
  p2: vec4<f32>,       // push-out radius, unused, max particle speed, unused
};

const FLUID: u32 = 0u;
const AIR: u32 = 1u;
const SOLID: u32 = 2u;
// WGSL has no float atomics: P2G accumulates in 16.16 fixed point with atomicAdd on i32.
const FIXED: f32 = 65536.0;
const STRIDE: u32 = 8u; // accum: 0-2 sum(w*v_c), 3-5 sum(w), 6 density weight, 7 particle count

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> accum: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> grid: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> gridPrev: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> cellType: array<u32>;
@group(0) @binding(7) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(8) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(9) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(10) var<storage, read_write> gridTmp: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> posTmp: array<vec4<f32>>;
@group(0) @binding(20) var<uniform> HP: HashParams;
@group(0) @binding(21) var<storage, read_write> hStart: array<u32>;
@group(0) @binding(22) var<storage, read_write> hSorted: array<u32>;

fn NX() -> u32 { return P.dims.x + 1u; }
fn NY() -> u32 { return P.dims.y + 1u; }
fn numNodes() -> u32 { return (P.dims.x + 1u) * (P.dims.y + 1u) * (P.dims.z + 1u); }
fn node(i: u32, j: u32, k: u32) -> u32 { return i + NX() * (j + NY() * k); }
fn nodeI(c: vec3<i32>) -> u32 { return node(u32(c.x), u32(c.y), u32(c.z)); }
fn coordOf(idx: u32) -> vec3<u32> {
  let nx = NX(); let ny = NY();
  return vec3<u32>(idx % nx, (idx / nx) % ny, idx / (nx * ny));
}
fn inCells(c: vec3<i32>) -> bool {
  return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(P.dims.xyz));
}
fn typeAt(c: vec3<i32>) -> u32 {
  if (!inCells(c)) { return SOLID; }
  return cellType[nodeI(c)];
}
fn h() -> f32 { return P.p0.x; }
fn gridCoord(worldPos: vec3<f32>) -> vec3<f32> { return worldPos * P.p0.y + vec3<f32>(1.0); }

// ---------------------------------------------------------------- P2G
@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= numNodes()) { return; }
  for (var c = 0u; c < STRIDE; c++) { atomicStore(&accum[id.x * STRIDE + c], 0); }
}

fn splat(slot: u32, g: vec3<f32>, value: f32, wantValue: bool) {
  let b = clamp(vec3<i32>(floor(g)), vec3<i32>(0), vec3<i32>(P.dims.xyz) - vec3<i32>(1));
  let f = clamp(g - vec3<f32>(b), vec3<f32>(0.0), vec3<f32>(1.0));
  for (var dk = 0; dk < 2; dk++) {
    for (var dj = 0; dj < 2; dj++) {
      for (var di = 0; di < 2; di++) {
        let w = select(1.0 - f.x, f.x, di == 1) * select(1.0 - f.y, f.y, dj == 1) * select(1.0 - f.z, f.z, dk == 1);
        let idx = nodeI(b + vec3<i32>(di, dj, dk)) * STRIDE;
        if (wantValue) {
          atomicAdd(&accum[idx + slot], i32(round(value * w * FIXED)));
          atomicAdd(&accum[idx + slot + 3u], i32(round(w * FIXED)));
        } else {
          atomicAdd(&accum[idx + 6u], i32(round(w * FIXED)));
        }
      }
    }
  }
}

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.dims.w) { return; }
  let g = gridCoord(pos[i].xyz);
  let vmax = P.p2.z;
  let v = clamp(vel[i].xyz, vec3<f32>(-vmax), vec3<f32>(vmax));
  splat(0u, g - vec3<f32>(0.0, 0.5, 0.5), v.x, true);
  splat(1u, g - vec3<f32>(0.5, 0.0, 0.5), v.y, true);
  splat(2u, g - vec3<f32>(0.5, 0.5, 0.0), v.z, true);
  splat(0u, g - vec3<f32>(0.5, 0.5, 0.5), 0.0, false);
  let c = clamp(vec3<i32>(floor(g)), vec3<i32>(0), vec3<i32>(P.dims.xyz) - vec3<i32>(1));
  atomicAdd(&accum[nodeI(c) * STRIDE + 7u], 1);
}

// normalise the weighted sums, save a copy for the FLIP delta, classify cells
@compute @workgroup_size(64)
fn gridNormalize(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= numNodes()) { return; }
  let b = idx * STRIDE;
  var v = vec4<f32>(0.0);
  for (var c = 0u; c < 3u; c++) {
    let ws = f32(atomicLoad(&accum[b + 3u + c]));
    if (ws > 0.0) { v[c] = f32(atomicLoad(&accum[b + c])) / ws; }
  }
  v.w = f32(atomicLoad(&accum[b + 6u])) / FIXED;
  grid[idx] = v;
  gridPrev[idx] = v;
  let q = coordOf(idx);
  var t = AIR;
  if (q.x >= P.dims.x || q.y >= P.dims.y || q.z >= P.dims.z) {
    t = SOLID;
  } else if (q.x == 0u || q.y == 0u || q.z == 0u || q.x == P.dims.x - 1u || q.y == P.dims.y - 1u || q.z == P.dims.z - 1u) {
    t = SOLID;
  } else if (atomicLoad(&accum[b + 7u]) > 0) {
    t = FLUID;
  }
  cellType[idx] = t;
}

// ---------------------------------------------------------------- forces & boundaries
fn isWallFace(c: vec3<i32>, axis: u32) -> bool {
  var o = vec3<i32>(0);
  o[axis] = 1;
  return typeAt(c - o) == SOLID || typeAt(c) == SOLID;
}

fn faceWorld(c: vec3<u32>, axis: u32) -> vec3<f32> {
  var off = vec3<f32>(0.5);
  off[axis] = 0.0;
  return (vec3<f32>(c) + off - vec3<f32>(1.0)) * h();
}

@compute @workgroup_size(64)
fn applyForces(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= numNodes()) { return; }
  let q = coordOf(idx);
  let c = vec3<i32>(q);
  var v = grid[idx];
  let dt = P.p0.z;
  for (var a = 0u; a < 3u; a++) {
    var comp = v[a] + P.gravity[a] * dt;
    // mouse interaction: relax face velocities towards the cursor velocity inside a sphere
    if (P.mouseVel.w > 0.0) {
      let d = length(faceWorld(q, a) - P.mousePos.xyz);
      if (d < P.mousePos.w) {
        let s = P.mouseVel.w * (1.0 - d / P.mousePos.w);
        comp = mix(comp, P.mouseVel[a], s);
      }
    }
    if (isWallFace(c, a)) { comp = 0.0; }
    v[a] = comp;
  }
  grid[idx] = v;
}

// explicit viscosity: u += nu dt Laplacian(u), stability-limited to nu dt / h^2 <= 1/6
@compute @workgroup_size(64)
fn diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= numNodes()) { return; }
  let c = vec3<i32>(coordOf(idx));
  let v = grid[idx];
  var o = v;
  let k = min(P.p1.x * P.p0.z * P.p0.y * P.p0.y, 1.0 / 6.0);
  let lim = vec3<i32>(P.dims.xyz);
  for (var a = 0u; a < 3u; a++) {
    if (isWallFace(c, a)) { continue; }
    var lap = 0.0;
    for (var e = 0u; e < 3u; e++) {
      var d = vec3<i32>(0);
      d[e] = 1;
      let cp = c + d; let cm = c - d;
      if (all(cp <= lim)) { lap += grid[nodeI(cp)][a] - v[a]; }
      if (all(cm >= vec3<i32>(0))) { lap += grid[nodeI(cm)][a] - v[a]; }
    }
    o[a] = v[a] + k * lap;
  }
  gridTmp[idx] = o;
}

@compute @workgroup_size(64)
fn copyTmpToGrid(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= numNodes()) { return; }
  grid[id.x] = vec4<f32>(gridTmp[id.x].xyz, grid[id.x].w);
}

// ---------------------------------------------------------------- pressure
@compute @workgroup_size(64)
fn divergence(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= numNodes()) { return; }
  let c = vec3<i32>(coordOf(idx));
  if (typeAt(c) != FLUID) {
    rhs[idx] = 0.0; pressure[idx] = 0.0; pressureOut[idx] = 0.0;
    return;
  }
  let v = grid[idx];
  var D = (grid[nodeI(c + vec3<i32>(1, 0, 0))].x - v.x)
        + (grid[nodeI(c + vec3<i32>(0, 1, 0))].y - v.y)
        + (grid[nodeI(c + vec3<i32>(0, 0, 1))].z - v.z);
  // drift correction (volume preservation): push apart over-dense regions
  let rest = P.p1.z;
  if (rest > 0.0 && P.p1.w > 0.0) {
    let compression = (v.w - rest) / rest;
    if (compression > 0.0) { D -= P.p1.w * compression * h() / P.p0.z; }
  }
  rhs[idx] = -D;
}

fn neighbourSum(c: vec3<i32>) -> vec2<f32> {
  var diag = 0.0;
  var s = 0.0;
  for (var e = 0u; e < 3u; e++) {
    for (var sgn = -1; sgn <= 1; sgn += 2) {
      var d = vec3<i32>(0);
      d[e] = sgn;
      let n = c + d;
      let t = typeAt(n);
      if (t != SOLID) { diag += 1.0; }
      if (t == FLUID) { s += pressure[nodeI(n)]; }
    }
  }
  return vec2<f32>(diag, s);
}

// Jacobi: pressureOut = (rhs + sum p_nb) / (#non-solid neighbours). Ping-pong by swapping bind groups.
@compute @workgroup_size(64)
fn jacobi(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= numNodes()) { return; }
  let c = vec3<i32>(coordOf(idx));
  if (typeAt(c) != FLUID) { pressureOut[idx] = 0.0; return; }
  let ns = neighbourSum(c);
  pressureOut[idx] = select(0.0, (rhs[idx] + ns.y) / ns.x, ns.x > 0.0);
}

// red-black Gauss-Seidel with over-relaxation, in place
fn rbUpdate(idx: u32, parity: u32) {
  if (idx >= numNodes()) { return; }
  let q = coordOf(idx);
  if (((q.x + q.y + q.z) & 1u) != parity) { return; }
  let c = vec3<i32>(q);
  if (typeAt(c) != FLUID) { return; }
  let ns = neighbourSum(c);
  if (ns.x == 0.0) { return; }
  let gs = (rhs[idx] + ns.y) / ns.x;
  let w = P.p1.y;
  pressure[idx] = (1.0 - w) * pressure[idx] + w * gs;
}
@compute @workgroup_size(64)
fn rbRed(@builtin(global_invocation_id) id: vec3<u32>) { rbUpdate(id.x, 0u); }
@compute @workgroup_size(64)
fn rbBlack(@builtin(global_invocation_id) id: vec3<u32>) { rbUpdate(id.x, 1u); }

fn pAt(c: vec3<i32>) -> f32 {
  if (typeAt(c) == FLUID) { return pressure[nodeI(c)]; }
  return 0.0;
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= numNodes()) { return; }
  let c = vec3<i32>(coordOf(idx));
  var v = grid[idx];
  let pc = pAt(c);
  for (var a = 0u; a < 3u; a++) {
    var o = vec3<i32>(0);
    o[a] = 1;
    let ta = typeAt(c - o);
    let tb = typeAt(c);
    if (ta == SOLID || tb == SOLID) {
      v[a] = 0.0;
    } else if (ta == FLUID || tb == FLUID) {
      v[a] = v[a] - (pc - pAt(c - o));
    }
  }
  grid[idx] = v;
}

// ---------------------------------------------------------------- G2P + advection
fn faceValid(c: vec3<i32>, axis: u32) -> bool {
  var o = vec3<i32>(0);
  o[axis] = 1;
  return typeAt(c - o) == FLUID || typeAt(c) == FLUID;
}

// trilinear interpolation of one velocity component using only valid faces; returns (value, weight)
fn sampleComp(gWorld: vec3<f32>, axis: u32, old: bool) -> vec2<f32> {
  var off = vec3<f32>(0.5);
  off[axis] = 0.0;
  let g = gWorld - off;
  let b = clamp(vec3<i32>(floor(g)), vec3<i32>(0), vec3<i32>(P.dims.xyz) - vec3<i32>(1));
  let f = clamp(g - vec3<f32>(b), vec3<f32>(0.0), vec3<f32>(1.0));
  var s = 0.0;
  var ws = 0.0;
  for (var dk = 0; dk < 2; dk++) {
    for (var dj = 0; dj < 2; dj++) {
      for (var di = 0; di < 2; di++) {
        let c = b + vec3<i32>(di, dj, dk);
        let w = select(1.0 - f.x, f.x, di == 1) * select(1.0 - f.y, f.y, dj == 1) * select(1.0 - f.z, f.z, dk == 1);
        if (w > 0.0 && faceValid(c, axis)) {
          var val = 0.0;
          if (old) { val = gridPrev[nodeI(c)][axis]; } else { val = grid[nodeI(c)][axis]; }
          s += w * val;
          ws += w;
        }
      }
    }
  }
  return vec2<f32>(s, ws);
}

fn sampleVel(worldPos: vec3<f32>, fallback: vec3<f32>, old: bool) -> vec3<f32> {
  let g = gridCoord(worldPos);
  var v = fallback;
  for (var a = 0u; a < 3u; a++) {
    let r = sampleComp(g, a, old);
    if (r.y > 0.0) { v[a] = r.x / r.y; }
  }
  return v;
}

fn boxMax() -> vec3<f32> { return vec3<f32>(P.dims.xyz - vec3<u32>(2u)) * h(); }

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.dims.w) { return; }
  let x = pos[i].xyz;
  let vp = vel[i].xyz;
  let vNew = sampleVel(x, vp, false);
  let vOld = sampleVel(x, vp, true);
  let flip = vp + (vNew - vOld);
  var v = mix(vNew, flip, P.p0.w);
  // RK2 (midpoint) advection through the projected grid field
  let dt = P.p0.z;
  let v1 = sampleVel(x, v, false);
  let xm = x + 0.5 * dt * v1;
  let v2 = sampleVel(xm, v, false);
  var xn = x + dt * v2;
  let r = P.p2.x;
  let lo = vec3<f32>(r);
  let hi = boxMax() - vec3<f32>(r);
  for (var a = 0u; a < 3u; a++) {
    if (xn[a] < lo[a]) { xn[a] = lo[a]; v[a] = max(v[a], 0.0); }
    if (xn[a] > hi[a]) { xn[a] = hi[a]; v[a] = min(v[a], 0.0); }
  }
  let vmax = P.p2.z;
  v = clamp(v, vec3<f32>(-vmax), vec3<f32>(vmax));
  pos[i] = vec4<f32>(xn, 1.0);
  vel[i] = vec4<f32>(v, 0.0);
  // positive floats order like their bit patterns -> atomicMax gives max speed for the CFL controller
  atomicMax(&stats[0], bitcast<u32>(length(v)));
}

// ---------------------------------------------------------------- particle push-out (anti-clumping)
@compute @workgroup_size(64)
fn pushOut(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.dims.w) { return; }
  let xi = pos[i].xyz;
  let r = P.p2.x;
  let minD = 2.0 * r;
  let c0 = hashCoord(xi);
  var corr = vec3<f32>(0.0);
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let cc = c0 + vec3<i32>(dx, dy, dz);
        if (!hashInside(cc)) { continue; }
        let cell = hashIndex(cc);
        for (var s = hStart[cell]; s < hStart[cell + 1u]; s++) {
          let j = hSorted[s];
          if (j == i) { continue; }
          let d = xi - pos[j].xyz;
          let l2 = dot(d, d);
          if (l2 < minD * minD && l2 > 1e-12) {
            let l = sqrt(l2);
            corr += d / l * (0.5 * (minD - l));
          }
        }
      }
    }
  }
  let hi = boxMax() - vec3<f32>(r);
  posTmp[i] = vec4<f32>(clamp(xi + corr, vec3<f32>(r), hi), 1.0);
}

@compute @workgroup_size(64)
fn copyPos(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= P.dims.w) { return; }
  pos[id.x] = posTmp[id.x];
}
