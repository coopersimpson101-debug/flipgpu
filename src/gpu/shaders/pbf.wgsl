// Position Based Fluids (Macklin & Mueller, SIGGRAPH 2013) -- WebGPU compute kernels.
// CPU reference (unit-tested): PBF3D in src/cpu/sph.ts
// Unit particle mass; rest density rho0 = poly6 sum over the initial lattice.

struct Params {
  dims: vec4<u32>,     // unused, unused, unused, numParticles
  p0: vec4<f32>,       // kernel radius h, dt, rho0, epsilon (CFM relaxation, 1/m^2)
  p1: vec4<f32>,       // s_corr k, 1/W(dq), XSPH c, particle radius
  p2: vec4<f32>,       // poly6 coef, spiky grad coef, max speed, unused
  gravity: vec4<f32>,  // body acceleration
  box: vec4<f32>,      // tank interior size
  mousePos: vec4<f32>, // xyz, radius
  mouseVel: vec4<f32>, // xyz, strength
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> posPred: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> posTmp: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> velTmp: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> lambda: array<f32>;
@group(0) @binding(7) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(20) var<uniform> HP: HashParams;
@group(0) @binding(21) var<storage, read_write> hStart: array<u32>;
@group(0) @binding(22) var<storage, read_write> hSorted: array<u32>;

fn N() -> u32 { return P.dims.w; }

fn poly6(r2: f32) -> f32 {
  let h = P.p0.x;
  let d = h * h - r2;
  if (d <= 0.0) { return 0.0; }
  return P.p2.x * d * d * d;
}
// dW_spiky/dr (<= 0)
fn spikyDeriv(r: f32) -> f32 {
  let h = P.p0.x;
  if (r >= h) { return 0.0; }
  let d = h - r;
  return P.p2.y * d * d;
}

fn clampBox(x: vec3<f32>) -> vec3<f32> {
  let r = P.p1.w;
  return clamp(x, vec3<f32>(r), P.box.xyz - vec3<f32>(r));
}

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= N()) { return; }
  let dt = P.p0.y;
  var v = vel[i].xyz + P.gravity.xyz * dt;
  let x = pos[i].xyz;
  if (P.mouseVel.w > 0.0) {
    let d = length(x - P.mousePos.xyz);
    if (d < P.mousePos.w) { v = mix(v, P.mouseVel.xyz, P.mouseVel.w * (1.0 - d / P.mousePos.w)); }
  }
  vel[i] = vec4<f32>(v, 0.0);
  posPred[i] = vec4<f32>(clampBox(x + v * dt), 1.0);
}

// lambda_i = -C_i / (sum_k |grad_k C_i|^2 + eps),  C_i = max(rho_i / rho0 - 1, 0)
@compute @workgroup_size(64)
fn computeLambda(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= N()) { return; }
  let xi = posPred[i].xyz;
  let h = P.p0.x;
  let rho0 = P.p0.z;
  var rho = 0.0;
  var gradI = vec3<f32>(0.0);
  var sum2 = 0.0;
  let c0 = hashCoord(xi);
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let cc = c0 + vec3<i32>(dx, dy, dz);
        if (!hashInside(cc)) { continue; }
        let cell = hashIndex(cc);
        for (var s = hStart[cell]; s < hStart[cell + 1u]; s++) {
          let j = hSorted[s];
          let d = xi - posPred[j].xyz;
          let r2 = dot(d, d);
          if (r2 >= h * h) { continue; }
          rho += poly6(r2);
          if (j != i && r2 > 1e-12) {
            let r = sqrt(r2);
            let g = d * (spikyDeriv(r) / (r * rho0));
            gradI += g;
            sum2 += dot(g, g);
          }
        }
      }
    }
  }
  // unilateral: only compression is corrected (bilateral C causes surface clumping + energy gain)
  let C = max(rho / rho0 - 1.0, 0.0);
  lambda[i] = -C / (sum2 + dot(gradI, gradI) + P.p0.w);
}

// delta p_i = 1/rho0 sum_j (lambda_i + lambda_j + s_corr) grad W(p_i - p_j)
@compute @workgroup_size(64)
fn computeDelta(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= N()) { return; }
  let xi = posPred[i].xyz;
  let h = P.p0.x;
  let rho0 = P.p0.z;
  let li = lambda[i];
  var dp = vec3<f32>(0.0);
  let c0 = hashCoord(xi);
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let cc = c0 + vec3<i32>(dx, dy, dz);
        if (!hashInside(cc)) { continue; }
        let cell = hashIndex(cc);
        for (var s = hStart[cell]; s < hStart[cell + 1u]; s++) {
          let j = hSorted[s];
          if (j == i) { continue; }
          let d = xi - posPred[j].xyz;
          let r2 = dot(d, d);
          if (r2 >= h * h || r2 < 1e-12) { continue; }
          let r = sqrt(r2);
          let q = poly6(r2) * P.p1.y;
          let sc = -P.p1.x * q * q * q * q; // artificial pressure (tensile instability fix), n = 4
          dp += d * ((li + lambda[j] + sc) * spikyDeriv(r) / (r * rho0));
        }
      }
    }
  }
  posTmp[i] = vec4<f32>(clampBox(xi + dp), 1.0);
}

@compute @workgroup_size(64)
fn applyDelta(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= N()) { return; }
  posPred[id.x] = posTmp[id.x];
}

@compute @workgroup_size(64)
fn updateVelocity(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= N()) { return; }
  vel[i] = vec4<f32>((posPred[i].xyz - pos[i].xyz) / P.p0.y, 0.0);
}

// XSPH artificial viscosity: v_i += c sum_j (v_j - v_i) W_ij / rho0
@compute @workgroup_size(64)
fn xsph(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= N()) { return; }
  let xi = posPred[i].xyz;
  let vi = vel[i].xyz;
  let h = P.p0.x;
  var acc = vec3<f32>(0.0);
  let c0 = hashCoord(xi);
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let cc = c0 + vec3<i32>(dx, dy, dz);
        if (!hashInside(cc)) { continue; }
        let cell = hashIndex(cc);
        for (var s = hStart[cell]; s < hStart[cell + 1u]; s++) {
          let j = hSorted[s];
          if (j == i) { continue; }
          let d = xi - posPred[j].xyz;
          let r2 = dot(d, d);
          if (r2 >= h * h) { continue; }
          acc += (vel[j].xyz - vi) * poly6(r2);
        }
      }
    }
  }
  var v = vi + P.p1.z * acc / P.p0.z;
  let vmax = P.p2.z;
  if (length(v) > vmax) { v = normalize(v) * vmax; }
  velTmp[i] = vec4<f32>(v, 0.0);
}

@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= N()) { return; }
  let v = velTmp[i];
  vel[i] = v;
  pos[i] = posPred[i];
  atomicMax(&stats[0], bitcast<u32>(length(v.xyz)));
}
