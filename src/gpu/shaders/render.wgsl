// Rendering: ray-cast sphere impostors (with correct depth) and a screen-space fluid surface
// (view-depth splat -> separable bilateral smoothing -> normal reconstruction -> Fresnel shading).

struct Camera {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  invProj: mat4x4<f32>,
  invView: mat4x4<f32>,
  viewport: vec4<f32>,  // width, height, 1/width, 1/height
  params: vec4<f32>,    // sphere radius, speed for colour saturation, filter radius (world), depth falloff
  light: vec4<f32>,     // view-space direction to light
  box: vec4<f32>,       // tank size
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(1) @binding(0) var texA: texture_2d<f32>;
@group(1) @binding(1) var texB: texture_2d<f32>;

// ------------------------------------------------------------------ sphere impostors
struct SphereVO {
  @builtin(position) clip: vec4<f32>,
  @location(0) viewPos: vec3<f32>,
  @location(1) @interpolate(flat) center: vec3<f32>,
  @location(2) @interpolate(flat) speed: f32,
};

@vertex
fn vsSphere(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SphereVO {
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  let c = (cam.view * vec4<f32>(positions[ii].xyz, 1.0)).xyz;
  let r = cam.params.x;
  // camera-facing quad through the centre, enlarged to cover the perspective silhouette;
  // the fragment shader ray-casts the exact sphere.
  let p = c + vec3<f32>(quad[vi] * r * 1.5, 0.0);
  var o: SphereVO;
  o.clip = cam.proj * vec4<f32>(p, 1.0);
  o.viewPos = p;
  o.center = c;
  o.speed = length(velocities[ii].xyz);
  return o;
}

struct Hit { pos: vec3<f32>, normal: vec3<f32>, chord: f32, ok: bool };

fn raySphere(viewPos: vec3<f32>, c: vec3<f32>) -> Hit {
  var h: Hit;
  let r = cam.params.x;
  let rd = normalize(viewPos);   // eye at the origin in view space
  let b = dot(rd, c);
  let disc = b * b - dot(c, c) + r * r;
  h.ok = disc > 0.0;
  let s = sqrt(max(disc, 0.0));
  h.pos = rd * (b - s);
  h.normal = (h.pos - c) / r;
  h.chord = 2.0 * s;
  return h;
}

fn clipDepth(viewPos: vec3<f32>) -> f32 {
  let cp = cam.proj * vec4<f32>(viewPos, 1.0);
  return cp.z / cp.w;
}

fn speedColor(speed: f32) -> vec3<f32> {
  let t = clamp(speed / cam.params.y, 0.0, 1.0);
  let deep = vec3<f32>(0.05, 0.22, 0.62);
  let mid = vec3<f32>(0.10, 0.55, 0.90);
  let foam = vec3<f32>(0.85, 0.95, 1.0);
  return select(mix(mid, foam, (t - 0.5) * 2.0), mix(deep, mid, t * 2.0), t < 0.5);
}

struct ColorDepth { @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 };

@fragment
fn fsSphereColor(in: SphereVO) -> ColorDepth {
  let hit = raySphere(in.viewPos, in.center);
  if (!hit.ok) { discard; }
  let n = hit.normal;
  let l = normalize(cam.light.xyz);
  let v = normalize(-hit.pos);
  let diff = max(dot(n, l), 0.0);
  let spec = pow(max(dot(normalize(l + v), n), 0.0), 60.0);
  let base = speedColor(in.speed);
  var o: ColorDepth;
  o.color = vec4<f32>(base * (0.28 + 0.72 * diff) + vec3<f32>(0.5 * spec), 1.0);
  o.depth = clipDepth(hit.pos);
  return o;
}

struct FloatDepth { @location(0) value: vec4<f32>, @builtin(frag_depth) depth: f32 };

// nearest surface view-space z (negative); 0 is reserved for "no fluid"
@fragment
fn fsSphereDepth(in: SphereVO) -> FloatDepth {
  let hit = raySphere(in.viewPos, in.center);
  if (!hit.ok) { discard; }
  var o: FloatDepth;
  o.value = vec4<f32>(hit.pos.z, 0.0, 0.0, 1.0);
  o.depth = clipDepth(hit.pos);
  return o;
}

// additive chord length -> fluid thickness along the view ray (for Beer-Lambert absorption)
@fragment
fn fsThickness(in: SphereVO) -> @location(0) vec4<f32> {
  let hit = raySphere(in.viewPos, in.center);
  if (!hit.ok) { discard; }
  return vec4<f32>(hit.chord, 0.0, 0.0, 1.0);
}

// ------------------------------------------------------------------ full-screen passes
struct FullVO { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> FullVO {
  var pts = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: FullVO;
  o.clip = vec4<f32>(pts[vi], 0.0, 1.0);
  o.uv = pts[vi] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return o;
}

fn skyWorld(d: vec3<f32>) -> vec3<f32> {
  let t = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
  let ground = vec3<f32>(0.34, 0.35, 0.38);
  let horizon = vec3<f32>(0.62, 0.70, 0.78);
  let zenith = vec3<f32>(0.20, 0.36, 0.62);
  var c = select(mix(ground, horizon, smoothstep(0.35, 0.5, t)), mix(horizon, zenith, smoothstep(0.5, 1.0, t)), t > 0.5);
  // a soft "sun" for highlights in reflections
  let sun = normalize((cam.invView * vec4<f32>(cam.light.xyz, 0.0)).xyz);
  c += vec3<f32>(1.0, 0.95, 0.85) * pow(max(dot(d, sun), 0.0), 400.0) * 3.0;
  return c;
}

fn viewRay(fragXY: vec2<f32>) -> vec3<f32> {
  let ndc = vec2<f32>(fragXY.x * cam.viewport.z * 2.0 - 1.0, 1.0 - fragXY.y * cam.viewport.w * 2.0);
  return vec3<f32>(ndc.x / cam.proj[0][0], ndc.y / cam.proj[1][1], -1.0);
}

fn worldDir(viewDir: vec3<f32>) -> vec3<f32> {
  return normalize((cam.invView * vec4<f32>(viewDir, 0.0)).xyz);
}

fn backgroundAt(fragXY: vec2<f32>) -> vec3<f32> {
  return skyWorld(worldDir(normalize(viewRay(fragXY))));
}

struct ColorOnly { @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 };

@fragment
fn fsBackground(in: FullVO) -> ColorOnly {
  var o: ColorOnly;
  o.color = vec4<f32>(backgroundAt(in.clip.xy), 1.0);
  o.depth = 1.0;
  return o;
}

fn bilateral(fragXY: vec2<f32>, dir: vec2<i32>) -> f32 {
  let p = vec2<i32>(fragXY);
  let d0 = textureLoad(texA, p, 0).r;
  if (d0 >= 0.0) { return 0.0; }
  let size = vec2<i32>(textureDimensions(texA));
  // filter radius in pixels = world radius projected at this depth
  let pxPerMeter = 0.5 * cam.viewport.y * cam.proj[1][1] / (-d0);
  let R = clamp(i32(cam.params.z * pxPerMeter), 1, 20);
  let sigma = f32(R) * 0.5;
  let falloff = cam.params.w;
  var sum = 0.0;
  var wsum = 0.0;
  for (var i = -R; i <= R; i++) {
    let q = clamp(p + dir * i, vec2<i32>(0), size - vec2<i32>(1));
    let d = textureLoad(texA, q, 0).r;
    if (d >= 0.0) { continue; }
    let fi = f32(i);
    let dz = (d - d0) / falloff;
    let w = exp(-fi * fi / (2.0 * sigma * sigma)) * exp(-dz * dz);
    sum += d * w;
    wsum += w;
  }
  return sum / max(wsum, 1e-6);
}

@fragment
fn fsBlurX(in: FullVO) -> @location(0) vec4<f32> { return vec4<f32>(bilateral(in.clip.xy, vec2<i32>(1, 0)), 0.0, 0.0, 1.0); }
@fragment
fn fsBlurY(in: FullVO) -> @location(0) vec4<f32> { return vec4<f32>(bilateral(in.clip.xy, vec2<i32>(0, 1)), 0.0, 0.0, 1.0); }

fn viewPosAt(p: vec2<i32>) -> vec3<f32> {
  let d = textureLoad(texA, p, 0).r;
  return viewRay(vec2<f32>(p) + vec2<f32>(0.5)) * (-d);
}

@fragment
fn fsComposite(in: FullVO) -> ColorOnly {
  let p = vec2<i32>(in.clip.xy);
  let size = vec2<i32>(textureDimensions(texA));
  let d = textureLoad(texA, p, 0).r;
  var o: ColorOnly;
  o.depth = 1.0;
  if (d >= 0.0) {
    o.color = vec4<f32>(backgroundAt(in.clip.xy), 1.0);
    return o;
  }
  let pos = viewPosAt(p);
  // central differences with the smaller one-sided difference across silhouettes
  let px1 = clamp(p + vec2<i32>(1, 0), vec2<i32>(0), size - 1);
  let px0 = clamp(p - vec2<i32>(1, 0), vec2<i32>(0), size - 1);
  let py1 = clamp(p + vec2<i32>(0, 1), vec2<i32>(0), size - 1);
  let py0 = clamp(p - vec2<i32>(0, 1), vec2<i32>(0), size - 1);
  var ddx = viewPosAt(px1) - pos;
  let ddx2 = pos - viewPosAt(px0);
  if (textureLoad(texA, px1, 0).r >= 0.0 || (textureLoad(texA, px0, 0).r < 0.0 && abs(ddx2.z) < abs(ddx.z))) { ddx = ddx2; }
  var ddy = viewPosAt(py1) - pos;
  let ddy2 = pos - viewPosAt(py0);
  if (textureLoad(texA, py1, 0).r >= 0.0 || (textureLoad(texA, py0, 0).r < 0.0 && abs(ddy2.z) < abs(ddy.z))) { ddy = ddy2; }
  // screen y grows downwards, so cross(ddy, ddx) points towards the camera
  var n = normalize(cross(ddy, ddx));
  if (n.z < 0.0) { n = -n; }
  let v = normalize(-pos);
  let l = normalize(cam.light.xyz);
  let thickness = textureLoad(texB, p, 0).r;
  // Schlick Fresnel, water F0 = 0.02
  let cosT = clamp(dot(n, v), 0.0, 1.0);
  let F = 0.02 + 0.98 * pow(1.0 - cosT, 5.0);
  let reflected = skyWorld(worldDir(reflect(-v, n)));
  let refrDir = refract(-v, n, 1.0 / 1.333);
  let behind = skyWorld(worldDir(refrDir));
  // Beer-Lambert absorption (red absorbed most strongly)
  let sigmaA = vec3<f32>(1.1, 0.38, 0.16);
  let transmit = exp(-sigmaA * thickness);
  let waterTint = vec3<f32>(0.04, 0.22, 0.42);
  let refracted = mix(waterTint, behind, transmit);
  let spec = pow(max(dot(normalize(l + v), n), 0.0), 120.0);
  let diffuse = 0.75 + 0.25 * max(dot(n, l), 0.0);
  let col = mix(refracted * diffuse, reflected, F) + vec3<f32>(spec * 0.8);
  o.color = vec4<f32>(col, 1.0);
  o.depth = clipDepth(pos);
  return o;
}

// ------------------------------------------------------------------ tank outline
@vertex
fn vsLines(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var e = array<u32, 24>(0u, 1u, 1u, 3u, 3u, 2u, 2u, 0u, 4u, 5u, 5u, 7u, 7u, 6u, 6u, 4u, 0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u);
  let k = e[vi];
  let corner = vec3<f32>(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
  return cam.proj * cam.view * vec4<f32>(corner * cam.box.xyz, 1.0);
}

@fragment
fn fsLines() -> @location(0) vec4<f32> { return vec4<f32>(0.85, 0.9, 0.95, 1.0); }
