/** Minimal column-major 4x4 matrix / vector helpers (WebGPU clip space: z in [0, 1]). */
export type Mat4 = Float32Array;
export type V3 = [number, number, number];

export const identity = (): Mat4 => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  return o;
}

/** Right-handed perspective projection, camera looks down -z, depth mapped to [0,1]. */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const o = new Float32Array(16);
  o[0] = f / aspect;
  o[5] = f;
  o[10] = far / (near - far);
  o[11] = -1;
  o[14] = (near * far) / (near - far);
  return o;
}

export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const length = (a: V3) => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a: V3): V3 => scale(a, 1 / (length(a) || 1));

export function lookAt(eye: V3, target: V3, up: V3): Mat4 {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

export function transformPoint(m: Mat4, p: V3): [number, number, number, number] {
  const o: [number, number, number, number] = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) o[r] = m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r];
  return o;
}

/** General 4x4 inverse (cofactor expansion). Returns null if singular. */
export function invert(m: Mat4): Mat4 | null {
  const a = m, o = new Float32Array(16);
  const b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4], b02 = a[0] * a[7] - a[3] * a[4];
  const b03 = a[1] * a[6] - a[2] * a[5], b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6];
  const b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12], b08 = a[8] * a[15] - a[11] * a[12];
  const b09 = a[9] * a[14] - a[10] * a[13], b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-20) return null;
  const id = 1 / det;
  o[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * id;
  o[1] = (a[2] * b10 - a[1] * b11 - a[3] * b09) * id;
  o[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * id;
  o[3] = (a[10] * b04 - a[9] * b05 - a[11] * b03) * id;
  o[4] = (a[6] * b08 - a[4] * b11 - a[7] * b07) * id;
  o[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * id;
  o[6] = (a[14] * b02 - a[12] * b05 - a[15] * b01) * id;
  o[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * id;
  o[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * id;
  o[9] = (a[1] * b08 - a[0] * b10 - a[3] * b06) * id;
  o[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * id;
  o[11] = (a[9] * b02 - a[8] * b04 - a[11] * b00) * id;
  o[12] = (a[5] * b07 - a[4] * b09 - a[6] * b06) * id;
  o[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * id;
  o[14] = (a[13] * b01 - a[12] * b03 - a[14] * b00) * id;
  o[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * id;
  return o;
}

/** World-space ray through normalised device coords (x,y in [-1,1]). */
export function screenRay(invViewProj: Mat4, ndcX: number, ndcY: number): { origin: V3; dir: V3 } {
  const n = transformPoint(invViewProj, [ndcX, ndcY, 0]);
  const f = transformPoint(invViewProj, [ndcX, ndcY, 1]);
  const a: V3 = [n[0] / n[3], n[1] / n[3], n[2] / n[3]];
  const b: V3 = [f[0] / f[3], f[1] / f[3], f[2] / f[3]];
  return { origin: a, dir: normalize(sub(b, a)) };
}

/** Intersection parameter t of a ray with the plane dot(n, x) = d (null if parallel). */
export function rayPlane(origin: V3, dir: V3, n: V3, d: number): number | null {
  const den = dot(n, dir);
  if (Math.abs(den) < 1e-9) return null;
  return (d - dot(n, origin)) / den;
}

export class OrbitCamera {
  yaw = -0.55;
  pitch = 0.42;
  distance = 4.6;
  target: V3 = [1.5, 0.55, 0.6];
  fovY = (45 * Math.PI) / 180;
  near = 0.05;
  far = 60;

  eye(): V3 {
    const cp = Math.cos(this.pitch);
    return [
      this.target[0] + this.distance * cp * Math.sin(this.yaw),
      this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * cp * Math.cos(this.yaw),
    ];
  }
  view(): Mat4 { return lookAt(this.eye(), this.target, [0, 1, 0]); }
  proj(aspect: number): Mat4 { return perspective(this.fovY, aspect, this.near, this.far); }
  orbit(dx: number, dy: number) {
    this.yaw -= dx * 0.005;
    this.pitch = Math.min(1.45, Math.max(-0.2, this.pitch + dy * 0.005));
  }
  zoom(f: number) { this.distance = Math.min(20, Math.max(0.8, this.distance * f)); }
}
