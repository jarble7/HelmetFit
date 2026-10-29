import * as THREE from 'three';
import type { HeadMeasurements } from './types';

// Procedural "sample head" built from a handful of tape-measure numbers.
// The head is lofted from horizontal cross-sections (superellipses with separate
// front/back extents), topped with a superellipsoid dome, and then facial
// features (nose, brow, eye sockets, lips, chin, ears) are pushed in/out.
// Frame: +Y up, +Z forward (face), origin at ear-canal level between the ears.

export const DEFAULT_MEASUREMENTS: HeadMeasurements = {
  circumference: +(23.2 * 25.4).toFixed(2), // 23.2 in (589.28 mm) — Head-3
  length: +(8.5 * 25.4).toFixed(2),         // 8.5 in (215.90 mm) — Axis Long
  breadth: +(6.1 * 25.4).toFixed(2),        // 6.1 in (154.94 mm) — Axis Short
  height: 232,
  earToTop: 130,
  faceWidth: 148,
  jawWidth: 120,
  neckCircumference: 390,
  noseProtrusion: 22,
  earProtrusion: 18,
};

function presetFromInches(circumferenceIn: number, axisLongIn: number, axisShortIn: number): HeadMeasurements {
  const r2 = (v: number) => Math.round(v * 100) / 100;
  const circumference = r2(circumferenceIn * 25.4);
  return {
    ...scaleFromCircumference(circumference),
    circumference,
    length: r2(axisLongIn * 25.4),
    breadth: r2(axisShortIn * 25.4),
  };
}

export const HEAD_PRESETS: Record<string, HeadMeasurements> = {
  'Head-1 (Hat 6-7/8)': presetFromInches(21.7, 7.9, 5.8),
  'Head-2 (Hat 7-1/8)': presetFromInches(22.4, 8.1, 6.0),
  'Head-3 (Hat 7-3/8)': { ...DEFAULT_MEASUREMENTS },
  'Head-4 (Hat 7-1/4)': presetFromInches(22.9, 8.3, 6.0),
};

/** Fill in typical proportions from just a head circumference. */
export function scaleFromCircumference(circumference: number): HeadMeasurements {
  const k = circumference / DEFAULT_MEASUREMENTS.circumference;
  const d = DEFAULT_MEASUREMENTS;
  const r1 = (v: number) => Math.round(v * 10) / 10;
  return {
    circumference,
    length: r1(d.length * k),
    breadth: r1(d.breadth * k),
    height: r1(d.height * k),
    earToTop: r1(d.earToTop * k),
    faceWidth: r1(d.faceWidth * k),
    jawWidth: r1(d.jawWidth * k),
    neckCircumference: r1(d.neckCircumference * k),
    noseProtrusion: d.noseProtrusion,
    earProtrusion: d.earProtrusion,
  };
}

export const MEASUREMENT_INFO: Record<keyof HeadMeasurements, { label: string; help: string; min: number; max: number }> = {
  circumference: { label: 'Head circumference', help: 'Soft tape around the widest part: just above the eyebrows and ears, over the bump at the back.', min: 380, max: 720 },
  length: { label: 'Head length (Axis Long)', help: 'Front to back: from between the eyebrows to the furthest point at the back of the head (calipers or two books against a wall).', min: 140, max: 250 },
  breadth: { label: 'Head width (Axis Short)', help: 'Side to side at the widest point above the ears.', min: 110, max: 200 },
  height: { label: 'Chin to crown', help: 'Bottom of the chin to the very top of the head, measured vertically (stand against a wall, use a book on your head).', min: 160, max: 290 },
  earToTop: { label: 'Ear to crown', help: 'Vertical distance from the ear hole to the top of the head.', min: 90, max: 170 },
  faceWidth: { label: 'Cheekbone width', help: 'Across the face at the widest cheekbones.', min: 90, max: 180 },
  jawWidth: { label: 'Jaw width', help: 'Across the corners of the jaw (below the ears).', min: 70, max: 160 },
  neckCircumference: { label: 'Neck circumference', help: 'Around the neck, below the Adam\'s apple.', min: 220, max: 520 },
  noseProtrusion: { label: 'Nose depth', help: 'How far the tip of the nose sticks out in front of the cheeks.', min: 5, max: 45 },
  earProtrusion: { label: 'Ear stick-out', help: 'How far the ears stick out from the side of the head.', min: 3, max: 40 },
};

interface Level { y: number; hw: number; zf: number; zb: number; n: number }

export interface HeadShape {
  m: HeadMeasurements;
  T: number; // ear to top
  C: number; // ear to chin
  c0: number; // height of the widest (tape) level
  cranN: number;
  levels: Level[];
}

const N_AROUND = 160;
const DOME_LEVELS = 44;
const DOME_P = 2.4;

function monotoneCubic(xs: number[], ys: number[]) {
  // Fritsch–Carlson monotone cubic interpolation; xs ascending.
  const n = xs.length;
  const d: number[] = [], m: number[] = new Array(n).fill(0);
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
    if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
  }
  return (x: number) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (x > xs[i + 1]) i++;
    const h = xs[i + 1] - xs[i], t = (x - xs[i]) / h;
    const t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
  };
}

function buildLevels(m: HeadMeasurements, cranN: number): HeadShape {
  const T = m.earToTop;
  const C = Math.max(40, m.height - m.earToTop);
  const B = m.breadth / 2;
  const Lf = m.length * 0.46, Lb = m.length * 0.54;
  const c0 = 0.3 * T;
  const neckR = m.neckCircumference / (2 * Math.PI);
  const zN = -0.05 * m.length;
  const neckBack = neckR - zN;
  const neckFront = zN + neckR;
  const faceHW = m.faceWidth / 2, jawHW = m.jawWidth / 2;

  // Keyframes from the bottom of the neck up to the tape level (y ascending).
  const keys: Level[] = [
    { y: -C - 75, hw: neckR, zf: neckFront, zb: neckBack, n: 2 },
    { y: -C - 18, hw: neckR, zf: neckFront, zb: neckBack, n: 2 },
    { y: -C, hw: Math.max(neckR * 1.05, jawHW * 0.6), zf: Lf * 0.72, zb: neckBack, n: 2.2 },
    { y: -0.75 * C, hw: jawHW, zf: Lf * 0.8, zb: Math.max(neckBack, Lb * 0.5), n: 2.3 },
    { y: -0.45 * C, hw: (faceHW + jawHW) / 2, zf: Lf * 0.88, zb: Math.max(neckBack, Lb * 0.6), n: 2.4 },
    { y: -0.2 * C, hw: Math.min(faceHW, B * 0.98), zf: Lf * 0.9, zb: Lb * 0.78, n: 2.4 },
    { y: 0, hw: B * 0.94, zf: Lf * 0.93, zb: Lb * 0.92, n: (2.4 + cranN) / 2 },
    { y: c0, hw: B, zf: Lf, zb: Lb, n: cranN },
  ];
  const ky = keys.map((k) => k.y);
  const fhw = monotoneCubic(ky, keys.map((k) => k.hw));
  const fzf = monotoneCubic(ky, keys.map((k) => k.zf));
  const fzb = monotoneCubic(ky, keys.map((k) => k.zb));
  const fn = monotoneCubic(ky, keys.map((k) => k.n));

  const levels: Level[] = [];
  const bottom = keys[0].y;
  const steps = Math.ceil((c0 - bottom) / 2.5);
  for (let i = 0; i < steps; i++) {
    const y = bottom + ((c0 - bottom) * i) / steps;
    levels.push({ y, hw: fhw(y), zf: fzf(y), zb: fzb(y), n: fn(y) });
  }
  // Superellipsoid dome above the tape line; the last level collapses to the crown.
  for (let i = 0; i < DOME_LEVELS; i++) {
    const phi = (Math.PI / 2) * (i / DOME_LEVELS);
    const sy = Math.pow(Math.sin(phi), 2 / DOME_P);
    const s = Math.pow(Math.cos(phi), 2 / DOME_P);
    levels.push({ y: c0 + (T - c0) * sy, hw: B * s, zf: Lf * s, zb: Lb * s, n: cranN });
  }
  return { m, T, C, c0, cranN, levels };
}

/** Where the sample head's eye sockets are centred (±x, y). */
export function sampleHeadEyeXY(m: HeadMeasurements): { x: number; y: number } {
  return { x: 0.23 * m.faceWidth, y: 0.02 * m.earToTop };
}

const gauss = (v: number, s: number) => Math.exp(-(v / s) * (v / s));

/** Radial (horizontal) feature displacement at a base surface point. */
function featureOffset(shape: HeadShape, x: number, y: number, z: number, cosT: number, sinT: number): number {
  const { m, T, C } = shape;
  let d = 0;
  if (cosT > 0.3) {
    const front = Math.min(1, (cosT - 0.3) / 0.3);
    // Nose: rises from the bridge to the tip, then drops to the upper lip.
    const yRoot = 0.12 * T, yTip = -0.32 * C, yBase = -0.42 * C;
    let prof = 0;
    if (y <= yRoot && y >= yTip) prof = 0.25 + 0.75 * ((yRoot - y) / (yRoot - yTip));
    else if (y < yTip && y > yBase) prof = 1 - (yTip - y) / (yTip - yBase);
    if (prof > 0) {
      const sig = 6 + 6 * Math.min(1, Math.max(0, (yRoot - y) / (yRoot - yTip)));
      d += m.noseProtrusion * prof * gauss(x, sig);
    }
    // Eye sockets.
    const eye = sampleHeadEyeXY(m);
    d -= 8 * front * (gauss(x - eye.x, 14) + gauss(x + eye.x, 14)) * gauss(y - eye.y, 10);
    // Brow ridge.
    d += 3 * front * gauss(x, 45) * gauss(y - 0.18 * T, 7);
    // Lips and chin.
    d += 4 * front * gauss(x, 22) * gauss(y + 0.55 * C, 8);
    d += 6 * front * gauss(x, 16) * gauss(y + 0.88 * C, 9);
  }
  if (Math.abs(sinT) > 0.6) {
    // Ears: flat-topped oval bumps on the sides, centred just behind the ear canal.
    const q = ((z + 12) / 17) ** 2 + ((y - 2) / (0.19 * T)) ** 2;
    if (q < 1) {
      const side = Math.min(1, (Math.abs(sinT) - 0.6) / 0.25);
      d += m.earProtrusion * side * Math.pow(1 - q, 0.6);
    }
  }
  return d;
}

function ringPoint(shape: HeadShape, lv: Level, theta: number, out: THREE.Vector3) {
  const s = Math.sin(theta), c = Math.cos(theta);
  const e = 2 / lv.n;
  const x = lv.hw * Math.sign(s) * Math.pow(Math.abs(s), e);
  const z = (c >= 0 ? lv.zf : lv.zb) * Math.sign(c) * Math.pow(Math.abs(c), e);
  const r = Math.hypot(x, z) || 1;
  const d = featureOffset(shape, x, lv.y, z, c, s);
  return out.set(x + (x / r) * d, lv.y, z + (z / r) * d);
}

/** Perimeter of the convex hull of 2D points (what a tape measure reads). */
export function hullPerimeter(pts: Array<[number, number]>): number {
  if (pts.length < 3) return 0;
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Array<[number, number]> = [], upper: Array<[number, number]> = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  let per = 0;
  for (let i = 0; i < hull.length; i++) { const a = hull[i], b = hull[(i + 1) % hull.length]; per += Math.hypot(a[0] - b[0], a[1] - b[1]); }
  return per;
}

function tapeCircumference(shape: HeadShape): number {
  const lv = shape.levels.find((l) => Math.abs(l.y - shape.c0) < 1e-6) ?? { y: shape.c0, hw: shape.m.breadth / 2, zf: shape.m.length / 2, zb: shape.m.length / 2, n: shape.cranN };
  const v = new THREE.Vector3();
  const pts: Array<[number, number]> = [];
  for (let j = 0; j < 720; j++) { ringPoint(shape, lv, (j / 720) * Math.PI * 2, v); pts.push([v.x, v.z]); }
  return hullPerimeter(pts);
}

export interface HeadBuildResult {
  geometry: THREE.BufferGeometry;
  /** Circumference the model actually has (may differ if the inputs are inconsistent). */
  modelCircumference: number;
  /** Warning when the measurements can't all be honoured. */
  warning: string | null;
}

let sampleHeadTemplate: THREE.BufferGeometry | null = null;

/** Set the base 3D mesh (Head-3.stl) used for the sample head. */
export function setSampleHeadTemplate(geometry: THREE.BufferGeometry | null) {
  sampleHeadTemplate = geometry;
}

/** Scale the Head-3 sample head mesh to match `m` (falls back to the procedural head if not loaded). */
export function buildSampleHead(m: HeadMeasurements): HeadBuildResult {
  if (!sampleHeadTemplate) return buildParametricHead(m);
  const ref = DEFAULT_MEASUREMENTS;
  const sx = m.breadth / ref.breadth;
  const sy = m.height / ref.height;
  const sz = m.length / ref.length;
  const geometry = sampleHeadTemplate.clone();
  geometry.scale(sx, sy, sz);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const axesCirc = ref.circumference * Math.hypot(m.breadth, m.length) / Math.hypot(ref.breadth, ref.length);
  const diff = m.circumference - axesCirc;
  const warning = Math.abs(diff) > 8
    ? 'Circumference differs from Axis Long and Axis Short; click "Estimate rest from circumference" to scale length and width to match.'
    : null;
  return { geometry, modelCircumference: warning ? axesCirc : m.circumference, warning };
}

/** Build the procedural head mesh. The cranium squareness is solved so the tape circumference matches. */
export function buildParametricHead(m: HeadMeasurements): HeadBuildResult {
  let lo = 1.6, hi = 4.0;
  const circ = (n: number) => tapeCircumference(buildLevels(m, n));
  let n = 2.2;
  const cLo = circ(lo), cHi = circ(hi);
  let warning: string | null = null;
  if (m.circumference <= cLo) { n = lo; warning = 'Circumference is small for this length and width; the model will be larger than the circumference you entered.'; }
  else if (m.circumference >= cHi) { n = hi; warning = 'Circumference is large for this length and width; the model will be smaller than the circumference you entered.'; }
  else {
    for (let i = 0; i < 30; i++) { n = (lo + hi) / 2; if (circ(n) < m.circumference) lo = n; else hi = n; }
  }
  const shape = buildLevels(m, n);
  const geometry = meshFromShape(shape);
  return { geometry, modelCircumference: tapeCircumference(shape), warning };
}

function meshFromShape(shape: HeadShape): THREE.BufferGeometry {
  const { levels, T } = shape;
  const rings = levels.length;
  const pos = new Float32Array((rings * N_AROUND + 2) * 3);
  const v = new THREE.Vector3();
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < N_AROUND; j++) {
      ringPoint(shape, levels[i], (j / N_AROUND) * Math.PI * 2, v);
      pos.set([v.x, v.y, v.z], (i * N_AROUND + j) * 3);
    }
  }
  const top = rings * N_AROUND, bottom = top + 1;
  pos.set([0, T, 0], top * 3);
  // Bottom cap centre sits under the middle of the neck.
  pos.set([0, levels[0].y, (levels[0].zf - levels[0].zb) / 2], bottom * 3);
  const idx: number[] = [];
  const at = (i: number, j: number) => i * N_AROUND + (j % N_AROUND);
  for (let i = 0; i < rings - 1; i++) {
    for (let j = 0; j < N_AROUND; j++) {
      idx.push(at(i, j), at(i, j + 1), at(i + 1, j));
      idx.push(at(i, j + 1), at(i + 1, j + 1), at(i + 1, j));
    }
  }
  for (let j = 0; j < N_AROUND; j++) {
    idx.push(at(rings - 1, j), at(rings - 1, j + 1), top);
    idx.push(bottom, at(0, j + 1), at(0, j));
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}
