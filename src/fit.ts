import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';
import type { FitSettings, FitStats } from './types';
import { HEAD_TOP_Y, measureHead } from './measure';
import { detectHeadEyes, detectHelmetHoles, HELMET_VIEW_AXES, helmetEyesFor, type EyePair, type HelmetEyes, type ViewHoles } from './landmarks';

// Auto-fit: find the helmet transform (position, rotation, scale) that puts the
// helmet's inner wall a target gap away from the head everywhere it covers,
// without touching the head.
//
// Gaps are measured radially from a point in the middle of the cranium: for a
// direction u, R(u) is how far the head surface is and H(u) the first helmet
// hit. gap = H - R (negative means the helmet cuts into the head). Directions
// where the ray never hits the helmet are "uncovered" (face openings, neck).

/** Point inside the cranium that rays are cast from. */
export const RAY_ORIGIN = new THREE.Vector3(0, HEAD_TOP_Y - 115, 0);
const MAX_RAY = 400;

export function fibonacciDirections(n: number): THREE.Vector3[] {
  const out: THREE.Vector3[] = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * (i + 0.5)) / n;
    const r = Math.sqrt(1 - y * y);
    out.push(new THREE.Vector3(Math.cos(ga * i) * r, y, Math.sin(ga * i) * r));
  }
  return out;
}

export class HeadField {
  readonly bvh: MeshBVH;
  readonly origin = RAY_ORIGIN.clone();
  /** Eye positions (canonical frame), sorted by x. Null if unknown. */
  eyes: EyePair | null;
  /**
   * Stand-in neck for head models cut off at the chin. Without it a helmet
   * tipped back can "fit" with the face poking out of its neck hole.
   */
  readonly virtualNeck: VirtualNeck | null;
  /** Pass `eyes` to override detection (known for the sample head, or picked by hand). */
  constructor(readonly geometry: THREE.BufferGeometry, eyes?: EyePair | null) {
    this.bvh = new MeshBVH(geometry);
    const m = measureHead(geometry);
    this.eyes = eyes === undefined ? detectHeadEyes(this.bvh, m.length) : eyes;
    geometry.computeBoundingBox();
    const bottom = geometry.boundingBox!.min.y;
    this.virtualNeck = m.meshHeight < 250
      ? { x: 0, z: -0.1 * m.length, radius: 0.36 * m.breadth, top: bottom + 10, bottom: Math.max(bottom - 130, HEAD_TOP_Y - 300) }
      : null;
  }
  /** Distance from the origin to the outermost head surface along dir (null if none). */
  radius(dir: THREE.Vector3): number | null {
    const hits = this.bvh.raycast(new THREE.Ray(this.origin, dir), THREE.DoubleSide, 0, MAX_RAY);
    let r = -1;
    for (const h of hits) r = Math.max(r, h.distance);
    return r > 0 ? r : null;
  }
}

/** `neck`: a stand-in neck point — only an obstacle, not part of the fit statistics. */
export interface Sample { dir: THREE.Vector3; r: number; neck?: boolean }

export interface VirtualNeck { x: number; z: number; radius: number; top: number; bottom: number }

export function headSamples(field: HeadField, n: number): Sample[] {
  const out: Sample[] = [];
  const neck = field.virtualNeck;
  if (neck) {
    const around = Math.max(12, Math.round(n / 40));
    const p = new THREE.Vector3();
    for (let y = neck.top; y >= neck.bottom; y -= 8) {
      for (let k = 0; k < around; k++) {
        const t = (k / around) * Math.PI * 2;
        p.set(neck.x + neck.radius * Math.sin(t), y, neck.z + neck.radius * Math.cos(t)).sub(field.origin);
        const r = p.length();
        // Skip neck points hidden inside the head model.
        const hr = field.radius(p.clone().divideScalar(r));
        if (hr != null && hr > r) continue;
        out.push({ dir: p.clone().divideScalar(r), r, neck: true });
      }
    }
  }
  for (const dir of fibonacciDirections(n)) {
    const r = field.radius(dir);
    // Ignore directions that run out the bottom of the neck.
    if (r == null) continue;
    const p = field.origin.clone().addScaledVector(dir, r);
    if (p.y < HEAD_TOP_Y - 300) continue;
    out.push({ dir, r });
  }
  return out;
}

/**
 * Helmet assembly geometry re-centred on its bounding-box centre so that
 * rotations pivot about the helmet itself. Poses used with this field are for
 * the centred geometry; convert with toCentred / fromCentred.
 */
export class HelmetField {
  readonly bvh: MeshBVH;
  readonly centre: THREE.Vector3;
  readonly geometry: THREE.BufferGeometry;
  constructor(assemblyGeometry: THREE.BufferGeometry) {
    assemblyGeometry.computeBoundingBox();
    this.centre = assemblyGeometry.boundingBox!.getCenter(new THREE.Vector3());
    this.geometry = assemblyGeometry.translate(-this.centre.x, -this.centre.y, -this.centre.z);
    this.geometry.computeBoundingBox();
    this.bvh = new MeshBVH(this.geometry);
  }
  private viewsByAxis = new Map<number, ViewHoles>();
  private open: THREE.Vector3 | null | undefined;
  /**
   * Direction (centred-local) in which the helmet is most open: the neck hole
   * of a helmet, the open back of a mask. Rays from the middle of the helmet
   * that escape without hitting it are averaged. Null if it's closed all round.
   */
  openDirection(): THREE.Vector3 | null {
    if (this.open !== undefined) return this.open;
    const c = this.geometry.boundingBox!.getCenter(new THREE.Vector3());
    const sum = new THREE.Vector3();
    let n = 0;
    const dirs = fibonacciDirections(1500);
    for (const d of dirs) {
      if (!this.bvh.raycastFirst(new THREE.Ray(c, d), THREE.DoubleSide)) { sum.add(d); n++; }
    }
    this.open = n >= 0.03 * dirs.length && sum.length() > 0.2 * n ? sum.normalize() : null;
    return this.open;
  }
  /** Eye holes / visor for an orientation of the centred geometry (null if none found). */
  eyesFor(q: THREE.Quaternion, tilted = false): HelmetEyes | null {
    const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(q.clone().invert());
    let bestIdx = 0, bestDot = -2;
    for (let i = 0; i < HELMET_VIEW_AXES.length; i++) {
      const d = HELMET_VIEW_AXES[i].dot(fwd);
      if (d > bestDot) { bestDot = d; bestIdx = i; }
    }
    if (bestDot < (tilted ? 0.7 : 0.9)) return null;
    let vh = this.viewsByAxis.get(bestIdx);
    if (!vh) {
      // Soup vertices are unique per triangle, so a hit's triangle is face.a / 3.
      const lens = this.geometry.userData.lensTriangles as Uint8Array | null | undefined;
      vh = detectHelmetHoles(
        this.bvh,
        this.geometry.boundingBox!,
        lens ? (a) => lens[Math.floor(a / 3)] === 1 : undefined,
        [HELMET_VIEW_AXES[bestIdx]],
      )[0];
      this.viewsByAxis.set(bestIdx, vh);
    }
    return helmetEyesFor([vh], q, this.geometry.boundingBox!, tilted);
  }
  /** Assembly pose -> pose of the centred geometry. */
  toCentred(p: Pose): Pose {
    const off = this.centre.clone().multiply(p.scale).applyQuaternion(p.quaternion);
    return { position: p.position.clone().add(off), quaternion: p.quaternion.clone(), scale: p.scale.clone() };
  }
  fromCentred(p: Pose): Pose {
    const off = this.centre.clone().multiply(p.scale).applyQuaternion(p.quaternion);
    return { position: p.position.clone().sub(off), quaternion: p.quaternion.clone(), scale: p.scale.clone() };
  }
  /** World matrix for the centred geometry, given the assembly's world matrix. */
  matrixFor(assemblyMatrix: THREE.Matrix4, out = new THREE.Matrix4()): THREE.Matrix4 {
    return out.multiplyMatrices(assemblyMatrix, new THREE.Matrix4().makeTranslation(this.centre.x, this.centre.y, this.centre.z));
  }
}

/**
 * Merge pieces (each with its own matrix inside the assembly) into one
 * assembly-space triangle soup. Triangles of `lens` pieces are flagged in
 * userData.lensTriangles so eye-hole detection can look through them.
 */
export function mergePieces(pieces: Array<{ geometry: THREE.BufferGeometry; matrix: THREE.Matrix4; lens?: boolean }>): THREE.BufferGeometry {
  let total = 0;
  const soups = pieces.map((p) => {
    const g = p.geometry.getIndex() ? p.geometry.toNonIndexed() : p.geometry.clone();
    g.applyMatrix4(p.matrix);
    total += g.getAttribute('position').count;
    return g;
  });
  const pos = new Float32Array(total * 3);
  const lens = pieces.some((p) => p.lens) ? new Uint8Array(total / 3) : null;
  let o = 0;
  soups.forEach((g, i) => {
    const a = g.getAttribute('position').array as Float32Array;
    pos.set(a, o);
    if (lens && pieces[i].lens) lens.fill(1, o / 9, (o + a.length) / 9);
    o += a.length;
    g.dispose();
  });
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  out.userData.lensTriangles = lens;
  out.computeBoundingBox();
  return out;
}

/** Pose parameters: helmet world matrix = T(t) · R(base·delta) · S(s). */
export interface Pose { position: THREE.Vector3; quaternion: THREE.Quaternion; scale: THREE.Vector3 }

export function poseMatrix(p: Pose, out = new THREE.Matrix4()) {
  return out.compose(p.position, p.quaternion, p.scale);
}

const _ray = new THREE.Ray();
const _inv = new THREE.Matrix4();
const _o = new THREE.Vector3();
const _d = new THREE.Vector3();
const _hit = new THREE.Vector3();

/**
 * Gap along each sample direction for a helmet pose. Returns NaN for uncovered
 * directions. Rays are transformed into helmet space so the BVH never rebuilds.
 */
export function gapsForPose(helmet: HelmetField, origin: THREE.Vector3, samples: Sample[], pose: THREE.Matrix4, out?: Float64Array): Float64Array {
  const gaps = out ?? new Float64Array(samples.length);
  _inv.copy(pose).invert();
  _o.copy(origin).applyMatrix4(_inv);
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    _d.copy(origin).add(s.dir).applyMatrix4(_inv).sub(_o);
    const lenScale = _d.length();
    _d.divideScalar(lenScale);
    _ray.origin.copy(_o); _ray.direction.copy(_d);
    const h = helmet.bvh.raycastFirst(_ray, THREE.DoubleSide, 0, MAX_RAY * lenScale);
    if (!h) { gaps[i] = NaN; continue; }
    // Convert the local hit distance back to world mm.
    _hit.copy(h.point).applyMatrix4(pose);
    gaps[i] = _hit.distanceTo(origin) - s.r;
  }
  return gaps;
}

export function statsFromGaps(gaps: Float64Array, settings: FitSettings, samples?: Sample[]): FitStats {
  let n = 0, total = 0, sum = 0, min = Infinity, max = -Infinity, tight = 0;
  for (let i = 0; i < gaps.length; i++) {
    if (samples?.[i].neck) continue;
    total++;
    const g = gaps[i];
    if (Number.isNaN(g)) continue;
    n++; sum += g; min = Math.min(min, g); max = Math.max(max, g);
    if (g < settings.minGap) tight++;
  }
  return {
    coverage: total ? n / total : 0,
    minGap: n ? min : NaN,
    meanGap: n ? sum / n : NaN,
    maxGap: n ? max : NaN,
    tightFraction: n ? tight / n : 0,
    penetration: n && min < 0 ? -min : 0,
    samples: total,
  };
}

/**
 * Cost of a set of gaps: snug at target, never closer than minGap. Averaged
 * over covered directions only (so a mask isn't rewarded for being blown up to
 * cover more of the head), plus a mild bonus for covering more.
 */
export function costFromGaps(gaps: Float64Array, settings: FitSettings, samples?: Sample[]): number {
  const { targetGap, minGap } = settings;
  let cost = 0, n = 0, total = 0;
  for (let i = 0; i < gaps.length; i++) {
    const g = gaps[i];
    if (samples?.[i].neck) {
      // Stand-in neck: only penalise the helmet cutting through it.
      if (g < 0) cost += 20 * g * g + 100;
      continue;
    }
    total++;
    if (Number.isNaN(g)) continue;
    n++;
    if (g >= targetGap) {
      cost += Math.min((g - targetGap) ** 2, 900);
    } else {
      const d = targetGap - g;
      cost += 2 * d * d;
      if (g < minGap) { const p = minGap - g; cost += 60 * p * p + 400; }
    }
  }
  if (n === 0) return 1e6;
  const coverage = n / total;
  return cost / n + 60 * (1 - coverage);
}

// ---------------------------------------------------------------------------
// Optimiser

function nelderMead(f: (x: number[]) => number, x0: number[], step: number[], iters: number): { x: number[]; fx: number } {
  const n = x0.length;
  let simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) { const p = x0.slice(); p[i] += step[i]; simplex.push(p); }
  let vals = simplex.map(f);
  for (let k = 0; k < iters; k++) {
    const order = vals.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
    simplex = order.map((i) => simplex[i]); vals = order.map((i) => vals[i]);
    const c = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += simplex[i][j] / n;
    const worst = simplex[n];
    const at = (t: number) => c.map((cj, j) => cj + t * (worst[j] - cj));
    const xr = at(-1), fr = f(xr);
    if (fr < vals[0]) {
      const xe = at(-2), fe = f(xe);
      if (fe < fr) { simplex[n] = xe; vals[n] = fe; } else { simplex[n] = xr; vals[n] = fr; }
    } else if (fr < vals[n - 1]) {
      simplex[n] = xr; vals[n] = fr;
    } else {
      const xc = fr < vals[n] ? at(-0.5) : at(0.5), fc = f(xc);
      if (fc < Math.min(fr, vals[n])) { simplex[n] = xc; vals[n] = fc; }
      else {
        for (let i = 1; i <= n; i++) { simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j])); vals[i] = f(simplex[i]); }
      }
    }
  }
  let bi = 0;
  for (let i = 1; i < vals.length; i++) if (vals[i] < vals[bi]) bi = i;
  return { x: simplex[bi], fx: vals[bi] };
}

/** The 24 rotations that map axes onto axes. */
export function axisRotations(): THREE.Quaternion[] {
  const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1)];
  const out: THREE.Quaternion[] = [];
  for (const up of axes) for (const fwd of axes) {
    if (Math.abs(up.dot(fwd)) > 0.5) continue;
    const x = new THREE.Vector3().crossVectors(up, fwd);
    // Columns = where helmet-local X, Y, Z go. We want local `up` -> world +Y and local `fwd` -> world +Z.
    const m = new THREE.Matrix4().makeBasis(x, up, fwd).transpose();
    out.push(new THREE.Quaternion().setFromRotationMatrix(m));
  }
  return out;
}

export interface FitInput {
  head: HeadField;
  helmet: HelmetField;
  settings: FitSettings;
  /** Current assembly pose (used when locking scale / starting from current). */
  current: Pose;
  /** Head size info for initial guesses. */
  headBox: THREE.Box3;
  onProgress?: (fraction: number, message: string) => void;
  /** Abort flag checked between steps. */
  signal?: { cancelled: boolean };
  debug?: (msg: string) => void;
}

/** `pose` is the assembly pose (not the centred one). */
export interface FitResult { pose: Pose; stats: FitStats; cost: number; eyes: EyeAlignment | null }

const yieldFrame = () => new Promise<void>((r) => setTimeout(r, 0));

/** `q0` is the starting orientation the candidate was refined from. */
interface Candidate { q: THREE.Quaternion; pos: THREE.Vector3; scale: THREE.Vector3; cost: number; q0?: THREE.Quaternion; tilted?: boolean }

/** Parameter vector: [tx, ty, tz, rx, ry, rz (deg, small, about base), ls | lsx, lsy, lsz (log scale)]. */
function makeEvaluator(input: FitInput, samples: Sample[], base: Candidate, opts: { rotate: boolean; scaleMode: 'lock' | 'uniform' | 'xyz'; helmetHasEyes: boolean }) {
  const gaps = new Float64Array(samples.length);
  const m = new THREE.Matrix4();
  const e = new THREE.Euler();
  const dq = new THREE.Quaternion();
  const toPose = (x: number[]): Pose => {
    const position = new THREE.Vector3(base.pos.x + x[0], base.pos.y + x[1], base.pos.z + x[2]);
    let k = 3;
    const quaternion = base.q.clone();
    if (opts.rotate) {
      e.set(THREE.MathUtils.degToRad(x[3]), THREE.MathUtils.degToRad(x[4]), THREE.MathUtils.degToRad(x[5]));
      dq.setFromEuler(e);
      quaternion.premultiply(dq);
      k = 6;
    }
    const scale = base.scale.clone();
    if (opts.scaleMode === 'uniform') scale.multiplyScalar(Math.exp(x[k]));
    else if (opts.scaleMode === 'xyz') scale.set(scale.x * Math.exp(x[k]), scale.y * Math.exp(x[k + 1]), scale.z * Math.exp(x[k + 2]));
    return { position, quaternion, scale };
  };
  const dims = 3 + (opts.rotate ? 3 : 0) + (opts.scaleMode === 'uniform' ? 1 : opts.scaleMode === 'xyz' ? 3 : 0);
  const steps = [8, 8, 8, ...(opts.rotate ? [6, 6, 6] : []), ...(opts.scaleMode === 'uniform' ? [0.05] : opts.scaleMode === 'xyz' ? [0.05, 0.05, 0.05] : [])];
  const eyes = input.settings.useEyes && input.head.eyes ? input.helmet.eyesFor(base.q, base.tilted) : null;
  const headEyes = input.head.eyes;
  const open = input.helmet.openDirection();
  const _open = new THREE.Vector3();
  const w0 = new THREE.Vector3(), w1 = new THREE.Vector3();
  const cost = (x: number[]) => {
    const p = toPose(x);
    poseMatrix(p, m);
    gapsForPose(input.helmet, input.head.origin, samples, m, gaps);
    let c = costFromGaps(gaps, input.settings, samples);
    if (eyes && headEyes) {
      w0.copy(eyes.points[0]).applyMatrix4(m); w1.copy(eyes.points[1]).applyMatrix4(m);
      c += eyeCost(w0, w1, headEyes);
    } else if (headEyes && opts.helmetHasEyes && input.settings.useEyes) {
      // The helmet has eye openings, but this way round they don't face the face.
      c += MISSING_EYES_COST;
    }
    if (opts.rotate) {
      // Fine-tuning may tilt a starting orientation a little (files are rarely
      // exactly square), but not swing it round into a different one.
      const drift = THREE.MathUtils.radToDeg(p.quaternion.angleTo(base.q0 ?? base.q));
      if (drift > MAX_DRIFT_DEG) c += DRIFT_W * (drift - MAX_DRIFT_DEG) ** 2;
    }
    if (open) {
      // The helmet's big opening must face down (neck) or back (mask), never
      // up or forwards — otherwise a helmet tipped back can "fit" with the face
      // poking out of the neck hole.
      const o = _open.copy(open).applyQuaternion(p.quaternion);
      const ok = Math.max(-o.y, -o.z);
      if (ok < 0.5) c += OPEN_W * (0.5 - ok) ** 2;
    }
    if (opts.scaleMode === 'xyz') {
      // Mild pull towards uniform scale so stretching is only used when it helps.
      const lx = Math.log(p.scale.x), ly = Math.log(p.scale.y), lz = Math.log(p.scale.z);
      c += ANISO_W * ((lx - ly) ** 2 + (ly - lz) ** 2 + (lx - lz) ** 2);
    }
    return c;
  };
  return { toPose, cost, dims, steps, eyes };
}

const EYE_W = 2;
const EYE_SPACING_W = 0.1;
const ANISO_W = 100;
/** Same as eye holes ~9 mm off: enough to break near-ties in shape, not to override a bad fit. */
const MISSING_EYES_COST = EYE_W * 75;
const MAX_DRIFT_DEG = 20;
const DRIFT_W = 2;
/** Opening pointing straight up costs 400·1.5² = 900; pointing sideways 100. */
const OPEN_W = 400;

/** Penalty (mm²) for helmet eye points being off the head's eyes: centre left/right + up/down, and spacing. */
function eyeCost(h0: THREE.Vector3, h1: THREE.Vector3, e: EyePair): number {
  const dx = (h0.x + h1.x - e[0].x - e[1].x) / 2;
  const dy = (h0.y + h1.y - e[0].y - e[1].y) / 2;
  const ds = Math.abs(h1.x - h0.x) - Math.abs(e[1].x - e[0].x);
  return EYE_W * (dx * dx + dy * dy) + EYE_SPACING_W * ds * ds;
}

export interface EyeAlignment {
  kind: HelmetEyes['kind'];
  /** World positions of the helmet's eye points. */
  helmet: EyePair;
  /** Centre offset (helmet minus eyes), mm. */
  dx: number;
  dy: number;
  /** Eye-hole spacing minus eye spacing, mm. */
  spacing: number;
}

/** How well the helmet's eye holes line up with the head's eyes for the current placement. */
export function eyeAlignment(head: HeadField, helmet: HelmetField, assemblyMatrix: THREE.Matrix4): EyeAlignment | null {
  if (!head.eyes) return null;
  const m = helmet.matrixFor(assemblyMatrix);
  const q = new THREE.Quaternion();
  m.decompose(new THREE.Vector3(), q, new THREE.Vector3());
  const he = helmet.eyesFor(q) ?? helmet.eyesFor(q, true);
  if (!he) return null;
  const h0 = he.points[0].clone().applyMatrix4(m), h1 = he.points[1].clone().applyMatrix4(m);
  const [a, b] = h0.x <= h1.x ? [h0, h1] : [h1, h0];
  const e = head.eyes;
  return {
    kind: he.kind, helmet: [a, b],
    dx: (a.x + b.x - e[0].x - e[1].x) / 2, dy: (a.y + b.y - e[0].y - e[1].y) / 2,
    spacing: Math.abs(b.x - a.x) - Math.abs(e[1].x - e[0].x),
  };
}

/** Initial placements for one orientation: helmet-style (crown aligned) and mask-style (front aligned). */
function initialPlacements(input: FitInput, q: THREE.Quaternion, lockedScale: THREE.Vector3 | null, tilted = false): Candidate[] {
  const hb = input.helmet.geometry.boundingBox!.clone();
  const rotated = hb.clone().applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(q));
  const size = rotated.getSize(new THREE.Vector3());
  const head = input.headBox;
  const headSize = head.getSize(new THREE.Vector3());
  const pad = input.settings.targetGap + 4; // gap + typical wall thickness
  let s = lockedScale ? lockedScale.x : (headSize.x + 2 * pad) / Math.max(size.x, 1e-6);
  if (!Number.isFinite(s) || s <= 0) s = 1;
  const scale = lockedScale ? lockedScale.clone() : new THREE.Vector3(s, s, s);
  const out: Candidate[] = [];
  const centre = rotated.getCenter(new THREE.Vector3());
  // Position such that rotated+scaled box sits where we want. World box = s*rotated + pos.
  const scaledMin = rotated.min.clone().multiply(scale), scaledMax = rotated.max.clone().multiply(scale);
  const scaledCentre = centre.clone().multiply(scale);
  const headCentreX = (head.min.x + head.max.x) / 2;
  const cranZ = 0; // imported and sample heads are centred on the cranium
  // Helmet: crown of cavity a pad above the crown of the head, centred front/back.
  out.push({ q, scale, tilted, cost: Infinity, pos: new THREE.Vector3(headCentreX - scaledCentre.x, head.max.y + pad - scaledMax.y, cranZ - scaledCentre.z) });
  // Mask / face piece: back of the piece a pad in front of the face, centred at eye level.
  if ((scaledMax.z - scaledMin.z) < headSize.z * 0.75) {
    out.push({ q, scale, tilted, cost: Infinity, pos: new THREE.Vector3(headCentreX - scaledCentre.x, -10 - scaledCentre.y, head.max.z + pad - scaledMax.z) });
    // Back piece.
    out.push({ q, scale, tilted, cost: Infinity, pos: new THREE.Vector3(headCentreX - scaledCentre.x, 40 - scaledCentre.y, head.min.z - pad - scaledMin.z) });
  }
  // Generic: box centres coincide.
  const headCentre = head.getCenter(new THREE.Vector3());
  out.push({ q, scale, tilted, cost: Infinity, pos: headCentre.clone().sub(scaledCentre) });
  // Eyes: put the eye holes over the eyes, a pad in front of them.
  const eyes = input.settings.useEyes && input.head.eyes ? input.helmet.eyesFor(q, tilted) : null;
  if (eyes && input.head.eyes) {
    const e = input.head.eyes;
    const rotM = new THREE.Matrix4().compose(new THREE.Vector3(), q, scale);
    const h0 = eyes.points[0].clone().applyMatrix4(rotM), h1 = eyes.points[1].clone().applyMatrix4(rotM);
    const hMid = h0.clone().add(h1).multiplyScalar(0.5);
    const eMid = e[0].clone().add(e[1]).multiplyScalar(0.5);
    const z = Math.max(e[0].z, e[1].z) + input.settings.targetGap + 12;
    out.push({ q, scale, tilted, cost: Infinity, pos: new THREE.Vector3(eMid.x - hMid.x, eMid.y - hMid.y, z - hMid.z) });
    // Same, but sized so the eye-hole spacing matches the eye spacing.
    const spacing = Math.abs(h1.x - h0.x);
    if (!lockedScale && spacing > 1) {
      const k = Math.abs(e[1].x - e[0].x) / spacing;
      if (k > 0.6 && k < 1.6) {
        const s2 = scale.clone().multiplyScalar(k);
        const hm2 = hMid.clone().multiplyScalar(k);
        out.push({ q, scale: s2, tilted, cost: Infinity, pos: new THREE.Vector3(eMid.x - hm2.x, eMid.y - hm2.y, z - hm2.z) });
      }
    }
  }
  return out;
}

/**
 * Starting orientations from the helmet's own shape, whatever angle it
 * arrived at (e.g. pieces assembled in one part's print orientation): its
 * left-right mirror plane gives left-right exactly, and the neck opening gives
 * roughly which way is down. Front/back and tilt are left to the fit.
 */
function symmetryOrientations(input: FitInput): THREE.Quaternion[] {
  const helmet = input.helmet;
  const open = helmet.openDirection();
  if (!open) return [];
  const pos = helmet.geometry.getAttribute('position');
  const step = Math.max(1, Math.floor(pos.count / 2500));
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < pos.count; i += step) pts.push(new THREE.Vector3().fromBufferAttribute(pos, i));
  const c = pts.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(pts.length);
  const MAXD = 10;
  const score = (n: THREE.Vector3) => {
    let sum = 0;
    const r = new THREE.Vector3();
    for (const p of pts) {
      r.copy(p).addScaledVector(n, -2 * p.clone().sub(c).dot(n)); // mirror across the plane through c
      const hit = helmet.bvh.closestPointToPoint(r, undefined, 0, MAXD);
      sum += hit ? hit.distance : MAXD;
    }
    return sum / pts.length;
  };
  // The mirror plane contains the up/down direction, so its normal is ⟂ the opening.
  const e1 = new THREE.Vector3(1, 0, 0).cross(open).lengthSq() > 0.1 ? new THREE.Vector3(1, 0, 0).cross(open).normalize() : new THREE.Vector3(0, 1, 0).cross(open).normalize();
  const e2 = new THREE.Vector3().crossVectors(open, e1);
  let best = { s: Infinity, n: e1.clone() };
  for (let k = 0; k < 36; k++) {
    const a = (k / 36) * Math.PI;
    const n = e1.clone().multiplyScalar(Math.cos(a)).addScaledVector(e2, Math.sin(a));
    const sc = score(n);
    if (sc < best.s) best = { s: sc, n };
  }
  // Refine: small tilts of the normal in any direction.
  for (const d of [3, 1.5, 0.75]) {
    for (const axis of [e1, e2, open]) for (const sgn of [-1, 1]) {
      const n = best.n.clone().applyAxisAngle(axis, THREE.MathUtils.degToRad(sgn * d)).normalize();
      const sc = score(n);
      if (sc < best.s) best = { s: sc, n };
    }
  }
  if (best.s > 3) return []; // not a mirror-symmetric shape
  const out: THREE.Quaternion[] = [];
  for (const x of [best.n, best.n.clone().negate()]) {
    const up = open.clone().negate().addScaledVector(x, open.dot(x)).normalize();
    const front = new THREE.Vector3().crossVectors(x, up);
    const base = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, up, front).transpose());
    // The neck opening usually slopes back, so "up" is only rough: try a spread of tilts.
    for (const deg of [-40, -30, -20, -10, 0, 10, 20, 30, 40]) {
      out.push(base.clone().premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad(deg))));
    }
  }
  return out;
}

/**
 * Extra starting orientations from the helmet's eye holes: the line between
 * them fixes left-right exactly, so only the forward tilt is unknown — try a
 * spread of tilts. Handles helmets exported (or assembled) at odd angles that
 * the 24 square-on orientations can't reach.
 */
function eyeOrientations(input: FitInput): THREE.Quaternion[] {
  if (!input.settings.useEyes || !input.head.eyes) return [];
  const out: THREE.Quaternion[] = [];
  const seen: THREE.Quaternion[] = [];
  for (const q of axisRotations()) {
    const e = input.helmet.eyesFor(q);
    if (!e) continue;
    const qi = q.clone().invert();
    const x = e.points[1].clone().sub(e.points[0]).normalize(); // helmet-local direction that should become world +X
    const f0 = new THREE.Vector3(0, 0, 1).applyQuaternion(qi);
    const f = f0.addScaledVector(x, -f0.dot(x)).normalize();
    const up = new THREE.Vector3().crossVectors(f, x);
    const base = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, up, f).transpose());
    if (seen.some((s) => s.angleTo(base) < 0.05)) continue;
    seen.push(base);
    for (const deg of [-45, -30, -15, 15, 30, 45]) {
      out.push(base.clone().premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad(deg))));
    }
    out.push(base);
  }
  return out;
}

export async function autoFit(input: FitInput): Promise<FitResult> {
  const { settings } = input;
  const progress = input.onProgress ?? (() => {});
  const coarse = headSamples(input.head, 260);
  const medium = headSamples(input.head, 700);
  const fine = headSamples(input.head, 2400);
  const scaleMode: 'lock' | 'uniform' | 'xyz' = settings.lockScale ? 'lock' : settings.nonUniformScale ? 'xyz' : 'uniform';
  const current = input.helmet.toCentred(input.current);
  const lockedScale = settings.lockScale ? current.scale.clone() : null;

  let candidates: Candidate[] = [];
  if (settings.startFromCurrent) {
    candidates.push({ q: current.quaternion.clone(), pos: current.position.clone(), scale: current.scale.clone(), cost: Infinity });
  }
  if (!settings.startFromCurrent || settings.searchRotation) {
    const rots = settings.searchRotation ? axisRotations() : [current.quaternion.clone()];
    for (const q of rots) candidates.push(...initialPlacements(input, q, lockedScale));
    if (settings.searchRotation) {
      for (const q of eyeOrientations(input)) candidates.push(...initialPlacements(input, q, lockedScale, true));
      for (const q of symmetryOrientations(input)) candidates.push(...initialPlacements(input, q, lockedScale, true));
    }
  }

  const helmetHasEyes = settings.useEyes && !!input.head.eyes && axisRotations().some((q) => input.helmet.eyesFor(q));

  // Stage 1: quick position/scale settle for every candidate.
  const total = candidates.length;
  for (let i = 0; i < candidates.length; i++) {
    if (input.signal?.cancelled) throw new Error('cancelled');
    const c = candidates[i];
    const ev = makeEvaluator(input, coarse, c, { rotate: false, scaleMode: scaleMode === 'xyz' ? 'uniform' : scaleMode, helmetHasEyes });
    const r = nelderMead(ev.cost, new Array(ev.dims).fill(0), ev.steps.map((s) => s * 2), 45);
    const p = ev.toPose(r.x);
    candidates[i] = { q: p.quaternion, pos: p.position, scale: p.scale, cost: r.fx, q0: c.q0 ?? c.q, tilted: c.tilted };
    if (i % 4 === 3) { progress((0.5 * (i + 1)) / total, `Trying orientations ${i + 1}/${total}`); await yieldFrame(); }
  }
  candidates.sort((a, b) => a.cost - b.cost);
  const finalists = candidates.slice(0, 3);
  input.debug?.(`fit: ${total} candidates, helmetHasEyes=${helmetHasEyes}, head eyes=${!!input.head.eyes}; best: ` + candidates.slice(0, 6).map((c) => `${c.cost.toFixed(0)}${c.tilted ? 't' : ''}@${c.scale.toArray().map((v) => v.toFixed(2)).join('/')}`).join(' '));

  // Stage 2: refine the best few including small rotations.
  let best: Candidate | null = null;
  for (let i = 0; i < finalists.length; i++) {
    if (input.signal?.cancelled) throw new Error('cancelled');
    let c = finalists[i];
    for (let round = 0; round < 3; round++) {
      const ev = makeEvaluator(input, medium, c, { rotate: true, scaleMode, helmetHasEyes });
      const r = nelderMead(ev.cost, new Array(ev.dims).fill(0), ev.steps.map((s) => s / (round + 1)), 160);
      const p = ev.toPose(r.x);
      c = { q: p.quaternion, pos: p.position, scale: p.scale, cost: r.fx, q0: c.q0, tilted: c.tilted };
      progress(0.5 + (0.45 * (i * 3 + round + 1)) / (finalists.length * 3), `Refining best fit ${i + 1}/${finalists.length}`);
      await yieldFrame();
    }
    if (!best || c.cost < best.cost) best = c;
  }

  const centred: Pose = { position: best!.pos, quaternion: best!.q, scale: best!.scale };
  const gaps = gapsForPose(input.helmet, input.head.origin, fine, poseMatrix(centred));
  progress(1, 'Done');
  const pose = input.helmet.fromCentred(centred);
  const eyes = eyeAlignment(input.head, input.helmet, poseMatrix(pose));
  return { pose, stats: statsFromGaps(gaps, settings, fine), cost: costFromGaps(gaps, settings, fine), eyes };
}

/** Gap at every head vertex (NaN = not covered) for the clearance heat map. `pose` is the assembly world matrix. */
export function vertexGaps(head: HeadField, helmet: HelmetField, assemblyMatrix: THREE.Matrix4): Float64Array {
  const pose = helmet.matrixFor(assemblyMatrix);
  const pos = head.geometry.getAttribute('position');
  const samples: Sample[] = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).sub(head.origin);
    const r = v.length();
    samples.push({ dir: r > 1e-6 ? v.clone().divideScalar(r) : new THREE.Vector3(0, 1, 0), r });
  }
  return gapsForPose(helmet, head.origin, samples, pose);
}

/** Fit statistics for the assembly at `assemblyMatrix`. */
export function evaluatePose(head: HeadField, helmet: HelmetField, assemblyMatrix: THREE.Matrix4, settings: FitSettings, samples: Sample[]): FitStats {
  return statsFromGaps(gapsForPose(helmet, head.origin, samples, helmet.matrixFor(assemblyMatrix)), settings, samples);
}

/**
 * Gap along each sample direction against individual piece meshes (each with its
 * own cached local-space MeshBVH and world matrix), so moving or scaling individual
 * pieces or groups never needs to merge geometries or rebuild a BVH.
 */
export function gapsForPieces(meshes: THREE.Mesh[], origin: THREE.Vector3, samples: Sample[], out?: Float64Array): Float64Array {
  const gaps = out ?? new Float64Array(samples.length);
  gaps.fill(NaN);
  for (const m of meshes) {
    let bvh = m.geometry.userData.bvh as MeshBVH | undefined;
    if (!bvh) {
      bvh = new MeshBVH(m.geometry);
      m.geometry.userData.bvh = bvh;
    }
    const worldMat = m.matrixWorld;
    _inv.copy(worldMat).invert();
    _o.copy(origin).applyMatrix4(_inv);
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      _d.copy(origin).add(s.dir).applyMatrix4(_inv).sub(_o);
      const lenScale = _d.length();
      _d.divideScalar(lenScale);
      _ray.origin.copy(_o);
      _ray.direction.copy(_d);
      const h = bvh.raycastFirst(_ray, THREE.DoubleSide, 0, MAX_RAY * lenScale);
      if (!h) continue;
      _hit.copy(h.point).applyMatrix4(worldMat);
      const g = _hit.distanceTo(origin) - s.r;
      if (Number.isNaN(gaps[i]) || g < gaps[i]) gaps[i] = g;
    }
  }
  return gaps;
}

export function evaluatePiecesPose(head: HeadField, meshes: THREE.Mesh[], settings: FitSettings, samples: Sample[]): FitStats {
  return statsFromGaps(gapsForPieces(meshes, head.origin, samples), settings, samples);
}

export function vertexGapsForPieces(head: HeadField, meshes: THREE.Mesh[]): Float64Array {
  const pos = head.geometry.getAttribute('position');
  const samples: Sample[] = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).sub(head.origin);
    const r = v.length();
    samples.push({ dir: r > 1e-6 ? v.clone().divideScalar(r) : new THREE.Vector3(0, 1, 0), r });
  }
  return gapsForPieces(meshes, head.origin, samples);
}
