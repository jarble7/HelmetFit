import * as THREE from 'three';
import { hullPerimeter } from './headModel';

// Measuring arbitrary head meshes (imported scans) the same way a tape measure
// would, plus a best-guess auto-orientation for scans exported in odd axes.

/** Head top is placed at this height so imported heads line up with the sample head (ear level ≈ 0). */
export const HEAD_TOP_Y = 130;

/** Triangle-soup positions (9 floats per triangle). */
export function triangleSoup(g: THREE.BufferGeometry): Float32Array {
  const p = g.getAttribute('position');
  const idx = g.getIndex();
  const n = idx ? idx.count : p.count;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const k = idx ? idx.getX(i) : i;
    out[i * 3] = p.getX(k); out[i * 3 + 1] = p.getY(k); out[i * 3 + 2] = p.getZ(k);
  }
  return out;
}

/**
 * Points where the mesh crosses the plane dot(p, up) = h, expressed in the 2D
 * frame (e1, e2). Up/e1/e2 must be orthonormal.
 */
export function slicePoints(soup: Float32Array, up: THREE.Vector3, e1: THREE.Vector3, e2: THREE.Vector3, h: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  const hv = [0, 0, 0];
  for (let t = 0; t < soup.length; t += 9) {
    for (let k = 0; k < 3; k++) hv[k] = soup[t + k * 3] * up.x + soup[t + k * 3 + 1] * up.y + soup[t + k * 3 + 2] * up.z - h;
    if ((hv[0] > 0 && hv[1] > 0 && hv[2] > 0) || (hv[0] < 0 && hv[1] < 0 && hv[2] < 0)) continue;
    for (let k = 0; k < 3; k++) {
      const a = k, b = (k + 1) % 3;
      if ((hv[a] < 0) === (hv[b] < 0) || hv[a] === hv[b]) continue;
      const s = hv[a] / (hv[a] - hv[b]);
      const x = soup[t + a * 3] + s * (soup[t + b * 3] - soup[t + a * 3]);
      const y = soup[t + a * 3 + 1] + s * (soup[t + b * 3 + 1] - soup[t + a * 3 + 1]);
      const z = soup[t + a * 3 + 2] + s * (soup[t + b * 3 + 2] - soup[t + a * 3 + 2]);
      out.push([x * e1.x + y * e1.y + z * e1.z, x * e2.x + y * e2.y + z * e2.z]);
    }
  }
  return out;
}

const X = new THREE.Vector3(1, 0, 0), Y = new THREE.Vector3(0, 1, 0), Z = new THREE.Vector3(0, 0, 1);

export interface HeadMeasureResult {
  circumference: number;
  /** Height (y) at which the circumference was taken. */
  circumferenceY: number;
  length: number;
  breadth: number;
  meshHeight: number;
  top: number;
}

/** Measure a head already in the canonical frame (+Y up, +Z forward). */
export function measureHead(g: THREE.BufferGeometry): HeadMeasureResult {
  const soup = triangleSoup(g);
  g.computeBoundingBox();
  const bb = g.boundingBox!;
  const top = bb.max.y;
  let best = { c: 0, y: top - 90 };
  let length = 0, breadth = 0;
  // Tape line: widest loop that's still above the ears.
  for (let d = 50; d <= 94; d += 2) {
    const pts = slicePoints(soup, Y, X, Z, top - d);
    if (pts.length < 3) continue;
    const c = hullPerimeter(pts);
    if (c > best.c) best = { c, y: top - d };
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const [x, z] of pts) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z); }
    breadth = Math.max(breadth, maxX - minX);
    length = Math.max(length, maxZ - minZ);
  }
  return { circumference: best.c, circumferenceY: best.y, length, breadth, meshHeight: bb.max.y - bb.min.y, top };
}

/**
 * Translate a canonical-orientation head so the crown is at HEAD_TOP_Y and the
 * cranium is centred on the Y axis. Mutates and returns the geometry.
 */
export function centreHead(g: THREE.BufferGeometry): THREE.BufferGeometry {
  g.computeBoundingBox();
  const top = g.boundingBox!.max.y;
  const pts = slicePoints(triangleSoup(g), Y, X, Z, top - 90);
  let cx = (g.boundingBox!.min.x + g.boundingBox!.max.x) / 2, cz = (g.boundingBox!.min.z + g.boundingBox!.max.z) / 2;
  if (pts.length >= 3) {
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const [x, z] of pts) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z); }
    cx = (minX + maxX) / 2; cz = (minZ + maxZ) / 2;
  }
  g.translate(-cx, HEAD_TOP_Y - top, -cz);
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

/**
 * Guess the rotation that brings a head scan into +Y up / +Z forward.
 * Up: the end whose cross-sections grow like a skull dome. Forward: the side
 * with the nose sticking out. Returns a quaternion to apply to the geometry.
 */
export function guessHeadOrientation(g: THREE.BufferGeometry): THREE.Quaternion {
  const soup = triangleSoup(g);
  const axes = [X, Y, Z, X.clone().negate(), Y.clone().negate(), Z.clone().negate()];
  // Most head models are cut flat under the chin / at the neck. If there's a
  // big flat face, that's the bottom — far more reliable than shape guessing.
  const cut = dominantFlatNormal(soup);
  let ups = axes;
  if (cut) {
    const down = cut.clone();
    ups = [axes.reduce((a, b) => (b.dot(down) < a.dot(down) ? b : a))];
  }
  let best = { score: -Infinity, q: new THREE.Quaternion() };
  for (const up of ups) {
    const others = axes.filter((a) => Math.abs(a.dot(up)) < 0.5);
    let top = -Infinity;
    for (let t = 0; t < soup.length; t += 3) top = Math.max(top, soup[t] * up.x + soup[t + 1] * up.y + soup[t + 2] * up.z);
    const e1 = others[0], e2 = new THREE.Vector3().crossVectors(e1, up);
    const p20 = hullPerimeter(slicePoints(soup, up, e1, e2, top - 20));
    const p60 = hullPerimeter(slicePoints(soup, up, e1, e2, top - 60));
    const p90pts = slicePoints(soup, up, e1, e2, top - 90);
    const p90 = hullPerimeter(p90pts);
    if (!cut && (p60 <= 0 || p90 <= 0)) continue;
    // A skull dome: ~0.73 ratio at 20mm, still growing at 90mm, roughly oval at 90mm.
    let dome = -Math.abs(p20 / p60 - 0.73) * 4 - Math.max(0, p60 / p90 - 1.02) * 6;
    if (p90 < 380 || p90 > 760) dome -= 2;
    // Centre of the cranium slice, then look for the nose in the face band.
    let c1 = 0, c2 = 0;
    {
      let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
      for (const [u, v] of p90pts) { a = Math.min(a, u); b = Math.max(b, u); c = Math.min(c, v); d = Math.max(d, v); }
      c1 = (a + b) / 2; c2 = (c + d) / 2;
    }
    const reach = new Map<THREE.Vector3, number>();
    for (const f of others) reach.set(f, 0);
    for (let dd = 140; dd <= 180; dd += 8) {
      for (const [u, v] of slicePoints(soup, up, e1, e2, top - dd)) {
        const w = new THREE.Vector3().addScaledVector(e1, u - c1).addScaledVector(e2, v - c2);
        for (const f of others) reach.set(f, Math.max(reach.get(f)!, w.dot(f)));
      }
    }
    let face = others[0], faceReach = -Infinity;
    for (const f of others) if (reach.get(f)! > faceReach) { faceReach = reach.get(f)!; face = f; }
    const opposite = reach.get(others.find((f) => f.dot(face) < -0.5)!)!;
    const nose = Math.min(1.5, Math.max(-1, (faceReach - opposite) / 25));
    const score = (cut ? 0 : dome) + nose;
    if (score > best.score) {
      // Rotation taking (face -> +Z, up -> +Y).
      const xAxis = new THREE.Vector3().crossVectors(up, face);
      const m = new THREE.Matrix4().makeBasis(xAxis, up, face).transpose();
      best = { score, q: new THREE.Quaternion().setFromRotationMatrix(m) };
    }
  }
  return best.q;
}

/**
 * Normal of the largest flat region if it's a sizeable part of the surface
 * (≥ 4% of the area), else null. Normals are binned ~3° apart and weighted by area.
 */
export function dominantFlatNormal(soup: Float32Array): THREE.Vector3 | null {
  const bins = new Map<string, { area: number; n: THREE.Vector3 }>();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), n = new THREE.Vector3();
  let total = 0;
  for (let t = 0; t < soup.length; t += 9) {
    a.fromArray(soup, t); b.fromArray(soup, t + 3); c.fromArray(soup, t + 6);
    n.crossVectors(b.sub(a), c.sub(a));
    const area = n.length() / 2;
    if (!(area > 0)) continue;
    n.divideScalar(area * 2);
    total += area;
    const key = `${Math.round(n.x * 20)},${Math.round(n.y * 20)},${Math.round(n.z * 20)}`;
    const e = bins.get(key);
    if (e) { e.area += area; e.n.addScaledVector(n, area); } else bins.set(key, { area, n: n.clone().multiplyScalar(area) });
  }
  let best: { area: number; n: THREE.Vector3 } | null = null;
  for (const e of bins.values()) if (!best || e.area > best.area) best = e;
  return best && best.area >= 0.04 * total ? best.n.normalize() : null;
}
