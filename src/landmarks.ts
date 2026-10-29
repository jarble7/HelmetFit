import * as THREE from 'three';
import type { MeshBVH } from 'three-mesh-bvh';

// Eye landmarks: where the eyes are on the head, and where the eye holes (or a
// visor slot) are on the helmet. Auto fit uses them to line the two up.

export type EyePair = [THREE.Vector3, THREE.Vector3]; // sorted: smaller x first

// ---------------------------------------------------------------------------
// Head

/**
 * Find the eyes on a head in the canonical frame (+Y up, +Z forward, ear level
 * ≈ y 0). Eye height is the band where the face is recessed between the brow
 * above and the cheek below. Left/right position comes from typical human
 * proportions (pupils ≈ 16% of head length either side of the midline) —
 * shape alone can't reliably tell the pupil from the inner corner of the eye.
 */
export function detectHeadEyes(bvh: MeshBVH, headLength: number): EyePair | null {
  const ray = new THREE.Ray(new THREE.Vector3(), new THREE.Vector3(0, 0, -1));
  const front = (x: number, y: number) => {
    ray.origin.set(x, y, 600);
    const h = bvh.raycastFirst(ray, THREE.DoubleSide);
    return h ? h.point.z : NaN;
  };
  const xEye = 0.16 * headLength;
  const rows: Array<{ y: number; depth: number }> = [];
  for (let y = -25; y <= 45; y += 2) {
    let sum = 0, n = 0;
    for (const side of [-1, 1]) {
      for (let k = 0.6; k <= 1.41; k += 0.2) {
        const x = side * xEye * k;
        const z = front(x, y), up = front(x, y + 16), down = front(x, y - 16);
        if (![z, up, down].every(Number.isFinite) || z < 30) continue;
        sum += (up + down) / 2 - z; n++;
      }
    }
    if (n >= 6) rows.push({ y, depth: sum / n });
  }
  if (!rows.length) return null;
  const peak = rows.reduce((a, b) => (b.depth > a.depth ? b : a));
  if (peak.depth < 2) return null;
  let sw = 0, sy = 0;
  for (const r of rows) {
    const w = r.depth - 0.5 * peak.depth;
    if (w <= 0 || Math.abs(r.y - peak.y) > 15) continue;
    sw += w; sy += w * r.y;
  }
  const y = sy / sw;
  const l = surfacePointFront(bvh, -xEye, y), r = surfacePointFront(bvh, xEye, y);
  return l && r ? [l, r] : null;
}

/** Front surface point at (x, y), for analytic eye positions on the sample head. */
export function surfacePointFront(bvh: MeshBVH, x: number, y: number): THREE.Vector3 | null {
  const h = bvh.raycastFirst(new THREE.Ray(new THREE.Vector3(x, y, 600), new THREE.Vector3(0, 0, -1)), THREE.DoubleSide);
  return h ? h.point.clone() : null;
}

// ---------------------------------------------------------------------------
// Helmet

interface Hole {
  /** In-plane cell centres (a along u, b along v). */
  cells: Array<[number, number]>;
  area: number;
  centre: [number, number];
  /** Average depth (along the view axis) of the solid rim around the hole. */
  rimDepth: number;
}

export interface ViewHoles {
  front: THREE.Vector3;
  u: THREE.Vector3;
  v: THREE.Vector3;
  cell: number;
  /** Openings for each in-plane "up" direction (they depend on which way is up). */
  byUp: Array<{ up: THREE.Vector3; holes: Hole[] }>;
}

export const HELMET_VIEW_AXES = [
  new THREE.Vector3(1, 0, 0), new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0),
  new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1),
];
const AXES = HELMET_VIEW_AXES;

/**
 * Look at the helmet (in its own coordinates) from each of the 6 axis
 * directions and find openings in the near half: eye holes, visors, and
 * Corinthian-style eye openings that run down into the gap between the cheek
 * guards. A ray through an eye opening passes the front of the helmet and
 * either escapes (mask) or hits the inside of the back (full helmet).
 *
 * An opening is any empty region with helmet to its left, right and above —
 * so which way is "up" matters, and each view is analysed for all 4.
 */
export function detectHelmetHoles(
  bvh: MeshBVH,
  box: THREE.Box3,
  isLensVertex?: (vertexIndex: number) => boolean,
  axes: readonly THREE.Vector3[] = AXES,
): ViewHoles[] {
  // First hit that isn't a see-through lens piece.
  const firstHit = (r: THREE.Ray) => {
    const first = bvh.raycastFirst(r, THREE.DoubleSide);
    if (!first || !isLensVertex || !isLensVertex(first.face!.a)) return first;
    let best: THREE.Intersection | null = null;
    for (const h of bvh.raycast(r, THREE.DoubleSide)) if (!isLensVertex(h.face!.a) && (!best || h.distance < best.distance)) best = h;
    return best;
  };
  const out: ViewHoles[] = [];
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const ray = new THREE.Ray();
  for (const front of axes) {
    const u = Math.abs(front.y) > 0.5 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0).cross(front).normalize();
    const v = new THREE.Vector3().crossVectors(front, u);
    const ext = (d: THREE.Vector3) => Math.abs(d.x) * size.x + Math.abs(d.y) * size.y + Math.abs(d.z) * size.z;
    const eu = ext(u) / 2, ev = ext(v) / 2, ef = ext(front) / 2;
    const cell = Math.max(1.5, Math.max(eu, ev) / 70);
    const nu = Math.ceil((2 * eu) / cell) + 2, nv = Math.ceil((2 * ev) / cell) + 2;
    const solid = new Uint8Array(nu * nv);
    const depth = new Float32Array(nu * nv);
    ray.direction.copy(front).negate();
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        const a = -eu - cell + i * cell, b = -ev - cell + j * cell;
        ray.origin.copy(centre).addScaledVector(u, a).addScaledVector(v, b).addScaledVector(front, ef + 10);
        const h = firstHit(ray);
        if (!h) continue;
        const d = h.point.clone().sub(centre).dot(front);
        depth[i * nv + j] = d;
        // Solid only if the first hit is in the near part of the helmet; a ray
        // through an eye hole of a full helmet hits the inside of the back.
        if (d > -0.2 * ef) solid[i * nv + j] = 1;
      }
    }
    const byUp: ViewHoles['byUp'] = [];
    // Eye openings are on the outside of a helmet. Looking into the hollow
    // side of a mask its recesses look like openings, so skip that view. From
    // the hollow side the outline (silhouette edge) is nearer than the rest;
    // from outside any helmet the outline curves away.
    const hit = (i: number, j: number) => i >= 0 && j >= 0 && i < nu && j < nv && depth[i * nv + j] !== 0;
    let bSum = 0, bN = 0, aSum = 0, aN = 0;
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
      if (!hit(i, j)) continue;
      const d = depth[i * nv + j];
      aSum += d; aN++;
      if (!hit(i + 1, j) || !hit(i - 1, j) || !hit(i, j + 1) || !hit(i, j - 1)) { bSum += d; bN++; }
    }
    // Eye openings sit in a broad face of the helmet; a view that's mostly
    // empty (looking at a mask's rim edge-on) only produces lookalikes.
    let solidN = 0;
    for (let k = 0; k < nu * nv; k++) solidN += solid[k];
    const edgeOn = solidN < 0.3 * nu * nv;
    if (edgeOn || (bN && aN && bSum / bN > aSum / aN + 0.15 * ef)) {
      out.push({ front: front.clone(), u, v, cell, byUp: [v, v.clone().negate(), u, u.clone().negate()].map((up) => ({ up, holes: [] })) });
      continue;
    }
    for (const [up, alongV, sign] of [[v, true, 1], [v.clone().negate(), true, -1], [u, false, 1], [u.clone().negate(), false, -1]] as const) {
      byUp.push({ up: up.clone(), holes: findOpenings(solid, depth, nu, nv, cell, eu, ev, ef, alongV, sign) });
    }
    out.push({ front: front.clone(), u, v, cell, byUp });
  }
  return out;
}

/** Empty cells with solid somewhere left, right and above (for the given up), grouped into openings. */
function findOpenings(solid: Uint8Array, depth: Float32Array, nu: number, nv: number, cell: number, eu: number, ev: number, ef: number, alongV: boolean, sign: number): Hole[] {
  // Map (row, col) with row increasing upwards onto grid (i, j).
  const rows = alongV ? nv : nu, cols = alongV ? nu : nv;
  const idx = (r: number, c: number) => {
    const rr = sign > 0 ? r : rows - 1 - r;
    return alongV ? c * nv + rr : rr * nv + c;
  };
  const cand = new Uint8Array(nu * nv);
  for (let r = 0; r < rows; r++) {
    let first = -1, last = -1;
    for (let c = 0; c < cols; c++) if (solid[idx(r, c)]) { if (first < 0) first = c; last = c; }
    for (let c = first + 1; c < last; c++) if (!solid[idx(r, c)]) cand[idx(r, c)] = 1;
  }
  for (let c = 0; c < cols; c++) {
    let seen = false; // solid somewhere above?
    for (let r = rows - 1; r >= 0; r--) {
      const k = idx(r, c);
      if (solid[k]) seen = true; else if (!seen) cand[k] = 0;
    }
  }
  const label = new Uint8Array(nu * nv);
  const holes: Hole[] = [];
  const stack: number[] = [];
  for (let k0 = 0; k0 < nu * nv; k0++) {
    if (!cand[k0] || label[k0]) continue;
    const ks: number[] = [];
    stack.push(k0); label[k0] = 1;
    while (stack.length) {
      const k = stack.pop()!; ks.push(k);
      const i = Math.floor(k / nv), j = k % nv;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= nu || jj >= nv) continue;
        const kk = ii * nv + jj;
        if (!cand[kk] || label[kk]) continue;
        label[kk] = 1; stack.push(kk);
      }
    }
    const area = ks.length * cell * cell;
    if (area < 100) continue; // smaller than ~1 cm²: vents, noise
    let sa = 0, sb = 0, rim = 0, rimN = 0;
    const cells: Array<[number, number]> = [];
    for (const kk of ks) {
      const i = Math.floor(kk / nv), j = kk % nv;
      const a = -eu - cell + i * cell, b = -ev - cell + j * cell;
      cells.push([a, b]); sa += a; sb += b;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= nu || jj >= nv) continue;
        const n2 = ii * nv + jj;
        if (solid[n2]) { rim += depth[n2]; rimN++; }
      }
    }
    holes.push({ cells, area, centre: [sa / ks.length, sb / ks.length], rimDepth: rimN ? rim / rimN : ef });
  }
  return holes;
}

export interface HelmetEyes {
  /** Centred-helmet-local eye points, sorted so [0] maps to world -X. */
  points: EyePair;
  kind: 'holes' | 'visor';
}

/**
 * Eye targets on the helmet for an orientation q (centred helmet-local ->
 * world rotation). Picks the view whose front points at world +Z, then a pair
 * of side-by-side holes, or failing that one wide visor slot.
 */
/** `tilted`: allow the helmet to be up to ~45° off a square-on view (for tilted starting orientations). */
export function helmetEyesFor(views: ViewHoles[], q: THREE.Quaternion, box: THREE.Box3, tilted = false): HelmetEyes | null {
  const centre = box.getCenter(new THREE.Vector3());
  // Helmet extent in the wearer's frame for this orientation.
  const worldBox = box.clone().applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(q));
  const wMin = worldBox.min, wSize = worldBox.getSize(new THREE.Vector3()), wCentre = worldBox.getCenter(new THREE.Vector3());
  /** Eye points must be side by side, at eye height, and on the front of the helmet. */
  const plausible = (pts: EyePair) => {
    const [a, b] = pts.map((p) => p.clone().applyQuaternion(q));
    const dx = Math.abs(a.x - b.x);
    if (dx < 15 || Math.abs(a.y - b.y) > 0.3 * dx || Math.abs(a.z - b.z) > 0.5 * dx) return false;
    const height = ((a.y + b.y) / 2 - wMin.y) / wSize.y;
    // Low bound is generous: tall ears or crests (Batman) put the eyes low in the overall height.
    if (height < 0.15 || height > 0.85) return false;
    return (a.z + b.z) / 2 > wCentre.z - 0.1 * wSize.z;
  };
  const qi = q.clone().invert();
  const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(qi);
  let view = views[0], bestDot = -2;
  for (const vh of views) { const d = vh.front.dot(fwd); if (d > bestDot) { bestDot = d; view = vh; } }
  // Tilted: up to ~45° off a view still works — eye points are 3D, and the checks below use q.
  if (bestDot < (tilted ? 0.7 : 0.9)) return null;
  const hDir = new THREE.Vector3(1, 0, 0).applyQuaternion(qi);
  const vDir = new THREE.Vector3(0, 1, 0).applyQuaternion(qi);
  const holes = view.byUp.reduce((a, b) => (b.up.dot(vDir) > a.up.dot(vDir) ? b : a)).holes;
  // In-plane coordinates of a cell in the "horizontal / vertical" frame of the wearer.
  const hu = hDir.dot(view.u), hv = hDir.dot(view.v), vu = vDir.dot(view.u), vv = vDir.dot(view.v);
  const toHV = (a: number, b: number): [number, number] => [a * hu + b * hv, a * vu + b * vv];
  const toPoint = (hc: number, vc: number, depth: number) => {
    // Invert the 2x2 [hu hv; vu vv] (orthonormal up to sign) to get back (a, b).
    const det = hu * vv - hv * vu;
    const a = (hc * vv - vc * hv) / det, b = (vc * hu - hc * vu) / det;
    return centre.clone().addScaledVector(view.u, a).addScaledVector(view.v, b).addScaledVector(view.front, depth);
  };
  const info = holes.map((h) => {
    const [ch, cv] = toHV(h.centre[0], h.centre[1]);
    let minH = Infinity, maxH = -Infinity, minV = Infinity, maxV = -Infinity;
    for (const [a, b] of h.cells) { const [x, y] = toHV(a, b); minH = Math.min(minH, x); maxH = Math.max(maxH, x); minV = Math.min(minV, y); maxV = Math.max(maxV, y); }
    return { h, ch, cv, w: maxH - minH, ht: maxV - minV };
  });
  // Pair of eye holes: similar size, side by side, eye-spacing apart.
  const pairs: Array<{ s: number; a: typeof info[0]; b: typeof info[0] }> = [];
  for (let i = 0; i < info.length; i++) for (let j = i + 1; j < info.length; j++) {
    const a = info[i], b = info[j];
    const dh = Math.abs(a.ch - b.ch), dv = Math.abs(a.cv - b.cv);
    const ratio = Math.max(a.h.area, b.h.area) / Math.min(a.h.area, b.h.area);
    if (dh < 25 || dh > 160 || dv > 0.35 * dh || ratio > 3) continue;
    if (a.w > 110 || b.w > 110) continue;
    pairs.push({ s: Math.min(a.h.area, b.h.area) / ratio + (a.cv + b.cv), a, b }); // prefer higher, matched holes
  }
  for (const p of pairs.sort((x, y) => y.s - x.s)) {
    const [l, r] = p.a.ch < p.b.ch ? [p.a, p.b] : [p.b, p.a];
    const points: EyePair = [toPoint(l.ch, l.cv, l.h.rimDepth), toPoint(r.ch, r.cv, r.h.rimDepth)];
    if (plausible(points)) return { kind: 'holes', points };
  }
  // Visor: one opening whose widest band (the bar of a T-visor, or the eye
  // openings of a Corinthian helmet) is a wide, shallow slot. Tall rectangular
  // face openings don't count. Eyes sit on that band, a quarter of its width
  // in from each end.
  for (const vz of info.filter((x) => x.w >= 70).sort((a, b) => b.cv - a.cv)) {
    const rows = new Map<number, [number, number]>();
    for (const [a, b] of vz.h.cells) {
      const [x, y] = toHV(a, b);
      const key = Math.round(y / view.cell);
      const r = rows.get(key) ?? [Infinity, -Infinity];
      rows.set(key, [Math.min(r[0], x), Math.max(r[1], x)]);
    }
    let widest = 0;
    for (const [, [lo, hi]] of rows) widest = Math.max(widest, hi - lo);
    const bar = [...rows].filter(([, [lo, hi]]) => hi - lo >= 0.6 * widest);
    const barHeight = bar.length * view.cell;
    if (barHeight > 55 || widest < 1.5 * barHeight) continue;
    let sy = 0, sl = 0, sr = 0;
    for (const [key, [lo, hi]] of bar) { sy += key * view.cell; sl += lo; sr += hi; }
    const y = sy / bar.length, lo = sl / bar.length, hi = sr / bar.length, w = hi - lo;
    const points: EyePair = [toPoint(lo + w / 4, y, vz.h.rimDepth), toPoint(hi - w / 4, y, vz.h.rimDepth)];
    if (plausible(points)) return { kind: 'visor', points };
  }
  return null;
}

/**
 * A lens / eye-slot insert: a thin piece tucked inside the outer shell (its
 * box is inset from the whole helmet's box on every side). Eye detection
 * looks through these.
 */
export function looksLikeLens(piece: THREE.Box3, assembly: THREE.Box3): boolean {
  const s = piece.getSize(new THREE.Vector3());
  const dims = [s.x, s.y, s.z].sort((a, b) => a - b);
  if (dims[0] > 0.25 * dims[2]) return false;
  const inset = 5;
  return piece.min.x > assembly.min.x + inset && piece.min.y > assembly.min.y + inset && piece.min.z > assembly.min.z + inset
    && piece.max.x < assembly.max.x - inset && piece.max.y < assembly.max.y - inset && piece.max.z < assembly.max.z - inset;
}
