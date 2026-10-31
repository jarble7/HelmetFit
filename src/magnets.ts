import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';
import { findCutFaces, rigidFit } from './assemble';

// Magnet pockets and alignment connectors: blind round magnet holes (e.g. 6×3,
// 8×2, 10×3, 20×3 mm magnets), cylindrical studs, and 4-walled rectangular
// pegs/knobs + sockets/holes. Mating pieces have connectors that face each
// other with the same spacing, so they work as alignment marks even when
// pieces meet along curved seams instead of flat cuts.

export type ConnectorKind = 'pocket' | 'stud' | 'peg' | 'socket';

export interface MagnetHole {
  piece: number;
  /** Centre of the pocket's opening or peg/socket midpoint (piece coordinates). */
  opening: THREE.Vector3;
  /** Outward direction (for a pocket/socket: out of the opening; for a peg/stud: toward the tip). */
  axis: THREE.Vector3;
  radius: number;
  depth: number;
  kind?: ConnectorKind;
  /** True if a socket goes all the way through the piece (can mate from either side). */
  through?: boolean;
}

function compatibleKinds(a: ConnectorKind = 'pocket', b: ConnectorKind = 'pocket'): boolean {
  if (a === 'peg' || b === 'peg') return (a === 'peg' && b === 'socket') || (a === 'socket' && b === 'peg');
  if (a === 'stud' || b === 'stud') return (a === 'stud' && b === 'pocket') || (a === 'pocket' && b === 'stud');
  return a === b;
}

/**
 * Find magnet pockets, cylindrical studs, and rectangular pegs/sockets on a piece.
 */
const bvhCache = new WeakMap<THREE.BufferGeometry, MeshBVH>();
/** Hit-testing structure for pocket checks, built once per geometry. */
function treeFor(g: THREE.BufferGeometry): MeshBVH {
  let t = bvhCache.get(g);
  if (!t) { t = new MeshBVH(g.clone()); bvhCache.set(g, t); }
  return t;
}
/** Forget cached pocket data after a geometry was changed in place. */
export function forgetPocketData(g: THREE.BufferGeometry) { bvhCache.delete(g); }

/** Magnet size the model was designed for, if the user knows it (mm). */
export interface MagnetSize { diameter: number; thickness: number }

/** Pocket radius we'd expect for a magnet (pockets are printed a little oversize). */
const pocketRadiusFits = (r: number, m: MagnetSize) => r >= m.diameter / 2 - 0.6 && r <= m.diameter / 2 + 1.0;

export function findMagnetHoles(geometry: THREE.BufferGeometry, piece: number, bvh?: MeshBVH, size?: MagnetSize): MagnetHole[] {
  const tree = bvh ?? treeFor(geometry);
  const out: MagnetHole[] = [];
  const faces = findCutFaces(geometry, piece, 3.0, 0.985, 0.25);
  const ray = new THREE.Ray();

  // 1. Round magnet pockets and cylindrical studs (radius ~1.4–13.5 mm).
  for (const f of faces) {
    if (f.area < Math.PI * 1.4 * 1.4 * 0.6 || f.area > Math.PI * 13.5 * 13.5) continue;
    // Floor/cap must be a filled disc: outline points all at about the same radius, area ≈ πr².
    const radii = f.outline.map(([x, y]) => Math.hypot(x, y));
    let r = radii.reduce((a, b) => a + b, 0) / radii.length;
    const spread = Math.sqrt(radii.reduce((a, b) => a + (b - r) ** 2, 0) / radii.length);
    if (r < 1.4 || r > 13.5) continue;
    // A clean round floor is the usual sign of a pocket. Floors with a chamfer,
    // ridge or text aren't quite round — still a pocket if the wall is clear.
    // Known magnet size: only pockets of that size count, and their floors may be rougher.
    if (size && !pocketRadiusFits(r, size)) continue;
    const roundFloor = spread <= 0.085 * r && Math.abs(f.area - Math.PI * r * r) <= 0.25 * Math.PI * r * r;
    const roughFloor = !roundFloor && spread <= (size ? 0.35 : 0.25) * r && f.area >= (size ? 0.5 : 0.6) * Math.PI * r * r && f.area <= (size ? 1.6 : 1.4) * Math.PI * r * r;
    if (!roundFloor && !roughFloor) continue;
    const axis = f.normal.clone();
    const u = f.u, v = f.v;
    // Walk up +axis: radial rays hitting a wall at ≈ r indicate a recessed pocket.
    let depth = 0, steady = 0;
    const hits: Array<[number, number]> = []; // wall hit points (u, v) for a circle fit
    for (let h = 0.3; h < 15; h += 0.3) {
      const c = f.centre.clone().addScaledVector(axis, h);
      let wall = 0;
      for (let k = 0; k < 16; k++) {
        const a = (k / 16) * Math.PI * 2;
        ray.origin.copy(c);
        ray.direction.copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
        const hit = tree.raycastFirst(ray, THREE.DoubleSide, 0, r * 1.6);
        if (hit && Math.abs(hit.distance - r) < 0.2 * r + 0.15) {
          wall++;
          hits.push([Math.cos(a) * hit.distance, Math.sin(a) * hit.distance]);
        }
      }
      // A rough floor needs a near-complete wall all the way up (13 of 16 probes); a clean one 10.
      if (wall >= (roundFloor || size ? 10 : 13)) { depth = h; steady++; } else break;
    }
    if (depth >= (size ? Math.max(0.9, 0.6 * size.thickness) : roundFloor ? 0.9 : 1.2)) {
      // Centre and radius from the wall itself (least-squares circle), not the floor outline.
      const fit = fitCircle(hits);
      if (fit && Math.abs(fit.r - r) < 0.3 * r) {
        f.centre.addScaledVector(u, fit.cx).addScaledVector(v, fit.cy);
        r = fit.r;
      }
      void steady;
      // The opening must be open: looking out along the axis from the floor, nothing in the way close by.
      ray.origin.copy(f.centre).addScaledVector(axis, 0.05);
      ray.direction.copy(axis);
      if (!tree.raycastFirst(ray, THREE.DoubleSide, 0, depth + 0.5)) {
        out.push({ piece, opening: f.centre.clone().addScaledVector(axis, depth), axis, radius: r, depth, kind: 'pocket' });
        continue;
      }
    }
    // Check for a protruding round stud: cap at f.centre, cylindrical wall extending along -axis.
    if (r <= 6.5) {
      ray.origin.copy(f.centre).addScaledVector(axis, 0.05);
      ray.direction.copy(axis);
      if (tree.raycastFirst(ray, THREE.DoubleSide, 0, 3)) continue;
      let studDepth = 0;
      for (let h = 0.3; h < 12; h += 0.3) {
        const c = f.centre.clone().addScaledVector(axis, -h);
        let wall = 0;
        for (let k = 0; k < 8; k++) {
          const a = (k / 8) * Math.PI * 2;
          ray.direction.copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
          ray.origin.copy(c).addScaledVector(ray.direction, r * 1.45);
          ray.direction.negate();
          const hit = tree.raycastFirst(ray, THREE.DoubleSide, 0, r * 0.9);
          if (hit && Math.abs(hit.distance - 0.45 * r) < 0.18 * r + 0.15) wall++;
        }
        if (wall >= 6) studDepth = h; else break;
      }
      if (studDepth >= 1.2) {
        out.push({ piece, opening: f.centre.clone().addScaledVector(axis, -studDepth * 0.5), axis, radius: r, depth: studDepth, kind: 'stud' });
      }
    }
  }

  // 2. Rectangular/square pegs (knobs/studs) and sockets (holes) formed by 2 orthogonal pairs of opposite flat side walls.
  const small = faces.filter((c) => c.area >= 4 && c.area <= 150);
  const pairs: Array<{ a: typeof small[0]; b: typeof small[0]; mid: THREE.Vector3; dist: number; kind: 'peg' | 'socket' }> = [];
  for (let i = 0; i < small.length; i++) {
    for (let j = i + 1; j < small.length; j++) {
      const a = small[i], b = small[j];
      if (a.normal.dot(b.normal) > -0.94) continue;
      if (Math.abs(a.area - b.area) / Math.max(a.area, b.area) > 0.65) continue;
      const ab = b.centre.clone().sub(a.centre);
      const dist = ab.length();
      if (dist < 1.8 || dist > 12) continue;
      const along = ab.dot(a.normal); // > 0: normals point toward each other (socket); < 0: away (peg)
      const lateral = Math.sqrt(Math.max(0, dist * dist - along * along));
      if (lateral > 2.0) continue;
      pairs.push({ a, b, mid: a.centre.clone().add(b.centre).multiplyScalar(0.5), dist: Math.abs(along), kind: along > 0 ? 'socket' : 'peg' });
    }
  }
  const used = new Uint8Array(pairs.length);
  for (let i = 0; i < pairs.length; i++) {
    if (used[i]) continue;
    for (let j = i + 1; j < pairs.length; j++) {
      if (used[j] || pairs[i].kind !== pairs[j].kind) continue;
      if (pairs[i].mid.distanceTo(pairs[j].mid) > 2.5) continue;
      const n1 = pairs[i].a.normal.clone().sub(pairs[i].b.normal).normalize();
      const n2 = pairs[j].a.normal.clone().sub(pairs[j].b.normal).normalize();
      if (Math.abs(n1.dot(n2)) > 0.25) continue;
      used[i] = used[j] = 1;
      const pos = pairs[i].mid.clone().add(pairs[j].mid).multiplyScalar(0.5);
      const axis = new THREE.Vector3().crossVectors(n1, n2).normalize();
      const radius = (pairs[i].dist + pairs[j].dist) / 4;
      const dPos = tree.raycastFirst(new THREE.Ray(pos, axis), THREE.DoubleSide, 0.2, 20)?.distance ?? Infinity;
      const dNeg = tree.raycastFirst(new THREE.Ray(pos, axis.clone().negate()), THREE.DoubleSide, 0.2, 20)?.distance ?? Infinity;
      if (dPos === Infinity && dNeg === Infinity) {
        const draft = pairs[i].a.normal.clone().add(pairs[i].b.normal).add(pairs[j].a.normal).add(pairs[j].b.normal).dot(axis);
        if (draft < 0) axis.negate();
        out.push({ piece, opening: pos, axis, radius, depth: 3, kind: pairs[i].kind, through: true });
      } else {
        const sign = pairs[i].kind === 'socket' ? (dPos > dNeg ? 1 : -1) : (dPos < dNeg ? 1 : -1);
        axis.multiplyScalar(sign);
        out.push({ piece, opening: pos, axis, radius, depth: 3, kind: pairs[i].kind });
      }
      break;
    }
  }
  return out;
}

export interface MagnetMatch {
  /** Moves piece b so its pockets/connectors mate with piece a's (b coordinates -> a coordinates). */
  bToA: THREE.Matrix4;
  /** Connector pairs that line up face to face. */
  pairs: Array<[MagnetHole, MagnetHole]>;
  rms: number;
}

/**
 * Ways to join piece b to piece a so that their magnet pockets or pegs/sockets
 * line up (compatible kind and size, facing each other). Tries every pair of
 * connectors with matching spacing as a hypothesis, keeps those where other
 * connectors also line up, best (most pairs) first.
 */
/** `minPairs`: fewest connectors that must line up (3 for automatic assembly; 2 when the user picked the two pieces). */
export function matchMagnets(ha: MagnetHole[], hb: MagnetHole[], max = 6, minPairs = 3): MagnetMatch[] {
  if (ha.length < 2 || hb.length < 2) return [];
  const TOL = 1.5;
  // Spatial hash of a's openings for fast "is there a connector here?" checks.
  const cell = 4;
  const grid = new Map<string, MagnetHole[]>();
  const key = (p: THREE.Vector3) => `${Math.floor(p.x / cell)},${Math.floor(p.y / cell)},${Math.floor(p.z / cell)}`;
  for (const h of ha) { const k = key(h.opening); grid.set(k, [...(grid.get(k) ?? []), h]); }
  const nearA = (p: THREE.Vector3, axis: THREE.Vector3, r: number, kindB?: ConnectorKind, throughB?: boolean) => {
    let best: MagnetHole | null = null, bd = TOL;
    const cx = Math.floor(p.x / cell), cy = Math.floor(p.y / cell), cz = Math.floor(p.z / cell);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) {
      for (const h of grid.get(`${cx + i},${cy + j},${cz + k}`) ?? []) {
        if (!compatibleKinds(h.kind, kindB)) continue;
        const rTol = (h.kind ?? 'pocket') === 'pocket' && (kindB ?? 'pocket') === 'pocket' ? 0.45 : 0.85;
        if (Math.abs(h.radius - r) >= rTol) continue;
        const d = h.opening.distanceTo(p);
        const dot = h.axis.dot(axis);
        const axisOk = h.through || throughB ? Math.abs(dot) > 0.82 : dot < -0.82;
        if (d < bd && axisOk) { bd = d; best = h; }
      }
    }
    return best;
  };
  // Pairs on b, bucketed by spacing.
  const bPairs = new Map<number, Array<[MagnetHole, MagnetHole]>>();
  for (let i = 0; i < hb.length; i++) for (let j = 0; j < hb.length; j++) {
    if (i === j) continue;
    const d = hb[i].opening.distanceTo(hb[j].opening);
    if (d < 6) continue;
    const k = Math.round(d / TOL);
    bPairs.set(k, [...(bPairs.get(k) ?? []), [hb[i], hb[j]]]);
  }
  const found: MagnetMatch[] = [];
  const tmp = new THREE.Vector3(), ax = new THREE.Vector3();
  const L = 5; // lever length for the axis direction, mm
  for (let i = 0; i < ha.length; i++) for (let j = i + 1; j < ha.length; j++) {
    const a1 = ha[i], a2 = ha[j];
    const d = a1.opening.distanceTo(a2.opening);
    if (d < 6) continue;
    const angA = a1.axis.angleTo(a2.axis);
    const k = Math.round(d / TOL);
    for (const kk of [k - 1, k, k + 1]) for (const [b1, b2] of bPairs.get(kk) ?? []) {
      if (!compatibleKinds(a1.kind, b1.kind) || !compatibleKinds(a2.kind, b2.kind)) continue;
      if (Math.abs(b1.opening.distanceTo(b2.opening) - d) > TOL) continue;
      const anyThrough = a1.through || a2.through || b1.through || b2.through;
      const angB = b1.axis.angleTo(b2.axis);
      if (!anyThrough && Math.abs(angB - angA) > 0.28) continue;
      const signs: number[] = anyThrough ? [-1, 1] : [-1];
      for (const sgn of signs) {
        const src = [b1.opening, b2.opening, b1.opening.clone().addScaledVector(b1.axis, L), b2.opening.clone().addScaledVector(b2.axis, L)];
        const dst = [a1.opening, a2.opening, a1.opening.clone().addScaledVector(a1.axis, sgn * L), a2.opening.clone().addScaledVector(a2.axis, sgn * L)];
        const fit = rigidFit(src, dst);
        if (fit.rms > 1.4) continue;
        // Count every b connector that lands on an a connector.
        const pairs: Array<[MagnetHole, MagnetHole]> = [];
        const rot = new THREE.Matrix3().setFromMatrix4(fit.matrix);
        for (const hbk of hb) {
          tmp.copy(hbk.opening).applyMatrix4(fit.matrix);
          ax.copy(hbk.axis).applyMatrix3(rot);
          const hak = nearA(tmp, ax, hbk.radius, hbk.kind, hbk.through);
          if (hak && !pairs.some((p) => p[0] === hak)) pairs.push([hak, hbk]);
        }
        if (pairs.length < minPairs) continue;
        // Skip near-duplicates of a hypothesis we already have.
        if (found.some((f) => f.pairs.length >= pairs.length && pairs.every(([pa, pb]) => f.pairs.some(([qa, qb]) => qa === pa && qb === pb)))) continue;
        found.push({ bToA: fit.matrix, pairs, rms: fit.rms });
      }
    }
  }
  // Refine each on all its connector pairs, best first.
  const rotTmp = new THREE.Matrix3();
  for (const f of found) {
    rotTmp.setFromMatrix4(f.bToA);
    const src: THREE.Vector3[] = [], dst: THREE.Vector3[] = [];
    for (const [a, b] of f.pairs) {
      const sgn = (a.through || b.through) && b.axis.clone().applyMatrix3(rotTmp).dot(a.axis) > 0 ? 1 : -1;
      src.push(b.opening, b.opening.clone().addScaledVector(b.axis, L));
      dst.push(a.opening, a.opening.clone().addScaledVector(a.axis, sgn * L));
    }
    const r = rigidFit(src, dst);
    f.bToA = r.matrix; f.rms = r.rms;
  }
  return found.sort((x, y) => y.pairs.length - x.pairs.length || x.rms - y.rms).slice(0, max);
}

/** Least-squares circle through 2D points (Kåsa fit). */
function fitCircle(pts: Array<[number, number]>): { cx: number; cy: number; r: number } | null {
  if (pts.length < 6) return null;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0, sz = 0;
  for (const [x, y] of pts) { const z = x * x + y * y; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; sxz += x * z; syz += y * z; sz += z; }
  const n = pts.length;
  // Solve [sxx sxy sx; sxy syy sy; sx sy n] [a b c] = [sxz syz sz]; centre (a/2, b/2), r = sqrt(c + cx² + cy²).
  const M = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]], R = [sxz, syz, sz];
  const det = (m: number[][]) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const d = det(M);
  if (Math.abs(d) < 1e-9) return null;
  const col = (k: number) => M.map((row, i) => row.map((v, j) => (j === k ? R[i] : v)));
  const a = det(col(0)) / d, b = det(col(1)) / d, c = det(col(2)) / d;
  const cx = a / 2, cy = b / 2, rr = c + cx * cx + cy * cy;
  return rr > 0 ? { cx, cy, r: Math.sqrt(rr) } : null;
}

/**
 * Measure a magnet pocket at a clicked spot (piece coordinates) when it wasn't
 * found automatically: treat the clicked surface as the pocket floor, find the
 * wall with sideways probes, fit a circle, then walk up to the opening.
 */
export function measurePocket(geometry: THREE.BufferGeometry, point: THREE.Vector3, normal: THREE.Vector3, piece: number, size?: MagnetSize): MagnetHole | null {
  const tree = treeFor(geometry);
  const axis = normal.clone().normalize();
  const u = Math.abs(axis.x) < 0.9 ? new THREE.Vector3(1, 0, 0).cross(axis).normalize() : new THREE.Vector3(0, 1, 0).cross(axis).normalize();
  const v = new THREE.Vector3().crossVectors(axis, u);
  const ray = new THREE.Ray();
  const maxR = size ? size.diameter / 2 + 2 : 14;
  const probe = (centre: THREE.Vector3, h: number) => {
    const pts: Array<[number, number]> = [];
    const c = centre.clone().addScaledVector(axis, h);
    for (let k = 0; k < 24; k++) {
      const a = (k / 24) * Math.PI * 2;
      ray.origin.copy(c);
      ray.direction.copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
      const hit = tree.raycastFirst(ray, THREE.DoubleSide, 0, maxR);
      if (hit) pts.push([Math.cos(a) * hit.distance, Math.sin(a) * hit.distance]);
    }
    return pts;
  };
  // Twice: the click may be off-centre, so re-centre on the first circle and measure again.
  let centre = point.clone();
  let fit: { cx: number; cy: number; r: number } | null = null;
  for (let pass = 0; pass < 2; pass++) {
    const pts = probe(centre, 0.4);
    if (pts.length < 16) return null; // not surrounded by a wall: not inside a pocket
    fit = fitCircle(pts);
    if (!fit) return null;
    centre = centre.clone().addScaledVector(u, fit.cx).addScaledVector(v, fit.cy);
  }
  const r = fit!.r;
  if (r < 1.4 || r > 13.5 || (size && !pocketRadiusFits(r, size))) return null;
  // Walk up while the wall is still there at about the same radius.
  let depth = 0.4;
  for (let h = 0.7; h < 15; h += 0.3) {
    const pts = probe(centre, h);
    const ok = pts.filter(([x, y]) => Math.abs(Math.hypot(x, y) - r) < 0.2 * r + 0.15).length;
    if (ok >= 18) depth = h; else break;
  }
  if (depth < 0.9) return null;
  return { piece, opening: centre.clone().addScaledVector(axis, depth), axis, radius: r, depth, kind: 'pocket' } as MagnetHole;
}
