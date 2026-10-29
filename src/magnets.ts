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
export function findMagnetHoles(geometry: THREE.BufferGeometry, piece: number, bvh?: MeshBVH): MagnetHole[] {
  const tree = bvh ?? new MeshBVH(geometry.clone());
  const out: MagnetHole[] = [];
  const faces = findCutFaces(geometry, piece, 3.0, 0.985, 0.25);
  const ray = new THREE.Ray();

  // 1. Round magnet pockets and cylindrical studs (radius ~1.4–13.5 mm).
  for (const f of faces) {
    if (f.area < Math.PI * 1.4 * 1.4 * 0.6 || f.area > Math.PI * 13.5 * 13.5) continue;
    // Floor/cap must be a filled disc: outline points all at about the same radius, area ≈ πr².
    const radii = f.outline.map(([x, y]) => Math.hypot(x, y));
    const r = radii.reduce((a, b) => a + b, 0) / radii.length;
    const spread = Math.sqrt(radii.reduce((a, b) => a + (b - r) ** 2, 0) / radii.length);
    if (r < 1.4 || r > 13.5 || spread > 0.085 * r) continue;
    if (Math.abs(f.area - Math.PI * r * r) > 0.25 * Math.PI * r * r) continue;
    const axis = f.normal.clone();
    const u = f.u, v = f.v;
    // Walk up +axis: radial rays hitting a wall at ≈ r indicate a recessed pocket.
    let depth = 0;
    for (let h = 0.3; h < 15; h += 0.3) {
      const c = f.centre.clone().addScaledVector(axis, h);
      let wall = 0;
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        ray.origin.copy(c);
        ray.direction.copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
        const hit = tree.raycastFirst(ray, THREE.DoubleSide, 0, r * 1.6);
        if (hit && Math.abs(hit.distance - r) < 0.15 * r + 0.15) wall++;
      }
      if (wall >= 6) depth = h; else break;
    }
    if (depth >= 0.9) {
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
