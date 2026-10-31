import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';
import { findMagnetHoles, matchMagnets, type MagnetHole } from './magnets';
import { analyseSurface, flatRegions } from './surface';

// Auto-assembly of pieces that were exported "print-ready" (each turned and
// laid flat on the bed, so they no longer share coordinates).
//
// Pieces of a helmet that was cut into parts with flat cuts each carry flat
// "cut faces". Two mating cut faces have the same outline. We find every
// piece's cut faces, pair up faces with matching area and outline, work out
// the move that presses one onto the other, and join pieces greedily (best
// matches first), rejecting joins that would make pieces overlap.

export interface CutFace {
  piece: number;
  normal: THREE.Vector3;
  /** A point on the face (centroid). */
  centre: THREE.Vector3;
  area: number;
  /** In-plane frame (u, v, normal). */
  u: THREE.Vector3;
  v: THREE.Vector3;
  /** Outline sampled every ~0.5 mm, in (u, v) coordinates relative to centre. */
  outline: Array<[number, number]>;
  /** The face's triangles (piece coordinates), for highlighting it. */
  triangles: Float32Array;
  /** Outline length in mm. */
  perimeter: number;
}

export interface Match {
  a: CutFace;
  b: CutFace;
  /** Transform that moves piece b so face b presses onto face a (in piece a's frame). */
  bToA: THREE.Matrix4;
  rms: number;
}

export interface AssembleResult {
  /** Per-piece matrix (relative to the first piece of its group). */
  matrices: THREE.Matrix4[];
  /** Group id per piece; pieces in the same group were joined together. */
  group: number[];
  joins: Array<{ a: number; b: number; rms: number; magnets: number }>;
  faces: number;
  /** Magnet pockets found on all pieces. */
  pockets: number;
}

/** A way to join piece pb onto piece pa: by a matching cut face, or by lined-up magnet pockets. */
interface Join {
  pa: number; pb: number;
  bToA: THREE.Matrix4;
  rms: number;
  /** Cut faces / pockets this join uses up (each can only be used once). */
  uses: object[];
  /** Number of magnet pockets lined up (0 for a cut-face join). */
  magnets: number;
}

// ---------------------------------------------------------------------------
// Cut faces

/** `normalDot` / `planeTol`: how flat a region must be (strict for cut faces, looser for magnet-pocket floors on bumpy meshes). */
export function findCutFaces(geometry: THREE.BufferGeometry, piece: number, minArea = 40, normalDot = 0.9995, planeTol = 0.05): CutFace[] {
  const surf = analyseSurface(geometry); // remembered per geometry, shared with pocket detection
  const { pos, idx, N, C, A } = surf;
  const faces: CutFace[] = [];
  for (const { tris, area, seed } of flatRegions(surf, minArea, normalDot, planeTol)) {
    const n0 = new THREE.Vector3(N[seed * 3], N[seed * 3 + 1], N[seed * 3 + 2]);
    // Boundary edges of the region = its outline.
    const edgeCount = new Map<number, [number, number, number]>();
    for (const t of tris) for (let k = 0; k < 3; k++) {
      const i = idx[3 * t + k], j = idx[3 * t + ((k + 1) % 3)];
      const key = i < j ? i * 4294967296 + j : j * 4294967296 + i;
      const e = edgeCount.get(key);
      if (e) e[2]++; else edgeCount.set(key, [i, j, 1]);
    }
    const centre = new THREE.Vector3();
    let wsum = 0;
    for (const t of tris) { centre.x += C[t * 3] * A[t]; centre.y += C[t * 3 + 1] * A[t]; centre.z += C[t * 3 + 2] * A[t]; wsum += A[t]; }
    centre.divideScalar(wsum);
    const u = Math.abs(n0.x) < 0.9 ? new THREE.Vector3(1, 0, 0).cross(n0).normalize() : new THREE.Vector3(0, 1, 0).cross(n0).normalize();
    const v = new THREE.Vector3().crossVectors(n0, u);
    const outline: Array<[number, number]> = [];
    const p = new THREE.Vector3(), q = new THREE.Vector3();
    for (const [i, j, count] of edgeCount.values()) {
      if (count !== 1) continue;
      p.fromArray(pos, i * 3).sub(centre); q.fromArray(pos, j * 3).sub(centre);
      const len = p.distanceTo(q), steps = Math.max(1, Math.ceil(len / 0.5));
      for (let k = 0; k < steps; k++) {
        const w = p.clone().lerp(q, k / steps);
        outline.push([w.dot(u), w.dot(v)]);
      }
    }
    const perimeter = outline.length * 0.5;
    const triangles = new Float32Array(tris.length * 9);
    tris.forEach((t, k) => { for (let m = 0; m < 3; m++) { const vi = idx[3 * t + m] * 3; triangles[k * 9 + m * 3] = pos[vi]; triangles[k * 9 + m * 3 + 1] = pos[vi + 1]; triangles[k * 9 + m * 3 + 2] = pos[vi + 2]; } });
    if (outline.length >= 8) faces.push({ piece, normal: n0, centre, area, u, v, outline, triangles, perimeter });
  }
  return faces;
}

// ---------------------------------------------------------------------------
// 2D outline registration

class Grid2 {
  private cells = new Map<number, number[]>();
  constructor(private pts: Array<[number, number]>, private h: number) {
    pts.forEach(([x, y], i) => {
      const k = Grid2.key(Math.floor(x / h), Math.floor(y / h));
      const c = this.cells.get(k); if (c) c.push(i); else this.cells.set(k, [i]);
    });
  }
  private static key(i: number, j: number) { return (i + 50000) * 100003 + (j + 50000); }
  nearest(x: number, y: number, maxR = 3): [number, number] {
    const cx = Math.floor(x / this.h), cy = Math.floor(y / this.h);
    let best = -1, bd = Infinity;
    for (let r = 0; r <= maxR; r++) {
      for (let i = cx - r; i <= cx + r; i++) for (let j = cy - r; j <= cy + r; j++) {
        if (Math.max(Math.abs(i - cx), Math.abs(j - cy)) !== r) continue;
        const c = this.cells.get(Grid2.key(i, j)); if (!c) continue;
        for (const k of c) { const d = (this.pts[k][0] - x) ** 2 + (this.pts[k][1] - y) ** 2; if (d < bd) { bd = d; best = k; } }
      }
      if (best >= 0 && Math.sqrt(bd) <= r * this.h) break;
    }
    return [best, Math.sqrt(bd)];
  }
}

function principalAngle(pts: Array<[number, number]>): number {
  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of pts) { sxx += x * x; syy += y * y; sxy += x * y; }
  return 0.5 * Math.atan2(2 * sxy, sxx - syy);
}

interface Fit2 { theta: number; tx: number; ty: number; rms: number }

/**
 * Rigid 2D fits of `src` onto `dst` (rotation θ, translation t) from several
 * starting angles, best first. RMS is of the symmetric nearest distances.
 * `thorough` refines every start at full detail (for hand snapping, where we
 * want alternatives); otherwise a quick coarse pass picks one to refine.
 */
function registerCandidates(src: Array<[number, number]>, dst: Array<[number, number]>, thorough: boolean): Fit2[] {
  const grid = new Grid2(dst, 2);
  const icp = (pts: Array<[number, number]>, theta0: number, tx0: number, ty0: number, iters: number) => {
    let theta = theta0, tx = tx0, ty = ty0;
    for (let it = 0; it < iters; it++) {
      const c = Math.cos(theta), s = Math.sin(theta);
      // Pair each moved src point with its nearest dst point, then solve the best rigid 2D fit.
      let mx = 0, my = 0, nx = 0, ny = 0, n = 0;
      const pairs: Array<[number, number, number, number]> = [];
      for (const [x, y] of pts) {
        const X = c * x - s * y + tx, Y = s * x + c * y + ty;
        const [k, d] = grid.nearest(X, Y, 3);
        if (k < 0 || d > 6) continue;
        pairs.push([x, y, dst[k][0], dst[k][1]]);
        mx += x; my += y; nx += dst[k][0]; ny += dst[k][1]; n++;
      }
      if (n < 8) return null;
      mx /= n; my /= n; nx /= n; ny /= n;
      let sxx = 0, sxy = 0;
      for (const [x, y, X, Y] of pairs) { const ax = x - mx, ay = y - my, bx = X - nx, by = Y - ny; sxx += ax * bx + ay * by; sxy += ax * by - ay * bx; }
      const nt = Math.atan2(sxy, sxx);
      const c2 = Math.cos(nt), s2 = Math.sin(nt);
      const ntx = nx - (c2 * mx - s2 * my), nty = ny - (s2 * mx + c2 * my);
      const done = Math.abs(nt - theta) < 1e-5 && Math.abs(ntx - tx) < 1e-4 && Math.abs(nty - ty) < 1e-4;
      theta = nt; tx = ntx; ty = nty;
      if (done) break;
    }
    return { theta, tx, ty };
  };
  // Symmetric RMS (src->dst and dst->src) so a part-outline can't match a bigger one.
  const score = (theta: number, tx: number, ty: number, pts: Array<[number, number]>, dpts: Array<[number, number]>) => {
    const c = Math.cos(theta), s = Math.sin(theta);
    const moved = pts.map(([x, y]) => [c * x - s * y + tx, s * x + c * y + ty] as [number, number]);
    const g2 = new Grid2(moved, 2);
    let sum = 0, n = 0;
    for (const [x, y] of moved) { const [, d] = grid.nearest(x, y, 3); sum += Math.min(d, 6) ** 2; n++; }
    for (const [x, y] of dpts) { const [, d] = g2.nearest(x, y, 3); sum += Math.min(d, 6) ** 2; n++; }
    return Math.sqrt(sum / n);
  };
  const a0 = principalAngle(dst) - principalAngle(src);
  const starts = thorough
    ? Array.from({ length: 16 }, (_, k) => (k * Math.PI) / 8)
    : [0, Math.PI, Math.PI / 2, -Math.PI / 2, Math.PI / 4, -Math.PI / 4, (3 * Math.PI) / 4, (-3 * Math.PI) / 4];
  if (thorough) {
    const out: Fit2[] = [];
    for (const extra of starts) {
      const r = icp(src, a0 + extra, 0, 0, 40);
      if (!r) continue;
      const rms = score(r.theta, r.tx, r.ty, src, dst);
      // Keep distinct fits only (different angle or position).
      const dup = out.find((o) => Math.abs(Math.atan2(Math.sin(o.theta - r.theta), Math.cos(o.theta - r.theta))) < 0.09 && Math.hypot(o.tx - r.tx, o.ty - r.ty) < 2);
      if (dup) { if (rms < dup.rms) Object.assign(dup, r, { rms }); continue; }
      out.push({ ...r, rms });
    }
    return out.sort((x, y) => x.rms - y.rms);
  }
  // Coarse pass on every 4th point from several starting angles, then refine the best at full detail.
  const coarse = src.filter((_, i) => i % 4 === 0), coarseDst = dst.filter((_, i) => i % 4 === 0);
  let best: Fit2 | null = null;
  for (const extra of starts) {
    const r = icp(coarse, a0 + extra, 0, 0, 25);
    if (!r) continue;
    const rms = score(r.theta, r.tx, r.ty, coarse, coarseDst);
    if (!best || rms < best.rms) best = { ...r, rms };
    if (best.rms < 0.3) break;
  }
  if (!best || best.rms > 2) return [];
  const r = icp(src, best.theta, best.tx, best.ty, 30) ?? best;
  return [{ ...r, rms: score(r.theta, r.tx, r.ty, src, dst) }];
}

/** Transform moving face fb's piece so fb presses onto fa with in-plane fit `reg`. */
function mateTransform(fa: CutFace, fb: CutFace, reg: { theta: number; tx: number; ty: number }): THREE.Matrix4 {
  // b-frame -> mated a-frame: b's (u, v, n) map to rotate(θ) within (u_a, -v_a), and n -> -n_a.
  const ua = fa.u, va = fa.v.clone().negate(), na = fa.normal.clone().negate();
  const c = Math.cos(reg.theta), s = Math.sin(reg.theta);
  const ub = ua.clone().multiplyScalar(c).addScaledVector(va, s);
  const vb = ua.clone().multiplyScalar(-s).addScaledVector(va, c);
  const target = new THREE.Matrix4().makeBasis(ub, vb, na);
  target.setPosition(fa.centre.clone().addScaledVector(ua, reg.tx).addScaledVector(va, reg.ty));
  const source = new THREE.Matrix4().makeBasis(fb.u, fb.v, fb.normal);
  source.setPosition(fb.centre);
  return target.multiply(source.invert());
}

/**
 * Ways to press face fb onto face fa for hand snapping, best first. Always
 * returns at least one: if the outlines don't match well (pegs, keys, uneven
 * cuts) the faces are still put flat together, centred, with matched long axes.
 */
export function snapCandidates(fa: CutFace, fb: CutFace): Array<{ bToA: THREE.Matrix4; rms: number }> {
  const dst = fa.outline.map(([x, y]) => [x, -y] as [number, number]);
  const fits = registerCandidates(fb.outline, dst, true).slice(0, 8);
  const a0 = principalAngle(dst) - principalAngle(fb.outline);
  for (const theta of [a0, a0 + Math.PI]) {
    if (!fits.some((f) => Math.abs(Math.atan2(Math.sin(f.theta - theta), Math.cos(f.theta - theta))) < 0.09)) fits.push({ theta, tx: 0, ty: 0, rms: Infinity });
  }
  return fits.map((f) => ({ bToA: mateTransform(fa, fb, f), rms: f.rms }));
}

/** The cut face of a piece at a clicked point (piece coordinates), if any. */
export function cutFaceAt(faces: CutFace[], point: THREE.Vector3, normal: THREE.Vector3): CutFace | null {
  let best: CutFace | null = null, bd = Infinity;
  for (const f of faces) {
    if (Math.abs(f.normal.dot(normal)) < 0.95) continue;
    const off = Math.abs(point.clone().sub(f.centre).dot(f.normal));
    if (off > 0.6) continue;
    const d = point.distanceTo(f.centre);
    if (d < bd) { bd = d; best = f; }
  }
  return best;
}

/**
 * Group pieces that are currently joined face to face (a cut face pressed on a
 * partner face). `faces[i]` in piece i's coordinates, `place[i]` its placement.
 */
export function joinedGroups(faces: CutFace[][], place: THREE.Matrix4[], joinedFaces?: Set<CutFace>): number[] {
  const n = faces.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const world = faces.flat().map((f) => {
    const m = place[f.piece];
    const sc = m.getMaxScaleOnAxis() || 1;
    return { f, sc, c: f.centre.clone().applyMatrix4(m), nrm: f.normal.clone().transformDirection(m) };
  });
  for (let i = 0; i < world.length; i++) for (let j = i + 1; j < world.length; j++) {
    const A = world[i], B = world[j];
    if (A.f.piece === B.f.piece || A.nrm.dot(B.nrm) > -0.99) continue;
    if (Math.abs(A.sc - B.sc) / Math.max(A.sc, B.sc) > 0.03) continue;
    // Same plane, overlapping area.
    if (Math.abs(B.c.clone().sub(A.c).dot(A.nrm)) > 0.6 * A.sc || A.c.distanceTo(B.c) > 0.5 * A.sc * Math.sqrt(Math.max(A.f.area, B.f.area))) continue;
    parent[find(A.f.piece)] = find(B.f.piece);
    joinedFaces?.add(A.f); joinedFaces?.add(B.f);
  }
  return parent.map((_, i) => find(i));
}

/**
 * Transform moving piece b so its face fb presses onto face fa of piece a
 * (normals opposite, outlines coincide). Both faces in their own piece frames.
 */
export function matchFaces(fa: CutFace, fb: CutFace): Match | null {
  if (Math.abs(fa.area - fb.area) / Math.max(fa.area, fb.area) > 0.06) return null;
  // Outline lengths (samples are ~0.5 mm apart) must agree too.
  if (Math.abs(fa.perimeter - fb.perimeter) / Math.max(fa.perimeter, fb.perimeter) > 0.25) return null;
  // Seen from the mating side, face a's outline is mirrored: use frame (u, -v, -n) for a.
  const dst = fa.outline.map(([x, y]) => [x, -y] as [number, number]);
  const reg = registerCandidates(fb.outline, dst, false)[0];
  // Real cut-face pairs match to ≲0.35 mm; looser is a lookalike flat surface.
  if (!reg || !(reg.rms < 0.4)) return null;
  return { a: fa, b: fb, bToA: mateTransform(fa, fb, reg), rms: reg.rms };
}

// ---------------------------------------------------------------------------
// Assembly

/** Do pieces of group B (placed by mB) overlap group A, apart from along the joint plane? */
function overlaps(
  bvhs: MeshBVH[], samples: THREE.Vector3[][], groupA: number[], groupB: number[], place: THREE.Matrix4[], isMagnet = false,
): number {
  let bad = 0, total = 0;
  const p = new THREE.Vector3(), local = new THREE.Vector3();
  const ray = new THREE.Ray(local, new THREE.Vector3(0.371, 0.529, 0.763));
  const inv = groupA.map((i) => place[i].clone().invert());
  for (const j of groupB) {
    for (const s of samples[j]) {
      p.copy(s).applyMatrix4(place[j]);
      total++;
      for (let k = 0; k < groupA.length; k++) {
        local.copy(p).applyMatrix4(inv[k]);
        if (isMagnet) {
          // Magnet/peg joins press curved surfaces together (distance ≈ 0 along the seam);
          // only count points that penetrate > 1 mm deep inside the other piece's volume.
          const rHit = bvhs[groupA[k]].raycastFirst(ray, THREE.DoubleSide);
          if (rHit && rHit.face && rHit.face.normal.dot(ray.direction) > 0 && !bvhs[groupA[k]].closestPointToPoint(local, undefined, 0, 1.0)) {
            bad++; break;
          }
        } else {
          const hit = bvhs[groupA[k]].closestPointToPoint(local, undefined, 0, 0.6);
          if (hit) { bad++; break; }
        }
      }
    }
  }
  return total ? bad / total : 0;
}

/**
 * `seed`: current placements. Pieces already joined face to face in it (e.g.
 * snapped by hand) stay joined; the rest are fitted around them.
 */
/**
 * `fixed[i]`: piece i is locked where it is (in `seed`) — all locked pieces
 * count as one already-assembled group and are never moved; only the other
 * pieces are placed, onto them or onto each other.
 * `userGroups[i]`: optional group ID per piece — pieces with the same ID stay
 * locked together as a rigid sub-assembly so groups can be auto-assembled onto
 * each other without taking apart their internal parts.
 */
export interface AssembleOptions {
  /** What the pieces join by — a model uses one or the other. Default: both. */
  mode?: 'faces' | 'magnets';
  /** Already-found cut faces / pockets per piece (same coordinates as `geometries`), to skip finding them again. */
  faces?: CutFace[][];
  holes?: MagnetHole[][];
}

export async function autoAssemble(geometries: THREE.BufferGeometry[], onProgress?: (msg: string) => void, debug?: (msg: string) => void, seed?: THREE.Matrix4[], fixed?: boolean[], userGroups?: number[], options: AssembleOptions = {}): Promise<AssembleResult> {
  const progress = onProgress ?? (() => {});
  const yieldNow = () => new Promise<void>((r) => setTimeout(r, 0));
  const T = (label: string, t0: number) => debug?.(`${label}: ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  const n = geometries.length;
  let t0 = performance.now();
  const useFaces = options.mode !== 'magnets', useMagnets = options.mode !== 'faces';
  const faces: CutFace[][] = [];
  if (useFaces) {
    progress('Finding cut faces…');
    await yieldNow();
    for (let i = 0; i < n; i++) { faces.push(options.faces?.[i] ?? findCutFaces(geometries[i], i)); await yieldNow(); }
  } else for (let i = 0; i < n; i++) faces.push([]);
  T('cut faces', t0); t0 = performance.now();
  // Magnet pockets: pieces that interlock or overlap instead of meeting at flat cuts.
  const holes: MagnetHole[][] = [];
  if (useMagnets) {
    progress('Looking for magnet pockets & alignment pegs…');
    await yieldNow();
    for (let i = 0; i < n; i++) { holes.push(options.holes?.[i] ?? findMagnetHoles(geometries[i], i)); await yieldNow(); }
  } else for (let i = 0; i < n; i++) holes.push([]);
  T('pockets', t0); t0 = performance.now();

  const seedFaces = new Set<CutFace>();
  let seedGroup = seed ? joinedGroups(faces, seed, seedFaces) : (userGroups ? Array.from({ length: n }, (_, i) => i) : null);
  if (seed && seedGroup && useMagnets) {
    // Pieces already joined by magnets (pockets face to face where they are now) stay joined.
    const parent = seedGroup.slice();
    const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    const world = holes.map((hs, i) => hs.map((h) => ({ c: h.opening.clone().applyMatrix4(seed[i]), a: h.axis.clone().transformDirection(seed[i]) })));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (world[i].some((x) => world[j].some((y) => x.c.distanceTo(y.c) < 1.5 && x.a.dot(y.a) < -0.85))) parent[find(j)] = find(i);
    }
    seedGroup = parent.map((_, i) => find(i));
  }
  if (seedGroup && userGroups) {
    const parent = seedGroup.slice();
    const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (userGroups[i] !== undefined && userGroups[i] >= 0 && userGroups[i] === userGroups[j]) {
        parent[find(j)] = find(i);
      }
    }
    seedGroup = parent.map((_, i) => find(i));
  }

  const all = faces.flat();
  progress(`Matching ${all.length} cut faces…`);
  await yieldNow();
  const matches: Match[] = [];
  for (let i = 0; i < all.length; i++) {
    for (let j = 0; j < all.length; j++) {
      if (all[i].piece >= all[j].piece) continue;
      if (seedGroup && seedGroup[all[i].piece] === seedGroup[all[j].piece]) continue;
      const m = matchFaces(all[i], all[j]);
      if (m) matches.push(m);
    }
    if (i % 10 === 9) await yieldNow();
  }
  matches.sort((x, y) => x.rms - y.rms);
  T('matching', t0); t0 = performance.now();

  const magnetJoins: Join[] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (seedGroup && seedGroup[i] === seedGroup[j]) continue;
    for (const m of matchMagnets(holes[i], holes[j])) {
      magnetJoins.push({ pa: i, pb: j, bToA: m.bToA, rms: m.rms, uses: m.pairs.flat(), magnets: m.pairs.length });
    }
  }
  // If user/seed groups contain multiple pieces, also match their combined connectors in seed space.
  if (seed && seedGroup) {
    const byGroup = new Map<number, number[]>();
    seedGroup.forEach((g, i) => byGroup.set(g, [...(byGroup.get(g) ?? []), i]));
    const groups = [...byGroup.values()];
    if (groups.some((m) => m.length > 1)) {
      const groupHoles = groups.map((members) => {
        const rep = members[0];
        const invRep = seed[rep].clone().invert();
        const combined: Array<{ hole: MagnetHole; orig: MagnetHole }> = [];
        for (const k of members) {
          const toRep = k === rep ? new THREE.Matrix4() : invRep.clone().multiply(seed[k]);
          const rotRep = new THREE.Matrix3().setFromMatrix4(toRep);
          for (const h of holes[k]) {
            combined.push({
              orig: h,
              hole: { ...h, piece: rep, opening: h.opening.clone().applyMatrix4(toRep), axis: h.axis.clone().applyMatrix3(rotRep).normalize() },
            });
          }
        }
        return { rep, members, combined };
      });
      for (let gi = 0; gi < groupHoles.length; gi++) for (let gj = gi + 1; gj < groupHoles.length; gj++) {
        const A = groupHoles[gi], B = groupHoles[gj];
        if (A.members.length === 1 && B.members.length === 1) continue;
        const mapOrig = new Map<MagnetHole, MagnetHole>();
        for (const x of A.combined) mapOrig.set(x.hole, x.orig);
        for (const x of B.combined) mapOrig.set(x.hole, x.orig);
        for (const m of matchMagnets(A.combined.map((x) => x.hole), B.combined.map((x) => x.hole))) {
          magnetJoins.push({
            pa: A.rep, pb: B.rep, bToA: m.bToA, rms: m.rms,
            uses: m.pairs.flat().map((h) => mapOrig.get(h) ?? h),
            magnets: m.pairs.length,
          });
        }
      }
    }
  }
  T(`magnets (${holes.flat().length} pockets, ${magnetJoins.length} joins)`, t0); t0 = performance.now();
  // Magnet joins with several pockets lined up are near-certain, so try them first.
  const faceJoins: Join[] = matches.map((m) => ({ pa: m.a.piece, pb: m.b.piece, bToA: m.bToA, rms: m.rms, uses: [m.a, m.b], magnets: 0 }));
  const joinsToTry: Join[] = [
    ...magnetJoins.sort((x, y) => y.magnets - x.magnets || x.rms - y.rms),
    ...faceJoins,
  ];

  // Greedy union, best outline matches first.
  const bvhs = geometries.map((g) => new MeshBVH(g.clone()));
  // Overlap test points: skip any on a piece's own cut faces — those touch
  // neighbouring pieces legitimately, possibly several at once when two
  // groups join.
  const samples = geometries.map((g, i) => {
    const pos = g.getAttribute('position');
    const step = Math.max(1, Math.floor(pos.count / 500));
    const out: THREE.Vector3[] = [];
    const v = new THREE.Vector3();
    for (let k = 0; k < pos.count; k += step) {
      v.fromBufferAttribute(pos, k);
      if (faces[i].some((f) => Math.abs(v.clone().sub(f.centre).dot(f.normal)) < 2)) continue;
      // Around magnet pockets/pegs the pieces press or interlock together.
      if (holes[i].some((h) => v.distanceTo(h.opening) < Math.max(8, h.radius + 4))) continue;
      out.push(v.clone());
    }
    return out;
  });
  /** One greedy pass, best outline matches first, never using a banned match. */
  const firstFixed = fixed ? fixed.findIndex(Boolean) : -1;
  if (seedGroup && firstFixed >= 0) {
    // Everything locked (and anything already joined to it) is one group.
    const lockedGroups = new Set(seedGroup.filter((_, i) => fixed![i]));
    const g0 = seedGroup[firstFixed];
    seedGroup = seedGroup.map((g) => (lockedGroups.has(g) ? g0 : g));
  }
  const isFixedGroup = (group: number[], g: number) => firstFixed >= 0 && group[firstFixed] === g;
  /** Best alignment of group B onto group A by all their pockets (world frame), if it lines up more than `atLeast`. */
  const groupMatchMemo = new Map<string, THREE.Matrix4 | null>();
  const groupMagnetMatch = (membersA: number[], membersB: number[], place: THREE.Matrix4[], atLeast: number): THREE.Matrix4 | null => {
    const key = [membersA, membersB].map((ms) => ms.map((k) => `${k}:${place[k].elements.map((e) => e.toFixed(1)).join(',')}`).join(';')).join('|');
    if (groupMatchMemo.has(key)) return groupMatchMemo.get(key)!;
    const inWorld = (ms: number[]) => ms.flatMap((k) => holes[k].map((h) => ({ ...h, opening: h.opening.clone().applyMatrix4(place[k]), axis: h.axis.clone().transformDirection(place[k]) })));
    const best = matchMagnets(inWorld(membersA), inWorld(membersB), 1, 3)[0];
    const out = best && best.pairs.length > atLeast ? best.bToA : null;
    groupMatchMemo.set(key, out);
    return out;
  };
  const greedy = async (banned: Set<Join>, order: Join[] = joinsToTry) => {
    const group = seedGroup ? seedGroup.slice() : Array.from({ length: n }, (_, i) => i);
    const place = seed ? seed.map((m) => m.clone()) : Array.from({ length: n }, () => new THREE.Matrix4());
    const usedFace = new Set<object>(seedFaces);
    const used: Join[] = [];
    for (const m of order) {
      if (banned.has(m)) continue;
      let ga = group[m.pa], gb = group[m.pb];
      if (ga === gb || m.uses.some((u) => usedFace.has(u))) continue;
      if (isFixedGroup(group, ga) && isFixedGroup(group, gb)) continue;
      let membersA = group.map((g, i) => (g === ga ? i : -1)).filter((i) => i >= 0);
      let membersB = group.map((g, i) => (g === gb ? i : -1)).filter((i) => i >= 0);
      const scaleA = place[m.pa].getMaxScaleOnAxis() || 1;
      const scaleB = place[m.pb].getMaxScaleOnAxis() || 1;
      // Move all of b's group (resizing it as a whole if needed) so b sits on a — unless b's group is locked
      // or (when neither is locked) b's group was explicitly scaled / has more assembled parts than a's.
      const keepB = isFixedGroup(group, gb) || (!isFixedGroup(group, ga) && (
        (Math.abs(scaleB - 1) > 0.01 && Math.abs(scaleA - 1) <= 0.01) ||
        (Math.abs(scaleA - scaleB) <= 0.01 && membersB.length > membersA.length)
      ));
      let move = place[m.pa].clone().multiply(m.bToA).multiply(place[m.pb].clone().invert());
      if (keepB) {
        move = place[m.pb].clone().multiply(m.bToA.clone().invert()).multiply(place[m.pa].clone().invert());
        [ga, gb] = [gb, ga];
        [membersA, membersB] = [membersB, membersA];
      }
      if (m.magnets > 0 && membersA.length + membersB.length > 2) {
        // Match all pockets of both groups as they stand now: a piece joining a
        // group can line up with several of its pieces at once, which pins its
        // orientation far better than its pockets on one neighbour.
        const better = groupMagnetMatch(membersA, membersB, place, m.magnets);
        if (better) move = better;
      }
      const trial = place.map((p, i) => (group[i] === gb ? move.clone().multiply(p) : p));
      const ov = overlaps(bvhs, samples, membersA, membersB, trial, m.magnets > 0);
      debug?.(`try ${m.pa}+${m.pb} ${m.magnets ? `${m.magnets} magnets` : 'face'} rms ${m.rms.toFixed(3)} overlap ${(ov * 100).toFixed(1)}%`);
      // Real clashes measure ≥ ~3.5%; below that is contact noise where three pieces meet.
      if (ov > 0.03) continue;
      for (const i of membersB) { place[i] = trial[i]; group[i] = ga; }
      for (const u of m.uses) usedFace.add(u);
      used.push(m);
    }
    await yieldNow();
    return { group, place, used, score: scoreAssembly(group, place) };
  };

  /**
   * Whole-assembly score: biggest group first, then how many cut faces end up
   * pressed against a partner face (a right assembly closes almost all of them).
   */
  const scoreAssembly = (group: number[], place: THREE.Matrix4[]) => {
    const counts = new Map<number, number>();
    for (const g of group) counts.set(g, (counts.get(g) ?? 0) + 1);
    const biggest = Math.max(...counts.values());
    const world = all.map((f) => {
      const m = place[f.piece];
      return { f, sc: m.getMaxScaleOnAxis() || 1, c: f.centre.clone().applyMatrix4(m), n: f.normal.clone().transformDirection(m) };
    });
    let closed = 0;
    for (let i = 0; i < world.length; i++) for (let j = i + 1; j < world.length; j++) {
      const A = world[i], B = world[j];
      if (A.f.piece === B.f.piece || group[A.f.piece] !== group[B.f.piece]) continue;
      if (A.n.dot(B.n) > -0.99 || A.c.distanceTo(B.c) > 2 * A.sc) continue;
      if (Math.abs(A.f.area - B.f.area) / Math.max(A.f.area, B.f.area) > 0.06) continue;
      closed++;
    }
    // Magnet pockets of different pieces that end up face to face.
    const pk = holes.flat().map((h) => {
      const m = place[h.piece];
      return { h, sc: m.getMaxScaleOnAxis() || 1, c: h.opening.clone().applyMatrix4(m), a: h.axis.clone().transformDirection(m) };
    });
    for (let i = 0; i < pk.length; i++) for (let j = i + 1; j < pk.length; j++) {
      const A = pk[i], B = pk[j];
      if (A.h.piece === B.h.piece || group[A.h.piece] !== group[B.h.piece]) continue;
      if (A.c.distanceTo(B.c) < 1.5 * A.sc && ((A.h.through || B.h.through) ? Math.abs(A.a.dot(B.a)) > 0.82 : A.a.dot(B.a) < -0.82)) closed++;
    }
    return biggest * 1000 + closed;
  };

  progress(`Joining pieces (${joinsToTry.length} candidate joins)…`);
  await yieldNow();
  let best = await greedy(new Set());
  let banned = new Set<Join>();
  debug?.(`greedy: score ${best.score.toFixed(1)}`);
  // Near-equal matches can come in any order (tiny rounding differences), and a
  // wrong early join sends the rest astray: try a few shuffled orders too.
  for (let r = 1; r <= 6; r++) {
    const jitter = (k: number) => (((k + 1) * 2654435761 * r) % 1000) / 1000 - 0.5;
    const order = joinsToTry
      .map((m, k) => ({ m, key: (m.magnets ? -1000 - m.magnets : 0) + m.rms + 0.08 * jitter(k) }))
      .sort((x, y) => x.key - y.key)
      .map((x) => x.m);
    const cand = await greedy(new Set(), order);
    debug?.(`shuffle ${r}: score ${cand.score.toFixed(1)}`);
    if (cand.score > best.score) best = cand;
  }
  // Local search: try forbidding each join of the best assembly in turn.
  for (let round = 0; round < 4; round++) {
    let improved: typeof best | null = null, improvedBan: Set<Join> | null = null;
    for (const m of best.used) {
      const ban = new Set(banned); ban.add(m);
      const cand = await greedy(ban);
      if (cand.score > (improved ?? best).score) { improved = cand; improvedBan = ban; }
    }
    if (!improved) break;
    best = improved; banned = improvedBan!;
    debug?.(`round ${round + 1}: score ${best.score.toFixed(1)}`);
    progress(`Checking alternatives… (${round + 1})`);
  }
  const { group, place } = best;
  const joins = best.used.map((m) => ({ a: m.pa, b: m.pb, rms: m.rms, magnets: m.magnets }));
  // Locked pieces must not have moved.
  if (firstFixed >= 0 && seed) for (let i = 0; i < n; i++) if (fixed![i]) place[i].copy(seed[i]);
  T('joining', t0);
  return { matrices: place, group, joins, faces: all.length, pockets: holes.flat().length };
}

// ---------------------------------------------------------------------------
// Point-pair fitting (hand alignment of parts that don't have flat cuts)

/**
 * Rigid transform (rotation + translation) moving `src` points onto `dst`.
 * 1 point: translate. 2: also turn the src segment onto the dst segment.
 * 3+: least-squares fit (Horn's quaternion method).
 */
export function rigidFit(src: THREE.Vector3[], dst: THREE.Vector3[]): { matrix: THREE.Matrix4; rms: number } {
  const n = src.length;
  const cs = src.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(n);
  const cd = dst.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(n);
  const q = new THREE.Quaternion();
  if (n === 2) {
    q.setFromUnitVectors(src[1].clone().sub(src[0]).normalize(), dst[1].clone().sub(dst[0]).normalize());
  } else if (n >= 3) {
    // Cross-covariance and Horn's symmetric 4x4 matrix; its top eigenvector is the rotation.
    let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
    for (let i = 0; i < n; i++) {
      const a = src[i].clone().sub(cs), b = dst[i].clone().sub(cd);
      Sxx += a.x * b.x; Sxy += a.x * b.y; Sxz += a.x * b.z;
      Syx += a.y * b.x; Syy += a.y * b.y; Syz += a.y * b.z;
      Szx += a.z * b.x; Szy += a.z * b.y; Szz += a.z * b.z;
    }
    const N = [
      [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
      [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
      [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
      [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
    ];
    // Eigenvector of the largest eigenvalue (Jacobi rotations on the symmetric 4x4).
    const A = N.map((r) => r.slice());
    const V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    for (let sweep = 0; sweep < 50; sweep++) {
      let off = 0;
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) off += A[i][j] * A[i][j];
      if (off < 1e-20) break;
      for (let p = 0; p < 4; p++) for (let r = p + 1; r < 4; r++) {
        if (Math.abs(A[p][r]) < 1e-30) continue;
        const theta = (A[r][r] - A[p][p]) / (2 * A[p][r]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < 4; k++) { // A = A·J
          const akp = A[k][p], akr = A[k][r];
          A[k][p] = c * akp - sn * akr; A[k][r] = sn * akp + c * akr;
        }
        for (let k = 0; k < 4; k++) { // A = Jᵀ·A
          const apk = A[p][k], ark = A[r][k];
          A[p][k] = c * apk - sn * ark; A[r][k] = sn * apk + c * ark;
        }
        for (let k = 0; k < 4; k++) {
          const vkp = V[k][p], vkr = V[k][r];
          V[k][p] = c * vkp - sn * vkr; V[k][r] = sn * vkp + c * vkr;
        }
      }
    }
    let best = 0;
    for (let i = 1; i < 4; i++) if (A[i][i] > A[best][best]) best = i;
    const v = [V[0][best], V[1][best], V[2][best], V[3][best]];
    q.set(v[1], v[2], v[3], v[0]).normalize();
  }
  const m = new THREE.Matrix4().makeRotationFromQuaternion(q);
  const t = cd.clone().sub(cs.clone().applyQuaternion(q));
  m.setPosition(t);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += src[i].clone().applyMatrix4(m).distanceToSquared(dst[i]);
  return { matrix: m, rms: Math.sqrt(sum / n) };
}

