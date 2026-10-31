import type * as THREE from 'three';

// Fast per-piece surface analysis shared by cut-face and magnet-pocket
// detection: welded vertices, per-triangle normal / centroid / area, and
// vertex -> triangle adjacency, all in flat typed arrays. Computed once per
// geometry and remembered (pieces can have millions of triangles).

export interface Surface {
  /** Welded vertex positions (xyz per vertex). */
  pos: Float32Array;
  /** Triangle vertex indices into `pos`. */
  idx: Uint32Array;
  nt: number;
  /** Unit normal per triangle (xyz). */
  N: Float32Array;
  /** Centroid per triangle (xyz). */
  C: Float32Array;
  /** Area per triangle. */
  A: Float32Array;
  /** Triangles touching vertex v: vtList[vtStart[v] .. vtStart[v + 1]). */
  vtStart: Uint32Array;
  vtList: Uint32Array;
}

const cache = new WeakMap<THREE.BufferGeometry, Surface>();

/** Forget cached analysis (call after changing a geometry in place, e.g. rescaling). */
export function forgetSurface(g: THREE.BufferGeometry) { cache.delete(g); }

export function analyseSurface(g: THREE.BufferGeometry): Surface {
  const hit = cache.get(g);
  if (hit) return hit;
  const src = g.getAttribute('position');
  const srcIdx = g.getIndex();
  const nCorners = srcIdx ? srcIdx.count : src.count;
  const nt = Math.floor(nCorners / 3);
  // Weld corners whose positions match to 1e-3 (open-addressing hash on quantised coords).
  let cap = 1;
  while (cap < src.count * 2) cap <<= 1;
  const table = new Int32Array(cap).fill(-1);
  const remap = new Uint32Array(src.count);
  const qx = new Int32Array(src.count), qy = new Int32Array(src.count), qz = new Int32Array(src.count);
  const posTmp = new Float32Array(src.count * 3);
  let nv = 0;
  const arr = src.array as Float32Array;
  const stride = (src as THREE.BufferAttribute).itemSize === 3 && !(src as unknown as { isInterleavedBufferAttribute?: boolean }).isInterleavedBufferAttribute ? 3 : 0;
  for (let i = 0; i < src.count; i++) {
    const x = stride ? arr[i * 3] : src.getX(i), y = stride ? arr[i * 3 + 1] : src.getY(i), z = stride ? arr[i * 3 + 2] : src.getZ(i);
    const xi = Math.round(x * 1000), yi = Math.round(y * 1000), zi = Math.round(z * 1000);
    let h = (Math.imul(xi, 73856093) ^ Math.imul(yi, 19349663) ^ Math.imul(zi, 83492791)) & (cap - 1);
    for (;;) {
      const id = table[h];
      if (id < 0) {
        table[h] = nv; qx[nv] = xi; qy[nv] = yi; qz[nv] = zi;
        posTmp[nv * 3] = x; posTmp[nv * 3 + 1] = y; posTmp[nv * 3 + 2] = z;
        remap[i] = nv++;
        break;
      }
      if (qx[id] === xi && qy[id] === yi && qz[id] === zi) { remap[i] = id; break; }
      h = (h + 1) & (cap - 1);
    }
  }
  const pos = posTmp.slice(0, nv * 3);
  const idx = new Uint32Array(nt * 3);
  for (let c = 0; c < nt * 3; c++) idx[c] = remap[srcIdx ? srcIdx.getX(c) : c];
  // Per-triangle normal, centroid, area.
  const N = new Float32Array(nt * 3), C = new Float32Array(nt * 3), A = new Float32Array(nt);
  for (let t = 0; t < nt; t++) {
    const a = idx[t * 3] * 3, b = idx[t * 3 + 1] * 3, c = idx[t * 3 + 2] * 3;
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    A[t] = len / 2;
    if (len > 0) { N[t * 3] = nx / len; N[t * 3 + 1] = ny / len; N[t * 3 + 2] = nz / len; }
    C[t * 3] = (pos[a] + pos[b] + pos[c]) / 3; C[t * 3 + 1] = (pos[a + 1] + pos[b + 1] + pos[c + 1]) / 3; C[t * 3 + 2] = (pos[a + 2] + pos[b + 2] + pos[c + 2]) / 3;
  }
  // Vertex -> triangle adjacency (CSR).
  const vtStart = new Uint32Array(nv + 1);
  for (let c = 0; c < nt * 3; c++) vtStart[idx[c] + 1]++;
  for (let v = 0; v < nv; v++) vtStart[v + 1] += vtStart[v];
  const fill = vtStart.slice(0, nv);
  const vtList = new Uint32Array(nt * 3);
  for (let c = 0; c < nt * 3; c++) vtList[fill[idx[c]]++] = (c / 3) | 0;
  const s: Surface = { pos, idx, nt, N, C, A, vtStart, vtList };
  cache.set(g, s);
  return s;
}

/**
 * Flat regions of a surface. Strict (cut faces): every triangle's own plane
 * must match the seed's — keeps gently curved shell areas out. Loose (pocket
 * floors on bumpy meshes): triangles stay within planeTol of the seed plane.
 * Returns triangle lists of regions with at least `minArea`.
 */
export function flatRegions(s: Surface, minArea: number, normalDot: number, planeTol: number): Array<{ tris: number[]; area: number; seed: number }> {
  const strict = normalDot >= 0.999;
  const { N, C, A, idx, vtStart, vtList, nt } = s;
  const seen = new Uint8Array(nt);
  const stack = new Int32Array(nt);
  const out: Array<{ tris: number[]; area: number; seed: number }> = [];
  for (let s0 = 0; s0 < nt; s0++) {
    if (seen[s0] || A[s0] === 0) continue;
    const nx = N[s0 * 3], ny = N[s0 * 3 + 1], nz = N[s0 * 3 + 2];
    const d0 = nx * C[s0 * 3] + ny * C[s0 * 3 + 1] + nz * C[s0 * 3 + 2];
    let sp = 0;
    stack[sp++] = s0; seen[s0] = 1;
    const tris: number[] = [];
    let area = 0;
    while (sp) {
      const t = stack[--sp];
      tris.push(t); area += A[t];
      for (let k = 0; k < 3; k++) {
        const v = idx[t * 3 + k];
        for (let e = vtStart[v]; e < vtStart[v + 1]; e++) {
          const u = vtList[e];
          if (seen[u] || A[u] === 0) continue;
          const ux = N[u * 3], uy = N[u * 3 + 1], uz = N[u * 3 + 2];
          if (ux * nx + uy * ny + uz * nz <= normalDot) continue;
          const cx = C[u * 3], cy = C[u * 3 + 1], cz = C[u * 3 + 2];
          const off = strict ? ux * cx + uy * cy + uz * cz - d0 : nx * cx + ny * cy + nz * cz - d0;
          if (Math.abs(off) < planeTol) { seen[u] = 1; stack[sp++] = u; }
        }
      }
    }
    if (area >= minArea) out.push({ tris, area, seed: s0 });
  }
  return out;
}
