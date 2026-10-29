// Fast mesh simplification by vertex clustering: snap vertices to a grid,
// merge the ones in the same cell, drop triangles that collapse. Used to keep
// display and fitting responsive for huge meshes; exports keep full detail.

export interface IndexedMesh { positions: Float32Array; index: Uint32Array }

export const DISPLAY_TRIANGLE_BUDGET = 2_500_000;

/** Simplify to roughly `target` triangles (no-op if already under). */
export function clusterSimplify(mesh: IndexedMesh, target: number): IndexedMesh {
  const nTri = mesh.index.length / 3;
  if (nTri <= target) return mesh;
  const p = mesh.positions;
  let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < p.length; i += 3) {
    const x = p[i], y = p[i + 1], z = p[i + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  // Total surface area guides the first cell size: occupied cells ≈ area / h².
  let area = 0;
  const ix = mesh.index;
  for (let t = 0; t < ix.length; t += 3) {
    const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    area += 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
  }
  let h = Math.sqrt(area / (target / 2)) * 0.7;
  for (let attempt = 0; attempt < 6; attempt++) {
    const out = clusterOnce(mesh, h, minX, minY, minZ, maxX, maxY, maxZ);
    if (out.index.length / 3 <= target * 1.15 || attempt === 5) return out;
    h *= 1.25;
  }
  return mesh;
}

function clusterOnce(mesh: IndexedMesh, h: number, minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): IndexedMesh {
  const p = mesh.positions;
  const nx = Math.max(1, Math.ceil((maxX - minX) / h) + 1), ny = Math.max(1, Math.ceil((maxY - minY) / h) + 1);
  void maxZ;
  const cellOf = new Map<number, number>();
  const remap = new Uint32Array(p.length / 3);
  const sums: number[] = [];
  const counts: number[] = [];
  for (let v = 0; v < p.length / 3; v++) {
    const i = Math.floor((p[v * 3] - minX) / h), j = Math.floor((p[v * 3 + 1] - minY) / h), k = Math.floor((p[v * 3 + 2] - minZ) / h);
    const key = (k * ny + j) * nx + i;
    let c = cellOf.get(key);
    if (c === undefined) { c = counts.length; cellOf.set(key, c); counts.push(0); sums.push(0, 0, 0); }
    remap[v] = c;
    counts[c]++;
    sums[c * 3] += p[v * 3]; sums[c * 3 + 1] += p[v * 3 + 1]; sums[c * 3 + 2] += p[v * 3 + 2];
  }
  const positions = new Float32Array(counts.length * 3);
  for (let c = 0; c < counts.length; c++) {
    positions[c * 3] = sums[c * 3] / counts[c]; positions[c * 3 + 1] = sums[c * 3 + 1] / counts[c]; positions[c * 3 + 2] = sums[c * 3 + 2] / counts[c];
  }
  const ix = mesh.index;
  const keep = new Uint32Array(ix.length);
  let n = 0;
  for (let t = 0; t < ix.length; t += 3) {
    const a = remap[ix[t]], b = remap[ix[t + 1]], c = remap[ix[t + 2]];
    if (a === b || b === c || a === c) continue;
    keep[n++] = a; keep[n++] = b; keep[n++] = c;
  }
  return { positions, index: keep.slice(0, n) };
}
