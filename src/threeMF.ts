import { Unzip, UnzipInflate, unzipSync } from 'three/examples/jsm/libs/fflate.module.js';
import type { IndexedMesh } from './simplify';

// Streaming 3MF reader. three.js's loader unzips the whole archive and builds
// a DOM of the XML — fine for small files, but a 160 MB 3MF holds ~1 GB of
// XML, more than a browser string can hold, so the tab dies. This reads the
// zip in chunks, scans the XML text tag by tag as it decompresses, and writes
// straight into typed arrays. Each build item (plate part) becomes its own part.

export interface MeshPart extends IndexedMesh {
  name: string;
  /**
   * Bambu/Orca record where each part sat in the file it came from
   * (source_offset). When present this matrix (column-major 4x4, mm) moves the
   * part from its plate position back to that original position — for pieces
   * that came from assembled files it rebuilds the model exactly.
   */
  sourceMatrix?: number[];
}

const UNIT_MM: Record<string, number> = { micron: 0.001, millimeter: 1, centimeter: 10, inch: 25.4, foot: 304.8, meter: 1000 };

/** Growable typed array built from fixed-size chunks (no big re-allocations while reading). */
class Chunked<T extends Float32Array | Uint32Array> {
  private chunks: T[] = [];
  private cur: T;
  private n = 0;
  length = 0;
  constructor(private make: (n: number) => T, private size = 1 << 20) { this.cur = make(size); }
  push(v: number) {
    if (this.n === this.size) { this.chunks.push(this.cur); this.cur = this.make(this.size); this.n = 0; }
    this.cur[this.n++] = v; this.length++;
  }
  done(): T {
    const out = this.make(this.length);
    let o = 0;
    for (const c of this.chunks) { out.set(c, o); o += c.length; }
    out.set(this.cur.subarray(0, this.n) as T, o);
    return out;
  }
}

interface ObjectDef {
  name: string;
  mesh?: { positions: Float32Array; index: Uint32Array };
  components: Array<{ ref: string; transform: number[] | null }>;
}

function attr(tag: string, name: string): string | null {
  // Matches name="..." with a word boundary before the name.
  let i = tag.indexOf(name + '="');
  while (i > 0 && /[\w:]/.test(tag[i - 1])) i = tag.indexOf(name + '="', i + 1);
  if (i < 0) return null;
  const s = i + name.length + 2;
  return tag.slice(s, tag.indexOf('"', s));
}

/** 3MF 3x4 matrix "m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32" (row vectors). */
function parseTransform(s: string | null): number[] | null {
  if (!s) return null;
  const m = s.trim().split(/\s+/).map(Number);
  return m.length === 12 && m.every(Number.isFinite) ? m : null;
}
function mul(a: number[] | null, b: number[] | null): number[] | null {
  // Apply a then b (row-vector convention): p·A·B.
  if (!a) return b; if (!b) return a;
  const r = new Array(12);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
    r[i * 3 + j] = (i < 3 ? 0 : b[9 + j]) + a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  }
  return r;
}

export function parse3MF(data: ArrayBuffer, onProgress?: (msg: string) => void): MeshPart[] {
  const objects = new Map<string, ObjectDef>();
  const items: Array<{ ref: string; transform: number[] | null; name: string | null }> = [];
  let unit = 1;
  let rootPath = '';

  const parseModel = (path: string, chunks: (cb: (text: string, final: boolean) => void) => void) => {
    let buf = '';
    let obj: ObjectDef | null = null;
    let objKey = '';
    let verts: Chunked<Float32Array> | null = null, tris: Chunked<Uint32Array> | null = null;
    const key = (p: string | null, id: string | null) => `${p ?? path}#${id}`;
    const handle = (tag: string) => {
      // Tag name without namespace prefix.
      const closing = tag[1] === '/';
      let e = closing ? 2 : 1;
      while (e < tag.length && !/[\s/>]/.test(tag[e])) e++;
      let name = tag.slice(closing ? 2 : 1, e);
      const colon = name.indexOf(':');
      if (colon >= 0) name = name.slice(colon + 1);
      if (closing) {
        if (name === 'mesh' && obj && verts && tris) {
          (obj as ObjectDef).mesh = { positions: verts.done(), index: tris.done() };
          verts = null; tris = null;
        } else if (name === 'object') obj = null;
        return;
      }
      switch (name) {
        case 'vertex':
          if (verts) { verts.push(parseFloat(attr(tag, 'x')!)); verts.push(parseFloat(attr(tag, 'y')!)); verts.push(parseFloat(attr(tag, 'z')!)); }
          break;
        case 'triangle':
          if (tris) { tris.push(+attr(tag, 'v1')!); tris.push(+attr(tag, 'v2')!); tris.push(+attr(tag, 'v3')!); }
          break;
        case 'model': {
          const u = attr(tag, 'unit');
          // Units come from the root model (applied when parts are built, so file order doesn't matter).
          if (u && path === rootPath) unit = UNIT_MM[u] ?? 1;
          break;
        }
        case 'object':
          obj = { name: attr(tag, 'name') ?? `object ${attr(tag, 'id')}`, components: [] };
          objKey = key(null, attr(tag, 'id'));
          objects.set(objKey, obj);
          if (tag.endsWith('/>')) obj = null;
          break;
        case 'mesh':
          verts = new Chunked((n) => new Float32Array(n));
          tris = new Chunked((n) => new Uint32Array(n));
          break;
        case 'component':
          obj?.components.push({ ref: key(attr(tag, 'p:path'), attr(tag, 'objectid')), transform: parseTransform(attr(tag, 'transform')) });
          break;
        case 'item':
          items.push({ ref: key(attr(tag, 'p:path'), attr(tag, 'objectid')), transform: parseTransform(attr(tag, 'transform')), name: null });
          break;
      }
    };
    chunks((text, final) => {
      buf += text;
      let i = 0;
      for (;;) {
        const s = buf.indexOf('<', i);
        if (s < 0) { i = buf.length; break; }
        const e = buf.indexOf('>', s);
        if (e < 0) { i = s; break; }
        handle(buf.slice(s, e + 1));
        i = e + 1;
      }
      buf = buf.slice(i);
      if (final) buf = '';
    });
  };

  // The root model is almost always 3D/3dmodel.model; parse every .model entry
  // (multi-file 3MFs from slicers keep each object in 3D/Objects/*.model).
  let modelCount = 0;
  let total = 0;
  let config = '';
  const unzip = new Unzip((file) => {
    if (/model_settings\.config$/i.test(file.name)) {
      // Bambu/Orca keep part names here.
      const dec = new TextDecoder();
      file.ondata = (err, chunk, final) => { if (err) throw err; config += dec.decode(chunk, { stream: !final }); };
      file.start();
      return;
    }
    if (!/\.model$/i.test(file.name)) return;
    // Decompressed XML goes straight into the text scanner chunk by chunk.
    const path = '/' + file.name.replace(/^\/+/, '');
    if (/3dmodel\.model$/i.test(file.name) && !rootPath) rootPath = path;
    const decoder = new TextDecoder();
    let push: ((text: string, final: boolean) => void) | null = null;
    parseModel(path, (cb) => { push = cb; });
    file.ondata = (err, chunk, final) => {
      if (err) throw err;
      total += chunk.length;
      push!(decoder.decode(chunk, { stream: !final }), final);
    };
    file.start();
    modelCount++;
  });
  unzip.register(UnzipInflate);
  const bytes = new Uint8Array(data);
  const STEP = 4 << 20;
  for (let o = 0; o < bytes.length; o += STEP) {
    unzip.push(bytes.subarray(o, Math.min(bytes.length, o + STEP)), o + STEP >= bytes.length);
    onProgress?.(`Unzipping 3MF… ${Math.round((100 * Math.min(bytes.length, o + STEP)) / bytes.length)}% (${(total / 1e6).toFixed(0)} MB unpacked)`);
  }
  if (!modelCount) throw new Error('No 3D model found inside this 3MF.');

  // Names from slicer config (Bambu/Orca): <object id="12"><metadata key="name" …/>
  // <part id="10"> <metadata key="name" value="…"/>, where part id = the component's objectid.
  const objectNames = new Map<string, string>(), partNames = new Map<string, string>();
  const sourceOffsets = new Map<string, [number, number, number]>();
  for (const om of config.matchAll(/<object id="(\d+)">([\s\S]*?)<\/object>/g)) {
    const own = om[2].split('<part')[0].match(/<metadata key="name" value="([^"]*)"/);
    if (own) objectNames.set(om[1], own[1]);
    for (const pm of om[2].matchAll(/<part id="(\d+)"[^>]*>([\s\S]*?)<\/part>/g)) {
      const nm = pm[2].match(/<metadata key="name" value="([^"]*)"/);
      if (nm) partNames.set(pm[1], nm[1]);
      const off = ['x', 'y', 'z'].map((a) => pm[2].match(new RegExp(`key="source_offset_${a}" value="([^"]*)"`)));
      if (off.every(Boolean)) sourceOffsets.set(pm[1], off.map((m) => parseFloat(m![1])) as [number, number, number]);
    }
  }
  const idOf = (ref: string) => ref.slice(ref.indexOf('#') + 1);

  // Resolve build items, flattening components and transforms. A build item
  // made of several meshes (a slicer "assembly" of parts) gives one part per
  // mesh, so the pieces stay separate.
  onProgress?.('Building parts…');
  const parts: MeshPart[] = [];
  type Leaf = { mesh: NonNullable<ObjectDef['mesh']>; t: number[] | null; name: string; offset?: [number, number, number] };
  const leaves = (ref: string, t: number[] | null, name: string, out: Leaf[], depth: number) => {
    const o = objects.get(ref);
    if (!o || depth > 16) return;
    if (o.mesh) out.push({ mesh: o.mesh, t, name, offset: sourceOffsets.get(idOf(ref)) });
    o.components.forEach((c, k) => {
      leaves(c.ref, mul(c.transform, t), partNames.get(idOf(c.ref)) ?? `${name} part ${k + 1}`, out, depth + 1);
    });
  };
  /** Plate position -> original file position: undo the placement t, then add the source offset (mm). */
  const sourceMatrix = (l: Leaf): number[] | undefined => {
    if (!l.offset) return undefined;
    const t = l.t ?? [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
    // Row-vector 3x4 t as a column-vector affine matrix A (p' = A p), in mm.
    const A = [t[0], t[3], t[6], t[9] * unit, t[1], t[4], t[7], t[10] * unit, t[2], t[5], t[8], t[11] * unit];
    // Invert the affine A = [R | c]: R^-1, -R^-1 c.
    const [a, b, c, d, e, f, g, h, i] = [A[0], A[1], A[2], A[4], A[5], A[6], A[8], A[9], A[10]];
    const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (Math.abs(det) < 1e-12) return undefined;
    const R = [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det, (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
    const tx = A[3], ty = A[7], tz = A[11];
    const o = l.offset.map((v) => v * unit);
    const T = [0, 1, 2].map((r) => -(R[r * 3] * tx + R[r * 3 + 1] * ty + R[r * 3 + 2] * tz) + o[r]);
    // Column-major for THREE.Matrix4.fromArray.
    return [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, T[0], T[1], T[2], 1];
  };
  const toPart = (name: string, list: Leaf[]): MeshPart => {
    let nv = 0, ni = 0;
    for (const l of list) { nv += l.mesh.positions.length; ni += l.mesh.index.length; }
    const positions = new Float32Array(nv), index = new Uint32Array(ni);
    let ov = 0, oi = 0;
    for (const { mesh, t } of list) {
      const p = mesh.positions, base = ov / 3;
      for (let i = 0; i < p.length; i += 3) {
        const x = p[i], y = p[i + 1], z = p[i + 2];
        if (t) {
          positions[ov++] = (x * t[0] + y * t[3] + z * t[6] + t[9]) * unit;
          positions[ov++] = (x * t[1] + y * t[4] + z * t[7] + t[10]) * unit;
          positions[ov++] = (x * t[2] + y * t[5] + z * t[8] + t[11]) * unit;
        } else { positions[ov++] = x * unit; positions[ov++] = y * unit; positions[ov++] = z * unit; }
      }
      for (const v of mesh.index) index[oi++] = v + base;
    }
    return { name, positions, index };
  };
  const roots = items.length ? items : [...objects.keys()].filter((k) => k.startsWith(rootPath + '#')).map((ref) => ({ ref, transform: null, name: null }));
  for (const it of roots) {
    const list: Leaf[] = [];
    const objName = objectNames.get(idOf(it.ref)) ?? objects.get(it.ref)?.name ?? `part ${parts.length + 1}`;
    leaves(it.ref, it.transform, objName, list, 0);
    const withTris = list.filter((l) => l.mesh.index.length >= 3);
    if (withTris.length > 1) for (const l of withTris) parts.push({ ...toPart(l.name, [l]), sourceMatrix: sourceMatrix(l) });
    else if (withTris.length === 1) parts.push({ ...toPart(objName, withTris), sourceMatrix: sourceMatrix(withTris[0]) });
  }
  if (!parts.length) throw new Error('This 3MF has no triangles.');
  // Source offsets are only useful if they're really there (all-zero means the
  // parts were made inside the slicer) and put the parts together compactly.
  const anyOffset = [...sourceOffsets.values()].some((o) => Math.hypot(...o) > 0.01);
  if (!anyOffset || !compactWhenRebuilt(parts)) for (const p of parts) delete p.sourceMatrix;
  return parts;
}

/** Do the parts, moved to their recorded source positions, form one compact object (not a spread-out plate)? */
function compactWhenRebuilt(parts: MeshPart[]): boolean {
  const withSrc = parts.filter((p) => p.sourceMatrix);
  if (withSrc.length < 2) return false;
  const all = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  let biggest = 0;
  for (const p of withSrc) {
    const m = p.sourceMatrix!;
    const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    const step = Math.max(3, Math.floor(p.positions.length / 3 / 2000) * 3);
    for (let i = 0; i < p.positions.length; i += step) {
      const x = p.positions[i], y = p.positions[i + 1], z = p.positions[i + 2];
      const q = [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]];
      for (let k = 0; k < 3; k++) { b[k] = Math.min(b[k], q[k]); b[k + 3] = Math.max(b[k + 3], q[k]); all[k] = Math.min(all[k], q[k]); all[k + 3] = Math.max(all[k + 3], q[k]); }
    }
    biggest = Math.max(biggest, Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]));
  }
  return Math.hypot(all[3] - all[0], all[4] - all[1], all[5] - all[2]) < 2.5 * biggest;
}

export interface ProjectImage { name: string; mime: string; data: Uint8Array }

/**
 * Pictures stored in a slicer project (designer photos, thumbnails, plate
 * previews), best first. Only these entries are decompressed.
 */
export function extract3MFImages(data: ArrayBuffer, max = 30): ProjectImage[] {
  const rank = (n: string) => {
    if (/(^|\/)(pick_|plate_no_light|top_)/i.test(n)) return -1; // selection masks / technical renders
    if (/Auxiliaries\/Model Pictures\//i.test(n)) return 0;
    if (/Auxiliaries\/Profile Pictures\/(?!plate_)/i.test(n)) return 1;
    if (/thumbnail_(middle|3mf)/i.test(n)) return 2;
    if (/Metadata\/plate_\d+\.png$/i.test(n)) return 3;
    return -1;
  };
  const files = unzipSync(new Uint8Array(data), { filter: (f) => /\.(png|jpe?g|webp)$/i.test(f.name) && rank(f.name) >= 0 && f.originalSize < 8e6 });
  return Object.entries(files)
    .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b, undefined, { numeric: true }))
    .slice(0, max)
    .map(([name, bytes]) => ({ name, data: bytes, mime: /\.png$/i.test(name) ? 'image/png' : /\.webp$/i.test(name) ? 'image/webp' : 'image/jpeg' }));
}
