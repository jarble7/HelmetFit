// Generates samples/demo-helmet-*.stl: a 3-piece helmet (crown, back, face with
// eye slot), deliberately exported Z-up and oversized so Auto fit has work to do.
// Run: npx vite-node scripts/make-sample-helmet.ts
import * as THREE from 'three';
import { writeFileSync } from 'node:fs';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import { buildParametricHead, HEAD_PRESETS } from '../src/headModel';

const head = buildParametricHead(HEAD_PRESETS['Head-3 (Hat 7-3/8)']).geometry;
const pos = head.getAttribute('position'), nrm = head.getAttribute('normal'), idx = head.getIndex()!;
const GAP = 14, WALL = 3;
const v = new THREE.Vector3(), n = new THREE.Vector3();

type Region = (p: THREE.Vector3) => boolean;
// Full-face helmet down to the jaw with a T-shaped visor.
const covered: Region = (p) => p.y > -70;
const eyeSlot: Region = (p) => p.z > 40 && ((p.y > -5 && p.y < 15 && Math.abs(p.x) < 55) || (p.y > -45 && p.y <= -5 && Math.abs(p.x) < 12));
// Stand-alone face mask with two round eye holes, exported upside-down-facing and 93%.
const eyeHole: Region = (p) => Math.hypot(Math.abs(p.x) - 0.23 * HEAD_PRESETS['Head-3 (Hat 7-3/8)'].faceWidth, p.y - 3) < 17;
const mask: Region = (p) => p.z > 25 && p.y > -110 && p.y < 75 && !eyeHole(p) && !(Math.abs(p.x) < 14 && p.y < -45 && p.y > -65 && p.z > 60);
const pieces: Record<string, Region> = {
  crown: (p) => covered(p) && p.y > 60,
  back: (p) => covered(p) && p.y <= 60 && p.z < 10,
  face: (p) => covered(p) && p.y <= 60 && p.z >= 10 && !eyeSlot(p),
};

const fileT = new THREE.Matrix4().compose(new THREE.Vector3(0, 0, 0), new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0)), new THREE.Vector3(1.08, 1.08, 1.08));

const maskT = new THREE.Matrix4().compose(new THREE.Vector3(0, 0, 0), new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, Math.PI, 0)), new THREE.Vector3(0.93, 0.93, 0.93));
for (const [name, keep, T, file] of [
  ...Object.entries(pieces).map(([n, k]) => [n, k, fileT, `samples/demo-helmet-${n}.stl`] as const),
  ['mask', mask, maskT, 'samples/demo-mask.stl'] as const,
]) {
  const out: number[] = [];
  const off = (i: number, d: number) => { v.fromBufferAttribute(pos, i); n.fromBufferAttribute(nrm, i); return v.clone().addScaledVector(n, d); };
  const edgeCount = new Map<string, number>();
  const tris: number[][] = [];
  for (let t = 0; t < idx.count; t += 3) {
    const ids = [idx.getX(t), idx.getX(t + 1), idx.getX(t + 2)];
    if (!ids.every((i) => keep(v.fromBufferAttribute(pos, i).clone()))) continue;
    tris.push(ids);
    for (let k = 0; k < 3; k++) { const a = ids[k], b = ids[(k + 1) % 3]; const key = a < b ? `${a},${b}` : `${b},${a}`; edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1); }
  }
  const push = (...ps: THREE.Vector3[]) => ps.forEach((p) => out.push(p.x, p.y, p.z));
  for (const [a, b, c] of tris) {
    push(off(a, GAP + WALL), off(b, GAP + WALL), off(c, GAP + WALL)); // outer, facing out
    push(off(a, GAP), off(c, GAP), off(b, GAP)); // inner, facing the head
    // Close open edges with a rim so each piece is a solid.
    for (const [p, q] of [[a, b], [b, c], [c, a]]) {
      const key = p < q ? `${p},${q}` : `${q},${p}`;
      if (edgeCount.get(key) === 1) { push(off(p, GAP), off(q, GAP), off(q, GAP + WALL)); push(off(p, GAP), off(q, GAP + WALL), off(p, GAP + WALL)); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(out, 3));
  g.applyMatrix4(T);
  const dv = new STLExporter().parse(new THREE.Mesh(g), { binary: true }) as DataView;
  writeFileSync(file, new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength));
  console.log(name, out.length / 9, 'triangles');
}
