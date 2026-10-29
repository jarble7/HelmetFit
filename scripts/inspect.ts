// Inspect a head and a helmet file the way the app would. Usage: npx vite-node scripts/inspect.ts <head> <helmet>
import * as THREE from 'three';
import { readFileSync } from 'node:fs';
import { parseMeshFile, guessUnit } from '../src/io';
import { guessHeadOrientation, centreHead, measureHead } from '../src/measure';
import { HeadField, HelmetField, mergePieces } from '../src/fit';
import { UNIT_TO_MM } from '../src/types';

const [headPath, helmetPath] = process.argv.slice(2);
const load = (p: string) => { const b = readFileSync(p); return parseMeshFile(p, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const fmtBox = (b: THREE.Box3) => `min ${b.min.toArray().map((v) => v.toFixed(1))} max ${b.max.toArray().map((v) => v.toFixed(1))} size ${b.getSize(new THREE.Vector3()).toArray().map((v) => v.toFixed(1))}`;

const h = load(headPath);
console.log('HEAD tris', h.getAttribute('position').count / 3, fmtBox(h.boundingBox!));
const hs = h.boundingBox!.getSize(new THREE.Vector3());
const hu = guessUnit(Math.max(hs.x, hs.y, hs.z), 250);
h.scale(UNIT_TO_MM[hu], UNIT_TO_MM[hu], UNIT_TO_MM[hu]);
const q = guessHeadOrientation(h);
console.log('head unit guess', hu, 'orientation euler deg', new THREE.Euler().setFromQuaternion(q).toArray().slice(0, 3).map((r) => (+r * 180 / Math.PI).toFixed(0)));
h.applyQuaternion(q); centreHead(h); h.computeVertexNormals();
console.log('canonical head', fmtBox(h.boundingBox!));
console.log('measure', measureHead(h));
const hf = new HeadField(h);
console.log('eyes', hf.eyes?.map((e) => e.toArray().map((v) => v.toFixed(1))));

const m = load(helmetPath);
console.log('HELMET tris', m.getAttribute('position').count / 3, fmtBox(m.boundingBox!));
const ms = m.boundingBox!.getSize(new THREE.Vector3());
const mu = guessUnit(Math.max(ms.x, ms.y, ms.z), 300);
console.log('helmet unit guess', mu);
m.scale(UNIT_TO_MM[mu], UNIT_TO_MM[mu], UNIT_TO_MM[mu]);
const field = new HelmetField(mergePieces([{ geometry: m, matrix: new THREE.Matrix4() }]));
const t0 = performance.now();
const views = (field as unknown as { eyesFor: (q: THREE.Quaternion) => unknown }).eyesFor(new THREE.Quaternion());
console.log('hole detection ms', (performance.now() - t0).toFixed(0));
const v = (field as unknown as { views: Array<{ front: THREE.Vector3; byUp: Array<{ up: THREE.Vector3; holes: Array<{ area: number; centre: number[] }> }> }> }).views;
for (const vh of v) console.log("view", vh.front.toArray(), vh.byUp.map((b) => `up${b.up.toArray().map(Math.round)}:` + b.holes.map((x) => `${x.area.toFixed(0)}@(${x.centre.map((c) => c.toFixed(0))})`).join(",")).join("  "));
void views;
