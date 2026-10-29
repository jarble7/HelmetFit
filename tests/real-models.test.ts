// Regression tests on real models in samples/ (skipped if the files aren't there).
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { parseMeshFile } from '../src/io';
import { guessHeadOrientation, centreHead } from '../src/measure';
import { HeadField, HelmetField, mergePieces, autoFit } from '../src/fit';
import { looksLikeLens } from '../src/landmarks';
import { DEFAULT_FIT_SETTINGS } from '../src/types';

const HEAD = 'samples/head+(1).stl';
const SPARTAN = 'samples/obj_1_Spartan_Sentinel_1108234153_generate.stl';
const IRONPOOL = 'samples/Ironpool';
const load = (p: string) => { const b = readFileSync(p); return parseMeshFile(p, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const deg = (a: THREE.Quaternion, b: THREE.Quaternion) => THREE.MathUtils.radToDeg(a.angleTo(b));

function loadHead() {
  const g = load(HEAD);
  const q = guessHeadOrientation(g);
  g.applyQuaternion(q); centreHead(g); g.computeVertexNormals(); g.computeBoundingBox();
  return { g, q };
}

async function fit(files: string[]) {
  const { g } = loadHead();
  const geoms = files.map(load);
  const asm = new THREE.Box3(); geoms.forEach((x) => asm.union(x.boundingBox!));
  const lens = geoms.map((x) => looksLikeLens(x.boundingBox!, asm));
  const helmet = new HelmetField(mergePieces(geoms.map((x, i) => ({ geometry: x, matrix: new THREE.Matrix4(), lens: lens[i] }))));
  const res = await autoFit({ head: new HeadField(g), helmet, settings: DEFAULT_FIT_SETTINGS, current: { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), scale: new THREE.Vector3(1, 1, 1) }, headBox: g.boundingBox!.clone() });
  return { res, lens };
}

describe.skipIf(!existsSync(HEAD))('real head scan', () => {
  it('auto-orients the Z-up, face −Y head upright (flat neck cut = bottom)', () => {
    const { q } = loadHead();
    expect(deg(q, new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)))).toBeLessThan(1);
  });
  it('finds both eyes level and about 53 mm apart', () => {
    const { g } = loadHead();
    const e = new HeadField(g).eyes!;
    expect(e).not.toBeNull();
    expect(Math.abs(e[0].y - e[1].y)).toBeLessThan(2);
    expect(Math.abs(e[1].x - e[0].x - 53)).toBeLessThan(6);
  });
});

describe.skipIf(!existsSync(HEAD) || !existsSync(SPARTAN))('Spartan (Corinthian) helmet', () => {
  it('fits facing forward with the eye openings on the eyes', async () => {
    const { res } = await fit([SPARTAN]);
    // File is Z-up, face −Y.
    expect(deg(res.pose.quaternion, new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)))).toBeLessThan(15);
    expect(res.eyes?.kind).toBe('visor');
    expect(Math.hypot(res.eyes!.dx, res.eyes!.dy)).toBeLessThan(3);
  }, 120_000);
});

describe.skipIf(!existsSync(HEAD) || !existsSync(IRONPOOL))('Ironpool 10-piece helmet with lens insert', () => {
  it('spots the lens, stays upright, and lines up the eye holes', async () => {
    const files = readdirSync(IRONPOOL).filter((f) => /\.stl$/i.test(f)).sort().map((f) => `${IRONPOOL}/${f}`);
    const { res, lens } = await fit(files);
    expect(files.filter((_, i) => lens[i]).map((f) => f.split('/').pop())).toEqual(['iron pool  helmet 5.stl']);
    // File is already Y-up, face +Z: must not tip back "looking at the sky".
    expect(deg(res.pose.quaternion, new THREE.Quaternion())).toBeLessThan(20);
    expect(res.eyes?.kind).toBe('holes');
    expect(Math.hypot(res.eyes!.dx, res.eyes!.dy)).toBeLessThan(3);
  }, 180_000);
});

const BATMAN = 'samples/JumbledBatman';
describe.skipIf(!existsSync(BATMAN))('Batman cowl exported laid out for printing', () => {
  it('auto-assembles all 14 pieces with every mirror pair joined', async () => {
    const { autoAssemble } = await import('../src/assemble');
    const files = readdirSync(BATMAN).filter((f) => /\.stl$/i.test(f)).sort((a, b) => parseInt(a.match(/\d+/)![0]) - parseInt(b.match(/\d+/)![0]));
    const res = await autoAssemble(files.map((f) => load(`${BATMAN}/${f}`)));
    expect(new Set(res.group).size).toBe(1);
    // The assembled cowl is left-right symmetric: each mirror pair's centres
    // should be equally far from the middle of the whole thing.
    const centres = files.map((f, i) => load(`${BATMAN}/${f}`).boundingBox!.getCenter(new THREE.Vector3()).applyMatrix4(res.matrices[i]));
    const mid = centres.reduce((a, c) => a.add(c), new THREE.Vector3()).divideScalar(centres.length);
    const idx = (n: number) => files.indexOf(`Part_${n}.stl`);
    for (const [a, b] of [[1, 11], [2, 10], [3, 9], [5, 14], [6, 13], [7, 12]]) {
      expect(Math.abs(centres[idx(a)].distanceTo(mid) - centres[idx(b)].distanceTo(mid))).toBeLessThan(3);
    }
  }, 180_000);

  it('snaps a knocked-out piece back by its cut face, and auto-assembles around existing joins', async () => {
    const { autoAssemble, findCutFaces, snapCandidates } = await import('../src/assemble');
    const files = readdirSync(BATMAN).filter((f) => /\.stl$/i.test(f)).sort((a, b) => parseInt(a.match(/\d+/)![0]) - parseInt(b.match(/\d+/)![0]));
    const geoms = files.map((f) => load(`${BATMAN}/${f}`));
    const good = await autoAssemble(geoms);
    const i5 = files.indexOf('Part_5.stl'), i6 = files.indexOf('Part_6.stl');
    // The face pair joining 5 and 6 in the good assembly.
    const f5s = findCutFaces(geoms[i5], i5), f6s = findCutFaces(geoms[i6], i6);
    const w = (f: typeof f5s[0], m: THREE.Matrix4) => ({ c: f.centre.clone().applyMatrix4(m), n: f.normal.clone().transformDirection(m) });
    let pair: [typeof f5s[0], typeof f5s[0]] | null = null;
    for (const a of f6s) for (const b of f5s) { const A = w(a, good.matrices[i6]), B = w(b, good.matrices[i5]); if (A.n.dot(B.n) < -0.99 && A.c.distanceTo(B.c) < 3) pair = [a, b]; }
    expect(pair).not.toBeNull();
    const snapped = good.matrices[i6].clone().multiply(snapCandidates(pair![0], pair![1])[0].bToA);
    const rel = good.matrices[i5].clone().invert().multiply(snapped);
    const t = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    rel.decompose(t, q, s);
    expect(t.length()).toBeLessThan(0.1);
    expect(deg(q, new THREE.Quaternion())).toBeLessThan(0.1);
    // Knock two pieces away; assembling from the current state must put them back next to their neighbours.
    const seed = good.matrices.map((m) => m.clone());
    for (const n of [9, 12]) seed[files.indexOf(`Part_${n}.stl`)].premultiply(new THREE.Matrix4().makeRotationZ(2)).premultiply(new THREE.Matrix4().makeTranslation(0, 200, 0));
    const res = await autoAssemble(geoms, undefined, undefined, seed);
    expect(new Set(res.group).size).toBe(1);
    for (const [n, nb] of [[9, 10], [12, 13]]) {
      const a = files.indexOf(`Part_${n}.stl`), b = files.indexOf(`Part_${nb}.stl`);
      const relGood = good.matrices[b].clone().invert().multiply(good.matrices[a]);
      const relNow = res.matrices[b].clone().invert().multiply(res.matrices[a]);
      relGood.invert().multiply(relNow).decompose(t, q, s);
      expect(t.length()).toBeLessThan(0.5);
    }
  }, 240_000);

  it('places only unlocked pieces onto locked ones, never moving the locked ones', async () => {
    const { autoAssemble } = await import('../src/assemble');
    const files = readdirSync(BATMAN).filter((f) => /\.stl$/i.test(f)).sort((a, b) => parseInt(a.match(/\d+/)![0]) - parseInt(b.match(/\d+/)![0]));
    const geoms = files.map((f) => load(`${BATMAN}/${f}`));
    const good = await autoAssemble(geoms);
    const loose = [files.indexOf('Part_9.stl'), files.indexOf('Part_12.stl')];
    const seed = good.matrices.map((m, i) => (loose.includes(i) ? m.clone().premultiply(new THREE.Matrix4().makeRotationY(1.4)).premultiply(new THREE.Matrix4().makeTranslation(250, 0, 0)) : m.clone()));
    const fixed = files.map((_, i) => !loose.includes(i));
    const res = await autoAssemble(geoms, undefined, undefined, seed, fixed);
    const t = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    files.forEach((_, i) => {
      // Locked: exactly where they were. Unlocked: back in their right place.
      const diff = (fixed[i] ? seed[i] : good.matrices[i]).clone().invert().multiply(res.matrices[i]);
      diff.decompose(t, q, s);
      expect(t.length()).toBeLessThan(fixed[i] ? 1e-6 : 0.5);
    });
  }, 240_000);
});
