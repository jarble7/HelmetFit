import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { rigidFit, autoAssemble } from '../src/assemble';
import { findMagnetHoles, matchMagnets } from '../src/magnets';

const LUIGI_HAT = 'E:/3D Files/Masks/Luigi Head/STLs/hat_big.stl';
const LUIGI_HAIR = 'E:/3D Files/Masks/Luigi Head/STLs/hair_3.stl';

describe('rigidFit (snap by points)', () => {
  it('recovers random rotations + translations from 3–6 point pairs', () => {
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
    for (let trial = 0; trial < 50; trial++) {
      const q = new THREE.Quaternion(rnd(), rnd(), rnd(), rnd()).normalize();
      const t = new THREE.Vector3(rnd() * 300, rnd() * 300, rnd() * 300);
      const truth = new THREE.Matrix4().compose(t, q, new THREE.Vector3(1, 1, 1));
      const n = 3 + (trial % 4);
      // Spread-out clicks (as the UI asks for): reject nearly collinear sets.
      let src: THREE.Vector3[];
      do src = Array.from({ length: n }, () => new THREE.Vector3(rnd() * 100, rnd() * 100, rnd() * 100));
      while (src[1].clone().sub(src[0]).cross(src[2].clone().sub(src[0])).length() / 2 < 2000);
      // Hand clicks are a little off: add up to ±0.5 mm noise.
      const dst = src.map((p) => p.clone().applyMatrix4(truth).add(new THREE.Vector3(rnd(), rnd(), rnd()).multiplyScalar(0.5)));
      const { matrix, rms } = rigidFit(src, dst);
      const p = new THREE.Vector3(), r = new THREE.Quaternion(), s = new THREE.Vector3();
      matrix.decompose(p, r, s);
      expect(THREE.MathUtils.radToDeg(r.angleTo(q))).toBeLessThan(1.5);
      expect(rms).toBeLessThan(1);
    }
  });
  it('one pair translates, two pairs align the segment', () => {
    const a = [new THREE.Vector3(0, 0, 0)], b = [new THREE.Vector3(10, 5, 2)];
    expect(a[0].clone().applyMatrix4(rigidFit(a, b).matrix).distanceTo(b[0])).toBeLessThan(1e-9);
    const s2 = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(10, 0, 0)], d2 = [new THREE.Vector3(5, 5, 5), new THREE.Vector3(5, 15, 5)];
    const m = rigidFit(s2, d2).matrix;
    s2.forEach((p, i) => expect(p.clone().applyMatrix4(m).distanceTo(d2[i])).toBeLessThan(1e-6));
  });
});

describe.skipIf(!existsSync(LUIGI_HAT) || !existsSync(LUIGI_HAIR))('pegs and sockets (Luigi hat_big + hair_3)', () => {
  it('detects the 4 pegs on hair_3 and 4 matching sockets on hat_big and snaps them together', () => {
    const parse = (p: string) => { const b = readFileSync(p); return new STLLoader().parse(b.buffer.slice(b.byteOffset, b.byteLength)); };
    const hat = parse(LUIGI_HAT), hair = parse(LUIGI_HAIR);
    const hatHoles = findMagnetHoles(hat, 0), hairHoles = findMagnetHoles(hair, 1);
    expect(hatHoles.filter((h) => h.kind === 'socket').length).toBeGreaterThanOrEqual(4);
    expect(hairHoles.filter((h) => h.kind === 'peg').length).toBe(4);
    const matches = matchMagnets(hatHoles, hairHoles);
    expect(matches.length).toBeGreaterThan(0);
    expect(matches[0].pairs.length).toBe(4);
    expect(matches[0].rms).toBeLessThan(0.1);
    const t = new THREE.Vector3().setFromMatrixPosition(matches[0].bToA);
    expect(Math.abs(t.x)).toBeLessThan(0.3);
    expect(Math.abs(t.y)).toBeLessThan(0.3);
    expect(Math.abs(t.z - 36.6)).toBeLessThan(0.3);
  });

  it('autoAssemble preserves relative transforms within userGroups while assembling groups together', async () => {
    const parse = (p: string) => { const b = readFileSync(p); return new STLLoader().parse(b.buffer.slice(b.byteOffset, b.byteLength)); };
    const hat = parse(LUIGI_HAT), hair = parse(LUIGI_HAIR);
    // Create a companion piece (e.g. emblem/visor) grouped with hat_big at a known offset in seed.
    const badge = new THREE.BoxGeometry(20, 20, 5);
    const seed = [
      new THREE.Matrix4().identity(),
      new THREE.Matrix4().makeTranslation(12, -34, 56), // companion piece in Group 0 with hat
      new THREE.Matrix4().makeTranslation(0, 0, -50),   // hair_3 in Group 2, displaced in Z
    ];
    const userGroups = [0, 0, 2]; // hat + badge in group 0; hair in group 2
    const res = await autoAssemble([hat, badge, hair], undefined, undefined, seed, undefined, userGroups);
    expect(res.joins.length).toBe(1);
    // Relative transform between piece 0 (hat) and piece 1 (badge) must be unchanged!
    const relBefore = seed[0].clone().invert().multiply(seed[1]);
    const relAfter = res.matrices[0].clone().invert().multiply(res.matrices[1]);
    for (let i = 0; i < 16; i++) {
      expect(relAfter.elements[i]).toBeCloseTo(relBefore.elements[i], 5);
    }
  });

  it('locked groups do not move or resize, while unlocked groups stay together and resize as a whole to match', async () => {
    const parse = (p: string) => { const b = readFileSync(p); return new STLLoader().parse(b.buffer.slice(b.byteOffset, b.byteLength)); };
    const hat = parse(LUIGI_HAT), hair = parse(LUIGI_HAIR);
    const badge = new THREE.BoxGeometry(20, 20, 5);
    // Group 0 (hat + badge) is unlocked at 1.0x; Group 1 (hair) is locked at 1.2x scale and offset in space.
    const lockedMat = new THREE.Matrix4().compose(
      new THREE.Vector3(10, 20, -40),
      new THREE.Quaternion(),
      new THREE.Vector3(1.2, 1.2, 1.2),
    );
    const seed = [
      new THREE.Matrix4().identity(),
      new THREE.Matrix4().makeTranslation(15, 25, 35),
      lockedMat.clone(),
    ];
    const fixed = [false, false, true];
    const userGroups = [0, 0, 1];
    const res = await autoAssemble([hat, badge, hair], undefined, undefined, seed, fixed, userGroups);
    expect(res.joins.length).toBe(1);
    // 1. Locked group (piece 2) must not move or resize at all.
    for (let i = 0; i < 16; i++) {
      expect(res.matrices[2].elements[i]).toBeCloseTo(lockedMat.elements[i], 5);
    }
    // 2. Unlocked group (pieces 0 and 1) must resize as a whole to 1.2x while preserving its internal relative transform.
    expect(res.matrices[0].getMaxScaleOnAxis()).toBeCloseTo(1.2, 4);
    expect(res.matrices[1].getMaxScaleOnAxis()).toBeCloseTo(1.2, 4);
    const relBefore = seed[0].clone().invert().multiply(seed[1]);
    const relAfter = res.matrices[0].clone().invert().multiply(res.matrices[1]);
    for (let i = 0; i < 16; i++) {
      expect(relAfter.elements[i]).toBeCloseTo(relBefore.elements[i], 5);
    }
  });
});
