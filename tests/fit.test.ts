import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { buildParametricHead, DEFAULT_MEASUREMENTS, scaleFromCircumference } from '../src/headModel';
import { measureHead, guessHeadOrientation, centreHead } from '../src/measure';
import { HeadField, HelmetField, autoFit, mergePieces } from '../src/fit';
import { DEFAULT_FIT_SETTINGS } from '../src/types';

/**
 * A helmet shell whose inner wall sits `gap` mm outside the head, kept only
 * where `keep(vertex)` is true. Inner and outer walls (3mm thick).
 */
function shellFromHead(head: THREE.BufferGeometry, gap: number, keep: (p: THREE.Vector3) => boolean): THREE.BufferGeometry {
  const pos = head.getAttribute('position');
  const nrm = head.getAttribute('normal');
  const idx = head.getIndex()!;
  const out: number[] = [];
  const a = new THREE.Vector3(), n = new THREE.Vector3();
  const off = (i: number, d: number) => { a.fromBufferAttribute(pos, i); n.fromBufferAttribute(nrm, i); return a.clone().addScaledVector(n, d); };
  for (let t = 0; t < idx.count; t += 3) {
    const ids = [idx.getX(t), idx.getX(t + 1), idx.getX(t + 2)];
    if (!ids.every((i) => keep(a.fromBufferAttribute(pos, i).clone()))) continue;
    for (const d of [gap, gap + 3]) for (const i of ids) { const p = off(i, d); out.push(p.x, p.y, p.z); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(out, 3));
  return g;
}

function setup(keep: (p: THREE.Vector3) => boolean, fileTransform: THREE.Matrix4, gap = 10) {
  const head = buildParametricHead(DEFAULT_MEASUREMENTS).geometry;
  const shell = shellFromHead(head, gap, keep).applyMatrix4(fileTransform);
  const headField = new HeadField(head.clone());
  // Two pieces: split the shell in half by triangle order to simulate a multi-piece helmet.
  const p = shell.getAttribute('position').array as Float32Array;
  const half = Math.floor(p.length / 18) * 9;
  const g1 = new THREE.BufferGeometry(); g1.setAttribute('position', new THREE.BufferAttribute(p.slice(0, half), 3));
  const g2 = new THREE.BufferGeometry(); g2.setAttribute('position', new THREE.BufferAttribute(p.slice(half), 3));
  const merged = mergePieces([{ geometry: g1, matrix: new THREE.Matrix4() }, { geometry: g2, matrix: new THREE.Matrix4() }]);
  head.computeBoundingBox();
  return { headField, helmet: new HelmetField(merged), headBox: head.boundingBox!.clone() };
}

const identityPose = () => ({ position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), scale: new THREE.Vector3(1, 1, 1) });

describe('parametric head', () => {
  it('matches the requested circumference, length and breadth', () => {
    for (const m of [DEFAULT_MEASUREMENTS, scaleFromCircumference(540), scaleFromCircumference(610)]) {
      const r = buildParametricHead(m);
      expect(r.warning).toBeNull();
      expect(Math.abs(r.modelCircumference - m.circumference)).toBeLessThan(2);
      const meas = measureHead(r.geometry);
      expect(Math.abs(meas.circumference - m.circumference) / m.circumference).toBeLessThan(0.02);
      expect(Math.abs(meas.length - m.length)).toBeLessThan(m.length * 0.06);
      expect(Math.abs(meas.breadth - m.breadth)).toBeLessThan(m.breadth * 0.06);
      r.geometry.computeBoundingBox();
      const bb = r.geometry.boundingBox!;
      expect(bb.max.y).toBeCloseTo(m.earToTop, 0);
    }
  });

  it('auto-orients a Z-up, X-forward scan back to Y-up, Z-forward', () => {
    const head = buildParametricHead(DEFAULT_MEASUREMENTS).geometry;
    const scramble = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, Math.PI / 2));
    const scan = head.clone().applyQuaternion(scramble);
    const fix = guessHeadOrientation(scan);
    const total = fix.clone().multiply(scramble);
    expect(total.angleTo(new THREE.Quaternion())).toBeLessThan(0.01);
    const centred = centreHead(scan.clone().applyQuaternion(fix));
    expect(centred.boundingBox!.max.y).toBeCloseTo(130, 3);
  });
});

describe('auto fit', () => {
  it('recovers scale, rotation and position of a full helmet (Z-up file, 120% oversize, offset)', async () => {
    const fileT = new THREE.Matrix4().compose(
      new THREE.Vector3(40, -25, 300),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0)),
      new THREE.Vector3(1.2, 1.2, 1.2),
    );
    // Helmet covers everything above the jaw except a face opening.
    const keep = (p: THREE.Vector3) => p.y > -60 && !(p.z > 40 && p.y < 20 && p.y > -55 && Math.abs(p.x) < 55);
    const { headField, helmet, headBox } = setup(keep, fileT);
    const res = await autoFit({ head: headField, helmet, settings: DEFAULT_FIT_SETTINGS, current: identityPose(), headBox });
    // X/Y/Z scaling is on by default, so allow ~2% per axis.
    for (const k of ['x', 'y', 'z'] as const) expect(Math.abs(res.pose.scale[k] * 1.2 - 1)).toBeLessThan(0.02);
    // Combined transform should be ~identity: pose · fileT ≈ I.
    const combined = new THREE.Matrix4().compose(res.pose.position, res.pose.quaternion, res.pose.scale).multiply(fileT);
    const pos = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    combined.decompose(pos, q, s);
    expect(pos.length()).toBeLessThan(3);
    expect(THREE.MathUtils.radToDeg(q.angleTo(new THREE.Quaternion()))).toBeLessThan(3);
    expect(res.stats.penetration).toBe(0);
    expect(res.stats.meanGap).toBeGreaterThan(7);
    expect(res.stats.meanGap).toBeLessThan(14);
  }, 120_000);

  it('places a face mask in front of the face', async () => {
    const fileT = new THREE.Matrix4().compose(
      new THREE.Vector3(0, 0, -120),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI, 0)),
      new THREE.Vector3(0.9, 0.9, 0.9),
    );
    const keep = (p: THREE.Vector3) => p.z > 25 && p.y > -95 && p.y < 60;
    const { headField, helmet, headBox } = setup(keep, fileT, 12);
    const res = await autoFit({ head: headField, helmet, settings: { ...DEFAULT_FIT_SETTINGS, targetGap: 12 }, current: identityPose(), headBox });
    const combined = new THREE.Matrix4().compose(res.pose.position, res.pose.quaternion, res.pose.scale).multiply(fileT);
    const pos = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    combined.decompose(pos, q, s);
    expect(s.x).toBeCloseTo(1, 1);
    expect(pos.length()).toBeLessThan(6);
    expect(THREE.MathUtils.radToDeg(q.angleTo(new THREE.Quaternion()))).toBeLessThan(5);
    expect(res.stats.penetration).toBe(0);
  }, 120_000);
});

// ---------------------------------------------------------------------------
// Eye alignment

import { sampleHeadEyeXY } from '../src/headModel';
import { detectHeadEyes, surfacePointFront } from '../src/landmarks';
import { eyeAlignment } from '../src/fit';

function sampleHeadField() {
  const head = buildParametricHead(DEFAULT_MEASUREMENTS).geometry;
  const f = new HeadField(head, null);
  const e = sampleHeadEyeXY(DEFAULT_MEASUREMENTS);
  f.eyes = [surfacePointFront(f.bvh, -e.x, e.y)!, surfacePointFront(f.bvh, e.x, e.y)!];
  return { head, f };
}

describe('eyes', () => {
  it('detects the eye sockets on the sample head', () => {
    const { f } = sampleHeadField();
    const found = detectHeadEyes(f.bvh, DEFAULT_MEASUREMENTS.length)!;
    expect(found).not.toBeNull();
    for (let i = 0; i < 2; i++) {
      expect(Math.abs(found[i].x - f.eyes![i].x)).toBeLessThan(3);
      expect(Math.abs(found[i].y - f.eyes![i].y)).toBeLessThan(4);
    }
  });

  const e = sampleHeadEyeXY(DEFAULT_MEASUREMENTS);
  const eyeHole = (p: THREE.Vector3, r: number) => Math.hypot(Math.abs(p.x) - e.x, p.y - e.y) < r;

  async function fitMask(keep: (p: THREE.Vector3) => boolean, fileT: THREE.Matrix4, settings = DEFAULT_FIT_SETTINGS) {
    const { head, f } = sampleHeadField();
    const shell = shellFromHead(head, settings.targetGap, keep).applyMatrix4(fileT);
    const helmet = new HelmetField(mergePieces([{ geometry: shell, matrix: new THREE.Matrix4() }]));
    head.computeBoundingBox();
    const res = await autoFit({ head: f, helmet, settings, current: identityPose(), headBox: head.boundingBox!.clone() });
    const combined = new THREE.Matrix4().compose(res.pose.position, res.pose.quaternion, res.pose.scale).multiply(fileT);
    const pos = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    combined.decompose(pos, q, s);
    return { res, pos, q, s, align: eyeAlignment(f, helmet, new THREE.Matrix4().compose(res.pose.position, res.pose.quaternion, res.pose.scale)) };
  }

  it('lines up a mask with two eye holes (rotated, shrunk, offset file)', async () => {
    const fileT = new THREE.Matrix4().compose(new THREE.Vector3(30, 80, -200), new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, Math.PI, 0)), new THREE.Vector3(0.85, 0.85, 0.85));
    const keep = (p: THREE.Vector3) => p.z > 20 && p.y > -95 && p.y < 70 && !eyeHole(p, 16);
    const { res, pos, q, s, align } = await fitMask(keep, fileT);
    expect(align?.kind).toBe('holes');
    expect(Math.abs(align!.dx)).toBeLessThan(2);
    expect(Math.abs(align!.dy)).toBeLessThan(2);
    expect(Math.abs(s.x - 1)).toBeLessThan(0.05);
    expect(Math.abs(s.y - 1)).toBeLessThan(0.05);
    expect(pos.length()).toBeLessThan(6);
    expect(THREE.MathUtils.radToDeg(q.angleTo(new THREE.Quaternion()))).toBeLessThan(4);
    expect(res.stats.penetration).toBe(0);
  }, 180_000);

  it('lines up a full helmet with a visor slot (Z-up, 110%)', async () => {
    const fileT = new THREE.Matrix4().compose(new THREE.Vector3(0, 0, 50), new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0)), new THREE.Vector3(1.1, 1.1, 1.1));
    const visor = (p: THREE.Vector3) => p.z > 30 && Math.abs(p.y - e.y) < 11 && Math.abs(p.x) < 58;
    const keep = (p: THREE.Vector3) => p.y > -75 && !visor(p) && !(p.z > 40 && p.y < -45 && Math.abs(p.x) < 40);
    const { res, pos, q, s, align } = await fitMask(keep, fileT);
    expect(align?.kind).toBe('visor');
    expect(Math.abs(align!.dx)).toBeLessThan(2);
    expect(Math.abs(align!.dy)).toBeLessThan(3);
    expect(Math.abs(s.y - 1)).toBeLessThan(0.05);
    expect(pos.length()).toBeLessThan(6);
    expect(THREE.MathUtils.radToDeg(q.angleTo(new THREE.Quaternion()))).toBeLessThan(4);
    expect(res.stats.penetration).toBe(0);
  }, 180_000);
});
