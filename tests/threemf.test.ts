import { describe, it, expect } from 'vitest';
import { zipSync, strToU8 } from 'three/examples/jsm/libs/fflate.module.js';
import { existsSync, readFileSync } from 'node:fs';
import { parse3MF } from '../src/threeMF';

const cube = (id: number) => `<object id="${id}" type="model"><mesh><vertices>
<vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/><vertex x="0" y="0" z="1"/>
</vertices><triangles><triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="1" v3="3"/><triangle v1="0" v2="3" v3="2"/><triangle v1="1" v2="2" v3="3"/></triangles></mesh></object>`;

describe('streaming 3MF reader', () => {
  it('splits a slicer assembly into named parts, applies transforms and units', () => {
    const objects = `<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>${cube(1)}${cube(2)}</resources></model>`;
    const root = `<?xml version="1.0"?><model unit="centimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"><resources>
<object id="5" type="model"><components>
<component p:path="/3D/Objects/object_1.model" objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/>
<component p:path="/3D/Objects/object_1.model" objectid="2" transform="2 0 0 0 2 0 0 0 2 0 0 0"/>
</components></object>
${cube(6)}
</resources><build><item objectid="5" transform="1 0 0 0 1 0 0 0 1 0 0 5"/><item objectid="6"/></build></model>`;
    const config = `<config><object id="5"><metadata key="name" value="Helmet"/><part id="1" subtype="normal_part"><metadata key="name" value="Left half"/></part><part id="2" subtype="normal_part"><metadata key="name" value="Right half"/></part></object><object id="6"><metadata key="name" value="Visor"/></object></config>`;
    const zip = zipSync({ '3D/3dmodel.model': strToU8(root), '3D/Objects/object_1.model': strToU8(objects), 'Metadata/model_settings.config': strToU8(config) });
    const parts = parse3MF(zip.buffer.slice(zip.byteOffset, zip.byteOffset + zip.byteLength));
    expect(parts.map((p) => p.name)).toEqual(['Left half', 'Right half', 'Visor']);
    // Left half: translated by 10 then by the item's 5 in z, in cm -> x 100..110 mm, z 50..60 mm.
    const xs = (i: number, axis: number) => { const v = parts[i].positions.filter((_, k) => k % 3 === axis); return [Math.min(...v), Math.max(...v)]; };
    expect(xs(0, 0)).toEqual([100, 110]);
    expect(xs(0, 2)).toEqual([50, 60]);
    expect(xs(1, 0)).toEqual([0, 20]); // scaled 2x, in cm
    expect(parts[0].index.length).toBe(12);
  });
});

describe.skipIf(!existsSync('samples/Batman-Sliced.3mf'))('real Bambu 3MF', () => {
  it('extracts all 7 parts of Batman-Sliced with their names', () => {
    const b = readFileSync('samples/Batman-Sliced.3mf');
    const parts = parse3MF(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    expect(parts.map((p) => p.name).sort()).toEqual(['Object_1', 'Object_2', 'Object_3_1', 'Object_3_2', 'Object_4_1', 'Object_4_2', 'Object_5']);
  }, 60_000);
});

describe.skipIf(!existsSync('samples/Batman-Sliced.3mf'))('magnet pockets', () => {
  it('finds the 5 mm magnet pockets and assembles mask + cowl + jaw by them', async () => {
    const THREE = await import('three');
    const { findMagnetHoles } = await import('../src/magnets');
    const { autoAssemble } = await import('../src/assemble');
    const b = readFileSync('samples/Batman-Sliced.3mf');
    const parts = parse3MF(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    const geoms = parts.map((p) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p.positions, 3)); g.setIndex(new THREE.BufferAttribute(p.index, 1)); return g.toNonIndexed(); });
    const idx = (n: string) => parts.findIndex((p) => p.name === n);
    const counts = ['Object_1', 'Object_2', 'Object_5'].map((n) => findMagnetHoles(geoms[idx(n)], idx(n)).length);
    expect(counts).toEqual([12, 28, 40]);
    const res = await autoAssemble(geoms);
    const g = (n: string) => res.group[idx(n)];
    expect(g('Object_1')).toBe(g('Object_5'));
    expect(g('Object_2')).toBe(g('Object_5'));
    expect(res.joins.filter((j) => j.magnets).reduce((k, j) => k + j.magnets, 0)).toBe(40);
  }, 180_000);
});

const IRON3MF = 'samples/Ironpool/100_+Scale(2).3mf';
describe.skipIf(!existsSync(IRON3MF))('rebuild from slicer project data', () => {
  it('puts all 9 Ironpool parts back where they came from (left/right mirror images)', async () => {
    const THREE = await import('three');
    const b = readFileSync(IRON3MF);
    const parts = parse3MF(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    expect(parts.length).toBe(9);
    expect(parts.every((p) => p.sourceMatrix)).toBe(true);
    const box = (name: string) => {
      const p = parts.find((q) => q.name === name)!;
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p.positions, 3));
      g.applyMatrix4(new THREE.Matrix4().fromArray(p.sourceMatrix!)); g.computeBoundingBox();
      return g.boundingBox!;
    };
    for (const [l, r] of [['Eye-L.stl', 'Eye-R.stl'], ['Side-L.stl', 'Side-R.stl'], ['Lens-L.stl', 'Lens-R.stl']]) {
      const a = box(l), c = box(r);
      // Mirror images across x = 0.
      expect(Math.abs(a.min.x + c.max.x)).toBeLessThan(0.5);
      expect(Math.abs(a.min.y - c.min.y)).toBeLessThan(0.5);
      expect(Math.abs(a.max.z - c.max.z)).toBeLessThan(0.5);
    }
  }, 120_000);
});
