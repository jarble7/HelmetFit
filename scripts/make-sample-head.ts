// samples/demo-head-scan.stl: the sample head exported Z-up, facing +X, in cm —
// a stand-in for a real scan, to test import / auto-orient.
import * as THREE from 'three';
import { writeFileSync } from 'node:fs';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import { buildParametricHead, scaleFromCircumference } from '../src/headModel';
const g = buildParametricHead(scaleFromCircumference(585)).geometry;
g.applyQuaternion(new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, Math.PI / 2)));
g.scale(0.1, 0.1, 0.1);
const dv = new STLExporter().parse(new THREE.Mesh(g), { binary: true }) as DataView;
writeFileSync('samples/demo-head-scan.stl', new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength));
