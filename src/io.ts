import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { ThreeMFLoader } from 'three/examples/jsm/loaders/3MFLoader.js';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import { zipSync } from 'three/examples/jsm/libs/fflate.module.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { LengthUnit } from './types';
import { clusterSimplify, DISPLAY_TRIANGLE_BUDGET } from './simplify';
import type { WorkerReply } from './meshWorker';

export const ACCEPTED_EXTENSIONS = '.stl,.obj,.3mf';

/** Parse a mesh file into one position-only geometry in the file's own units. */
export function parseMeshFile(fileName: string, data: ArrayBuffer): THREE.BufferGeometry {
  const ext = fileName.toLowerCase().split('.').pop();
  let geom: THREE.BufferGeometry;
  if (ext === 'stl') {
    geom = new STLLoader().parse(data);
  } else if (ext === 'obj') {
    geom = mergeObject(new OBJLoader().parse(new TextDecoder().decode(data)));
  } else if (ext === '3mf') {
    // Small files only; big ones go through loadMeshParts (streaming, in a worker).
    geom = mergeObject(new ThreeMFLoader().parse(data));
  } else {
    throw new Error(`Unsupported file type ".${ext}". Use STL, OBJ or 3MF.`);
  }
  const clean = new THREE.BufferGeometry();
  clean.setAttribute('position', geom.getAttribute('position'));
  if (geom.getIndex()) clean.setIndex(geom.getIndex());
  if (clean.getAttribute('position') == null || clean.getAttribute('position').count < 3) {
    throw new Error(`"${fileName}" has no triangles.`);
  }
  clean.computeVertexNormals();
  clean.computeBoundingBox();
  clean.computeBoundingSphere();
  return clean;
}

function mergeObject(root: THREE.Object3D): THREE.BufferGeometry {
  root.updateMatrixWorld(true);
  const parts: THREE.BufferGeometry[] = [];
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    let g = mesh.geometry.clone();
    if (g.getIndex()) g = g.toNonIndexed();
    const pos = g.getAttribute('position');
    const only = new THREE.BufferGeometry();
    only.setAttribute('position', pos);
    only.applyMatrix4(mesh.matrixWorld);
    parts.push(only);
  });
  if (!parts.length) throw new Error('File contains no meshes.');
  return parts.length === 1 ? parts[0] : mergeGeometries(parts)!;
}

/**
 * Guess what unit a model was exported in from its overall size.
 * `expectedMm` is roughly how big the thing should be (head ≈ 230, helmet ≈ 300).
 */
export function guessUnit(maxDimension: number, expectedMm: number): LengthUnit {
  const candidates: LengthUnit[] = ['mm', 'cm', 'in', 'm'];
  const factor: Record<LengthUnit, number> = { mm: 1, cm: 10, in: 25.4, m: 1000 };
  let best: LengthUnit = 'mm', bestErr = Infinity;
  for (const u of candidates) {
    const err = Math.abs(Math.log((maxDimension * factor[u]) / expectedMm));
    // Prefer mm unless another unit is clearly a better match.
    const biased = u === 'mm' ? err - 0.4 : err;
    if (biased < bestErr) { bestErr = biased; best = u; }
  }
  return best;
}

export function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return file.arrayBuffer();
}

/** Binary STL bytes of a geometry after applying a matrix. */
export function exportSTLBytes(geometry: THREE.BufferGeometry, matrix: THREE.Matrix4): Uint8Array {
  const g = geometry.clone().applyMatrix4(matrix);
  const mesh = new THREE.Mesh(g);
  const result = new STLExporter().parse(mesh, { binary: true }) as DataView;
  g.dispose();
  return new Uint8Array(result.buffer as ArrayBuffer, result.byteOffset, result.byteLength);
}

/** Binary STL of a geometry after applying a matrix. */
export function exportSTL(geometry: THREE.BufferGeometry, matrix: THREE.Matrix4): Blob {
  return new Blob([exportSTLBytes(geometry, matrix) as BlobPart], { type: 'model/stl' });
}

/** Pack multiple named binary files into a single ZIP archive Blob. */
export function exportZip(files: Record<string, Uint8Array>): Blob {
  const zipped = zipSync(files, { level: 6 });
  return new Blob([zipped as BlobPart], { type: 'application/zip' });
}

export function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

export interface LoadedPart {
  name: string;
  /** What's shown and fitted (simplified if the model is huge). */
  geometry: THREE.BufferGeometry;
  /** Full-detail mesh for downloads / assembly, when `geometry` was simplified. */
  full: THREE.BufferGeometry | null;
  /** From slicer project data: moves the part back to where it sat in its original file (rebuilds the model). */
  sourceMatrix?: THREE.Matrix4;
}

/** Pictures found in the last 3MF loaded (designer photos, plate previews). */
export const lastProjectImages: { file: string; images: Array<{ name: string; url: string }> } = { file: '', images: [] };

function indexedGeometry(positions: Float32Array, index: Uint32Array): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setIndex(new THREE.BufferAttribute(index, 1));
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

/** Faceted display copy (sharp edges stay sharp). */
function displayGeometry(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const d = g.getIndex() ? g.toNonIndexed() : g;
  d.computeVertexNormals();
  d.computeBoundingBox();
  d.computeBoundingSphere();
  return d;
}

/**
 * Load a mesh file as one or more parts. 3MF files are read by a streaming
 * reader in a background worker (a 3MF can hold several parts — each becomes
 * its own). Huge models get a simplified copy for display and fitting.
 */
export async function loadMeshParts(fileName: string, data: ArrayBuffer, onProgress?: (msg: string) => void): Promise<LoadedPart[]> {
  if (fileName.toLowerCase().endsWith('.3mf')) {
    const worker = new Worker(new URL('./meshWorker.ts', import.meta.url), { type: 'module' });
    try {
      const parts = await new Promise<Extract<WorkerReply, { type: 'done' }>['parts']>((resolve, reject) => {
        worker.onmessage = (e: MessageEvent<WorkerReply>) => {
          const m = e.data;
          if (m.type === 'progress') onProgress?.(m.msg);
          else if (m.type === 'done') {
            lastProjectImages.file = fileName;
            lastProjectImages.images = m.images.map((im) => ({ name: im.name.split('/').pop()!, url: URL.createObjectURL(new Blob([im.data as BlobPart], { type: im.mime })) }));
            resolve(m.parts);
          }
          else reject(new Error(`${fileName}: ${m.message}`));
        };
        worker.onerror = (e) => reject(new Error(`${fileName}: ${e.message || 'could not be read'}`));
        worker.postMessage({ data }); // copied, so the original stays usable for saving
      });
      return parts.map((p) => {
        const full = indexedGeometry(p.positions, p.index);
        const sourceMatrix = p.sourceMatrix ? new THREE.Matrix4().fromArray(p.sourceMatrix) : undefined;
        return p.display
          ? { name: p.name, geometry: displayGeometry(indexedGeometry(p.display.positions, p.display.index)), full, sourceMatrix }
          : { name: p.name, geometry: displayGeometry(full), full: null, sourceMatrix };
      });
    } finally {
      worker.terminate();
    }
  }
  const extLabel = fileName.toLowerCase().endsWith('.obj') ? 'OBJ' : 'STL';
  onProgress?.(`Loading ${extLabel} geometry…`);
  const g = parseMeshFile(fileName, data);
  const tris = (g.getIndex()?.count ?? g.getAttribute('position').count) / 3;
  if (tris <= DISPLAY_TRIANGLE_BUDGET) return [{ name: fileName.replace(/\.[^.]+$/, ''), geometry: g, full: null }];
  // Huge STL/OBJ: simplify a copy for display.
  onProgress?.(`Simplifying ${extLabel} (${Math.round(tris / 1000)}k triangles) for display…`);
  const welded = mergeIndexed(g);
  const s = clusterSimplify(welded, DISPLAY_TRIANGLE_BUDGET);
  return [{ name: fileName.replace(/\.[^.]+$/, ''), geometry: displayGeometry(indexedGeometry(s.positions, s.index)), full: g }];
}

/** Weld a triangle soup into an indexed mesh (exact duplicate vertices only). */
function mergeIndexed(g: THREE.BufferGeometry): { positions: Float32Array; index: Uint32Array } {
  const p = g.getAttribute('position');
  if (g.getIndex()) return { positions: p.array as Float32Array, index: new Uint32Array(g.getIndex()!.array) };
  const map = new Map<string, number>();
  const pos: number[] = [];
  const index = new Uint32Array(p.count);
  for (let i = 0; i < p.count; i++) {
    const k = `${p.getX(i)},${p.getY(i)},${p.getZ(i)}`;
    let v = map.get(k);
    if (v === undefined) { v = pos.length / 3; map.set(k, v); pos.push(p.getX(i), p.getY(i), p.getZ(i)); }
    index[i] = v;
  }
  return { positions: new Float32Array(pos), index };
}
