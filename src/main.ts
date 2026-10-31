import * as THREE from 'three';
import { Viewer, type GizmoMode, type SectionMode } from './viewer';
import { buildSampleHead, DEFAULT_MEASUREMENTS, HEAD_PRESETS, MEASUREMENT_INFO, sampleHeadEyeXY, scaleFromCircumference, setSampleHeadTemplate } from './headModel';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { detectHeadEyes, looksLikeLens, surfacePointFront, type EyePair } from './landmarks';
import { autoAssemble, cutFaceAt, findCutFaces, joinedGroups, rigidFit, snapCandidates, type CutFace } from './assemble';
import { findMagnetHoles, matchMagnets, measurePocket, forgetPocketData, type MagnetHole, type MagnetSize } from './magnets';
import { forgetSurface } from './surface';
import { centreHead, guessHeadOrientation, measureHead } from './measure';
import { guessUnit, exportSTL, exportSTLBytes, exportZip, downloadBlob, loadMeshParts, lastProjectImages, ACCEPTED_EXTENSIONS, type LoadedPart } from './io';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { HeadField, HelmetField, headSamples, mergePieces, evaluatePose, evaluatePiecesPose, vertexGaps, vertexGapsForPieces, autoFit, eyeAlignment, type EyeAlignment, type Sample } from './fit';
import * as store from './storage';
import {
  DEFAULT_FIT_SETTINGS, UNIT_TO_MM, identityTransform,
  type FitSettings, type FitStats, type HeadMeasurements, type HeadRecord, type LengthUnit, type PieceGroup, type TransformData, type ProjectRecord,
} from './types';

// ---------------------------------------------------------------------------
// DOM helpers

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const qsa = <T extends HTMLElement = HTMLElement>(sel: string) => Array.from(document.querySelectorAll<T>(sel));
const fmt = (v: number, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const status = (msg: string) => {
  const st = $('status');
  if (st) st.textContent = msg;
};

let hudTaskDepth = 0;
let hudStartTime = 0;
let hudHideTimer = 0;

const yieldHudFrame = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

function updateHudTask(msg: string, title?: string) {
  const st = $('status');
  if (st) st.textContent = msg;
  if (hudTaskDepth <= 0) return;
  const hud = $('assemble-hud');
  const titleEl = $('hud-title');
  const msgEl = $('hud-msg');
  if (title && titleEl) titleEl.textContent = title;
  if (msgEl) msgEl.textContent = msg;
  if (hud) {
    hud.hidden = false;
    hud.dataset.empty = 'false';
    hud.dataset.state = 'working';
  }
}

async function beginHudTask(title: string, msg: string) {
  if (hudHideTimer) {
    clearTimeout(hudHideTimer);
    hudHideTimer = 0;
  }
  if (hudTaskDepth === 0) {
    hudStartTime = performance.now();
  }
  hudTaskDepth++;
  updateHudTask(msg, title);
  await yieldHudFrame();
}

async function endHudTask() {
  hudTaskDepth = Math.max(0, hudTaskDepth - 1);
  if (hudTaskDepth > 0) return;
  const hideNow = () => {
    if (hudTaskDepth > 0) return;
    const hud = $('assemble-hud');
    const msgEl = $('hud-msg');
    if (hud) {
      hud.dataset.empty = 'true';
      hud.dataset.state = 'idle';
      hud.hidden = true;
    }
    if (msgEl) msgEl.textContent = '';
  };
  const elapsed = performance.now() - hudStartTime;
  if (elapsed < 280) {
    await new Promise<void>((r) => {
      hudHideTimer = window.setTimeout(() => {
        hudHideTimer = 0;
        hideNow();
        r();
      }, 280 - elapsed);
    });
  } else {
    hideNow();
  }
}

// ---------------------------------------------------------------------------
// State

interface Piece {
  name: string;
  fileName: string;
  filePath?: string;
  data: ArrayBuffer;
  colour: string;
  mesh: THREE.Mesh;
  /** See-through insert (lens / eye slot): eye detection looks through it. */
  lens: boolean;
  /** Full-detail mesh when the displayed one was simplified (huge models); used for downloads and assembly. */
  full: THREE.BufferGeometry | null;
  /** Which part of a multi-part file (3MF) this piece is. */
  part: number;
  /** Where the piece's centre was in its file. Geometry is stored centred so it rotates about its own middle. */
  origin: THREE.Vector3;
  /** Placed and confirmed: auto-assemble, snapping and the move handles leave it where it is. */
  locked?: boolean;
  /** ID of the PieceGroup this piece belongs to (null/undefined if ungrouped). */
  groupId?: string | null;
}

interface ImportedHead {
  fileName: string;
  filePath?: string;
  data: ArrayBuffer;
  unit: LengthUnit;
  orientation: THREE.Quaternion;
  extraScale: number;
  /** Hand-picked eyes; cleared whenever the scan is re-oriented or rescaled. */
  eyes: EyePair | null;
  /** The file's mesh (all parts merged, file units), read once. */
  base: THREE.BufferGeometry;
}

/** All parts of a file as one geometry (heads are one mesh). */
function mergedParts(parts: LoadedPart[]): THREE.BufferGeometry {
  if (parts.length === 1) return parts[0].geometry;
  const g = mergeGeometries(parts.map((p) => (p.geometry.getIndex() ? p.geometry.toNonIndexed() : p.geometry)))!;
  g.computeBoundingBox();
  return g;
}

/** Geometry to use for downloads and cut-face matching (full detail if we have it). */
const detailOf = (p: Piece) => p.full ?? p.mesh.geometry;

const PIECE_COLOURS = ['#6aa9ff', '#ffb35c', '#7bd88f', '#e98bd6', '#f2e35c', '#5fd4d4', '#ff7a7a', '#b39dff'];

const viewer = new Viewer($('canvas-host'));
let headTab: 'sample' | 'import' = 'sample';
let measurements: HeadMeasurements = { ...DEFAULT_MEASUREMENTS };
let imported: ImportedHead | null = null;
let currentHeadId: string | null = null;
let headField: HeadField | null = null;
let headSampleCache: Sample[] = [];
let helmetUnit: LengthUnit = 'mm';
const pieces: Piece[] = [];
const pieceGroups: PieceGroup[] = [];
let selected = -1;
let target: 'assembly' | number | `group:${string}` = 'assembly';
const groupPivot = new THREE.Object3D();
groupPivot.name = 'group-pivot';
viewer.assembly.add(groupPivot);
const groupPivotPrev = new THREE.Matrix4();
let gizmoMode: GizmoMode = 'translate';
let fitSettings: FitSettings = { ...DEFAULT_FIT_SETTINGS };
let helmetFieldCache: HelmetField | null = null;
let currentProjectId: string | null = null;
let fitSignal: { cancelled: boolean } | null = null;
let lastStats: FitStats | null = null;
let lastEyes: EyeAlignment | null = null;
let pickingEyes: THREE.Vector3[] | null = null;
/** Hand-picked eyes for the sample head; cleared when the measurements change. */
let sampleEyes: EyePair | null = null;

// ---------------------------------------------------------------------------
// Transforms <-> plain data

function readTransform(o: THREE.Object3D): TransformData {
  const r = o.rotation;
  return {
    position: [o.position.x, o.position.y, o.position.z],
    rotationDeg: [THREE.MathUtils.radToDeg(r.x), THREE.MathUtils.radToDeg(r.y), THREE.MathUtils.radToDeg(r.z)],
    scale: [o.scale.x, o.scale.y, o.scale.z],
  };
}

function writeTransform(o: THREE.Object3D, t: TransformData) {
  o.position.set(...t.position);
  o.rotation.set(...(t.rotationDeg.map((d) => THREE.MathUtils.degToRad(d)) as [number, number, number]));
  o.scale.set(...t.scale);
  o.updateMatrix();
  o.updateMatrixWorld(true);
}

const targetGroupId = (): string | null =>
  typeof target === 'string' && target.startsWith('group:') ? target.slice(6) : null;

function isGroupLocked(gid: string): boolean {
  const g = pieceGroups.find((grp) => grp.id === gid);
  const members = pieces.filter((p) => p.groupId === gid);
  return Boolean(g?.locked || (members.length > 0 && members.some((p) => p.locked)));
}

function isPieceLocked(p: Piece): boolean {
  return Boolean(p.locked || (p.groupId && isGroupLocked(p.groupId)));
}

function isPieceVisible(p: Piece): boolean {
  if (!p.mesh.visible) return false;
  if (!p.groupId) return true;
  const g = pieceGroups.find((grp) => grp.id === p.groupId);
  return g ? g.visible !== false : true;
}

function syncGroupPivot(gid: string): boolean {
  const members = pieces.filter((p) => p.groupId === gid);
  if (!members.length) return false;
  const box = new THREE.Box3();
  for (const p of members) {
    p.mesh.updateMatrix();
    box.union(p.mesh.geometry.boundingBox!.clone().applyMatrix4(p.mesh.matrix));
  }
  const c = box.getCenter(new THREE.Vector3());
  groupPivot.position.copy(c);
  groupPivot.rotation.set(0, 0, 0);
  const m0 = members[0].mesh;
  const qInv = m0.quaternion.clone().invert();
  const ax = new THREE.Vector3();
  groupPivot.scale.set(
    ax.set(1, 0, 0).applyQuaternion(qInv).multiply(m0.scale).length(),
    ax.set(0, 1, 0).applyQuaternion(qInv).multiply(m0.scale).length(),
    ax.set(0, 0, 1).applyQuaternion(qInv).multiply(m0.scale).length(),
  );
  groupPivot.updateMatrix();
  groupPivot.updateMatrixWorld(true);
  groupPivotPrev.copy(groupPivot.matrix);
  return true;
}

function applyGroupPivotDelta(gid: string) {
  if (isGroupLocked(gid)) return;
  groupPivot.updateMatrix();
  const p0 = new THREE.Vector3(), q0 = new THREE.Quaternion(), s0 = new THREE.Vector3();
  const p1 = new THREE.Vector3(), q1 = new THREE.Quaternion(), s1 = new THREE.Vector3();
  groupPivotPrev.decompose(p0, q0, s0);
  groupPivot.matrix.decompose(p1, q1, s1);
  const q0Inv = q0.clone().invert();
  const dq = q1.clone().multiply(q0Inv);
  const sr = new THREE.Vector3(
    s0.x ? s1.x / s0.x : 1,
    s0.y ? s1.y / s0.y : 1,
    s0.z ? s1.z / s0.z : 1,
  );
  const uniform = Math.abs(sr.x - sr.y) < 1e-6 && Math.abs(sr.y - sr.z) < 1e-6;
  const ax = new THREE.Vector3();
  for (const p of pieces) {
    if (p.groupId !== gid) continue;
    const offset = p.mesh.position.clone().sub(p0).applyQuaternion(q0Inv).multiply(sr).applyQuaternion(q1);
    p.mesh.position.copy(p1).add(offset);
    if (uniform) {
      p.mesh.scale.multiplyScalar(sr.x);
    } else {
      const qRel = q0Inv.clone().multiply(p.mesh.quaternion);
      const sx = ax.set(1, 0, 0).applyQuaternion(qRel).multiply(sr).length();
      const sy = ax.set(0, 1, 0).applyQuaternion(qRel).multiply(sr).length();
      const sz = ax.set(0, 0, 1).applyQuaternion(qRel).multiply(sr).length();
      p.mesh.scale.set(p.mesh.scale.x * sx, p.mesh.scale.y * sy, p.mesh.scale.z * sz);
    }
    p.mesh.quaternion.premultiply(dq);
    p.mesh.updateMatrix();
    p.mesh.updateMatrixWorld(true);
  }
  groupPivotPrev.copy(groupPivot.matrix);
}

const targetObject = (): THREE.Object3D | null => {
  if (target === 'assembly') return pieces.length ? viewer.assembly : null;
  const gid = targetGroupId();
  if (gid) return pieces.some((p) => p.groupId === gid) ? groupPivot : null;
  return typeof target === 'number' ? pieces[target]?.mesh ?? null : null;
};

// ---------------------------------------------------------------------------
// Undo / redo

interface Snapshot { assembly: TransformData; pieces: TransformData[]; groups: PieceGroup[]; pieceGroups: Array<string | null> }
const undoStack: Snapshot[] = [];
const redoStack: Snapshot[] = [];
const snapshot = (): Snapshot => ({
  assembly: readTransform(viewer.assembly),
  pieces: pieces.map((p) => readTransform(p.mesh)),
  groups: pieceGroups.map((g) => ({ ...g })),
  pieceGroups: pieces.map((p) => p.groupId ?? null),
});

function restore(s: Snapshot) {
  writeTransform(viewer.assembly, s.assembly);
  s.pieces.forEach((t, i) => pieces[i] && writeTransform(pieces[i].mesh, t));
  pieceGroups.length = 0;
  pieceGroups.push(...s.groups.map((g) => ({ ...g })));
  s.pieceGroups.forEach((gid, i) => { if (pieces[i]) pieces[i].groupId = gid; });
  const gid = targetGroupId();
  if (gid && !pieceGroups.some((g) => g.id === gid)) target = 'assembly';
  else if (gid) syncGroupPivot(gid);
  helmetFieldCache = null;
  renderPieceList();
  afterTransformChange(true);
}

/** Call BEFORE changing transforms. */
function pushHistory() {
  undoStack.push(snapshot());
  if (undoStack.length > 100) undoStack.shift();
  redoStack.length = 0;
  updateUndoButtons();
}

function undo() { if (!undoStack.length) return; redoStack.push(snapshot()); restore(undoStack.pop()!); updateUndoButtons(); }
function redo() { if (!redoStack.length) return; undoStack.push(snapshot()); restore(redoStack.pop()!); updateUndoButtons(); }
function updateUndoButtons() {
  ($('undo') as HTMLButtonElement).disabled = !undoStack.length;
  ($('redo') as HTMLButtonElement).disabled = !redoStack.length;
}

// ---------------------------------------------------------------------------
// Head

function setHeadGeometry(g: THREE.BufferGeometry) {
  viewer.setHead(g);
  headField = new HeadField(g, null);
  headField.eyes = headEyesFor(headField);
  viewer.setVirtualNeck(headField.virtualNeck);
  updateEyeStatus();
  headSampleCache = headSamples(headField, 900);
  showMeasured(g);
  afterTransformChange(false);
}

/** Sample head: detect eyes on the Head-3 mesh (or fall back to analytic positions). Scan: hand-picked if set, else detected. */
function headEyesFor(f: HeadField): EyePair | null {
  if (headTab === 'sample') {
    if (sampleEyes) return sampleEyes;
    const detected = detectHeadEyes(f.bvh, measurements.length);
    if (detected) return detected;
    const e = sampleHeadEyeXY(measurements);
    const l = surfacePointFront(f.bvh, -e.x, e.y), r = surfacePointFront(f.bvh, e.x, e.y);
    return l && r ? [l, r] : null;
  }
  return imported?.eyes ?? detectHeadEyes(f.bvh, measureHead(f.geometry).length);
}

function updateEyeStatus() {
  const el = $('eye-status');
  if (!headField) { el.textContent = ''; return; }
  if (pickingEyes) { el.textContent = `Click the ${pickingEyes.length === 0 ? 'first' : 'second'} eye on the head…`; return; }
  const manual = headTab === 'sample' ? sampleEyes : imported?.eyes;
  const src = manual ? 'set by hand' : headTab === 'sample' ? 'from measurements' : 'auto-detected';
  el.textContent = headField.eyes ? `Eyes: ${src} (cyan dots).` : 'Eyes not found — use "Set eyes by clicking".';
}

function startEyePick() {
  if (!headField) return;
  pickingEyes = [];
  updateEyeStatus();
  viewer.setView('front');
  viewer.onHeadPick = (p) => {
    pickingEyes!.push(p);
    if (pickingEyes!.length < 2) { updateEyeStatus(); return; }
    const [a, b] = pickingEyes!.sort((u, v) => u.x - v.x);
    pickingEyes = null;
    viewer.onHeadPick = null;
    if (headTab === 'sample') sampleEyes = [a, b];
    else if (imported) imported.eyes = [a, b];
    headField!.eyes = [a, b];
    updateEyeStatus();
    afterTransformChange(true);
    status('Eyes set. Save the head to keep them.');
  };
}

function rebuildSampleHead() {
  const res = buildSampleHead(measurements);
  const warn = $('head-warning');
  if (warn) {
    warn.hidden = !res.warning;
    warn.textContent = res.warning ?? '';
  }
  const circ = $('model-circ');
  if (circ) circ.textContent = `Model circumference: ${fmtLen(res.modelCircumference, measUnit())}`;
  setHeadGeometry(res.geometry);
}

/** Rebuild the scan head. Any change to orientation/units/scale invalidates hand-picked eyes. */
function rebuildImportedHead(autoOrient = false, keepEyes = true) {
  if (!imported) return;
  if (!keepEyes || autoOrient) imported.eyes = null;
  const g = imported.base.clone();
  const k = UNIT_TO_MM[imported.unit];
  g.scale(k, k, k);
  if (autoOrient) imported.orientation = guessHeadOrientation(g);
  g.applyQuaternion(imported.orientation);
  if (imported.extraScale !== 1) g.scale(imported.extraScale, imported.extraScale, imported.extraScale);
  centreHead(g);
  g.computeVertexNormals();
  $('head-file-name').textContent = imported.fileName;
  setHeadGeometry(g);
  viewer.frameAll();
}

function showMeasured(g: THREE.BufferGeometry) {
  const m = measureHead(g);
  const u = measUnit();
  $('head-measured').innerHTML = `<table>
    <tr><td>Tape circumference</td><td>${fmtLen(m.circumference, u)}</td></tr>
    <tr><td>Length (front–back)</td><td>${fmtLen(m.length, u)}</td></tr>
    <tr><td>Width (side–side)</td><td>${fmtLen(m.breadth, u)}</td></tr>
    <tr><td>Model height</td><td>${fmtLen(m.meshHeight, u)}</td></tr>
  </table><div class="hint">Measured on the 3D model${headTab === 'sample' ? ' (width includes ears)' : ''}.${m.meshHeight < 250 ? ' This model stops at the chin, so a faint stand-in neck is used when fitting.' : ''}</div>`;
}

const measUnit = () => ($('meas-unit') as HTMLSelectElement).value as LengthUnit;
const fmtLen = (mm: number, u: LengthUnit) => `${fmt(mm / UNIT_TO_MM[u], u === 'mm' ? 0 : u === 'cm' ? 1 : 2)} ${u}`;

function renderMeasurementFields() {
  const host = $('meas-fields');
  const u = measUnit();
  host.innerHTML = '';
  for (const key of Object.keys(MEASUREMENT_INFO) as Array<keyof HeadMeasurements>) {
    const info = MEASUREMENT_INFO[key];
    const label = document.createElement('label');
    const k = UNIT_TO_MM[u];
    const dec = u === 'mm' ? 0 : u === 'cm' ? 1 : 2;
    label.innerHTML = `<span class="field-label">${info.label} (${u}) <span class="help" title="${info.help.replace(/"/g, '&quot;')}">?</span></span>`;
    const input = document.createElement('input');
    input.type = 'number';
    input.step = u === 'mm' ? '1' : u === 'cm' ? '0.1' : '0.05';
    input.min = String(+(info.min / k).toFixed(dec));
    input.max = String(+(info.max / k).toFixed(dec));
    input.value = (measurements[key] / k).toFixed(dec);
    input.dataset.key = key;
    input.addEventListener('input', () => {
      const v = parseFloat(input.value) * UNIT_TO_MM[measUnit()];
      if (!Number.isFinite(v) || v < info.min || v > info.max) { input.style.borderColor = 'var(--danger)'; return; }
      input.style.borderColor = '';
      measurements = { ...measurements, [key]: v };
      sampleEyes = null;
      debounce('head', 200, rebuildSampleHead);
    });
    label.appendChild(input);
    host.appendChild(label);
  }
}

function switchHeadTab(tab: 'sample' | 'import') {
  headTab = tab;
  qsa('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  qsa('[data-tab-body]').forEach((b) => { b.hidden = b.dataset.tabBody !== tab; });
  viewer.showFrontArrow(tab === 'import');
  if (tab === 'sample') rebuildSampleHead();
  else if (imported) rebuildImportedHead();
}

async function refreshHeadList(selectId: string | null = currentHeadId) {
  const list = await store.heads.list();
  const sel = $('head-select') as HTMLSelectElement;
  sel.innerHTML = '<option value="">— not saved —</option>' + list.map((h) => `<option value="${h.id}">${escapeHtml(h.name)} (${h.kind === 'parametric' ? 'measurements' : 'scan'})</option>`).join('');
  sel.value = selectId ?? '';
}

async function loadHeadRecord(rec: HeadRecord) {
  currentHeadId = rec.id;
  ($('head-name') as HTMLInputElement).value = rec.name;
  localStorage.setItem('helmetfit.lastHead', rec.id);
  if (rec.kind === 'parametric') {
    measurements = { ...DEFAULT_MEASUREMENTS, ...rec.measurements };
    sampleEyes = rec.eyes ? [new THREE.Vector3(...rec.eyes[0]), new THREE.Vector3(...rec.eyes[1])] : null;
    renderMeasurementFields();
    switchHeadTab('sample');
  } else {
    const is3mf = rec.fileName.toLowerCase().endsWith('.3mf');
    const badge = is3mf ? 'UNZIPPING 3MF…' : 'LOADING STL…';
    await beginHudTask(badge, is3mf ? `Unzipping head scan ${rec.fileName}…` : `Loading head scan ${rec.fileName}…`);
    try {
      const o = rec.orientation.rotationDeg.map((d) => THREE.MathUtils.degToRad(d)) as [number, number, number];
      const eyes: EyePair | null = rec.eyes ? [new THREE.Vector3(...rec.eyes[0]), new THREE.Vector3(...rec.eyes[1])] : null;
      let base: THREE.BufferGeometry;
      try {
        base = mergedParts(await loadMeshParts(rec.fileName, rec.data, (m) => updateHudTask(`${rec.fileName}: ${m}`, badge)));
      } catch (e) {
        alert((e as Error).message);
        return;
      }
      imported = { fileName: rec.fileName, data: rec.data, unit: rec.unit, orientation: new THREE.Quaternion().setFromEuler(new THREE.Euler(...o)), extraScale: rec.extraScale, eyes, base };
      ($('head-unit') as HTMLSelectElement).value = rec.unit;
      switchHeadTab('import');
    } finally {
      await endHudTask();
    }
  }
  status(`Loaded head "${rec.name}"`);
}

async function saveHead() {
  const name = ($('head-name') as HTMLInputElement).value.trim();
  if (!name) { alert('Give the head a name first.'); return; }
  const existing = (await store.heads.list()).find((h) => h.name === name);
  if (existing && existing.id !== currentHeadId && !confirm(`Replace the saved head "${name}"?`)) return;
  const id = existing?.id ?? store.newId();
  let rec: HeadRecord;
  if (headTab === 'sample') {
    rec = { id, name, kind: 'parametric', measurements: { ...measurements }, eyes: sampleEyes ? [sampleEyes[0].toArray(), sampleEyes[1].toArray()] : null, updatedAt: Date.now() };
  } else {
    if (!imported) { alert('Import a head scan first.'); return; }
    const e = new THREE.Euler().setFromQuaternion(imported.orientation);
    rec = {
      id, name, kind: 'mesh', fileName: imported.fileName, data: imported.data, unit: imported.unit,
      orientation: { position: [0, 0, 0], rotationDeg: [e.x, e.y, e.z].map((r) => THREE.MathUtils.radToDeg(r)) as [number, number, number], scale: [1, 1, 1] },
      extraScale: imported.extraScale, updatedAt: Date.now(),
      eyes: imported.eyes ? [imported.eyes[0].toArray(), imported.eyes[1].toArray()] : null,
    };
  }
  await store.heads.put(rec);
  currentHeadId = id;
  localStorage.setItem('helmetfit.lastHead', id);
  await refreshHeadList(id);
  status(`Saved head "${name}"`);
}

// ---------------------------------------------------------------------------
// Helmet pieces

async function addPieceFiles(files: File[]) {
  const accepted = files.filter((f) => ACCEPTED_EXTENSIONS.split(',').some((ext) => f.name.toLowerCase().endsWith(ext)))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })); // Part_2 before Part_10
  if (!accepted.length) { alert('Use STL, OBJ or 3MF files.'); return; }
  const firstLoad = pieces.length === 0;
  const firstIs3mf = accepted[0].name.toLowerCase().endsWith('.3mf');
  await beginHudTask(
    firstIs3mf ? 'UNZIPPING 3MF…' : 'LOADING STL…',
    firstIs3mf ? `Unzipping ${accepted[0].name}…` : `Loading ${accepted[0].name}…`,
  );
  try {
    const loaded: Array<{ file: File; data: ArrayBuffer; name: string; geom: THREE.BufferGeometry; full: THREE.BufferGeometry | null; part: number; source?: THREE.Matrix4 }> = [];
    for (let idx = 0; idx < accepted.length; idx++) {
      const file = accepted[idx];
      const is3mf = file.name.toLowerCase().endsWith('.3mf');
      const badge = is3mf ? 'UNZIPPING 3MF…' : 'LOADING STL…';
      const stepSuffix = accepted.length > 1 ? ` (${idx + 1}/${accepted.length})` : '';
      updateHudTask(is3mf ? `Unzipping ${file.name}${stepSuffix}…` : `Loading ${file.name}${stepSuffix}…`, badge);
      await yieldHudFrame();
      try {
        const data = await file.arrayBuffer();
        const parts = await loadMeshParts(file.name, data, (m) => updateHudTask(`${file.name}: ${m}`, badge));
        const base = file.name.replace(/\.[^.]+$/, '');
        parts.forEach((p, i) => loaded.push({ file, data, name: parts.length > 1 ? p.name : base, geom: p.geometry, full: p.full, part: i, source: p.sourceMatrix }));
        fileBufferCache.set(file.name, data);
        void rememberMeshFile(file.name, data);
        if (lastProjectImages.file === file.name) showProjectImages();
      } catch (e) {
        alert(`${file.name}: ${(e as Error).message}`);
      }
    }
    if (!loaded.length) return;
    if (firstLoad && accepted.some((f) => f.name.toLowerCase().endsWith('.3mf'))) {
      // 3MF files say their units; the reader has already converted to mm.
      helmetUnit = 'mm';
      ($('helmet-unit') as HTMLSelectElement).value = helmetUnit;
    } else if (firstLoad) {
      // Guess units from the whole assembly's size.
      const box = new THREE.Box3();
      for (const l of loaded) box.union(l.geom.boundingBox!);
      const size = box.getSize(new THREE.Vector3());
      helmetUnit = guessUnit(Math.max(size.x, size.y, size.z), 300);
      ($('helmet-unit') as HTMLSelectElement).value = helmetUnit;
    }
    pushHistory();
    let rebuilt = 0;
    let appliedFromProject = 0;
    for (const l of loaded) {
      const k = UNIT_TO_MM[helmetUnit];
      l.geom.scale(k, k, k);
      l.full?.scale(k, k, k);
      // Slicer projects can record where each part sat in its original file: put it back there.
      let t = identityTransform();
      if (l.source) {
        const o = new THREE.Object3D();
        l.source.decompose(o.position, o.quaternion, o.scale);
        t = readTransform(o);
        rebuilt++;
      }
      const saved = pendingProjectDoc?.pieces?.find((sp) => sp.name === l.name || (sp.fileName === l.file.name && (sp.part ?? 0) === l.part));
      const col = saved?.color ?? PIECE_COLOURS[pieces.length % PIECE_COLOURS.length];
      const vis = saved ? (saved.visible ?? true) : true;
      const lens = saved ? (saved.lens ?? false) : false;
      addPiece(l.name, l.file.name, l.data, l.geom, col, saved ? saved.transform : t, vis, lens, l.full, l.part, saved ? (saved.centred ?? true) : false);
      const added = pieces[pieces.length - 1];
      added.filePath = filePathCache.get(l.file.name);
      if (saved) {
        added.locked = !!saved.locked;
        added.groupId = saved.groupId && pieceGroups.some((grp) => grp.id === saved.groupId) ? saved.groupId : null;
        appliedFromProject++;
      }
    }
    helmetFieldCache = null;
    renderPieceList();
    const unitMsg = firstLoad && helmetUnit !== 'mm' ? ` — looks like the file is in ${helmetUnit}, change "File units" if not` : '';
    const simplified = loaded.some((l) => l.full);
    status(`Added ${loaded.length} piece(s)${unitMsg}${simplified ? ' — very detailed model: showing a simplified copy, downloads keep full detail' : ''}`);
    if (appliedFromProject > 0) {
      if (pendingProjectDoc?.assembly) writeTransform(viewer.assembly, pendingProjectDoc.assembly);
      // The loaded project is complete now: keep it in the history list.
      if (pendingProjectDoc && !pendingProjectDoc.pieces.some((sp) => !pieces.some((q) => q.name === sp.name))) await rememberProject(pendingProjectDoc.name || 'Project');
      status(`Applied project settings & positions to ${appliedFromProject} piece(s).`);
      afterTransformChange(true);
      return;
    }
    if (rebuilt) {
      status(`Rebuilt ${rebuilt} part${rebuilt === 1 ? '' : 's'} from the project file's record of where each part came from.`);
    } else if (firstLoad && pieces.length >= 3 && ($('auto-assemble-on-load') as HTMLInputElement).checked && looksPrintReady()) {
      // Pieces exported laid out for printing don't line up; put them back together first.
      updateHudTask(`Added ${loaded.length} pieces — auto-organizing print plate…`, 'AUTO-ORGANIZING…');
      await runAutoAssemble(false, true);
    }
    const lensNames = refreshLens(pieces.slice(pieces.length - loaded.length));
    if (lensNames.length) status(`${$('status').textContent} Treating ${lensNames.join(', ')} as a lens (eye detection looks through it).`);
    recentreAssembly();
    if (firstLoad && headField) await runAutoFit();
    else afterTransformChange(true);
  } finally {
    await endHudTask();
  }
}

/** Spot lens inserts (the overall box matters, so re-check after pieces move). Returns names of lens pieces. */
function refreshLens(which: Piece[] = pieces): string[] {
  // Only shown pieces count (hidden ones may be lying anywhere, not yet placed).
  const asm = new THREE.Box3();
  for (const p of pieces) { if (!p.mesh.visible) continue; p.mesh.updateMatrix(); asm.union(p.mesh.geometry.boundingBox!.clone().applyMatrix4(p.mesh.matrix)); }
  // A lens by shape (thin, tucked inside the shell), or by name ("Lens-L.stl", "visor", "glass").
  for (const p of which) p.lens = p.mesh.visible && (/lens|visor|glass/i.test(p.name) || looksLikeLens(p.mesh.geometry.boundingBox!.clone().applyMatrix4(p.mesh.matrix), asm));
  helmetFieldCache = null;
  renderPieceList();
  return which.filter((p) => p.lens).map((p) => p.name);
}

// ---------------------------------------------------------------------------
// Snap pieces together by hand: click a cut face on the piece that stays put,
// then the matching face on the piece to move.

const cutFaceCache = new Map<THREE.BufferGeometry, CutFace[]>();
// ---------------------------------------------------------------------------
// How pieces join (one or the other per model) and the magnet size, if known.

let joinBy: 'magnets' | 'faces' = localStorage.getItem('helmetfit.joinBy') === 'faces' ? 'faces' : 'magnets';
let magnetSize: MagnetSize | undefined = parseMagnetSize(localStorage.getItem('helmetfit.magnetSize') ?? '');
function parseMagnetSize(v: string): MagnetSize | undefined {
  const m = v.match(/^(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)$/);
  return m ? { diameter: +m[1], thickness: +m[2] } : undefined;
}
/** Cut faces / pockets already found for a piece (without finding them now). */
const cachedFaces = (p: Piece) => cutFaceCache.get(p.mesh.geometry) ?? [];
const cachedPockets = (p: Piece) => pocketCache.get(p.mesh.geometry) ?? [];

/** Find cut faces or pockets for every piece that doesn't have them yet, showing progress and keeping the page responsive. */
async function prepareFeatures(kind: 'faces' | 'magnets', which: Piece[] = pieces) {
  const todo = which.filter((p) => !(kind === 'faces' ? cutFaceCache : pocketCache).has(p.mesh.geometry));
  for (let k = 0; k < todo.length; k++) {
    status(`${kind === 'faces' ? 'Finding cut faces' : 'Finding magnet pockets'}… ${k + 1}/${todo.length} (${todo[k].name})`);
    await new Promise((r) => setTimeout(r, 0));
    if (kind === 'faces') cutFacesOf(todo[k]); else pocketsOf(todo[k]);
  }
}

function cutFacesOf(p: Piece): CutFace[] {
  let f = cutFaceCache.get(p.mesh.geometry);
  if (!f) { f = findCutFaces(detailOf(p), 0); cutFaceCache.set(p.mesh.geometry, f); }
  return f;
}

interface SnapJoin {
  a: number; fa: CutFace; fb: CutFace;
  candidates: Array<{ bToA: THREE.Matrix4; rms: number; magnets?: number }>;
  index: number; angle: number;
  moved: number[]; before: THREE.Matrix4[]; b: number;
}
let snapMode = false;
/** Snap by matching cut faces, or by clicking matching points (for parts that overlap or interlock). */
let snapKind: 'faces' | 'points' | 'magnets' = 'faces';
let magnetFirst = -1;
const pocketCache = new Map<THREE.BufferGeometry, MagnetHole[]>();
function pocketsOf(p: Piece): MagnetHole[] {
  let h = pocketCache.get(p.mesh.geometry);
  if (!h) { h = findMagnetHoles(detailOf(p), 0, undefined, magnetSize); pocketCache.set(p.mesh.geometry, h); }
  return h;
}
function pocketMarkers(p: Piece): THREE.Group {
  const g = new THREE.Group();
  g.name = 'cutface-pockets';
  for (const h of pocketsOf(p)) {
    const m = new THREE.Mesh(new THREE.TorusGeometry(h.radius, 0.6, 6, 20), new THREE.MeshBasicMaterial({ color: '#ff3df5', depthTest: false }));
    m.position.copy(h.opening);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), h.axis);
    m.renderOrder = 20;
    g.add(m);
  }
  return g;
}
/** Point pairs, each stored in its piece's own coordinates so they move with the pieces. */
let pointPairs: Array<{ a: THREE.Vector3; b: THREE.Vector3 | null }> = [];
let pointA = -1, pointB = -1;
let snapFirst: { piece: number; face: CutFace } | null = null;
let snapJoin: SnapJoin | null = null;

function faceOverlay(faces: CutFace[], colour: string, name: string): THREE.Mesh {
  const n = faces.reduce((k, f) => k + f.triangles.length, 0);
  const pos = new Float32Array(n);
  let o = 0;
  for (const f of faces) { pos.set(f.triangles, o); o += f.triangles.length; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.75, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, depthWrite: false }));
  m.name = name;
  m.renderOrder = 5;
  return m;
}

function clearOverlays(name?: string, only?: Piece) {
  for (const p of only ? [only] : pieces) for (const c of [...p.mesh.children]) {
    if (c.name.startsWith('cutface') && (!name || c.name === name)) {
      p.mesh.remove(c);
      // Overlays can be a Group (magnet rings), so dispose every mesh inside.
      c.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); } });
    }
  }
  viewer.requestRender();
}

let snapModeToken = 0;
async function setSnapMode(on: boolean) {
  const token = ++snapModeToken;
  if (on && snapKind !== 'points') {
    // Find what this mode needs (once per piece; remembered after that) without freezing the page.
    $('snap-panel').hidden = false;
    $('snap-msg').textContent = snapKind === 'faces' ? 'Finding cut faces…' : 'Finding magnet pockets…';
    await prepareFeatures(snapKind);
    if (token !== snapModeToken) return; // mode changed meanwhile
  }
  snapMode = on;
  snapFirst = null; snapJoin = null;
  pointPairs = []; pointA = -1; pointB = -1; magnetFirst = -1;
  pocketPairs = []; pocketA = -1; pocketB = -1;
  clearOverlays();
  $('snap-panel').hidden = !on;
  $('snap-mode').classList.toggle('active', on);
  $('snap-adjust').hidden = true;
  qsa('[data-snap-kind]').forEach((b) => b.classList.toggle('active', b.dataset.snapKind === snapKind));
  $('points-clear').hidden = !(on && snapKind !== 'faces');
  if (on && snapKind === 'faces') {
    status('Finding cut faces…');
    for (const p of pieces) p.mesh.add(faceOverlay(cutFacesOf(p), '#ffd21f', 'cutface'));
    viewer.attachGizmo(null, gizmoMode);
    $('snap-msg').textContent = 'Click a yellow cut face on the piece that should stay where it is.';
    status('Snap mode: click a yellow cut face on the piece that stays still, then the matching face on the piece to move.');
  } else if (on && snapKind === 'magnets') {
    status('Finding magnet pockets…');
    let n = 0;
    for (const p of pieces) { const g = pocketMarkers(p); n += g.children.length; p.mesh.add(g); }
    viewer.attachGizmo(null, gizmoMode);
    $('snap-msg').textContent = n
      ? `Found ${n} magnet pockets (pink rings). Quick: click the piece that stays put, then the piece to join. Exact: click a pink ring on the fixed piece, then the ring it joins on the other piece — 2 pairs fix it.`
      : 'No magnet pockets found on these pieces. Try Cut faces or Points.';
    status(`Snap by magnets: ${n} pockets found.`);
  } else if (on) {
    viewer.attachGizmo(null, gizmoMode);
    $('snap-msg').textContent = 'Click a spot on the piece that stays put, then the same spot on the piece to move. Use 3 or more pairs spread out (corners, holes, edges) for an exact fit.';
    status('Snap by points: click a spot on the fixed piece, then the matching spot on the piece to move.');
  } else {
    recentreAssembly();
    updateGizmo();
    status('Snap mode off.');
  }
  viewer.requestRender();
}

const PAIR_COLOURS = ['#ff4d4d', '#4dff88', '#4da6ff', '#ffd24d', '#d24dff', '#4dfff0'];

function pointMarker(pLocal: THREE.Vector3, colour: string, owner: THREE.Mesh): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.SphereGeometry(2.5, 12, 8), new THREE.MeshBasicMaterial({ color: colour, depthTest: false }));
  m.position.copy(pLocal);
  // Keep the marker round even if the piece is scaled.
  m.scale.set(1 / owner.scale.x, 1 / owner.scale.y, 1 / owner.scale.z);
  m.name = 'cutface-pt';
  m.renderOrder = 20;
  return m;
}

function onPointPick(i: number, hit: { point: THREE.Vector3; normal: THREE.Vector3 }) {
  const p = pieces[i];
  p.mesh.updateMatrixWorld(true);
  const local = hit.point.clone().applyMatrix4(p.mesh.matrixWorld.clone().invert());
  const open = pointPairs.find((q) => !q.b);
  if (!open) {
    // New pair: point on the fixed piece.
    if (pointA >= 0 && i !== pointA) { $('snap-msg').textContent = `Click on ${pieces[pointA].name} (the fixed piece) to start the next pair, or "Clear points" to pick a different pair of pieces.`; return; }
    pointA = i;
    pointPairs.push({ a: local, b: null });
    p.mesh.add(pointMarker(local, PAIR_COLOURS[(pointPairs.length - 1) % PAIR_COLOURS.length], p.mesh));
    $('snap-msg').textContent = `Point ${pointPairs.length} on ${p.name}. Now click the same spot on the piece to move.`;
    viewer.requestRender();
    return;
  }
  if (i === pointA) { $('snap-msg').textContent = 'Click the matching spot on the other piece (the one to move).'; return; }
  if (pointB >= 0 && i !== pointB) { $('snap-msg').textContent = `Keep using ${pieces[pointB].name} as the moving piece, or "Clear points" to start over.`; return; }
  if (pointB < 0) {
    const moved = movingSet(pointA, i);
    if (!moved) return;
    pointB = i;
    pushHistory();
    snapJoin = { a: pointA, b: i, moved, before: moved.map((k) => pieces[k].mesh.matrix.clone()), fa: null as unknown as CutFace, fb: null as unknown as CutFace, candidates: [], index: 0, angle: 0 };
  }
  open.b = local;
  p.mesh.add(pointMarker(local, PAIR_COLOURS[(pointPairs.length - 1) % PAIR_COLOURS.length], p.mesh));
  applyPointFit();
}

/** Move the moving piece (and anything joined to it) so its points land on the fixed piece's points. */
function applyPointFit() {
  const j = snapJoin;
  if (!j) return;
  const A = pieces[j.a].mesh;
  A.updateMatrix();
  const bBefore = j.before[j.moved.indexOf(j.b)];
  const done = pointPairs.filter((q) => q.b) as Array<{ a: THREE.Vector3; b: THREE.Vector3 }>;
  // Work with the moving piece's points where they were when the pair started (its original placement).
  const src = done.map((q) => q.b.clone().applyMatrix4(bBefore));
  const dst = done.map((q) => q.a.clone().applyMatrix4(A.matrix));
  const fit = rigidFit(src, dst);
  const delta = fit.matrix;
  const pos = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
  j.moved.forEach((k, idx) => {
    delta.clone().multiply(j.before[idx]).decompose(pos, q, sc);
    const m = pieces[k].mesh;
    m.position.copy(pos); m.quaternion.copy(q); m.scale.copy(sc);
    m.updateMatrix(); m.updateMatrixWorld(true);
  });
  helmetFieldCache = null;
  afterTransformChange(true);
  const how = done.length === 1 ? 'moved so the points meet (add 2 more pairs to fix the angle)'
    : done.length === 2 ? 'lined up along the two points (add a 3rd pair to fix the twist)'
    : `best fit through ${done.length} pairs — points are ${fmt(fit.rms, 1)} mm apart on average`;
  $('snap-msg').textContent = `${pieces[j.b].name}${j.moved.length > 1 ? ` (+${j.moved.length - 1} joined)` : ''}: ${how}. Click another spot on ${pieces[j.a].name} to add a pair.`;
  rememberLink(j.a, j.b, done.map((d) => d.a), done.map((d) => d.b));
}

/** Pocket pairs picked by hand (each pocket in its own piece's coordinates). */
let pocketPairs: Array<{ a: MagnetHole; b: MagnetHole | null }> = [];
let pocketA = -1, pocketB = -1;

/** The pocket nearest a clicked point (piece coordinates), if the click was on/near one. */
function pocketAt(p: Piece, local: THREE.Vector3): MagnetHole | null {
  let best: MagnetHole | null = null, bd = Infinity;
  for (const h of pocketsOf(p)) { const d = h.opening.distanceTo(local); if (d < h.radius + 4 && d < bd) { bd = d; best = h; } }
  return best;
}

function highlightPocket(p: Piece, h: MagnetHole, colour: string) {
  const g = p.mesh.getObjectByName('cutface-pockets');
  const k = pocketsOf(p).indexOf(h);
  const ring = g?.children[k] as THREE.Mesh | undefined;
  if (ring) { (ring.material as THREE.MeshBasicMaterial).color.set(colour); ring.scale.setScalar(1.25); }
  viewer.requestRender();
}

/**
 * Joins made by snapping, remembered so a later snap never pulls them apart.
 * Each keeps matching anchor points on both pieces (pocket openings, face
 * centres, picked points); it only counts while those still coincide, so undo
 * and moving a piece away undo the join naturally.
 */
const snapLinks: Array<{ pa: Piece; pb: Piece; la: THREE.Vector3[]; lb: THREE.Vector3[] }> = [];
function rememberLink(a: number, b: number, la: THREE.Vector3[], lb: THREE.Vector3[]) {
  const pa = pieces[a], pb = pieces[b];
  const k = snapLinks.findIndex((l) => (l.pa === pa && l.pb === pb) || (l.pa === pb && l.pb === pa));
  if (k >= 0) snapLinks.splice(k, 1);
  if (la.length) snapLinks.push({ pa, pb, la: la.map((v) => v.clone()), lb: lb.map((v) => v.clone()) });
}

/**
 * Which pieces are currently connected: same group, a snap join still in
 * place, cut faces pressed together, or magnet pockets face to face. Returns
 * a group id per piece.
 */
function connectedGroups(): number[] {
  for (const q of pieces) { q.mesh.updateMatrix(); q.mesh.updateMatrixWorld(true); }
  const groups = joinedGroups(pieces.map(cachedFaces), pieces.map((q) => q.mesh.matrix));
  const find = (k: number): number => (groups[k] === k ? k : (groups[k] = find(groups[k])));
  const join = (u: number, v: number) => { groups[find(v)] = find(u); };
  for (let u = 0; u < pieces.length; u++) for (let v = u + 1; v < pieces.length; v++) {
    if (pieces[u].groupId && pieces[u].groupId === pieces[v].groupId) join(u, v);
  }
  // Magnet pockets face to face.
  const pk = pieces.map((q) => cachedPockets(q).map((h) => ({ c: h.opening.clone().applyMatrix4(q.mesh.matrix), a: h.axis.clone().transformDirection(q.mesh.matrix) })));
  for (let u = 0; u < pieces.length; u++) for (let v = u + 1; v < pieces.length; v++) {
    if (pk[u].some((x) => pk[v].some((y) => x.c.distanceTo(y.c) < 1.5 && x.a.dot(y.a) < -0.85))) join(u, v);
  }
  // Snap joins still in place.
  for (const l of snapLinks) {
    const u = pieces.indexOf(l.pa), v = pieces.indexOf(l.pb);
    if (u < 0 || v < 0) continue;
    const off = l.la.reduce((acc, p, k) => acc + p.clone().applyMatrix4(l.pa.mesh.matrix).distanceTo(l.lb[k].clone().applyMatrix4(l.pb.mesh.matrix)), 0) / l.la.length;
    if (off < 2) join(u, v);
  }
  for (let k = 0; k < pieces.length; k++) find(k);
  return groups;
}

/**
 * Pieces that move when piece `i` is joined onto piece `a`: `i` plus anything
 * already connected to it (unless that includes `a`). Null if that would move
 * a locked piece — locked pieces stay put.
 */
function movingSet(a: number, i: number): number[] | null {
  if (pieces[i].groupId && pieces[i].groupId === pieces[a].groupId) {
    $('snap-msg').textContent = `${pieces[i].name} and ${pieces[a].name} are in the same group — groups stay together as one section. Ungroup them first if you want to re-snap them internally.`;
    return null;
  }
  const groups = connectedGroups();
  // Already joined to the fixed piece (e.g. adjusting a snap): only this piece moves.
  const moved = groups[i] !== groups[a] ? pieces.map((_, k) => k).filter((k) => groups[k] === groups[i]) : [i];
  const locked = moved.filter((k) => isPieceLocked(pieces[k]));
  if (locked.length) {
    $('snap-msg').textContent = `${pieces[locked[0]].name} is locked, so it can't move. Click the locked piece/group first (as the one that stays put), or unlock it in the list.`;
    return null;
  }
  return moved;
}

function onMagnetPick(i: number, hit: { point: THREE.Vector3; normal: THREE.Vector3 }) {
  const p = pieces[i];
  p.mesh.updateMatrixWorld(true);
  const local = hit.point.clone().applyMatrix4(p.mesh.matrixWorld.clone().invert());
  let pocket = pocketAt(p, local);
  if (!pocket && (pocketPairs.length || pocketA >= 0)) {
    // No ring here: the pocket wasn't found automatically — measure it from its walls where clicked.
    const nLocal = hit.normal.clone().transformDirection(p.mesh.matrixWorld.clone().invert());
    pocket = measurePocket(detailOf(p), local, nLocal, 0, magnetSize);
    if (pocket) {
      pocketsOf(p).push(pocket);
      clearOverlays('cutface-pockets', p);
      p.mesh.add(pocketMarkers(p));
    } else { $('snap-msg').textContent = `No pocket found there on ${p.name} — click on the floor inside the magnet hole.`; return; }
  }
  // Clicked a pocket ring: build pocket pairs by hand.
  if (pocket) { onPocketPick(i, pocket); return; }
  if (pocketPairs.length) { $('snap-msg').textContent = 'Click a pink pocket ring (or "Clear points" to start over).'; return; }
  // Clicked a piece: match all pockets automatically.
  if (magnetFirst < 0 || magnetFirst === i) {
    magnetFirst = i;
    $('snap-msg').textContent = `${pieces[i].name} stays put (${pocketsOf(pieces[i]).length} pockets). Now click the piece to join to it.`;
    return;
  }
  const a = magnetFirst;
  magnetFirst = -1;
  // Match all pockets of everything connected to each side, as they sit now.
  const groups = connectedGroups();
  const sideA = pieces.map((_, k) => k).filter((k) => groups[k] === groups[a]);
  const sideB = groups[i] === groups[a] ? [i] : pieces.map((_, k) => k).filter((k) => groups[k] === groups[i]);
  const inAsm = (ks: number[]) => ks.flatMap((k) => pocketsOf(pieces[k]).map((h) => ({ ...h, opening: h.opening.clone().applyMatrix4(pieces[k].mesh.matrix), axis: h.axis.clone().transformDirection(pieces[k].mesh.matrix) })));
  const Am = pieces[a].mesh.matrix.clone(), Bm = pieces[i].mesh.matrix.clone();
  // matchMagnets gives T (moving side -> fixed side, assembly coords); as a b->a piece transform: A⁻¹·T·B.
  const found = matchMagnets(inAsm(sideA), inAsm(sideB), 6, 2).map((f) => ({ ...f, bToA: Am.clone().invert().multiply(f.bToA).multiply(Bm) }));
  if (!found.length) { $('snap-msg').textContent = `Couldn't match ${pieces[a].name} and ${pieces[i].name} automatically. Click the pocket rings instead: a pocket on ${pieces[a].name}, then the pocket it joins on ${pieces[i].name} (2 pairs fix it exactly).`; return; }
  const moved = movingSet(a, i);
  if (!moved) return;
  pushHistory();
  snapJoin = {
    a, b: i, fa: null as unknown as CutFace, fb: null as unknown as CutFace,
    candidates: found.map((f) => ({ bToA: f.bToA, rms: f.rms, magnets: f.pairs.length })),
    index: 0, angle: 0, moved, before: moved.map((k) => pieces[k].mesh.matrix.clone()),
  };
  applySnap();
  $('snap-adjust').hidden = false;
  $('snap-ccw').hidden = $('snap-cw').hidden = true;
}

function onPocketPick(i: number, pocket: MagnetHole) {
  const p = pieces[i];
  const open = pocketPairs.find((q) => !q.b);
  if (!open) {
    if (pocketA >= 0 && i !== pocketA) { $('snap-msg').textContent = `Start each pair on ${pieces[pocketA].name} (the fixed piece), or "Clear points" to pick other pieces.`; return; }
    pocketA = i;
    pocketPairs.push({ a: pocket, b: null });
    highlightPocket(p, pocket, PAIR_COLOURS[(pocketPairs.length - 1) % PAIR_COLOURS.length]);
    $('snap-msg').textContent = `Pocket ${pocketPairs.length} on ${p.name}. Now click the pocket it joins on the piece to move.`;
    return;
  }
  if (i === pocketA) { $('snap-msg').textContent = 'Now click the matching pocket on the other piece (the one to move).'; return; }
  if (pocketB >= 0 && i !== pocketB) { $('snap-msg').textContent = `Keep using ${pieces[pocketB].name} as the moving piece, or "Clear points" to start over.`; return; }
  if (pocketB < 0) {
    const moved = movingSet(pocketA, i);
    if (!moved) return;
    pocketB = i;
    pushHistory();
    snapJoin = { a: pocketA, b: i, moved, before: moved.map((k) => pieces[k].mesh.matrix.clone()), fa: null as unknown as CutFace, fb: null as unknown as CutFace, candidates: [], index: 0, angle: 0 };
  }
  open.b = pocket;
  highlightPocket(p, pocket, PAIR_COLOURS[(pocketPairs.length - 1) % PAIR_COLOURS.length]);
  applyPocketFit();
}

/** Move the moving piece so each picked pocket meets its partner, openings facing each other. */
function applyPocketFit() {
  const j = snapJoin;
  if (!j) return;
  const A = pieces[j.a].mesh;
  A.updateMatrix();
  const bBefore = j.before[j.moved.indexOf(j.b)];
  const done = pocketPairs.filter((q) => q.b) as Array<{ a: MagnetHole; b: MagnetHole }>;
  const L = 5;
  const src: THREE.Vector3[] = [], dst: THREE.Vector3[] = [];
  for (const { a, b } of done) {
    src.push(b.opening.clone().applyMatrix4(bBefore), b.opening.clone().addScaledVector(b.axis, L).applyMatrix4(bBefore));
    dst.push(a.opening.clone().applyMatrix4(A.matrix), a.opening.clone().addScaledVector(a.axis, -L).applyMatrix4(A.matrix));
  }
  let fit = rigidFit(src, dst);
  const place = (m: THREE.Matrix4) => {
    const pos = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
    j.moved.forEach((k, idx) => {
      m.clone().multiply(j.before[idx]).decompose(pos, q, sc);
      const mesh = pieces[k].mesh;
      mesh.position.copy(pos); mesh.quaternion.copy(q); mesh.scale.copy(sc);
      mesh.updateMatrix(); mesh.updateMatrixWorld(true);
    });
  };
  // Pockets lined up across the joint (pairs of a-pocket, b-pocket).
  const B = pieces[j.b].mesh;
  // Count/refine against every pocket on the fixed side (the fixed piece and anything connected to it).
  const groupsNow = connectedGroups();
  const fixedSide = pieces.map((_, k) => k).filter((k) => groupsNow[k] === groupsNow[j.a] && !j.moved.includes(k));
  const fixedPockets = fixedSide.flatMap((k) => pocketsOf(pieces[k]).map((h) => ({ h, k })));
  const linedUp = () => {
    const out: Array<[MagnetHole, MagnetHole]> = [];
    for (const hb of pocketsOf(pieces[j.b])) {
      const c = hb.opening.clone().applyMatrix4(B.matrix), ax = hb.axis.clone().transformDirection(B.matrix);
      const hit = fixedPockets.find(({ h, k }) => h.opening.clone().applyMatrix4(pieces[k].mesh.matrix).distanceTo(c) < 1.5 && h.axis.clone().transformDirection(pieces[k].mesh.matrix).dot(ax) < -0.85);
      // Keep the pair in the fixed piece's coordinates for the refit.
      if (hit) out.push([{ ...hit.h, opening: hit.h.opening.clone().applyMatrix4(pieces[hit.k].mesh.matrix).applyMatrix4(A.matrix.clone().invert()), axis: hit.h.axis.clone().transformDirection(pieces[hit.k].mesh.matrix).transformDirection(A.matrix.clone().invert()) }, hb]);
    }
    return out;
  };
  place(fit.matrix);
  let pairsNow = linedUp();
  if (done.length >= 2 && pairsNow.length > done.length) {
    // Refine through every pocket that lines up, not just the clicked ones.
    const s2: THREE.Vector3[] = [], d2: THREE.Vector3[] = [];
    for (const [a, b] of pairsNow) {
      s2.push(b.opening.clone().applyMatrix4(bBefore), b.opening.clone().addScaledVector(b.axis, L).applyMatrix4(bBefore));
      d2.push(a.opening.clone().applyMatrix4(A.matrix), a.opening.clone().addScaledVector(a.axis, -L).applyMatrix4(A.matrix));
    }
    fit = rigidFit(s2, d2);
    place(fit.matrix);
    pairsNow = linedUp();
  }
  helmetFieldCache = null;
  afterTransformChange(true);
  const lined = pairsNow.length;
  // Remember this join (by the pockets you picked) so later snaps keep it together.
  rememberLink(j.a, j.b, done.map((d) => d.a.opening), done.map((d) => d.b.opening));
  const how = done.length === 1
    ? 'pockets meet face to face — add a 2nd pair to fix the twist'
    : `fitted through ${lined > done.length ? `all ${lined} lined-up` : done.length} pocket pairs (${fmt(fit.rms, 1)} mm off)`;
  $('snap-msg').textContent = `${pieces[j.b].name}: ${how}. ${lined} pocket${lined === 1 ? '' : 's'} line up across the joint. Click another pocket on ${pieces[j.a].name} to add a pair.`;
}

function onSnapPick(i: number, hit?: { point: THREE.Vector3; normal: THREE.Vector3 }) {
  if (i < 0 || !hit) return;
  if (snapKind === 'points') { onPointPick(i, hit); return; }
  if (snapKind === 'magnets') { onMagnetPick(i, hit); return; }
  const p = pieces[i];
  p.mesh.updateMatrixWorld(true);
  const inv = p.mesh.matrixWorld.clone().invert();
  const local = hit.point.clone().applyMatrix4(inv);
  const nLocal = hit.normal.clone().transformDirection(inv);
  const face = cutFaceAt(cutFacesOf(p), local, nLocal);
  if (!face) { $('snap-msg').textContent = `That isn't a flat cut face on ${p.name} — click one of the yellow faces.`; return; }
  if (!snapFirst || snapFirst.piece === i) {
    clearOverlays('cutface-selected');
    p.mesh.add(faceOverlay([face], '#ff7a1a', 'cutface-selected'));
    snapFirst = { piece: i, face };
    $('snap-msg').textContent = `${p.name} stays put. Now click the matching cut face on the piece to move.`;
    return;
  }
  // Second click: press this face onto the first one.
  const a = snapFirst.piece;
  // Move the clicked piece together with anything already joined to it (unless that includes the fixed piece).
  const moved = movingSet(a, i);
  if (!moved) return;
  pushHistory();
  snapJoin = {
    a, fa: snapFirst.face, fb: face, b: i,
    candidates: snapCandidates(snapFirst.face, face),
    index: 0, angle: 0, moved, before: moved.map((k) => pieces[k].mesh.matrix.clone()),
  };
  applySnap();
  clearOverlays('cutface-selected');
  snapFirst = null;
  $('snap-adjust').hidden = false;
  $('snap-ccw').hidden = $('snap-cw').hidden = false;
}

function applySnap() {
  const j = snapJoin;
  if (!j) return;
  const A = pieces[j.a].mesh, B = pieces[j.b].mesh;
  A.updateMatrix();
  let target = A.matrix.clone().multiply(j.candidates[j.index].bToA);
  if (j.angle) {
    // Turn around the joint: axis = face normal through the face centre (helmet coordinates).
    const c = j.fa.centre.clone().applyMatrix4(A.matrix);
    const n = j.fa.normal.clone().transformDirection(A.matrix);
    const rot = new THREE.Matrix4().makeTranslation(c.x, c.y, c.z)
      .multiply(new THREE.Matrix4().makeRotationAxis(n, THREE.MathUtils.degToRad(j.angle)))
      .multiply(new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
    target = rot.multiply(target);
  }
  const bBefore = j.before[j.moved.indexOf(j.b)];
  const delta = target.multiply(bBefore.clone().invert());
  const pos = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
  j.moved.forEach((k, idx) => {
    delta.clone().multiply(j.before[idx]).decompose(pos, q, sc);
    const m = pieces[k].mesh;
    m.position.copy(pos); m.quaternion.copy(q); m.scale.copy(sc);
    m.updateMatrix(); m.updateMatrixWorld(true);
  });
  void B;
  helmetFieldCache = null;
  afterTransformChange(true);
  const c = j.candidates[j.index];
  // Remember the join: matched pockets (magnets) or the pressed-together face centres (cut faces).
  if (c.magnets) {
    const la: THREE.Vector3[] = [], lb: THREE.Vector3[] = [];
    for (const hb of cachedPockets(pieces[j.b])) {
      const w = hb.opening.clone().applyMatrix4(pieces[j.b].mesh.matrix);
      for (const ha of cachedPockets(pieces[j.a])) if (ha.opening.clone().applyMatrix4(A.matrix).distanceTo(w) < 1.5) { la.push(ha.opening); lb.push(hb.opening); break; }
    }
    rememberLink(j.a, j.b, la, lb);
  } else if (j.fa && j.fb) rememberLink(j.a, j.b, [j.fa.centre], [j.fb.centre]);
  if (c.magnets) {
    $('snap-msg').textContent = `Joined ${pieces[j.b].name}${j.moved.length > 1 ? ` (+${j.moved.length - 1} joined)` : ''} to ${pieces[j.a].name}: ${c.magnets} magnet pockets line up (within ${fmt(c.rms, 1)} mm). Fit ${j.index + 1} of ${j.candidates.length} — "Next fit" if it's wrong.`;
    return;
  }
  const fit = Number.isFinite(c.rms)
    ? c.rms < 0.3 ? `outlines match (within ${fmt(c.rms, 2)} mm)` : `outlines match loosely (${fmt(c.rms, 1)} mm)`
    : 'faces put flat together — outlines differ, so check it by eye';
  $('snap-msg').textContent = `Snapped ${pieces[j.b].name}${j.moved.length > 1 ? ` (+${j.moved.length - 1} joined)` : ''} onto ${pieces[j.a].name}: ${fit}. Fit ${j.index + 1} of ${j.candidates.length}${j.angle ? `, turned ${j.angle}°` : ''}. Click two more faces to keep going.`;
}

/** Print-ready sets: most pieces sit on one flat plane (the print bed). */
function looksPrintReady(): boolean {
  for (const axis of ['x', 'y', 'z'] as const) {
    for (const side of ['min', 'max'] as const) {
      const vals = pieces.map((p) => { p.mesh.updateMatrix(); return p.mesh.geometry.boundingBox!.clone().applyMatrix4(p.mesh.matrix)[side][axis]; });
      for (const v of vals) {
        const same = vals.filter((w) => Math.abs(w - v) < 0.5).length;
        if (same >= Math.max(3, 0.6 * pieces.length)) return true;
      }
    }
  }
  return false;
}

/** Put pieces that were laid out for printing back together by matching their cut faces. */
/**
 * `automatic`: run on load — only keep the result if most pieces actually fit
 * together (otherwise these parts aren't joined by flat cuts and a partial
 * result just scrambles them).
 */
async function runAutoAssemble(confirmFirst: boolean, automatic = false) {
  // Only shown pieces (and shown groups) take part; hidden ones are left exactly where they are.
  const use = pieces.filter((p) => isPieceVisible(p));
  if (use.length < 2) { alert(pieces.length < 2 ? 'Add at least two pieces first.' : 'Show at least two pieces (tick them in the list) to auto-assemble.'); return; }
  const fixedFlags = use.map((p) => isPieceLocked(p));
  const nLocked = fixedFlags.filter(Boolean).length;
  if (nLocked === use.length) { status('Every shown piece/group is locked — unlock the ones you want auto-assemble to place.'); return; }
  const nHidden = pieces.length - use.length;
  if (confirmFirst && !confirm((nLocked
    ? `Place the ${use.length - nLocked} unlocked pieces/groups onto the ${nLocked} locked ones? Locked pieces and locked groups won't move or resize. (You can undo.)`
    : `Re-arrange the ${use.length} shown pieces/groups by matching their cut faces and connectors? Groups stay together as pre-assembled sections. (You can undo.)`)
    + (nHidden ? `\n\n${nHidden} hidden piece${nHidden === 1 ? '' : 's'} won't be touched.` : ''))) return;
  const btn = $('assemble-run') as HTMLButtonElement;
  btn.disabled = true;
  await beginHudTask('AUTO-ORGANIZING…', `Auto-organizing ${use.length} pieces…`);
  try {
    pushHistory();
    // Start from where the pieces are now, so joins made by hand (Snap) and groups are kept.
    // Work in each file's own coordinates (pieces are stored centred on their middle) —
    // the assembler's results don't then depend on where each piece's pivot is.
    for (const p of use) p.mesh.updateMatrix();
    const toFile = (p: Piece) => new THREE.Matrix4().makeTranslation(p.origin.x, p.origin.y, p.origin.z);
    const groupIds = [...new Set(use.map((p) => p.groupId).filter((g): g is string => !!g))];
    const userGroups = use.map((p) => (p.groupId ? groupIds.indexOf(p.groupId) : -1));
    // Only search for what the model joins by, reusing anything already found (shifted into file coordinates).
    await prepareFeatures(joinBy, use);
    // (Cached features are tagged piece 0; the assembler needs each one's index in `use`.)
    const shiftFaces = (p: Piece, i: number) => cachedFaces(p).map((f) => ({ ...f, piece: i, centre: f.centre.clone().add(p.origin) }));
    const shiftPockets = (p: Piece, i: number) => cachedPockets(p).map((h) => ({ ...h, piece: i, opening: h.opening.clone().add(p.origin) }));
    const res = await autoAssemble(
      use.map((p) => detailOf(p).clone().applyMatrix4(toFile(p))),
      (msg) => updateHudTask(`Auto-organizing: ${msg}`, 'AUTO-ORGANIZING…'),
      (window as unknown as { HF_DEBUG?: boolean }).HF_DEBUG ? (m) => console.log(m) : undefined,
      use.map((p) => p.mesh.matrix.clone().multiply(toFile(p).invert())),
      fixedFlags,
      userGroups,
      { mode: joinBy, ...(joinBy === 'faces' ? { faces: use.map(shiftFaces) } : { holes: use.map(shiftPockets) }) },
    );
    res.matrices.forEach((m, i) => m.multiply(toFile(use[i])));
    const counts = new Map<number, number>();
    for (const g of res.group) counts.set(g, (counts.get(g) ?? 0) + 1);
    // With locked pieces/groups, "the helmet" is the group they're in.
    const lockedIdx = fixedFlags.findIndex(Boolean);
    const [main, mainSize] = lockedIdx >= 0 ? [res.group[lockedIdx], counts.get(res.group[lockedIdx])!] : [...counts].sort((a, b) => b[1] - a[1])[0];
    const magnetJoins = res.joins.filter((j) => j.magnets > 0);
    if (automatic && !magnetJoins.length && mainSize < Math.max(2, 0.6 * use.length)) {
      undoStack.pop(); updateUndoButtons();
      status(`These pieces don't join with matching flat cuts, so they were left as they are. Use "Snap pieces together…" → Points to line them up by hand.`);
      return;
    }
    const pos = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
    use.forEach((p, i) => {
      if (fixedFlags[i]) return; // Locked pieces and locked groups never move or resize.
      res.matrices[i].decompose(pos, q, sc);
      p.mesh.position.copy(pos); p.mesh.quaternion.copy(q); p.mesh.scale.copy(sc);
      p.mesh.updateMatrix(); p.mesh.updateMatrixWorld(true);
    });
    const left = res.joins.length > 0 ? use.filter((_, i) => !fixedFlags[i] && res.group[i] !== main).map((p) => p.name) : [];
    // Pieces it couldn't place are still lying where they were — hide them so they
    // don't throw off the fit; tick them in the list to show them and snap them on.
    if (res.joins.length > 0) {
      use.forEach((p, i) => { if (!fixedFlags[i] && res.group[i] !== main) p.mesh.visible = false; });
    }
    helmetFieldCache = null;
    if (lockedIdx < 0 && nHidden === 0) recentreAssembly();
    refreshLens();
    viewer.frameAll();
    afterTransformChange(true);
    const how = magnetJoins.length
      ? `using ${magnetJoins.reduce((k, j) => k + j.magnets, 0)} lined-up connectors${res.joins.length > magnetJoins.length ? ' and cut faces' : ''}`
      : 'from their cut faces';
    if (!res.joins.length && new Set(res.group).size > 1) {
      status('No matching cut faces or connectors found between the separate groups/pieces — use "Snap pieces together…" to join them.');
    } else if (lockedIdx >= 0) {
      const placed = use.filter((_, i) => !fixedFlags[i] && res.group[i] === main).length;
      status(left.length
        ? `Placed ${placed} of ${use.length - nLocked} unlocked pieces onto the locked ones ${how}. Couldn't place (hidden for now): ${left.join(', ')}.`
        : `Placed all ${use.length - nLocked} unlocked pieces onto the locked ones ${how}.`);
    } else status(left.length
      ? `Assembled ${use.length - left.length} of ${use.length} shown pieces ${how}. Couldn't place (hidden for now): ${left.join(', ')} — tick them in the list and use "Snap pieces together" to join them by hand.`
      : `Assembled all ${use.length} shown pieces ${how}.`);
  } catch (e) {
    status(`Auto-assemble failed: ${(e as Error).message}`);
  } finally {
    btn.disabled = false;
    await endHudTask();
  }
}

/**
 * Add a piece. Its geometry is re-centred on its own middle so rotating or
 * scaling it pivots there (file coordinates often put the origin far away —
 * e.g. a 3MF part's position on the print plate). `t` is the piece transform:
 * for the centred geometry if `tIsCentred`, else for the file's coordinates.
 */
function addPiece(name: string, fileName: string, data: ArrayBuffer, geomMm: THREE.BufferGeometry, colour: string, t: TransformData, visible: boolean, lens: boolean, full: THREE.BufferGeometry | null = null, part = 0, tIsCentred = false) {
  geomMm.computeBoundingBox();
  const origin = geomMm.boundingBox!.getCenter(new THREE.Vector3());
  geomMm.translate(-origin.x, -origin.y, -origin.z);
  geomMm.computeBoundingBox(); geomMm.computeBoundingSphere();
  if (full) { full.translate(-origin.x, -origin.y, -origin.z); full.computeBoundingBox(); }
  const mesh = viewer.makePieceObject(geomMm, colour);
  mesh.name = name;
  mesh.visible = visible;
  writeTransform(mesh, t);
  if (!tIsCentred) {
    // Same placement as before: (old transform) · (move centre back to where it was).
    const m = mesh.matrix.clone().multiply(new THREE.Matrix4().makeTranslation(origin.x, origin.y, origin.z));
    m.decompose(mesh.position, mesh.quaternion, mesh.scale);
    mesh.updateMatrix();
  }
  viewer.assembly.add(mesh);
  pieces.push({ name, fileName, data, colour, mesh, lens, full, part, origin });
}

/**
 * Put the whole helmet's pivot in the middle of its (shown) pieces, without
 * moving anything, so "Whole helmet" rotates about itself.
 */
function recentreAssembly() {
  // Middle as seen on screen (world box), expressed in the helmet's own coordinates.
  const a0 = viewer.assembly;
  a0.updateMatrixWorld(true);
  const box = new THREE.Box3();
  for (const p of pieces) { if (!p.mesh.visible) continue; p.mesh.updateMatrix(); box.union(p.mesh.geometry.boundingBox!.clone().applyMatrix4(p.mesh.matrixWorld)); }
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()).applyMatrix4(a0.matrixWorld.clone().invert());
  if (c.lengthSq() < 1e-6) return;
  for (const p of pieces) { p.mesh.position.sub(c); p.mesh.updateMatrix(); }
  const a = viewer.assembly;
  a.position.add(c.clone().multiply(a.scale).applyQuaternion(a.quaternion));
  a.updateMatrix(); a.updateMatrixWorld(true);
  helmetFieldCache = null;
  writeXformFields();
}

function removePiece(i: number) {
  const p = pieces[i];
  if (viewer.gizmo.object === p.mesh) viewer.attachGizmo(null, gizmoMode);
  viewer.assembly.remove(p.mesh);
  cutFaceCache.delete(p.mesh.geometry); pocketCache.delete(p.mesh.geometry);
  viewer.disposePieceObject(p.mesh);
  pieces.splice(i, 1);
  undoStack.length = 0; redoStack.length = 0; updateUndoButtons(); // piece indices changed
  if (selected >= pieces.length) selected = pieces.length - 1;
  if (typeof target === 'number') target = 'assembly';
  helmetFieldCache = null;
  renderPieceList();
  afterTransformChange(true);
}

function clearPieces() {
  if (fitSignal) fitSignal.cancelled = true;
  viewer.attachGizmo(null, gizmoMode);
  for (const p of pieces) { viewer.assembly.remove(p.mesh); cutFaceCache.delete(p.mesh.geometry); pocketCache.delete(p.mesh.geometry); viewer.disposePieceObject(p.mesh); }
  pieces.length = 0;
  pieceGroups.length = 0;
  selected = -1;
  target = 'assembly';
  helmetFieldCache = null;
}

// ---------------------------------------------------------------------------
// Project pictures (from 3MF files): a small viewer in the corner of the 3D view.

let picIndex = 0;
function showProjectImages() {
  const imgs = lastProjectImages.images;
  $('pics-open').hidden = imgs.length === 0;
  if (!imgs.length) { $('pics').hidden = true; return; }
  picIndex = Math.min(picIndex, imgs.length - 1);
  const im = imgs[picIndex];
  ($('pics-img') as HTMLImageElement).src = im.url;
  $('pics-caption').textContent = im.name;
  $('pics-caption').title = `${lastProjectImages.file}: ${im.name}`;
  $('pics-count').textContent = `${picIndex + 1} / ${imgs.length}`;
  $('pics').hidden = false;
}

/** "Remove all" button: start over with no pieces (saved projects aren't touched). */
function removeAllPieces() {
  if (!pieces.length) return;
  if (!confirm(`Remove ${pieces.length === 1 ? "the 1 piece" : `all ${pieces.length} pieces`}? This can't be undone (saved projects aren't affected).`)) return;
  if (snapMode) setSnapMode(false);
  clearPieces();
  writeTransform(viewer.assembly, identityTransform());
  undoStack.length = 0; redoStack.length = 0; updateUndoButtons();
  $('fit-msg').textContent = '';
  for (const im of lastProjectImages.images) URL.revokeObjectURL(im.url);
  lastProjectImages.images = []; lastProjectImages.file = '';
  showProjectImages();
  renderPieceList();
  afterTransformChange(true);
  status('Removed all pieces.');
}

function setHelmetUnit(u: LengthUnit) {
  cutFaceCache.clear(); pocketCache.clear(); // geometry is rescaled in place
  for (const p of pieces) for (const g of [p.mesh.geometry, p.full]) if (g) { forgetSurface(g); forgetPocketData(g); }
  const ratio = UNIT_TO_MM[u] / UNIT_TO_MM[helmetUnit];
  helmetUnit = u;
  for (const p of pieces) {
    p.full?.scale(ratio, ratio, ratio);
    p.mesh.geometry.scale(ratio, ratio, ratio);
    viewer.refreshPickData(p.mesh.geometry); // its hit-testing data was for the old size
    p.mesh.geometry.computeBoundingBox();
    p.mesh.geometry.computeBoundingSphere();
    delete p.mesh.geometry.userData.bvh;
    p.mesh.position.multiplyScalar(ratio);
    p.origin.multiplyScalar(ratio);
  }
  helmetFieldCache = null;
  afterTransformChange(true);
}

let draggedPieceIndex = -1;

function assignPieceToGroup(pieceIndex: number, gid: string | null) {
  const p = pieces[pieceIndex];
  if (!p || p.groupId === gid) return;
  pushHistory();
  const visBefore = isPieceVisible(p);
  p.groupId = gid;
  if (gid) {
    const g = pieceGroups.find((grp) => grp.id === gid);
    if (g) {
      g.collapsed = false;
      if (g.locked) p.locked = true;
      if (g.visible === false) p.mesh.visible = false;
    }
    status(`Moved ${p.name} into "${g?.name ?? 'group'}".`);
  } else {
    status(`Removed ${p.name} from its group.`);
  }
  const visAfter = isPieceVisible(p);
  renderPieceList();
  if (visBefore !== visAfter) {
    helmetFieldCache = null;
    afterTransformChange(true);
  } else {
    renderReport();
  }
}

function addGroup(name?: string, memberIndices?: number[]): PieceGroup {
  pushHistory();
  const assign = memberIndices ?? (selected >= 0 && pieces[selected] && !pieces[selected].groupId ? [selected] : []);
  const allLocked = assign.length > 0 && assign.every((idx) => pieces[idx]?.locked);
  const g: PieceGroup = {
    id: store.newId(),
    name: name?.trim() || `Group ${pieceGroups.length + 1}`,
    collapsed: false,
    locked: allLocked,
    visible: true,
  };
  pieceGroups.push(g);
  for (const idx of assign) if (pieces[idx]) pieces[idx].groupId = g.id;
  renderPieceList();
  status(`Created group "${g.name}"${assign.length ? ` with ${assign.length} piece${assign.length === 1 ? '' : 's'}` : ' — click and drag pieces onto it to add them'}.`);
  return g;
}

function renderPieceRow(p: Piece, i: number, inGroup: boolean): HTMLLIElement {
  const li = document.createElement('li');
  li.className = `${inGroup ? 'in-group ' : ''}${i === selected ? 'selected' : ''}`.trim();
  li.draggable = true;
  const effLocked = isPieceLocked(p);
  li.innerHTML = `<span class="grip" title="Click and drag onto a group folder to group, or drag out to ungroup">⋮⋮</span>
    <input type="checkbox" title="Show / include in fit and auto-assemble" ${p.mesh.visible ? 'checked' : ''} />
    <input type="color" value="${p.colour}" title="Colour" />
    <span class="name" title="${escapeHtml(p.fileName)} — drag onto a group to assign">${escapeHtml(p.name)}</span>
    <button class="lock ${effLocked ? 'active' : ''}" title="${effLocked ? 'Locked in place — click to unlock' : 'Lock this piece in place (preserves coordinates and size during auto-assemble)'}">${effLocked ? '🔒' : '🔓'}</button>
    <button class="lens ${p.lens ? 'active' : ''}" title="Lens / see-through insert: eye-hole detection looks through it">lens</button>
    <button class="danger remove" title="Remove piece">✕</button>`;
  const [vis, col] = li.querySelectorAll('input');
  vis.addEventListener('click', (e) => e.stopPropagation());
  vis.addEventListener('change', () => {
    p.mesh.visible = vis.checked;
    if (p.groupId) {
      const g = pieceGroups.find((grp) => grp.id === p.groupId);
      if (g) g.visible = pieces.filter((q) => q.groupId === g.id).some((q) => q.mesh.visible);
    }
    helmetFieldCache = null;
    renderPieceList();
    afterTransformChange(true);
  });
  col.addEventListener('click', (e) => e.stopPropagation());
  col.addEventListener('input', () => { p.colour = col.value; (p.mesh.material as THREE.MeshStandardMaterial).color.set(col.value); viewer.requestRender(); });
  li.querySelector('button.remove')!.addEventListener('click', (e) => { e.stopPropagation(); removePiece(i); });
  li.querySelector('button.lens')!.addEventListener('click', (e) => { e.stopPropagation(); p.lens = !p.lens; helmetFieldCache = null; renderPieceList(); afterTransformChange(true); });
  li.querySelector('button.lock')!.addEventListener('click', (e) => {
    e.stopPropagation();
    const next = !effLocked;
    if (p.groupId) {
      // Locking/unlocking inside a group locks/unlocks the whole group so it never gets disassociated.
      const g = pieceGroups.find((grp) => grp.id === p.groupId);
      if (g) g.locked = next;
      for (const q of pieces) if (q.groupId === p.groupId) q.locked = next;
      status(next ? `Locked group "${g?.name ?? p.name}".` : `Unlocked group "${g?.name ?? p.name}".`);
    } else {
      p.locked = next;
      status(p.locked ? `${p.name} locked in place.` : `${p.name} unlocked.`);
    }
    renderPieceList();
  });
  li.addEventListener('click', () => selectPiece(i));
  li.addEventListener('dragstart', (e) => {
    draggedPieceIndex = i;
    e.dataTransfer?.setData('text/plain', String(i));
  });
  li.addEventListener('dragend', () => { draggedPieceIndex = -1; });
  // Dropping onto a piece inside a group assigns the dragged piece to that group;
  // dropping onto an ungrouped piece removes the dragged piece from its group.
  li.addEventListener('dragover', (e) => {
    if (draggedPieceIndex >= 0 && draggedPieceIndex !== i) {
      e.preventDefault();
      e.stopPropagation();
      li.classList.add('drag-over');
    }
  });
  li.addEventListener('dragleave', () => li.classList.remove('drag-over'));
  li.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    li.classList.remove('drag-over');
    if (draggedPieceIndex >= 0 && draggedPieceIndex !== i) {
      assignPieceToGroup(draggedPieceIndex, inGroup ? (p.groupId ?? null) : null);
      draggedPieceIndex = -1;
    }
  });
  return li;
}

function renderPieceList() {
  if ($('pieces-clear')) $('pieces-clear').hidden = pieces.length === 0;
  if ($('lock-shown')) {
    $('lock-shown').hidden = pieces.length === 0;
    $('lock-shown').textContent = pieces.some((p) => isPieceVisible(p) && !isPieceLocked(p)) ? '🔒 Lock all shown' : '🔓 Unlock all shown';
  }
  if ($('group-add')) $('group-add').hidden = pieces.length === 0;
  const ul = $('piece-list');
  ul.innerHTML = '';

  // Render Photoshop-style layer groups first, with their member pieces indented underneath.
  for (const g of pieceGroups) {
    const members = pieces.map((p, i) => ({ p, i })).filter(({ p }) => p.groupId === g.id);
    const allVis = members.length > 0 ? members.every(({ p }) => p.mesh.visible) : g.visible !== false;
    const someVis = members.length > 0 ? members.some(({ p }) => p.mesh.visible) : g.visible !== false;
    const grpLocked = isGroupLocked(g.id);
    const hdr = document.createElement('li');
    hdr.className = `group-row ${target === `group:${g.id}` ? 'selected' : ''}`.trim();
    hdr.innerHTML = `<button class="fold" title="${g.collapsed ? 'Expand group' : 'Collapse group'}">${g.collapsed ? '▸' : '▾'}</button>
      <input type="checkbox" title="Show / hide all pieces in this group (hidden groups do not participate in auto-assemble)" ${someVis ? 'checked' : ''} />
      <span class="name" title="Layer group (${members.length} piece${members.length === 1 ? '' : 's'}) — drag pieces here to group, click to move/resize whole group, double-click to rename">📁 ${escapeHtml(g.name)} (${members.length})</span>
      <button class="rename" title="Rename group">✎</button>
      <button class="lock ${grpLocked ? 'active' : ''}" title="${grpLocked ? 'Unlock group' : 'Lock group (locked groups do not get resized or disassociated)'}">${grpLocked ? '🔒' : '🔓'}</button>
      <button class="danger ungroup" title="Ungroup (keeps pieces)">✕</button>`;
    const vis = hdr.querySelector('input') as HTMLInputElement;
    vis.indeterminate = someVis && !allVis;
    vis.addEventListener('click', (e) => e.stopPropagation());
    vis.addEventListener('change', () => {
      g.visible = vis.checked;
      for (const { p } of members) p.mesh.visible = vis.checked;
      helmetFieldCache = null;
      renderPieceList();
      afterTransformChange(true);
    });
    hdr.querySelector('button.fold')!.addEventListener('click', (e) => {
      e.stopPropagation();
      g.collapsed = !g.collapsed;
      renderPieceList();
    });
    const renameGroup = () => {
      const next = prompt('Group name', g.name)?.trim();
      if (next) { g.name = next; renderPieceList(); }
    };
    hdr.querySelector('button.rename')!.addEventListener('click', (e) => { e.stopPropagation(); renameGroup(); });
    hdr.querySelector('.name')!.addEventListener('dblclick', (e) => { e.stopPropagation(); renameGroup(); });
    hdr.querySelector('button.lock')!.addEventListener('click', (e) => {
      e.stopPropagation();
      const next = !grpLocked;
      g.locked = next;
      for (const { p } of members) p.locked = next;
      renderPieceList();
      status(next ? `Locked group "${g.name}" — it will not be moved, resized, or disassociated.` : `Unlocked group "${g.name}".`);
    });
    hdr.querySelector('button.ungroup')!.addEventListener('click', (e) => {
      e.stopPropagation();
      pushHistory();
      for (const { p } of members) p.groupId = null;
      const idx = pieceGroups.indexOf(g);
      if (idx >= 0) pieceGroups.splice(idx, 1);
      if (target === `group:${g.id}`) target = 'assembly';
      renderPieceList();
      status(`Ungrouped "${g.name}".`);
    });
    hdr.addEventListener('click', () => selectGroup(g.id));
    hdr.addEventListener('dragover', (e) => { if (draggedPieceIndex >= 0) { e.preventDefault(); e.stopPropagation(); hdr.classList.add('drag-over'); } });
    hdr.addEventListener('dragleave', () => hdr.classList.remove('drag-over'));
    hdr.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      hdr.classList.remove('drag-over');
      if (draggedPieceIndex >= 0 && pieces[draggedPieceIndex]) {
        assignPieceToGroup(draggedPieceIndex, g.id);
        draggedPieceIndex = -1;
      }
    });
    ul.appendChild(hdr);
    if (!g.collapsed) {
      for (const { p, i } of members) ul.appendChild(renderPieceRow(p, i, true));
    }
  }

  // Render ungrouped pieces.
  pieces.forEach((p, i) => {
    if (p.groupId && pieceGroups.some((g) => g.id === p.groupId)) return;
    ul.appendChild(renderPieceRow(p, i, false));
  });

  const sel = $('target-select') as HTMLSelectElement;
  let targetOpts = '<option value="assembly">Whole helmet</option>';
  for (const g of pieceGroups) {
    const members = pieces.map((p, i) => ({ p, i })).filter(({ p }) => p.groupId === g.id);
    targetOpts += `<option value="group:${escapeHtml(g.id)}">📁 ${escapeHtml(g.name)} (${members.length})</option>`;
    for (const { p, i } of members) {
      targetOpts += `<option value="${i}">&nbsp;&nbsp;&nbsp;↳ ${escapeHtml(p.name)}</option>`;
    }
  }
  pieces.forEach((p, i) => {
    if (p.groupId && pieceGroups.some((g) => g.id === p.groupId)) return;
    targetOpts += `<option value="${i}">${escapeHtml(p.name)}</option>`;
  });
  sel.innerHTML = targetOpts;
  sel.value = String(target);
  updateGizmo();
}

function selectGroup(gid: string) {
  if (selected >= 0 && pieces[selected]) viewer.setPieceHighlight(pieces[selected].mesh, false);
  selected = -1;
  target = `group:${gid}`;
  syncGroupPivot(gid);
  renderPieceList();
}

function selectPiece(i: number, switchTarget = true) {
  if (selected >= 0 && pieces[selected]) viewer.setPieceHighlight(pieces[selected].mesh, false);
  selected = i;
  if (i >= 0 && pieces[i]) viewer.setPieceHighlight(pieces[i].mesh, true);
  if (switchTarget && i >= 0) target = i;
  renderPieceList();
}

function isTargetLocked(): boolean {
  const gid = targetGroupId();
  if (gid) return isGroupLocked(gid);
  if (typeof target === 'number') return pieces[target] ? isPieceLocked(pieces[target]) : false;
  return false;
}

function updateGizmo() {
  const gid = targetGroupId();
  if (gid) syncGroupPivot(gid);
  const t = targetObject();
  const locked = isTargetLocked();
  if (locked && typeof target === 'number') status(`${pieces[target].name} is locked — unlock it in the list to move or resize it.`);
  else if (locked && gid) status(`Group "${pieceGroups.find((g) => g.id === gid)?.name}" is locked — unlock it in the list to move or resize it.`);
  viewer.attachGizmo(locked ? null : t, gizmoMode);
  qsa('[data-gizmo]').forEach((b) => b.classList.toggle('active', b.dataset.gizmo === gizmoMode));
  writeXformFields();
}

// ---------------------------------------------------------------------------
// Manual transform fields

function writeXformFields() {
  const o = targetObject();
  const locked = isTargetLocked();
  const t = o ? readTransform(o) : identityTransform();
  for (const input of qsa<HTMLInputElement>('[data-xf]')) {
    if (document.activeElement === input) continue;
    const kind = input.dataset.xf![0], idx = +input.dataset.xf![1];
    input.disabled = !o || locked;
    input.value = kind === 'p' ? fmt(t.position[idx], 1) : kind === 'r' ? fmt(t.rotationDeg[idx], 1) : fmt(t.scale[idx] * 100, 2);
  }
}

function readXformField(input: HTMLInputElement) {
  const o = targetObject();
  if (!o || isTargetLocked()) return;
  const v = parseFloat(input.value);
  if (!Number.isFinite(v)) return;
  const kind = input.dataset.xf![0], idx = +input.dataset.xf![1];
  pushHistory();
  const t = readTransform(o);
  if (kind === 'p') t.position[idx] = v;
  else if (kind === 'r') t.rotationDeg[idx] = v;
  else {
    const s = Math.max(1, v) / 100;
    // Locked: scale all axes by the same factor, keeping any X/Y/Z differences.
    if (uniformLocked()) { const k = s / t.scale[idx]; t.scale = t.scale.map((c) => c * k) as [number, number, number]; }
    else t.scale[idx] = s;
  }
  writeTransform(o, t);
  const gid = targetGroupId();
  if (gid) applyGroupPivotDelta(gid);
  if (o !== viewer.assembly) helmetFieldCache = null;
  afterTransformChange(true);
}

const uniformLocked = () => ($('lock-uniform') as HTMLInputElement).checked;

// Keep gizmo scaling uniform when locked.
let dragStartScale = new THREE.Vector3();
viewer.gizmo.addEventListener('mouseDown', () => {
  clearTimeout(timers.get('stats'));
  clearTimeout(timers.get('heat'));
  pushHistory();
  const gid = targetGroupId();
  if (gid) syncGroupPivot(gid);
  const o = targetObject();
  if (o) dragStartScale = o.scale.clone();
});
viewer.onGizmoChange = () => {
  const o = targetObject();
  if (!o || isTargetLocked()) return;
  if (gizmoMode === 'scale' && uniformLocked()) {
    const r = [o.scale.x / dragStartScale.x, o.scale.y / dragStartScale.y, o.scale.z / dragStartScale.z];
    let k = r[0];
    for (const v of r) if (Math.abs(v - 1) > Math.abs(k - 1)) k = v;
    o.scale.copy(dragStartScale).multiplyScalar(k);
  }
  const gid = targetGroupId();
  if (gid) applyGroupPivotDelta(gid);
  if (o !== viewer.assembly) helmetFieldCache = null;
  writeXformFields();
  viewer.requestRender();
};
viewer.onGizmoEnd = () => afterTransformChange(true);
viewer.pieceIndexOf = (o) => pieces.findIndex((p) => p.mesh === o);
viewer.isPickable = (o) => { const p = pieces.find((q) => q.mesh === o); return !!p && isPieceVisible(p); };
viewer.onPick = (i, hit) => {
  if (snapMode) { onSnapPick(i, hit); return; }
  if (i >= 0) {
    if (targetGroupId() !== null && pieces[i]?.groupId) selectGroup(pieces[i].groupId!);
    else selectPiece(i, target !== 'assembly');
  }
};

// ---------------------------------------------------------------------------
// Fit evaluation, heat map, report

function helmetField(): HelmetField | null {
  if (helmetFieldCache) return helmetFieldCache;
  const vis = pieces.filter((p) => isPieceVisible(p));
  if (!vis.length) return null;
  for (const p of vis) p.mesh.updateMatrix();
  helmetFieldCache = new HelmetField(mergePieces(vis.map((p) => ({ geometry: p.mesh.geometry, matrix: p.mesh.matrix, lens: p.lens }))));
  return helmetFieldCache;
}

/** Recompute stats (fast) and, if `full`, the heat map too. */
function afterTransformChange(full: boolean) {
  writeXformFields();
  viewer.requestRender();
  if (viewer.gizmo.dragging) return;
  debounce('stats', full ? 0 : 120, () => {
    if (viewer.gizmo.dragging) return;
    const vis = pieces.filter((p) => isPieceVisible(p));
    if (!vis.length || !headField) {
      lastStats = null; lastEyes = null; renderReport(); viewer.applyHeatmapColours(null);
      viewer.setEyeMarkers(headField?.eyes ?? null, null);
      return;
    }
    viewer.assembly.updateMatrixWorld(true);
    if (helmetFieldCache) {
      lastStats = evaluatePose(headField, helmetFieldCache, viewer.assembly.matrixWorld, fitSettings, headSampleCache);
      lastEyes = eyeAlignment(headField, helmetFieldCache, viewer.assembly.matrixWorld);
    } else {
      lastStats = evaluatePiecesPose(headField, vis.map((p) => p.mesh), fitSettings, headSampleCache);
      lastEyes = null;
    }
    viewer.setEyeMarkers(headField.eyes, lastEyes?.helmet ?? null);
    renderReport();
    if (heatmapOn()) debounce('heat', full ? 0 : 350, updateHeatmap);
  });
}

const heatmapOn = () => ($('show-heatmap') as HTMLInputElement).checked;

function updateHeatmap() {
  $('legend').hidden = !heatmapOn();
  const vis = pieces.filter((p) => isPieceVisible(p));
  if (!heatmapOn() || !vis.length || !headField) { viewer.applyHeatmapColours(null); return; }
  viewer.assembly.updateMatrixWorld(true);
  const gaps = helmetFieldCache
    ? vertexGaps(headField, helmetFieldCache, viewer.assembly.matrixWorld)
    : vertexGapsForPieces(headField, vis.map((p) => p.mesh));
  const cols = new Float32Array(gaps.length * 3);
  const c = new THREE.Color();
  for (let i = 0; i < gaps.length; i++) {
    gapColour(gaps[i], c);
    cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
  }
  viewer.applyHeatmapColours(cols);
  $('lg-min').textContent = `≤${fitSettings.minGap} mm`;
  $('lg-target').textContent = `${fitSettings.targetGap} mm`;
  $('lg-max').textContent = `≥${fitSettings.targetGap + 15} mm`;
}

// Gap colour scale: magenta (cuts in) → red (tight) → yellow → green (target) → blue (loose).
const STOPS: Array<[number, string]> = [];
function buildStops() {
  const { minGap, targetGap } = fitSettings;
  STOPS.length = 0;
  STOPS.push([-0.01, '#d000d0'], [0, '#ff2020'], [minGap, '#ff2020'], [(minGap + targetGap) / 2, '#ffd21f'], [targetGap, '#2fbf4a'], [targetGap + 7, '#2aa7c9'], [targetGap + 15, '#2a3fd0']);
  const grad = STOPS.slice(1).map(([g, col]) => `${col} ${(((g - 0) / (targetGap + 15)) * 100).toFixed(1)}%`).join(', ');
  (document.querySelector('.legend-bar') as HTMLElement).style.background = `linear-gradient(to right, ${grad})`;
}
const _ca = new THREE.Color(), _cb = new THREE.Color();
function gapColour(g: number, out: THREE.Color) {
  if (Number.isNaN(g)) return out.set('#9a9a9a');
  if (g < 0) return out.set(STOPS[0][1]);
  for (let i = 1; i < STOPS.length - 1; i++) {
    const [g0, c0] = STOPS[i], [g1, c1] = STOPS[i + 1];
    if (g <= g1) { const t = g1 > g0 ? (g - g0) / (g1 - g0) : 1; return out.copy(_ca.set(c0)).lerp(_cb.set(c1), t); }
  }
  return out.set(STOPS[STOPS.length - 1][1]);
}

function renderReport() {
  const host = $('report');
  if (!pieces.length) { host.innerHTML = '<p class="hint">Add helmet pieces to see how it fits.</p>'; return; }
  const s = lastStats;
  let verdict = '';
  if (!s || s.coverage < 0.02) verdict = '<div class="verdict bad">The helmet is not around the head yet — run Auto fit or move it.</div>';
  else if (s.penetration > 0.5) verdict = `<div class="verdict bad">Cuts into the head by up to ${fmt(s.penetration)} mm — make it bigger or move it.</div>`;
  else if (s.tightFraction > 0.03) verdict = `<div class="verdict warn">Pressure points: ${fmt(s.tightFraction * 100, 0)}% of the covered area is closer than ${fitSettings.minGap} mm.</div>`;
  else if (s.minGap < fitSettings.minGap) verdict = `<div class="verdict warn">Mostly good, but a small spot is only ${fmt(s.minGap)} mm from the head — turn on Gap colours to find it.</div>`;
  else if (s.meanGap > fitSettings.targetGap + 8) verdict = '<div class="verdict warn">Loose — print smaller or use thicker padding.</div>';
  else verdict = '<div class="verdict good">Good fit.</div>';
  let offAxis = false;
  const formatPieceScale = (p: Piece, idx?: number): string => {
    const sc = printScale(p);
    const dlBtn = idx !== undefined ? `<button type="button" class="dl-piece" data-dl-piece="${idx}" title="Save ${escapeHtml(p.name)} scaled STL individually">⬇</button>` : '';
    if (!sc) { offAxis = true; return `<td class="hint">use download ↓${dlBtn}</td>`; }
    const same = Math.abs(sc.x - sc.y) < 5e-4 && Math.abs(sc.y - sc.z) < 5e-4;
    const pct = same ? `${fmt(sc.x * 100, 1)}%` : `X ${fmt(sc.x * 100, 1)}% · Y ${fmt(sc.y * 100, 1)}% · Z ${fmt(sc.z * 100, 1)}%`;
    return `<td class="print-scale">${pct}${dlBtn}</td>`;
  };
  let scaleRows = '';
  for (const g of pieceGroups) {
    const members = pieces.map((p, idx) => ({ p, idx })).filter(({ p }) => p.groupId === g.id);
    if (!members.length) continue;
    const firstSc = printScale(members[0].p);
    const sameGrp = firstSc && members.every(({ p }) => {
      const mSc = printScale(p);
      return mSc && Math.abs(mSc.x - firstSc.x) < 5e-4 && Math.abs(mSc.y - firstSc.y) < 5e-4 && Math.abs(mSc.z - firstSc.z) < 5e-4;
    });
    const grpDlBtn = `<button type="button" class="dl-piece" data-dl-group="${escapeHtml(g.id)}" title="Download ${escapeHtml(g.name)} pieces as a Zip">⬇ Zip</button>`;
    const grpCell = sameGrp
      ? formatPieceScale(members[0].p).replace('</td>', `${grpDlBtn}</td>`)
      : `<td class="hint">per piece ↓${grpDlBtn}</td>`;
    scaleRows += `<tr class="group-scale-row"><td><strong>📁 ${escapeHtml(g.name)}</strong></td>${grpCell}</tr>`;
    for (const { p, idx } of members) {
      scaleRows += `<tr><td>&nbsp;&nbsp;&nbsp;↳ ${escapeHtml(p.name)}</td>${formatPieceScale(p, idx)}</tr>`;
    }
  }
  pieces.forEach((p, idx) => {
    if (p.groupId && pieceGroups.some((g) => g.id === p.groupId)) return;
    scaleRows += `<tr><td>${escapeHtml(p.name)}</td>${formatPieceScale(p, idx)}</tr>`;
  });
  host.innerHTML = `${verdict}
    ${s ? `<table>
      <tr><td>Head covered</td><td>${fmt(s.coverage * 100, 0)}%</td></tr>
      <tr><td>Closest gap</td><td>${fmt(s.minGap)} mm</td></tr>
      <tr><td>Average gap</td><td>${fmt(s.meanGap)} mm</td></tr>
      <tr><td>Largest gap</td><td>${fmt(s.maxGap)} mm</td></tr>
    </table>` : ''}
    <p class="hint">${eyeReport()}</p>
    <h3 class="hint">Print your original files at:</h3>
    <table>${scaleRows}</table>
    ${helmetUnit !== 'mm' ? `<p class="hint">Assumes your slicer opens the files in ${helmetUnit}. If it treats them as mm, multiply by ${UNIT_TO_MM[helmetUnit]}.</p>` : ''}
    ${offAxis ? '<p class="hint">Some pieces were turned for printing, so their stretch doesn\'t line up with the file\'s X/Y/Z — a slicer can\'t apply it. Use "Download Scaled (Zip)" (or ⬇ next to a piece) and print those at 100%.</p>' : '<p class="hint">Or use "Download Scaled (Zip)" (or ⬇ next to any piece) and print those at 100%.</p>'}`;
  host.querySelectorAll<HTMLButtonElement>('[data-dl-piece]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = +(btn.dataset.dlPiece ?? -1);
      if (idx >= 0 && pieces[idx]) exportPieces('print', [idx], 'files');
    });
  });
  host.querySelectorAll<HTMLButtonElement>('[data-dl-group]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const gid = btn.dataset.dlGroup;
      const idxs = pieces.map((p, i) => (p.groupId === gid ? i : -1)).filter((i) => i >= 0);
      if (idxs.length) {
        const gName = pieceGroups.find((grp) => grp.id === gid)?.name;
        exportPieces('print', idxs, 'zip', gName);
      }
    });
  });
}

function eyeReport(): string {
  if (!headField?.eyes) return 'Eyes: not set on the head.';
  if (!lastEyes) return 'Eye holes: none found facing the face (fine for helmets with an open face).';
  const what = lastEyes.kind === 'visor' ? 'Visor' : 'Eye holes';
  const off = Math.hypot(lastEyes.dx, lastEyes.dy);
  const dir = [
    Math.abs(lastEyes.dx) >= 1 ? `${fmt(Math.abs(lastEyes.dx))} mm ${lastEyes.dx > 0 ? 'toward your left ear' : 'toward your right ear'}` : '',
    Math.abs(lastEyes.dy) >= 1 ? `${fmt(Math.abs(lastEyes.dy))} mm ${lastEyes.dy > 0 ? 'high' : 'low'}` : '',
  ].filter(Boolean).join(', ');
  return off < 1.5 ? `${what} centred on your eyes (orange rings).` : `${what} are off your eyes: ${dir} (orange rings vs cyan dots).`;
}

/** Scale to apply to a piece's original file (per file axis). */
/**
 * Matrix that turns a piece's original file into the piece to print, in the
 * file's own orientation: the helmet's (possibly X/Y/Z) scale is applied in the
 * assembled helmet's axes, then the piece's placement rotation/offset is undone.
 * For pieces turned for printing that stretch may not line up with the file's
 * axes, so it isn't always expressible as a per-axis %.
 */
function printMatrix(p: Piece): THREE.Matrix4 {
  p.mesh.updateMatrix();
  const rigid = new THREE.Matrix4().compose(p.mesh.position, p.mesh.quaternion, new THREE.Vector3(1, 1, 1));
  const sAsm = new THREE.Matrix4().makeScale(viewer.assembly.scale.x, viewer.assembly.scale.y, viewer.assembly.scale.z);
  return rigid.invert().multiply(sAsm).multiply(p.mesh.matrix);
}

/** Per-axis print scale of a piece's file, or null if the stretch is off the file's axes. */
function printScale(p: Piece): THREE.Vector3 | null {
  const e = printMatrix(p).elements;
  const diag = new THREE.Vector3(e[0], e[5], e[10]);
  const off = Math.max(Math.abs(e[1]), Math.abs(e[2]), Math.abs(e[4]), Math.abs(e[6]), Math.abs(e[8]), Math.abs(e[9]));
  return off < 2e-3 ? diag : null;
}

// ---------------------------------------------------------------------------
// Auto fit

async function runAutoFit() {
  const hf = helmetField();
  if (!hf || !headField || !viewer.headMesh) { alert('Load a head and at least one visible helmet piece first.'); return; }
  const btn = $('fit-run') as HTMLButtonElement;
  const prog = $('fit-progress') as HTMLProgressElement;
  btn.disabled = true; $('fit-cancel').hidden = false; prog.hidden = false; prog.value = 0;
  fitSignal = { cancelled: false };
  const a = viewer.assembly;
  const before = snapshot();
  await beginHudTask(
    fitSettings.searchRotation ? 'AUTO-ROTATING…' : 'AUTO-FITTING…',
    fitSettings.searchRotation ? 'Searching orientations & fitting helmet to head…' : 'Fitting helmet to head…',
  );
  try {
    viewer.headMesh.geometry.computeBoundingBox();
    const res = await autoFit({
      head: headField, helmet: hf, settings: fitSettings,
      current: { position: a.position.clone(), quaternion: a.quaternion.clone(), scale: a.scale.clone() },
      headBox: viewer.headMesh.geometry.boundingBox!.clone(),
      signal: fitSignal,
      debug: (window as unknown as { HF_DEBUG?: boolean }).HF_DEBUG ? (m: string) => console.log(m) : undefined,
      onProgress: (f, msg) => {
        prog.value = f;
        $('fit-msg').textContent = msg;
        if (msg && msg !== 'Done') {
          updateHudTask(msg, /orientation/i.test(msg) ? 'AUTO-ROTATING…' : 'AUTO-FITTING…');
        }
      },
    });
    undoStack.push(before); redoStack.length = 0; updateUndoButtons();
    const hasLockedOrHidden = pieces.some((p) => !isPieceVisible(p) || isPieceLocked(p));
    if (hasLockedOrHidden) {
      a.updateMatrix();
      const targetAsm = new THREE.Matrix4().compose(res.pose.position, res.pose.quaternion, res.pose.scale);
      const delta = a.matrix.clone().invert().multiply(targetAsm);
      const pos = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
      for (const p of pieces) {
        if (!isPieceVisible(p) || isPieceLocked(p)) continue;
        p.mesh.updateMatrix();
        delta.clone().multiply(p.mesh.matrix).decompose(pos, q, sc);
        p.mesh.position.copy(pos); p.mesh.quaternion.copy(q); p.mesh.scale.copy(sc);
        p.mesh.updateMatrix(); p.mesh.updateMatrixWorld(true);
      }
      helmetFieldCache = null;
    } else {
      a.position.copy(res.pose.position);
      a.quaternion.copy(res.pose.quaternion);
      a.scale.copy(res.pose.scale);
      a.updateMatrixWorld(true);
      recentreAssembly(); // keep the helmet's pivot in its middle
    }
    const eyeMsg = res.eyes ? ` ${res.eyes.kind === 'visor' ? 'Visor' : 'Eye holes'} lined up (off by ${fmt(Math.hypot(res.eyes.dx, res.eyes.dy))} mm).` : fitSettings.useEyes ? ' No eye holes found — fitted by shape only.' : '';
    $('fit-msg').textContent = `Done. Average gap ${fmt(res.stats.meanGap)} mm, closest ${fmt(res.stats.minGap)} mm.${eyeMsg}`;
    viewer.frameAll();
    afterTransformChange(true);
  } catch (e) {
    $('fit-msg').textContent = (e as Error).message === 'cancelled' ? 'Cancelled.' : `Auto fit failed: ${(e as Error).message}`;
  } finally {
    btn.disabled = false; $('fit-cancel').hidden = true; prog.hidden = true; fitSignal = null;
    await endHudTask();
  }
}

function readFitSettings() {
  const num = (id: string, fallback: number) => { const v = parseFloat(($(id) as HTMLInputElement).value); return Number.isFinite(v) && v >= 0 ? v : fallback; };
  const chk = (id: string) => ($(id) as HTMLInputElement).checked;
  fitSettings = {
    targetGap: num('fit-target', DEFAULT_FIT_SETTINGS.targetGap),
    minGap: num('fit-min', DEFAULT_FIT_SETTINGS.minGap),
    searchRotation: chk('fit-rotation'),
    startFromCurrent: chk('fit-current'),
    lockScale: chk('fit-lockscale'),
    nonUniformScale: chk('fit-nonuniform'),
    useEyes: chk('fit-eyes'),
  };
  if (fitSettings.minGap > fitSettings.targetGap) fitSettings.minGap = fitSettings.targetGap;
  buildStops();
  afterTransformChange(true);
}

function writeFitSettings() {
  ($('fit-target') as HTMLInputElement).value = String(fitSettings.targetGap);
  ($('fit-min') as HTMLInputElement).value = String(fitSettings.minGap);
  ($('fit-rotation') as HTMLInputElement).checked = fitSettings.searchRotation;
  ($('fit-current') as HTMLInputElement).checked = fitSettings.startFromCurrent;
  ($('fit-lockscale') as HTMLInputElement).checked = fitSettings.lockScale;
  ($('fit-nonuniform') as HTMLInputElement).checked = fitSettings.nonUniformScale;
  ($('fit-eyes') as HTMLInputElement).checked = fitSettings.useEyes;
  buildStops();
}

// ---------------------------------------------------------------------------
// Export

let exportFormat: 'zip' | 'files' = 'zip';

function setExportFormat(fmtMode: 'zip' | 'files') {
  exportFormat = fmtMode;
  const printBtn = $('export-print');
  const wornBtn = $('export-worn');
  if (printBtn) {
    printBtn.textContent = fmtMode === 'zip' ? 'Download Scaled (Zip)' : 'Download Scaled pieces';
  }
  if (wornBtn) {
    wornBtn.textContent = fmtMode === 'zip' ? 'Download as worn (Zip)' : 'Download as worn';
  }
  qsa('[data-export-mode]').forEach((b) => b.classList.toggle('active', b.dataset.exportMode === fmtMode));
  const menu = $('export-mode-menu');
  const toggle = $('export-mode-toggle');
  if (menu) menu.hidden = true;
  if (toggle) toggle.setAttribute('aria-expanded', 'false');
}

function buildPieceExportMatrix(p: Piece, mode: 'print' | 'worn'): THREE.Matrix4 {
  p.mesh.updateMatrix();
  if (mode === 'worn') return viewer.assembly.matrixWorld.clone().multiply(p.mesh.matrix);
  // Scaled, back in the file's own (print) orientation, resting where the original did.
  const m = printMatrix(p);
  const orig = p.mesh.geometry.boundingBox!.clone().translate(p.origin); // where it sat in the file
  const moved = p.mesh.geometry.boundingBox!.clone().applyMatrix4(m);
  m.premultiply(new THREE.Matrix4().makeTranslation(orig.min.x - moved.min.x, orig.min.y - moved.min.y, orig.min.z - moved.min.z));
  return m;
}

function pieceExportFileName(p: Piece, mode: 'print' | 'worn'): string {
  const sc = printScale(p);
  const tag = mode === 'worn' ? 'as-worn' : sc ? `scaled-${fmt(sc.x * 100, 1)}pct` : 'scaled';
  const safe = p.name.replace(/[\\/:*?"<>|·]+/g, '_').trim() || 'piece';
  return `${safe}_${tag}.stl`;
}

async function exportPieces(mode: 'print' | 'worn', indices?: number[], forceFormat?: 'zip' | 'files', archiveLabel?: string) {
  if (!pieces.length) { alert('No pieces loaded.'); return; }
  const targetPieces = indices ? indices.map((i) => pieces[i]).filter(Boolean) : pieces;
  if (!targetPieces.length) return;
  const fmtMode = forceFormat ?? exportFormat;
  viewer.assembly.updateMatrixWorld(true);

  if (fmtMode === 'zip') {
    const zipEntries: Record<string, Uint8Array> = {};
    for (let i = 0; i < targetPieces.length; i++) {
      const p = targetPieces[i];
      const m = buildPieceExportMatrix(p, mode);
      let fileName = pieceExportFileName(p, mode);
      if (zipEntries[fileName]) {
        fileName = fileName.replace(/\.stl$/i, `_${i + 1}.stl`);
      }
      zipEntries[fileName] = exportSTLBytes(detailOf(p), m);
    }
    const current = currentProjectId ? await store.projects.get(currentProjectId) : undefined;
    const baseName = (archiveLabel ?? current?.name ?? pieces[0]?.name ?? 'helmet')
      .replace(/[\\/:*?"<>|·]+/g, '_')
      .trim() || 'helmet';
    const suffix = mode === 'worn' ? 'as-worn' : 'scaled';
    downloadBlob(exportZip(zipEntries), `${baseName}_${suffix}.zip`);
    status(`Downloaded ${targetPieces.length} piece(s) in "${baseName}_${suffix}.zip".`);
    return;
  }

  for (const p of targetPieces) {
    const m = buildPieceExportMatrix(p, mode);
    downloadBlob(exportSTL(detailOf(p), m), pieceExportFileName(p, mode));
    if (targetPieces.length > 1) await new Promise((r) => setTimeout(r, 350));
  }
  status(mode === 'print' ? 'Downloaded scaled pieces (millimetres) — print at 100%.' : 'Downloaded pieces as positioned on the head (millimetres).');
}

// ---------------------------------------------------------------------------
// Projects

async function refreshProjectList(selectId: string | null = currentProjectId) {
  const list = (await store.projects.list()).sort((a, b) => b.updatedAt - a.updatedAt); // most recent first
  const sel = $('project-select') as HTMLSelectElement;
  sel.innerHTML = '<option value="">— new project —</option>' + list.map((p) => `<option value="${p.id}">${escapeHtml(p.name)}</option>`).join('');
  sel.value = selectId ?? '';
}

/** The current workspace as a stored project (for the history list). */
function projectRecord(id: string, name: string): ProjectRecord {
  const rec: ProjectRecord = {
    id,
    name,
    headId: currentHeadId,
    groups: pieceGroups.map((g) => ({ ...g })),
    pieces: pieces.map((p) => ({ name: p.name, fileName: p.fileName, data: p.data, unit: helmetUnit, transform: readTransform(p.mesh), color: p.colour, visible: p.mesh.visible, lens: p.lens, part: p.part, centred: true, locked: !!p.locked, groupId: p.groupId ?? null })),
    assembly: readTransform(viewer.assembly),
    fitSettings: { ...fitSettings },
    updatedAt: Date.now(),
  };
  return rec;
}

/** Keep the current project in the history list (replacing an entry with the same name). */
async function rememberProject(name: string) {
  const existing = (await store.projects.list()).find((p) => p.name === name);
  const rec = projectRecord(existing?.id ?? currentProjectId ?? store.newId(), name);
  await store.projects.put(rec);
  currentProjectId = rec.id;
  currentProjectName = name;
  await refreshProjectList(rec.id);
}

/** File the project was last saved to / loaded from (Chrome/Edge), so Save updates it in place. */
let projectFileHandle: FileSystemFileHandle | null = null;
let currentProjectName: string | null = null;
type FilePickerWindow = Window & {
  showSaveFilePicker?: (o: unknown) => Promise<FileSystemFileHandle>;
  showOpenFilePicker?: (o: unknown) => Promise<FileSystemFileHandle[]>;
};
const PROJECT_FILE_TYPES = [{ description: 'HelmetFit project', accept: { 'application/json': ['.json'] } }];

/**
 * Save project: writes the project file (Chrome/Edge: to the file you pick
 * once, then updated in place; other browsers: a download) and keeps it in
 * the history list.
 */
async function saveProject() {
  if (!pieces.length && !currentProjectName) { alert('Add some pieces first.'); return; }
  const w = window as FilePickerWindow;
  const suggested = currentProjectName ?? (pieces[0]?.name ?? 'My helmet');
  let name = suggested;
  if (w.showSaveFilePicker) {
    try {
      // Must open straight from the click, so the file name you choose is the project name.
      if (!projectFileHandle) {
        projectFileHandle = await w.showSaveFilePicker({ suggestedName: `${safeFileName(suggested)}.helmetfit.json`, types: PROJECT_FILE_TYPES });
      }
      name = projectFileHandle.name.replace(/\.helmetfit\.json$|\.json$/i, '');
    } catch (e) {
      if ((e as Error).name === 'AbortError') return; // cancelled
      projectFileHandle = null; // fall back to a download
    }
  } else {
    const typed = prompt('Project name', suggested);
    if (!typed) return;
    name = typed;
  }
  if (!currentHeadId) status('Tip: save the head too, so the project remembers which head it was fitted to.');
  await rememberProject(name);
  const blob = await projectFileBlob(name);
  if (projectFileHandle) {
    const out = await (projectFileHandle as unknown as { createWritable: () => Promise<{ write: (b: Blob) => Promise<void>; close: () => Promise<void> }> }).createWritable();
    await out.write(blob);
    await out.close();
    status(`Saved project "${name}" to ${projectFileHandle.name}`);
  } else {
    downloadBlob(blob, `${safeFileName(name)}.helmetfit.json`);
    status(`Saved project "${name}" (downloaded ${safeFileName(name)}.helmetfit.json)`);
  }
}

const safeFileName = (n: string) => n.replace(/[\\/:*?"<>|·]+/g, '_').trim() || 'helmetfit-project';

/** Start over: no pieces, no project (saved projects stay in the history list). */
function newProject() {
  if (pieces.length && !confirm('Start a new project? Pieces in the current one are cleared — save first if you want to keep it. (Saved projects stay in the Project list.)')) return;
  if (snapMode) setSnapMode(false);
  clearPieces();
  pieceGroups.length = 0;
  snapLinks.length = 0;
  pendingProjectDoc = null;
  writeTransform(viewer.assembly, identityTransform());
  undoStack.length = 0; redoStack.length = 0; updateUndoButtons();
  currentProjectId = null; currentProjectName = null; projectFileHandle = null;
  ($('project-select') as HTMLSelectElement).value = '';
  $('fit-msg').textContent = '';
  for (const im of lastProjectImages.images) URL.revokeObjectURL(im.url);
  lastProjectImages.images = []; lastProjectImages.file = '';
  showProjectImages();
  renderPieceList();
  afterTransformChange(true);
  status('New project.');
}

async function loadProject(id: string) {
  const rec = await store.projects.get(id);
  if (!rec) return;
  await beginHudTask('LOADING PROJECT…', `Loading project "${rec.name}"…`);
  try {
    currentProjectId = id;
    currentProjectName = rec.name;
    projectFileHandle = null; // a different project: Save asks where to save it
    clearPieces();
    pieceGroups.push(...(rec.groups ?? []).map((g) => ({ ...g })));
    helmetUnit = rec.pieces[0]?.unit ?? 'mm';
    ($('helmet-unit') as HTMLSelectElement).value = helmetUnit;
    // Parts of one multi-part file share the same stored bytes: read each file once.
    const files = new Map<ArrayBuffer, Promise<LoadedPart[]>>();
    for (let i = 0; i < rec.pieces.length; i++) {
      const p = rec.pieces[i];
      try {
        if (!files.has(p.data)) {
          const is3mf = p.fileName.toLowerCase().endsWith('.3mf');
          const badge = is3mf ? 'UNZIPPING 3MF…' : 'LOADING PROJECT…';
          updateHudTask(is3mf ? `Unzipping ${p.fileName}…` : `Loading ${p.fileName} (${i + 1}/${rec.pieces.length})…`, badge);
          await yieldHudFrame();
          files.set(p.data, loadMeshParts(p.fileName, p.data, (m) => updateHudTask(`${p.fileName}: ${m}`, badge)));
        }
        const part = (await files.get(p.data)!)[p.part ?? 0];
        if (!part) throw new Error('part missing from file');
        // Each piece needs its own geometry (pieces get scaled in place on unit changes).
        const g = part.geometry.clone(), full = part.full?.clone() ?? null;
        const k = UNIT_TO_MM[p.unit];
        g.scale(k, k, k); full?.scale(k, k, k);
        addPiece(p.name, p.fileName, p.data, g, p.color, p.transform, p.visible, p.lens ?? false, full, p.part ?? 0, p.centred ?? false);
        const added = pieces[pieces.length - 1];
        added.locked = !!p.locked;
        added.groupId = p.groupId && pieceGroups.some((grp) => grp.id === p.groupId) ? p.groupId : null;
      } catch (e) {
        alert(`${p.fileName}: ${(e as Error).message}`);
      }
    }
    writeTransform(viewer.assembly, rec.assembly);
    recentreAssembly();
    showProjectImages();
    fitSettings = { ...DEFAULT_FIT_SETTINGS, ...rec.fitSettings };
    writeFitSettings();
    undoStack.length = 0; redoStack.length = 0; updateUndoButtons();
    if (rec.headId) {
      const h = await store.heads.get(rec.headId);
      if (h) { await loadHeadRecord(h); await refreshHeadList(h.id); }
    }
    helmetFieldCache = null;
    renderPieceList();
    afterTransformChange(true);
    viewer.frameAll();
    status(`Loaded project "${rec.name}"`);
  } finally {
    await endHudTask();
  }
}

interface ExportedProjectFile {
  format: 'helmetfit-project';
  version: 1;
  name: string;
  updatedAt: string;
  head: {
    tab: 'sample' | 'import';
    measUnit: 'mm' | 'cm' | 'in';
    measurements: HeadMeasurements;
    sampleEyes: [[number, number, number], [number, number, number]] | null;
    imported?: {
      fileName: string;
      filePath?: string;
      byteLength?: number;
      fileKey?: string;
      unit: LengthUnit;
      orientation: [number, number, number, number];
      extraScale: number;
      eyes: [[number, number, number], [number, number, number]] | null;
    };
  };
  helmetUnit: LengthUnit;
  assembly: TransformData;
  fitSettings: FitSettings;
  groups: PieceGroup[];
  pieces: Array<{
    name: string;
    fileName: string;
    filePath?: string;
    byteLength?: number;
    fileKey?: string;
    unit: LengthUnit;
    part: number;
    centred: boolean;
    geometry?: {
      boundingBox: {
        min: [number, number, number];
        max: [number, number, number];
        size: [number, number, number];
      };
      vertexCount: number;
      triangleCount: number;
      originOffset: [number, number, number];
    };
    transform: TransformData;
    printScale?: [number, number, number] | null;
    color: string;
    visible: boolean;
    lens: boolean;
    locked: boolean;
    groupId: string | null;
  }>;
  files?: Record<string, string>;
}

let pendingProjectDoc: ExportedProjectFile | null = null;
const fileBufferCache = new Map<string, ArrayBuffer>();
const filePathCache = new Map<string, string>();

async function rememberMeshFile(fileName: string, data: ArrayBuffer, knownPath?: string): Promise<string | undefined> {
  fileBufferCache.set(fileName, data);
  let resolvedPath = knownPath ?? filePathCache.get(fileName);
  if (!resolvedPath) {
    try {
      const resp = await fetch(`/api/local-mesh?name=${encodeURIComponent(fileName)}&size=${data.byteLength}&meta=1`);
      if (resp.ok) {
        const j = (await resp.json()) as { path?: string };
        if (j.path) resolvedPath = j.path;
      }
    } catch { /* ignore when offline or static build */ }
  }
  if (resolvedPath) {
    filePathCache.set(fileName, resolvedPath);
    for (const p of pieces) if (p.fileName === fileName && !p.filePath) p.filePath = resolvedPath;
    if (imported && imported.fileName === fileName && !imported.filePath) imported.filePath = resolvedPath;
  }
  try {
    await store.files.put({ fileName, filePath: resolvedPath, data, updatedAt: Date.now() });
  } catch { /* ignore IDB quota errors */ }
  return resolvedPath;
}

async function resolveMeshFileBuffer(
  fileName: string,
  hintPath?: string,
  expectedSize?: number,
  legacyB64?: string,
  existingMap?: Map<string, ArrayBuffer>,
): Promise<ArrayBuffer | null> {
  const fromMem = existingMap?.get(fileName) ?? fileBufferCache.get(fileName);
  if (fromMem) return fromMem;

  if (legacyB64) {
    const buf = base64ToBuffer(legacyB64);
    fileBufferCache.set(fileName, buf);
    existingMap?.set(fileName, buf);
    void rememberMeshFile(fileName, buf, hintPath);
    return buf;
  }

  try {
    const cached = await store.files.get(fileName);
    if (cached?.data) {
      fileBufferCache.set(fileName, cached.data);
      existingMap?.set(fileName, cached.data);
      if (cached.filePath) filePathCache.set(fileName, cached.filePath);
      return cached.data;
    }
  } catch { /* ignore IDB errors */ }

  try {
    const q = new URLSearchParams({ name: fileName });
    const p = hintPath ?? filePathCache.get(fileName);
    if (p) q.set('path', p);
    if (expectedSize && expectedSize > 0) q.set('size', String(expectedSize));
    const resp = await fetch(`/api/local-mesh?${q.toString()}`);
    if (resp.ok) {
      const hdrPath = resp.headers.get('X-Resolved-Path');
      const resolvedPath = hdrPath ? decodeURIComponent(hdrPath) : p;
      const buf = await resp.arrayBuffer();
      fileBufferCache.set(fileName, buf);
      existingMap?.set(fileName, buf);
      if (resolvedPath) filePathCache.set(fileName, resolvedPath);
      void rememberMeshFile(fileName, buf, resolvedPath);
      return buf;
    }
  } catch { /* ignore network errors */ }

  return null;
}

function base64ToBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function roundVec3(v: THREE.Vector3): [number, number, number] {
  return [+v.x.toFixed(4), +v.y.toFixed(4), +v.z.toFixed(4)];
}

function exportProjectData(name = 'helmetfit-project'): ExportedProjectFile {
  const measUnit = ($('meas-unit') as HTMLSelectElement).value as 'mm' | 'cm' | 'in';
  let importedHead: ExportedProjectFile['head']['imported'];
  if (imported) {
    const q = imported.orientation;
    const hPath = imported.filePath ?? filePathCache.get(imported.fileName);
    importedHead = {
      fileName: imported.fileName,
      ...(hPath ? { filePath: hPath } : {}),
      byteLength: imported.data.byteLength,
      unit: imported.unit,
      orientation: [+q.x.toFixed(6), +q.y.toFixed(6), +q.z.toFixed(6), +q.w.toFixed(6)],
      extraScale: imported.extraScale,
      eyes: imported.eyes ? [roundVec3(imported.eyes[0]), roundVec3(imported.eyes[1])] : null,
    };
  }
  const exportedPieces = pieces.map((p) => {
    const g = p.mesh.geometry;
    g.computeBoundingBox();
    const bb = g.boundingBox!;
    const sz = bb.getSize(new THREE.Vector3());
    const vCount = g.getAttribute('position')?.count ?? 0;
    const sc = printScale(p);
    const fPath = p.filePath ?? filePathCache.get(p.fileName);
    return {
      name: p.name,
      fileName: p.fileName,
      ...(fPath ? { filePath: fPath } : {}),
      byteLength: p.data.byteLength,
      unit: helmetUnit,
      part: p.part,
      centred: true,
      geometry: {
        boundingBox: {
          min: roundVec3(bb.min),
          max: roundVec3(bb.max),
          size: roundVec3(sz),
        },
        vertexCount: vCount,
        triangleCount: Math.floor(vCount / 3),
        originOffset: roundVec3(p.origin),
      },
      transform: readTransform(p.mesh),
      printScale: sc ? roundVec3(sc) : null,
      color: p.colour,
      visible: p.mesh.visible,
      lens: p.lens,
      locked: !!p.locked,
      groupId: p.groupId ?? null,
    };
  });
  return {
    format: 'helmetfit-project',
    version: 1,
    name,
    updatedAt: new Date().toISOString(),
    head: {
      tab: headTab,
      measUnit,
      measurements: { ...measurements },
      sampleEyes: sampleEyes ? [roundVec3(sampleEyes[0]), roundVec3(sampleEyes[1])] : null,
      ...(importedHead ? { imported: importedHead } : {}),
    },
    helmetUnit,
    assembly: readTransform(viewer.assembly),
    fitSettings: { ...fitSettings },
    groups: pieceGroups.map((g) => ({ ...g })),
    pieces: exportedPieces,
  };
}

/** The project file contents (settings, groups, pieces and where to find their mesh files). */
async function projectFileBlob(name: string): Promise<Blob> {
  for (const p of pieces) {
    if (!p.filePath && !filePathCache.has(p.fileName)) await rememberMeshFile(p.fileName, p.data);
  }
  return new Blob([JSON.stringify(exportProjectData(name), null, 2)], { type: 'application/json' });
}

async function importProjectData(raw: string | ExportedProjectFile) {
  const doc: ExportedProjectFile = typeof raw === 'string' ? JSON.parse(raw) : raw;
  if (!doc || doc.format !== 'helmetfit-project') {
    throw new Error('Not a valid HelmetFit project file.');
  }
  await beginHudTask('LOADING PROJECT…', `Loading project "${doc.name || 'project'}"…`);
  try {
    pendingProjectDoc = doc;
    // 1. Restore head settings & scan metadata if present.
    if (doc.head) {
      if (doc.head.measUnit) ($('meas-unit') as HTMLSelectElement).value = doc.head.measUnit;
      if (doc.head.measurements) measurements = { ...DEFAULT_MEASUREMENTS, ...doc.head.measurements };
      sampleEyes = doc.head.sampleEyes ? [new THREE.Vector3(...doc.head.sampleEyes[0]), new THREE.Vector3(...doc.head.sampleEyes[1])] : null;
      renderMeasurementFields();
      if (doc.head.imported) {
        const imp = doc.head.imported;
        if (!imported || imported.fileName !== imp.fileName) {
          const data = await resolveMeshFileBuffer(
            imp.fileName,
            imp.filePath,
            imp.byteLength,
            imp.fileKey ? doc.files?.[imp.fileKey] : undefined,
          );
          if (data) {
            const is3mf = imp.fileName.toLowerCase().endsWith('.3mf');
            const badge = is3mf ? 'UNZIPPING 3MF…' : 'LOADING STL…';
            updateHudTask(`Loading head scan ${imp.fileName}…`, badge);
            await yieldHudFrame();
            const base = mergedParts(await loadMeshParts(imp.fileName, data, (m) => updateHudTask(`${imp.fileName}: ${m}`, badge)));
            imported = {
              fileName: imp.fileName,
              filePath: imp.filePath ?? filePathCache.get(imp.fileName),
              data,
              unit: imp.unit,
              orientation: new THREE.Quaternion(...imp.orientation),
              extraScale: imp.extraScale ?? 1,
              eyes: imp.eyes ? [new THREE.Vector3(...imp.eyes[0]), new THREE.Vector3(...imp.eyes[1])] : null,
              base,
            };
          }
        } else if (imported && imported.fileName === imp.fileName) {
          imported.unit = imp.unit;
          imported.orientation.set(...imp.orientation);
          imported.extraScale = imp.extraScale ?? 1;
          imported.eyes = imp.eyes ? [new THREE.Vector3(...imp.eyes[0]), new THREE.Vector3(...imp.eyes[1])] : null;
        }
        if (imported) ($('head-unit') as HTMLSelectElement).value = imported.unit;
      }
      if (doc.head.tab === 'import' && imported) {
        headTab = 'import';
        qsa('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === 'import'));
        qsa('[data-tab-body]').forEach((b) => { b.hidden = b.dataset.tabBody !== 'import'; });
        viewer.showFrontArrow(true);
        rebuildImportedHead(false, false);
      } else {
        switchHeadTab('sample');
      }
    }
    // 2. Restore fit settings, helmet unit, and groups.
    fitSettings = { ...DEFAULT_FIT_SETTINGS, ...(doc.fitSettings ?? {}) };
    writeFitSettings();
    const nextUnit = doc.helmetUnit ?? doc.pieces?.[0]?.unit ?? 'mm';
    ($('helmet-unit') as HTMLSelectElement).value = nextUnit;
    if (pieces.length && nextUnit !== helmetUnit) setHelmetUnit(nextUnit);
    else helmetUnit = nextUnit;
    pieceGroups.length = 0;
    pieceGroups.push(...(doc.groups ?? []).map((g) => ({ ...g })));

    // 3. Apply transforms, groups, and properties to loaded pieces (or load from memory / IndexedDB / local hard drive).
    let matchedCount = 0;
    if (doc.pieces?.length) {
      const exactSamePieces = pieces.length === doc.pieces.length && doc.pieces.every((saved) =>
        pieces.some((p) => p.name === saved.name || (p.fileName === saved.fileName && p.part === (saved.part ?? 0))),
      );
      if (exactSamePieces) {
        updateHudTask(`Applying transforms & groups to ${doc.pieces.length} pieces…`, 'LOADING PROJECT…');
        for (const saved of doc.pieces) {
          const match = pieces.find((p) => p.name === saved.name || (p.fileName === saved.fileName && p.part === (saved.part ?? 0)));
          if (!match) continue;
          matchedCount++;
          writeTransform(match.mesh, saved.transform);
          match.colour = saved.color ?? match.colour;
          match.mesh.visible = saved.visible ?? true;
          match.lens = saved.lens ?? false;
          match.locked = !!saved.locked;
          match.groupId = saved.groupId && pieceGroups.some((grp) => grp.id === saved.groupId) ? saved.groupId : null;
          (match.mesh.material as THREE.MeshStandardMaterial).color.set(match.colour);
        }
      } else {
        // Resolve buffers from memory cache, IndexedDB (files & saved projects), or local hard drive (/api/local-mesh).
        const existingByFile = new Map<string, ArrayBuffer>();
        for (const [k, v] of fileBufferCache) existingByFile.set(k, v);
        for (const p of pieces) existingByFile.set(p.fileName, p.data);
        try {
          const savedProjs = await store.projects.list();
          for (const pr of savedProjs) {
            for (const pp of pr.pieces) if (pp.data && !existingByFile.has(pp.fileName)) existingByFile.set(pp.fileName, pp.data);
          }
        } catch { /* ignore IDB errors */ }

        const parsedFiles = new Map<string, Promise<LoadedPart[]>>();
        const resolvedBuffers = new Map<string, ArrayBuffer>();
        for (let i = 0; i < doc.pieces.length; i++) {
          const saved = doc.pieces[i];
          if (resolvedBuffers.has(saved.fileName)) continue;
          updateHudTask(`Locating ${saved.fileName} (${i + 1}/${doc.pieces.length})…`, 'LOADING PROJECT…');
          const buf = await resolveMeshFileBuffer(
            saved.fileName,
            saved.filePath,
            saved.byteLength,
            saved.fileKey ? doc.files?.[saved.fileKey] : undefined,
            existingByFile,
          );
          if (buf) resolvedBuffers.set(saved.fileName, buf);
        }

        if (resolvedBuffers.size > 0) {
          clearPieces();
          pieceGroups.push(...(doc.groups ?? []).map((g) => ({ ...g })));
          for (let i = 0; i < doc.pieces.length; i++) {
            const saved = doc.pieces[i];
            const buf = resolvedBuffers.get(saved.fileName);
            if (!buf) continue;
            if (!parsedFiles.has(saved.fileName)) {
              const is3mf = saved.fileName.toLowerCase().endsWith('.3mf');
              const badge = is3mf ? 'UNZIPPING 3MF…' : 'LOADING STL…';
              updateHudTask(
                is3mf ? `Unzipping ${saved.fileName}…` : `Loading ${saved.fileName} (${i + 1}/${doc.pieces.length})…`,
                badge,
              );
              await yieldHudFrame();
              parsedFiles.set(saved.fileName, loadMeshParts(saved.fileName, buf, (m) => updateHudTask(`${saved.fileName}: ${m}`, badge)));
            }
            const parts = await parsedFiles.get(saved.fileName)!;
            const part = parts[saved.part ?? 0];
            if (!part) continue;
            const g = part.geometry.clone(), full = part.full?.clone() ?? null;
            const k = UNIT_TO_MM[saved.unit ?? helmetUnit];
            g.scale(k, k, k); full?.scale(k, k, k);
            addPiece(saved.name, saved.fileName, buf, g, saved.color, saved.transform, saved.visible ?? true, saved.lens ?? false, full, saved.part ?? 0, saved.centred ?? true);
            const added = pieces[pieces.length - 1];
            added.filePath = saved.filePath ?? filePathCache.get(saved.fileName);
            added.locked = !!saved.locked;
            added.groupId = saved.groupId && pieceGroups.some((grp) => grp.id === saved.groupId) ? saved.groupId : null;
            matchedCount++;
          }
        }
      }
    }

    if (doc.assembly) {
      writeTransform(viewer.assembly, doc.assembly);
    }
    undoStack.length = 0; redoStack.length = 0; updateUndoButtons();
    helmetFieldCache = null;
    renderPieceList();
    afterTransformChange(true);
    viewer.frameAll();
    const totalExpected = doc.pieces?.length ?? 0;
    if (totalExpected > 0 && matchedCount < totalExpected) {
      const missing = doc.pieces.filter((sp) => !pieces.some((p) => p.name === sp.name || (p.fileName === sp.fileName && p.part === (sp.part ?? 0)))).map((sp) => sp.fileName);
      status(`Loaded project "${doc.name || 'project'}" (${matchedCount}/${totalExpected} pieces active — add ${missing.join(', ')} to place remaining pieces)`);
      alert(`Loaded project settings for "${doc.name || 'project'}", but could not automatically find these mesh files on your hard drive:\n\n• ${[...new Set(missing)].join('\n• ')}\n\nPlease select those STL/3MF files now and they will automatically orient and size to the project settings.`);
      ($('piece-files') as HTMLInputElement).click();
    } else {
      await rememberProject(doc.name || 'Project');
      status(`Loaded project "${doc.name || 'project'}"`);
    }
  } finally {
    await endHudTask();
  }
}

async function importProjectFile(file: File) {
  try {
    const text = await file.text();
    await importProjectData(text);
  } catch (e) {
    alert(`Could not load project file: ${(e as Error).message}`);
  }
}

// ---------------------------------------------------------------------------
// Utilities

const timers = new Map<string, number>();
function debounce(key: string, ms: number, fn: () => void) {
  clearTimeout(timers.get(key));
  timers.set(key, window.setTimeout(fn, ms));
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

// ---------------------------------------------------------------------------
// Wire up UI

function initUi() {
  // Head
  const presetSel = $('preset-select') as HTMLSelectElement;
  presetSel.innerHTML = '<option value="">Choose a preset…</option>' + Object.keys(HEAD_PRESETS).map((k) => `<option>${k}</option>`).join('');
  presetSel.addEventListener('change', () => {
    if (!presetSel.value) return;
    measurements = { ...HEAD_PRESETS[presetSel.value] };
    sampleEyes = null;
    renderMeasurementFields();
    rebuildSampleHead();
    presetSel.value = '';
  });
  $('meas-unit').addEventListener('change', () => {
    renderMeasurementFields();
    if (headTab === 'sample') rebuildSampleHead();
    else if (viewer.headMesh) showMeasured(viewer.headMesh.geometry);
  });
  $('fill-from-circ').addEventListener('click', () => {
    measurements = scaleFromCircumference(measurements.circumference);
    sampleEyes = null;
    renderMeasurementFields();
    rebuildSampleHead();
  });
  qsa('.tab').forEach((b) => b.addEventListener('click', () => switchHeadTab(b.dataset.tab as 'sample' | 'import')));
  ($('head-file') as HTMLInputElement).addEventListener('change', async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    const is3mf = f.name.toLowerCase().endsWith('.3mf');
    const badge = is3mf ? 'UNZIPPING 3MF…' : 'LOADING STL…';
    await beginHudTask(badge, is3mf ? `Unzipping head scan ${f.name}…` : `Loading head scan ${f.name}…`);
    try {
      const data = await f.arrayBuffer();
      let g: THREE.BufferGeometry;
      try {
        g = mergedParts(await loadMeshParts(f.name, data, (m) => updateHudTask(`${f.name}: ${m}`, badge)));
      } catch (err) {
        alert((err as Error).message);
        return;
      }
      const size = g.boundingBox!.getSize(new THREE.Vector3());
      const unit = is3mf ? 'mm' : guessUnit(Math.max(size.x, size.y, size.z), 250);
      ($('head-unit') as HTMLSelectElement).value = unit;
      imported = { fileName: f.name, data, unit, orientation: new THREE.Quaternion(), extraScale: 1, eyes: null, base: g };
      void rememberMeshFile(f.name, data);
      currentHeadId = null;
      ($('head-name') as HTMLInputElement).value = f.name.replace(/\.[^.]+$/, '');
      await refreshHeadList(null);
      updateHudTask(`Auto-rotating ${f.name} to upright & facing forward…`, 'AUTO-ROTATING…');
      await yieldHudFrame();
      rebuildImportedHead(true);
      status(`Imported ${f.name}. Check the blue arrow points out of the face; use the rotate buttons if not.`);
    } finally {
      (e.target as HTMLInputElement).value = '';
      await endHudTask();
    }
  });
  $('head-unit').addEventListener('change', () => { if (imported) { imported.unit = ($('head-unit') as HTMLSelectElement).value as LengthUnit; rebuildImportedHead(false, false); } });
  $('head-auto-orient').addEventListener('click', async () => {
    if (!imported) return;
    await beginHudTask('AUTO-ROTATING…', `Auto-rotating ${imported.fileName} to upright & facing forward…`);
    try {
      rebuildImportedHead(true);
    } finally {
      await endHudTask();
    }
  });
  qsa('[data-rot]').forEach((b) => b.addEventListener('click', () => {
    if (!imported) return;
    const axis = b.dataset.rot![0], sign = b.dataset.rot![1] === '+' ? -1 : 1;
    const v = new THREE.Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
    imported.orientation.premultiply(new THREE.Quaternion().setFromAxisAngle(v, (sign * Math.PI) / 2));
    rebuildImportedHead(false, false);
  }));
  $('head-apply-circ').addEventListener('click', () => {
    if (!imported || !viewer.headMesh) return;
    const want = parseFloat(($('head-target-circ') as HTMLInputElement).value);
    if (!Number.isFinite(want) || want < 300) { alert('Enter your head circumference in mm.'); return; }
    const have = measureHead(viewer.headMesh.geometry).circumference;
    imported.extraScale *= want / have;
    rebuildImportedHead(false, false);
  });
  $('head-clear-circ').addEventListener('click', () => { if (imported) { imported.extraScale = 1; rebuildImportedHead(false, false); } });
  $('eye-pick').addEventListener('click', startEyePick);
  $('eye-auto').addEventListener('click', () => {
    if (!headField) return;
    if (headTab === 'sample') sampleEyes = null;
    else if (imported) imported.eyes = null;
    headField.eyes = headEyesFor(headField);
    updateEyeStatus();
    afterTransformChange(true);
  });
  $('show-eyes').addEventListener('change', () => viewer.showEyeMarkers(($('show-eyes') as HTMLInputElement).checked));
  $('head-save').addEventListener('click', saveHead);
  $('head-select').addEventListener('change', async () => {
    const id = ($('head-select') as HTMLSelectElement).value;
    if (!id) return;
    const rec = await store.heads.get(id);
    if (rec) await loadHeadRecord(rec);
  });
  $('head-delete').addEventListener('click', async () => {
    const id = ($('head-select') as HTMLSelectElement).value;
    if (!id) return;
    const rec = await store.heads.get(id);
    if (!rec || !confirm(`Delete saved head "${rec.name}"? This can't be undone.`)) return;
    await store.heads.remove(id);
    if (currentHeadId === id) currentHeadId = null;
    await refreshHeadList(null);
    status(`Deleted head "${rec.name}"`);
  });
  $('head-export').addEventListener('click', () => {
    if (!viewer.headMesh) return;
    const name = ($('head-name') as HTMLInputElement).value.trim() || 'head';
    downloadBlob(exportSTL(viewer.headMesh.geometry, new THREE.Matrix4()), `${name}.stl`);
  });

  // Pieces
  ($('piece-files') as HTMLInputElement).addEventListener('change', async (e) => {
    const files = Array.from((e.target as HTMLInputElement).files ?? []);
    (e.target as HTMLInputElement).value = '';
    if (files.length) await addPieceFiles(files);
  });
  const grpBtn = $('group-add');
  grpBtn.addEventListener('click', () => addGroup());
  grpBtn.addEventListener('dragover', (e) => {
    if (draggedPieceIndex >= 0) { e.preventDefault(); grpBtn.classList.add('drag-over'); }
  });
  grpBtn.addEventListener('dragleave', () => grpBtn.classList.remove('drag-over'));
  grpBtn.addEventListener('drop', (e) => {
    e.preventDefault();
    grpBtn.classList.remove('drag-over');
    if (draggedPieceIndex >= 0 && pieces[draggedPieceIndex]) {
      addGroup(undefined, [draggedPieceIndex]);
      draggedPieceIndex = -1;
    }
  });
  const pl = $('piece-list');
  pl.addEventListener('dragover', (e) => {
    if (draggedPieceIndex >= 0 && !(e.target as HTMLElement).closest('.group-row, .in-group')) e.preventDefault();
  });
  pl.addEventListener('drop', (e) => {
    if (draggedPieceIndex >= 0 && !(e.target as HTMLElement).closest('.group-row, .in-group')) {
      e.preventDefault();
      assignPieceToGroup(draggedPieceIndex, null);
      draggedPieceIndex = -1;
    }
  });
  $('pieces-clear').addEventListener('click', removeAllPieces);
  $('lock-shown').addEventListener('click', () => {
    const shown = pieces.filter((p) => isPieceVisible(p));
    const lock = shown.some((p) => !isPieceLocked(p));
    for (const p of shown) {
      p.locked = lock;
      if (p.groupId) {
        const g = pieceGroups.find((grp) => grp.id === p.groupId);
        if (g) g.locked = lock;
      }
    }
    renderPieceList();
    status(lock ? `Locked ${shown.length} shown pieces/groups. Auto-assemble will now only place the others.` : 'Unlocked all shown pieces/groups.');
  });
  $('pics-prev').addEventListener('click', () => { const n = lastProjectImages.images.length; picIndex = (picIndex - 1 + n) % n; showProjectImages(); });
  $('pics-next').addEventListener('click', () => { picIndex = (picIndex + 1) % lastProjectImages.images.length; showProjectImages(); });
  $('pics-img').addEventListener('click', () => $('pics').classList.toggle('big'));
  $('pics-close').addEventListener('click', () => { $('pics').hidden = true; });
  $('pics-open').addEventListener('click', () => { picIndex = 0; showProjectImages(); });
  $('snap-mode').addEventListener('click', () => {
    if (!snapMode && pieces.length < 2) { alert('Add at least two pieces first.'); return; }
    if (!snapMode) snapKind = joinBy; // open in the model's joining method
    setSnapMode(!snapMode);
  });
  ($('join-by') as HTMLSelectElement).value = joinBy;
  $('join-by').addEventListener('change', () => {
    joinBy = ($('join-by') as HTMLSelectElement).value as typeof joinBy;
    localStorage.setItem('helmetfit.joinBy', joinBy);
    if (snapMode && snapKind !== 'points') { snapKind = joinBy; setSnapMode(true); }
  });
  ($('magnet-size') as HTMLSelectElement).value = magnetSize ? `${magnetSize.diameter}x${magnetSize.thickness}` : '';
  $('magnet-size').addEventListener('change', () => {
    const v = ($('magnet-size') as HTMLSelectElement).value;
    localStorage.setItem('helmetfit.magnetSize', v);
    magnetSize = parseMagnetSize(v);
    pocketCache.clear(); // look again with the new size
    if (snapMode && snapKind === 'magnets') setSnapMode(true);
  });
  $('snap-done').addEventListener('click', () => setSnapMode(false));
  qsa('[data-snap-kind]').forEach((b) => b.addEventListener('click', () => { snapKind = b.dataset.snapKind as typeof snapKind; setSnapMode(true); }));
  $('points-clear').addEventListener('click', () => setSnapMode(true));
  $('snap-next').addEventListener('click', () => { if (snapJoin) { snapJoin.index = (snapJoin.index + 1) % snapJoin.candidates.length; snapJoin.angle = 0; applySnap(); } });
  $('snap-ccw').addEventListener('click', () => { if (snapJoin) { snapJoin.angle -= 15; applySnap(); } });
  $('snap-cw').addEventListener('click', () => { if (snapJoin) { snapJoin.angle += 15; applySnap(); } });
  $('snap-undo').addEventListener('click', () => { if (snapJoin) { snapJoin = null; undo(); $('snap-adjust').hidden = true; $('snap-msg').textContent = 'Snap undone. Click a cut face on the piece that stays still.'; } });
  $('assemble-run').addEventListener('click', async () => {
    if (snapMode) setSnapMode(false);
    await runAutoAssemble(true);
    const hasLocked = pieces.some((p) => isPieceVisible(p) && isPieceLocked(p));
    if (headField && pieces.length && !hasLocked) await runAutoFit();
  });
  $('helmet-unit').addEventListener('change', () => { pushHistory(); setHelmetUnit(($('helmet-unit') as HTMLSelectElement).value as LengthUnit); });
  const autoAsmBox = $('auto-assemble-on-load') as HTMLInputElement;
  autoAsmBox.checked = localStorage.getItem('helmetfit.autoAssembleOnLoad') === '1';
  autoAsmBox.addEventListener('change', () => localStorage.setItem('helmetfit.autoAssembleOnLoad', autoAsmBox.checked ? '1' : '0'));
  const vp = $('viewport');
  let dragDepth = 0;
  vp.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; $('drop-hint').hidden = false; });
  vp.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('drop-hint').hidden = true; } });
  vp.addEventListener('dragover', (e) => e.preventDefault());
  vp.addEventListener('drop', async (e) => {
    e.preventDefault(); dragDepth = 0; $('drop-hint').hidden = true;
    const files = Array.from(e.dataTransfer?.files ?? []);
    if (files.length) await addPieceFiles(files);
  });

  // Manual adjust
  $('target-select').addEventListener('change', () => {
    const v = ($('target-select') as HTMLSelectElement).value;
    if (v === 'assembly') { target = 'assembly'; updateGizmo(); }
    else if (v.startsWith('group:')) selectGroup(v.slice(6));
    else selectPiece(+v);
  });
  qsa('[data-gizmo]').forEach((b) => b.addEventListener('click', () => { gizmoMode = b.dataset.gizmo as GizmoMode; updateGizmo(); }));
  qsa<HTMLInputElement>('[data-xf]').forEach((i) => i.addEventListener('change', () => readXformField(i)));
  $('xf-reset').addEventListener('click', () => {
    const o = targetObject();
    if (!o || isTargetLocked()) return;
    pushHistory();
    const gid = targetGroupId();
    if (gid) {
      const t = readTransform(o);
      t.rotationDeg = [0, 0, 0];
      t.scale = [1, 1, 1];
      writeTransform(o, t);
      applyGroupPivotDelta(gid);
      syncGroupPivot(gid);
    } else {
      writeTransform(o, identityTransform());
    }
    if (o !== viewer.assembly) helmetFieldCache = null;
    afterTransformChange(true);
  });
  $('xf-snap').addEventListener('click', () => {
    const o = targetObject();
    if (!o || isTargetLocked()) return;
    pushHistory();
    const t = readTransform(o);
    t.rotationDeg = t.rotationDeg.map((d) => Math.round(d / 90) * 90) as [number, number, number];
    writeTransform(o, t);
    const gid = targetGroupId();
    if (gid) {
      applyGroupPivotDelta(gid);
      syncGroupPivot(gid);
    }
    if (o !== viewer.assembly) helmetFieldCache = null;
    afterTransformChange(true);
  });

  // Fit
  ['fit-target', 'fit-min', 'fit-rotation', 'fit-current', 'fit-lockscale', 'fit-nonuniform', 'fit-eyes'].forEach((id) => $(id).addEventListener('change', readFitSettings));
  $('fit-run').addEventListener('click', runAutoFit);
  $('fit-cancel').addEventListener('click', () => { if (fitSignal) fitSignal.cancelled = true; });

  // View
  qsa('[data-view]').forEach((b) => b.addEventListener('click', () => viewer.setView(b.dataset.view as 'front')));
  $('frame-all').addEventListener('click', () => viewer.frameAll());
  $('show-head').addEventListener('change', () => viewer.setHeadVisible(($('show-head') as HTMLInputElement).checked));
  $('show-helmet').addEventListener('change', () => { viewer.assembly.visible = ($('show-helmet') as HTMLInputElement).checked; viewer.requestRender(); });
  $('show-heatmap').addEventListener('change', updateHeatmap);
  $('helmet-opacity').addEventListener('input', () => viewer.setHelmetOpacity(parseFloat(($('helmet-opacity') as HTMLInputElement).value)));
  const sectionChanged = () => {
    const mode = ($('section-mode') as HTMLSelectElement).value as SectionMode;
    const off = $('section-offset') as HTMLInputElement;
    off.disabled = mode === 'off';
    viewer.setSection(mode, parseFloat(off.value));
  };
  $('section-mode').addEventListener('change', sectionChanged);
  $('section-offset').addEventListener('input', sectionChanged);

  // Export
  $('export-print').addEventListener('click', () => exportPieces('print'));
  $('export-worn').addEventListener('click', () => exportPieces('worn'));
  const exportToggle = $('export-mode-toggle');
  const exportMenu = $('export-mode-menu');
  if (exportToggle && exportMenu) {
    exportToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      const open = !exportMenu.hidden;
      exportMenu.hidden = open;
      exportToggle.setAttribute('aria-expanded', String(!open));
    });
    qsa<HTMLButtonElement>('[data-export-mode]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const mode = (btn.dataset.exportMode as 'zip' | 'files') || 'zip';
        setExportFormat(mode);
      });
    });
    document.addEventListener('click', (e) => {
      if (!exportMenu.hidden && !(e.target as HTMLElement).closest('#export-split')) {
        exportMenu.hidden = true;
        exportToggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Projects
  $('project-save').addEventListener('click', saveProject);
  $('project-select').addEventListener('change', () => {
    const sel = $('project-select') as HTMLSelectElement;
    if (sel.value) { loadProject(sel.value); return; }
    newProject();
    if (pieces.length) sel.value = currentProjectId ?? ''; // cancelled: keep showing the current project
  });
  $('project-new').addEventListener('click', newProject);
  $('project-import').addEventListener('click', async () => {
    const w = window as FilePickerWindow;
    if (!w.showOpenFilePicker) { ($('project-file') as HTMLInputElement).click(); return; }
    try {
      // Chrome/Edge: keep the file so Save updates it in place.
      const [h] = await w.showOpenFilePicker({ types: PROJECT_FILE_TYPES });
      const f = await h.getFile();
      projectFileHandle = h;
      await importProjectFile(f);
    } catch (e) { if ((e as Error).name !== 'AbortError') alert(`Could not open the project: ${(e as Error).message}`); }
  });
  ($('project-file') as HTMLInputElement).addEventListener('change', async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    (e.target as HTMLInputElement).value = '';
    if (f) { projectFileHandle = null; await importProjectFile(f); }
  });
  $('undo').addEventListener('click', undo);
  $('redo').addEventListener('click', redo);

  window.addEventListener('keydown', (e) => {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); }
    else if (e.key === 'w') { gizmoMode = 'translate'; updateGizmo(); }
    else if (e.key === 'e') { gizmoMode = 'rotate'; updateGizmo(); }
    else if (e.key === 'r') { gizmoMode = 'scale'; updateGizmo(); }
  });
}

async function start() {
  initUi();
  writeFitSettings();
  renderMeasurementFields();
  updateUndoButtons();
  renderPieceList();
  try {
    const resp = await fetch('/sample-head.stl');
    if (resp.ok) {
      const g = new STLLoader().parse(await resp.arrayBuffer());
      g.computeVertexNormals();
      g.computeBoundingBox();
      g.computeBoundingSphere();
      setSampleHeadTemplate(g);
    }
  } catch { /* fall back to procedural head */ }
  await refreshProjectList();
  const lastId = localStorage.getItem('helmetfit.lastHead');
  const last = lastId ? await store.heads.get(lastId) : undefined;
  await refreshHeadList(last?.id ?? null);
  if (last) await loadHeadRecord(last);
  else switchHeadTab('sample');
  viewer.setView('iso');
  status('Enter your measurements (or import a head scan), then add helmet pieces on the right.');
}

start();

// Expose for automated checks in the browser console.
(window as unknown as { helmetfit: unknown }).helmetfit = {
  viewer, pieces, get groups() { return pieceGroups; }, get stats() { return lastStats; }, runAutoFit, addPieceFiles,
  addGroup, selectGroup, assignPieceToGroup, exportProjectData, importProjectData, renderPieceList,
  // For automated checks of snapping without screen clicks.
  cutFacesOf: (i: number) => cutFacesOf(pieces[i]), snapPick: onSnapPick, setSnapMode, runAutoAssemble,
};
