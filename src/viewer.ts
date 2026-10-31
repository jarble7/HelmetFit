import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { MeshBVH, acceleratedRaycast } from 'three-mesh-bvh';

THREE.Mesh.prototype.raycast = acceleratedRaycast;

export type GizmoMode = 'translate' | 'rotate' | 'scale';
export type SectionMode = 'off' | 'side' | 'front' | 'top';

const SKIN = new THREE.Color('#d9a88a');

/** Scene, camera, controls and the two things we show: the head and the helmet assembly. */
export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly orbit: OrbitControls;
  readonly gizmo: TransformControls;
  readonly assembly = new THREE.Group();
  readonly headMaterial = new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.75, metalness: 0, side: THREE.DoubleSide });
  headMesh: THREE.Mesh | null = null;
  private frontArrow: THREE.ArrowHelper;
  private section = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
  private sectionMode: SectionMode = 'off';
  private helmetOpacity = 0.55;
  private pieceMaterials = new Set<THREE.MeshStandardMaterial>();
  private needsRender = true;
  /** Called continuously while the gizmo is dragged. */
  onGizmoChange: (() => void) | null = null;
  /** Called once when a gizmo drag finishes. */
  onGizmoEnd: (() => void) | null = null;
  /** Called when a helmet piece is clicked (index of child in assembly), or -1 for empty space; with the hit point and surface normal (world). */
  onPick: ((pieceIndex: number, hit?: { point: THREE.Vector3; normal: THREE.Vector3 }) => void) | null = null;
  /** When set, clicks pick a point on the head instead of selecting pieces. */
  onHeadPick: ((p: THREE.Vector3) => void) | null = null;
  readonly eyeMarkers = new THREE.Group();
  private eyeMarkersOn = true;

  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    this.renderer.setClearColor('#1d2127');
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(35, 1, 5, 20000);
    this.camera.position.set(420, 160, 560);

    this.scene.add(new THREE.HemisphereLight('#ffffff', '#445566', 1.6));
    const key = new THREE.DirectionalLight('#ffffff', 2.2);
    key.position.set(300, 500, 400);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight('#aaccff', 0.9);
    rim.position.set(-400, 200, -300);
    this.scene.add(rim);

    const grid = new THREE.GridHelper(800, 40, '#55606e', '#323a44');
    grid.position.y = -200;
    this.scene.add(grid);
    this.scene.add(this.assembly);

    this.frontArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 120), 70, 0x3b82f6, 20, 12);
    this.frontArrow.visible = false;
    this.scene.add(this.frontArrow);
    this.eyeMarkers.renderOrder = 10;
    this.scene.add(this.eyeMarkers);

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.target.set(0, 0, 0);
    this.orbit.enableDamping = true;
    this.orbit.addEventListener('change', () => this.requestRender());

    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmo.setSize(0.9);
    this.gizmo.addEventListener('dragging-changed', (e) => {
      this.orbit.enabled = !(e as unknown as { value: boolean }).value;
      if (!(e as unknown as { value: boolean }).value) this.onGizmoEnd?.();
    });
    this.gizmo.addEventListener('objectChange', () => this.onGizmoChange?.());
    this.gizmo.addEventListener('change', () => this.requestRender());
    this.scene.add(this.gizmo.getHelper());

    this.installPicking();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    const loop = () => {
      requestAnimationFrame(loop);
      this.orbit.update();
      if (this.needsRender) { this.needsRender = false; this.renderer.render(this.scene, this.camera); }
    };
    loop();
  }

  requestRender() { this.needsRender = true; }

  private resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  private installPicking() {
    const el = this.renderer.domElement;
    let down: { x: number; y: number } | null = null;
    el.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
    el.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4 || this.gizmo.dragging) { down = null; return; }
      down = null;
      const r = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      const rc = new THREE.Raycaster();
      rc.setFromCamera(ndc, this.camera);
      if (this.onHeadPick) {
        const hit = this.headMesh ? rc.intersectObject(this.headMesh)[0] : undefined;
        if (hit) this.onHeadPick(hit.point.clone());
        return;
      }
      // Only real, shown pieces (the assembly also holds helpers such as the group pivot).
      const visible = this.assembly.children.filter((c) => c.visible && (this.isPickable ? this.isPickable(c) : true));
      const hits = rc.intersectObjects(visible, true).filter((h) => this.sectionMode === 'off' || this.section.distanceToPoint(h.point) >= 0);
      if (!hits.length) { this.onPick?.(-1); return; }
      // Skip helper overlays (cut-face highlights) and use the piece they belong to.
      let o: THREE.Object3D | null = hits[0].object;
      while (o && o.parent !== this.assembly) o = o.parent;
      const h = hits[0];
      const normal = h.face ? h.face.normal.clone().transformDirection(h.object.matrixWorld) : new THREE.Vector3();
      // Ask the app which piece this object is — never rely on its position among the assembly's children.
      const index = o ? (this.pieceIndexOf ? this.pieceIndexOf(o) : this.assembly.children.indexOf(o)) : -1;
      this.onPick?.(index, { point: h.point.clone(), normal });
    });
  }

  setHead(geometry: THREE.BufferGeometry | null) {
    if (this.headMesh) { this.scene.remove(this.headMesh); this.headMesh.geometry.dispose(); this.headMesh = null; }
    if (geometry) {
      this.headMesh = new THREE.Mesh(geometry, this.headMaterial);
      this.headMesh.name = 'head';
      this.scene.add(this.headMesh);
      geometry.computeBoundingBox();
      const bb = geometry.boundingBox!;
      this.frontArrow.position.set(0, 0, bb.max.z + 10); // eye level
    }
    this.applyHeatmapColours(null);
    this.requestRender();
  }

  /** Small markers: cyan on the head's eyes, orange on the helmet's eye holes. Drawn on top. */
  setEyeMarkers(head: THREE.Vector3[] | null, helmet: THREE.Vector3[] | null) {
    for (const c of [...this.eyeMarkers.children]) { this.eyeMarkers.remove(c); (c as THREE.Mesh).geometry.dispose(); ((c as THREE.Mesh).material as THREE.Material).dispose(); }
    const add = (p: THREE.Vector3, colour: string, ring: boolean) => {
      const g = ring ? new THREE.TorusGeometry(7, 1.6, 8, 24) : new THREE.SphereGeometry(3.5, 16, 12);
      const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: colour, depthTest: false, transparent: true, opacity: 0.95 }));
      m.position.copy(p);
      m.renderOrder = 10;
      this.eyeMarkers.add(m);
    };
    for (const p of head ?? []) add(p, '#22d3ee', false);
    for (const p of helmet ?? []) add(p, '#fb923c', true);
    this.eyeMarkers.visible = this.eyeMarkersOn;
    this.requestRender();
  }

  private neckMesh: THREE.Mesh | null = null;
  /** Faint stand-in neck (for head models cut off at the chin); null hides it. */
  setVirtualNeck(neck: { x: number; z: number; radius: number; top: number; bottom: number } | null) {
    if (this.neckMesh) { this.scene.remove(this.neckMesh); this.neckMesh.geometry.dispose(); (this.neckMesh.material as THREE.Material).dispose(); this.neckMesh = null; }
    if (neck) {
      const h = neck.top - neck.bottom;
      const g = new THREE.CylinderGeometry(neck.radius, neck.radius, h, 32, 1, true);
      this.neckMesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: '#d9a88a', transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false, clippingPlanes: this.headMaterial.clippingPlanes }));
      this.neckMesh.position.set(neck.x, (neck.top + neck.bottom) / 2, neck.z);
      this.neckMesh.name = 'virtual-neck';
      this.scene.add(this.neckMesh);
    }
    this.requestRender();
  }

  showEyeMarkers(v: boolean) { this.eyeMarkersOn = v; this.eyeMarkers.visible = v; this.requestRender(); }

  showFrontArrow(v: boolean) { this.frontArrow.visible = v; this.requestRender(); }
  setHeadVisible(v: boolean) { if (this.headMesh) this.headMesh.visible = v; this.requestRender(); }

  /** Per-vertex colours for the clearance map; null restores plain skin. */
  applyHeatmapColours(colours: Float32Array | null) {
    if (!this.headMesh) return;
    const g = this.headMesh.geometry;
    if (colours) {
      g.setAttribute('color', new THREE.BufferAttribute(colours, 3));
      this.headMaterial.vertexColors = true;
      this.headMaterial.color.set('#ffffff');
    } else {
      g.deleteAttribute('color');
      this.headMaterial.vertexColors = false;
      this.headMaterial.color.copy(SKIN);
    }
    this.headMaterial.needsUpdate = true;
    this.requestRender();
  }

  /** Which piece an assembly child is (index in the app's piece list), or -1. */
  pieceIndexOf: ((o: THREE.Object3D) => number) | null = null;
  /** Whether an assembly child can be clicked (e.g. not in a hidden group). */
  isPickable: ((o: THREE.Object3D) => boolean) | null = null;

  /** Rebuild a piece's fast hit-testing data after its geometry was changed in place (e.g. rescaled). */
  refreshPickData(geometry: THREE.BufferGeometry) {
    const bvh = new MeshBVH(geometry);
    geometry.userData.bvh = bvh;
    (geometry as THREE.BufferGeometry & { boundsTree?: MeshBVH }).boundsTree = bvh;
  }

  /** Create the display object for a piece; caller positions it. */
  makePieceObject(geometry: THREE.BufferGeometry, colour: string): THREE.Mesh {
    if (!geometry.userData.bvh) {
      const bvh = new MeshBVH(geometry);
      geometry.userData.bvh = bvh;
      (geometry as THREE.BufferGeometry & { boundsTree?: MeshBVH }).boundsTree = bvh;
    }
    const mat = new THREE.MeshStandardMaterial({
      color: colour, roughness: 0.5, metalness: 0.1, side: THREE.DoubleSide,
      transparent: true, opacity: this.helmetOpacity, depthWrite: this.helmetOpacity > 0.95,
      clippingPlanes: this.sectionMode === 'off' ? [] : [this.section],
    });
    this.pieceMaterials.add(mat);
    const m = new THREE.Mesh(geometry, mat);
    m.renderOrder = 2;
    return m;
  }

  disposePieceObject(m: THREE.Mesh) {
    const mat = m.material as THREE.MeshStandardMaterial;
    this.pieceMaterials.delete(mat);
    mat.dispose();
    m.geometry.dispose();
  }

  setHelmetOpacity(o: number) {
    this.helmetOpacity = o;
    for (const m of this.pieceMaterials) { m.opacity = o; m.transparent = o < 0.999; m.depthWrite = o > 0.95; m.needsUpdate = true; }
    this.requestRender();
  }

  setPieceHighlight(mesh: THREE.Mesh, on: boolean) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    mat.emissive.set(on ? '#335577' : '#000000');
    this.requestRender();
  }

  setSection(mode: SectionMode, offset: number) {
    this.sectionMode = mode;
    const normals: Record<Exclude<SectionMode, 'off'>, THREE.Vector3> = {
      side: new THREE.Vector3(-1, 0, 0), front: new THREE.Vector3(0, 0, -1), top: new THREE.Vector3(0, -1, 0),
    };
    const planes = mode === 'off' ? [] : [this.section];
    if (mode !== 'off') this.section.set(normals[mode], offset);
    this.headMaterial.clippingPlanes = planes;
    this.headMaterial.needsUpdate = true;
    for (const m of this.pieceMaterials) { m.clippingPlanes = planes; m.needsUpdate = true; }
    this.requestRender();
  }

  attachGizmo(obj: THREE.Object3D | null, mode: GizmoMode) {
    if (obj) { this.gizmo.attach(obj); this.gizmo.setMode(mode); } else this.gizmo.detach();
    this.requestRender();
  }

  setView(which: 'front' | 'side' | 'top' | 'back' | 'iso') {
    const t = new THREE.Vector3(0, 0, 0);
    const d = 700;
    const dirs = {
      front: new THREE.Vector3(0, 0.05, 1), back: new THREE.Vector3(0, 0.05, -1), side: new THREE.Vector3(1, 0.05, 0),
      top: new THREE.Vector3(0, 1, 0.001), iso: new THREE.Vector3(0.6, 0.25, 0.8),
    };
    this.orbit.target.copy(t);
    this.camera.position.copy(t).addScaledVector(dirs[which].normalize(), d);
    this.camera.lookAt(t);
    this.requestRender();
  }

  /** Frame the head and helmet. */
  frameAll() {
    const box = new THREE.Box3();
    if (this.headMesh) box.expandByObject(this.headMesh);
    for (const c of this.assembly.children) if (c.visible) box.expandByObject(c);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3());
    const r = box.getSize(new THREE.Vector3()).length() / 2;
    const dir = this.camera.position.clone().sub(this.orbit.target).normalize();
    this.orbit.target.copy(c);
    this.camera.position.copy(c).addScaledVector(dir, r / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.1);
    this.requestRender();
  }
}
