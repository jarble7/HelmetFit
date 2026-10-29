// Shared types. All lengths are millimetres. World frame: +Y up, +Z is the
// direction the face looks, +X is the wearer's left. The head origin sits at
// the ear-canal (tragion) level, centred between the ears.

export interface HeadMeasurements {
  /** Tape around the head just above the eyebrows and ears (widest loop). */
  circumference: number;
  /** Front-to-back: brow (between eyebrows) to the back of the skull. */
  length: number;
  /** Side-to-side: widest point above the ears. */
  breadth: number;
  /** Bottom of chin to top of head. */
  height: number;
  /** Ear canal to top of head (vertical). */
  earToTop: number;
  /** Cheekbone to cheekbone. */
  faceWidth: number;
  /** Jaw corner to jaw corner. */
  jawWidth: number;
  /** Around the neck. */
  neckCircumference: number;
  /** How far the nose tip sticks out from the face. */
  noseProtrusion: number;
  /** How far the ears stick out from the side of the head. */
  earProtrusion: number;
}

export type LengthUnit = 'mm' | 'cm' | 'in' | 'm';

export const UNIT_TO_MM: Record<LengthUnit, number> = { mm: 1, cm: 10, in: 25.4, m: 1000 };

/** Plain-data transform (serialisable). Rotation is Euler XYZ in degrees. */
export interface TransformData {
  position: [number, number, number];
  rotationDeg: [number, number, number];
  scale: [number, number, number];
}

export interface ParametricHeadRecord {
  id: string;
  name: string;
  kind: 'parametric';
  measurements: HeadMeasurements;
  /** Eye positions picked by hand, if the defaults were off. */
  eyes?: [[number, number, number], [number, number, number]] | null;
  updatedAt: number;
}

export interface MeshHeadRecord {
  id: string;
  name: string;
  kind: 'mesh';
  fileName: string;
  /** Original uploaded file bytes. */
  data: ArrayBuffer;
  unit: LengthUnit;
  /** Orientation fix applied after unit conversion (applied before auto-centring). */
  orientation: TransformData;
  /** Optional uniform rescale so the head matches a known circumference. */
  extraScale: number;
  /** Eye positions picked by hand (final head frame), if auto-detection was wrong. */
  eyes?: [[number, number, number], [number, number, number]] | null;
  updatedAt: number;
}

export type HeadRecord = ParametricHeadRecord | MeshHeadRecord;

export interface PieceGroup {
  id: string;
  name: string;
  collapsed?: boolean;
  locked?: boolean;
  visible?: boolean;
}

export interface PieceRecord {
  name: string;
  fileName: string;
  data: ArrayBuffer;
  unit: LengthUnit;
  /** Transform of the piece inside the helmet assembly. */
  transform: TransformData;
  color: string;
  visible: boolean;
  /** See-through insert; eye detection looks through it. */
  lens?: boolean;
  /** Part number within a multi-part file (3MF). Parts of one file share `data`. */
  part?: number;
  /** `transform` is for the piece re-centred on its middle (saved by newer versions). */
  centred?: boolean;
  /** Locked in place by the user. */
  locked?: boolean;
  /** ID of the PieceGroup this piece belongs to (null/undefined if ungrouped). */
  groupId?: string | null;
}

export interface ProjectRecord {
  id: string;
  name: string;
  headId: string | null;
  pieces: PieceRecord[];
  groups?: PieceGroup[];
  assembly: TransformData;
  fitSettings: FitSettings;
  updatedAt: number;
}

export interface FitSettings {
  /** Desired gap between head and helmet inner wall (padding / foam). */
  targetGap: number;
  /** Anything closer than this counts as a pressure point. */
  minGap: number;
  /** Allow different X/Y/Z scale factors. */
  nonUniformScale: boolean;
  /** Try all 24 axis-aligned orientations (otherwise keep current rotation). */
  searchRotation: boolean;
  /** Keep the current scale; only move/rotate. */
  lockScale: boolean;
  /** Start from the current placement instead of guessing one. */
  startFromCurrent: boolean;
  /** Line up the helmet's eye holes / visor with the head's eyes. */
  useEyes: boolean;
}

export const DEFAULT_FIT_SETTINGS: FitSettings = {
  targetGap: 10,
  minGap: 3,
  nonUniformScale: true,
  searchRotation: true,
  lockScale: false,
  startFromCurrent: false,
  useEyes: true,
};

export interface FitStats {
  /** Fraction (0..1) of head directions that the helmet covers. */
  coverage: number;
  minGap: number;
  meanGap: number;
  maxGap: number;
  /** Fraction of covered directions closer than minGap (incl. penetration). */
  tightFraction: number;
  /** Deepest penetration in mm (0 if none). */
  penetration: number;
  /** Number of sampled directions. */
  samples: number;
}

export function identityTransform(): TransformData {
  return { position: [0, 0, 0], rotationDeg: [0, 0, 0], scale: [1, 1, 1] };
}
