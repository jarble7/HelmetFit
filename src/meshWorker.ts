// Background reading of big 3MF files so the page stays responsive.
import { extract3MFImages, parse3MF, type ProjectImage } from './threeMF';
import { clusterSimplify, DISPLAY_TRIANGLE_BUDGET } from './simplify';

export interface WorkerPart { name: string; positions: Float32Array; index: Uint32Array; display: { positions: Float32Array; index: Uint32Array } | null; sourceMatrix?: number[] }
export type WorkerReply = { type: 'progress'; msg: string } | { type: 'done'; parts: WorkerPart[]; images: ProjectImage[] } | { type: 'error'; message: string };

self.onmessage = (e: MessageEvent<{ data: ArrayBuffer }>) => {
  const post = (m: WorkerReply, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);
  try {
    const parts = parse3MF(e.data.data, (msg) => post({ type: 'progress', msg }));
    const total = parts.reduce((k, p) => k + p.index.length / 3, 0);
    const out: WorkerPart[] = parts.map((p, i) => {
      let display: WorkerPart['display'] = null;
      if (total > DISPLAY_TRIANGLE_BUDGET) {
        post({ type: 'progress', msg: `Simplifying for display… ${i + 1}/${parts.length}` });
        const s = clusterSimplify(p, Math.max(2000, Math.round(((p.index.length / 3) * DISPLAY_TRIANGLE_BUDGET) / total)));
        if (s !== p) display = { positions: s.positions, index: s.index };
      }
      return { name: p.name, positions: p.positions, index: p.index, display, sourceMatrix: p.sourceMatrix };
    });
    let images: ProjectImage[] = [];
    try { images = extract3MFImages(e.data.data); } catch { /* pictures are optional */ }
    const transfer: Transferable[] = [];
    for (const p of out) { transfer.push(p.positions.buffer, p.index.buffer); if (p.display) transfer.push(p.display.positions.buffer, p.display.index.buffer); }
    for (const im of images) transfer.push(im.data.buffer);
    post({ type: 'done', parts: out, images }, transfer);
  } catch (err) {
    post({ type: 'error', message: (err as Error).message ?? String(err) });
  }
};
