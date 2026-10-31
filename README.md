# HelmetFit

Size a 3D-printed helmet or mask to your head, entirely in your browser (nothing is uploaded).

## Start it

```bash
npm install        # first time only
npm run dev        # serves http://127.0.0.1:5180
```

## How to use

1. **Your head** (left panel)
   - *Sample head + measurements*: pick a preset (`Head-1` to `Head-4`, or S/M/L/XL), or type your measurements (hover the **?** for how to measure). If you only know your head circumference, enter it and click *Estimate rest from circumference*.
   - *Import head scan*: load an STL/OBJ/3MF of a head. It's auto-oriented; the blue arrow must point out of the face — use the rotate buttons if it doesn't. Optionally rescale it to your real circumference.
   - Name it and click **Save head**. Saved heads are in the *Saved heads* list.
2. **Helmet / mask pieces** (right panel): add all piece files at once (or drag them onto the 3D view). Click **+ New group** and **click-and-drag** pieces onto a group folder (like Photoshop / Illustrator layers) to organize pre-assembled sections, or drag pieces out to ungroup.
3. **Auto fit** finds the size, position and rotation that leaves the requested gap (padding) between head and helmet without touching, and lines the helmet's eye holes (or visor slot) up with your eyes. X/Y/Z are scaled separately by default. Options: keep current size, start from where it is now, uniform scale only, ignore the eyes.
   - Eyes on the head show as cyan dots; the eye holes/visor the app found show as orange rings. If the head's eyes are wrong (e.g. on a scan), click **Set eyes by clicking** and click each eye.
4. **Adjust by hand**: move/rotate/scale the whole helmet, a layer group, or a single piece with the on-screen handles (W/E/R keys) or by typing numbers. Ctrl+Z / Ctrl+Y undo/redo.
5. **Fit report & print**: shows the coverage and gaps, and the **scale % to print your original files at**. *Download scaled pieces* gives STLs already scaled (print at 100%, original orientation). *Download as worn* gives them positioned on the head.

View tools: *Gap colours* paints the head red (too tight) → green (target) → blue (loose), grey where nothing covers it. *Cut-away* slices the view to see the gap in profile.

**Projects** (top bar): **Save project** saves a portable `.helmetfit.json` project file (your head settings, fit settings, layer groups and piece placements) named after the project; later saves update the same file. **Load project…** opens a project file. Every project you save or load is remembered in the project list, so you can switch back to it from there. **New project** starts an empty one (saved projects stay in the list).

## Auto-assemble: Join by and Magnet size

Before clicking **Auto-assemble pieces**, set **Join by** to how the model's pieces fit together — **Magnets** (round magnet pockets) or **Cut faces** (flat cuts). A model uses one or the other, and only that is searched for, which makes assembly and snapping much faster. **Magnet size** is optional: if you know the magnets the model was designed for (e.g. 8×2 mm), pick it and pockets are found more reliably; leave it on *Auto* otherwise. Both settings are remembered, and **Snap pieces together…** starts in the same mode.

## Groups & Auto-Assemble Rules

Groups let you combine pieces you've already fitted together into a single section (like a pre-assembled portion of a puzzle) using **+ New group** and click-and-drag. When **Auto-assemble pieces** runs, the following rules hold:

- **All groups stay together**, and do not get disassembled; unlocked groups can, however, get resized as a whole.
- **Locking** preserves coordinate status relative to other locked pieces and size.
- **Locked groups** do not get resized or disassociated.
- **Different groups are assembled together** with other groups — each group acts like a portion of an assembled puzzle, fitting multiple pre-assembled larger sections together.
- **Single items** not part of a group or locked maintain the same rules as before.
- **Hidden items or groups** do not participate in the Auto-Assemble.

## Demo files

`samples/` has a 3-piece full-face demo helmet with a T-visor (exported Z-up and oversized on purpose), a demo face mask with two eye holes (exported rotated and at 93%), and a demo head scan (in cm, sideways) to try the auto features. Regenerate with `npx vite-node scripts/make-sample-helmet.ts` / `scripts/make-sample-head.ts`.

## Development

- `npm test` — unit tests (head model accuracy, auto-orient, auto-fit recovery of known transforms)
- `node scripts/e2e.mjs` — headless browser run-through of the UI (needs `npm run dev` running); screenshots in `scripts/out/`
- `npm run build` — typecheck + production build

How the fit is measured: rays are cast from a point inside the skull; for each direction the gap is the distance from the head surface to the first helmet surface. The optimiser (Nelder–Mead, after trying all 24 axis-aligned orientations) minimises deviation from the target gap with a heavy penalty below the minimum gap.

## Limitations

- **Lock** pieces you've placed (🔒 in the piece list, or **Lock all shown**): auto-assemble then only places the unlocked pieces onto them, and snapping / the move handles leave locked pieces alone. Locks are saved with projects.
- **Layer groups** (**+ New group**): pieces inside the same group stay rigidly locked together relative to each other when auto-assembling or snapping groups onto other groups or pieces.
- Magnet pockets (small blind round holes) and alignment pegs/studs with matching sockets are found automatically; pieces whose connectors line up face to face with the same spacing are joined by them. Auto-assemble uses connectors or flat cuts, whichever **Join by** is set to; pieces it can't place are hidden until you place them by hand.
- Pieces laid out flat for printing are put back together automatically (also **Auto-assemble pieces** under the piece list) by matching their flat cut faces. It needs pieces that were cut apart with flat cuts; anything it can't place is listed. To fix or finish an assembly by hand, click **Snap pieces together…**: cut faces turn yellow; click a face on the piece that stays put, then the matching face on the piece to move, and it snaps face-to-face (pieces already joined to it come along). **Next fit** tries the next way the outlines line up, **⟲/⟳ 15°** turns it around the joint (for faces with pegs or uneven outlines). For parts with magnet pockets or pegs/sockets, **Match by → Magnets**: click the fixed piece, then the piece to join, and their connectors are lined up (pink rings show the connectors found). If that can't work it out, click the rings themselves — a connector on the fixed piece, then the connector it joins on the moving piece; 2 pairs fix the joint exactly, and it then refits through every connector that lines up. If a pocket has no ring (it wasn't detected), click inside it anyway and it's measured where you click. Joins you've already made are kept: pieces snapped together move and match as one, so a second join doesn't undo the first. For parts that overlap or interlock with no pockets, switch **Match by → Points**: click a spot on the fixed piece, then the same spot on the piece to move; 3+ spread-out pairs give an exact fit. **Auto-assemble** afterwards keeps your snaps and fills in the rest.
- Bambu Studio / Orca projects record where each part sat in the file it came from; when that's there, the parts are put straight back into place (no matching needed). The project's pictures (designer photos, plate previews) show in a small viewer in the top-right corner of the 3D view — ‹ › to step through, click to enlarge.
- Each piece rotates and scales about its own middle, and "Whole helmet" about the middle of the helmet, wherever the file's origin is.
- 3MF files (including Bambu Studio / Orca projects) are read in the background in chunks, so large files don't freeze or crash the page; each part inside becomes its own piece, named as in the slicer. Very detailed models (over ~2.5 million triangles) are shown and fitted with a simplified copy; downloads keep full detail. For pieces that were turned for printing, a stretched (X/Y/Z) fit can't be typed into a slicer as percentages — use **Download scaled pieces**, which keeps each piece in its print orientation.
- Eye openings are found by looking at the outside of the helmet for openings with helmet to their left, right and above: two side-by-side holes, or one slot no taller than ~55 mm (T-visors and Corinthian/Spartan eye openings included). A big open face isn't used for eye alignment, and helmets with a solid visor fit by shape only.
- Lens / eye-slot inserts (thin pieces tucked inside the shell) are spotted automatically and marked **lens** in the piece list; eye detection looks through them. Click **lens** to change it.
- If the head model stops at the chin, a faint stand-in neck is added while fitting, so a helmet can't "fit" tipped back with the face through its neck hole.
- Imported heads: if the model has a flat cut at the neck it's treated as the bottom; otherwise orientation is guessed from shape. Eye height is detected; eye left/right spacing uses typical proportions (pupils ≈ 16% of head length from the midline). Use "Set eyes by clicking" if they're off.
- Gaps are measured radially from the centre of the head, so they're approximate around the nose and ears.
- Very large scans (millions of triangles) make the live gap colours slow; decimate them first.
