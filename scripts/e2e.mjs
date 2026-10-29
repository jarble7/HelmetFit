// Headless smoke test of the real UI. Usage: node scripts/e2e.mjs (dev server must be running)
import { chromium } from 'playwright';
import { readdirSync, mkdirSync } from 'node:fs';
const base = process.env.HOME + '/.cache/ms-playwright/';
const dir = readdirSync(base).filter((d) => d.startsWith('chromium_headless_shell')).sort().pop();
const exe = `${base}${dir}/chrome-headless-shell-linux64/chrome-headless-shell`;
mkdirSync('scripts/out', { recursive: true });
const browser = await chromium.launch({ executablePath: exe, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));
page.on('dialog', (d) => { console.log('DIALOG:', d.message()); d.accept(d.type() === 'prompt' ? 'E2E project' : undefined); });
await page.goto('http://127.0.0.1:5180/');
await page.waitForTimeout(1500);
await page.screenshot({ path: 'scripts/out/1-start.png' });
console.log('measured:', (await page.textContent('#head-measured')).replace(/\s+/g, ' '));

// Add the 3 demo pieces -> auto-fit runs automatically on first load.
await page.setInputFiles('#piece-files', ['samples/demo-helmet-crown.stl', 'samples/demo-helmet-back.stl', 'samples/demo-helmet-face.stl']);
const t0 = Date.now();
await page.waitForFunction(() => document.querySelector('#fit-msg')?.textContent?.startsWith('Done') || document.querySelector('#fit-msg')?.textContent?.startsWith('Auto fit failed'), null, { timeout: 180000 });
console.log('fit took', ((Date.now() - t0) / 1000).toFixed(1), 's:', await page.textContent('#fit-msg'));
console.log('report:', (await page.textContent('#report')).replace(/\s+/g, ' '));
await page.check('#show-heatmap');
await page.waitForTimeout(800);
await page.screenshot({ path: 'scripts/out/2-fitted-heatmap.png' });
await page.click('[data-view="side"]');
await page.selectOption('#section-mode', 'side');
await page.waitForTimeout(500);
await page.screenshot({ path: 'scripts/out/3-section.png' });
await page.selectOption('#section-mode', 'off');

// Manual adjust: scale up 5% via numeric field, then undo.
const before = await page.inputValue('[data-xf="s0"]');
await page.fill('[data-xf="s0"]', String(parseFloat(before) * 1.05));
await page.press('[data-xf="s0"]', 'Enter');
await page.waitForTimeout(300);
console.log('scale', before, '->', await page.inputValue('[data-xf="s1"]'), '(uniform lock)');
await page.click('#undo');
await page.waitForTimeout(300);
console.log('after undo', await page.inputValue('[data-xf="s0"]'));

// Save head + project.
await page.fill('#head-name', 'E2E head');
await page.click('#head-save');
await page.waitForTimeout(300);
await page.click('#project-save');
await page.waitForTimeout(500);
console.log('heads:', await page.$$eval('#head-select option', (o) => o.map((x) => x.textContent)));
console.log('projects:', await page.$$eval('#project-select option', (o) => o.map((x) => x.textContent)));

// Import the scan head (cm, Z-up, facing +X).
await page.click('[data-tab="import"]');
await page.setInputFiles('#head-file', 'samples/demo-head-scan.stl');
await page.waitForTimeout(1500);
console.log('scan unit:', await page.inputValue('#head-unit'), '| measured:', (await page.textContent('#head-measured')).replace(/\s+/g, ' '));
await page.click('[data-view="iso"]');
await page.waitForTimeout(400);
await page.screenshot({ path: 'scripts/out/4-scan-head.png' });
console.log('report on scan head:', (await page.textContent('#report')).replace(/\s+/g, ' '));

// Reload and check persistence.
await page.reload();
await page.waitForTimeout(1500);
await page.selectOption('#project-select', { label: 'E2E project' });
await page.waitForTimeout(2500);
console.log('after reload pieces:', await page.$$eval('#piece-list .name', (o) => o.map((x) => x.textContent)));
console.log('report after reload:', (await page.textContent('#report')).replace(/\s+/g, ' '));
await page.screenshot({ path: 'scripts/out/5-reloaded.png' });
// Fresh page: a face mask with two eye holes.
await page.goto('http://127.0.0.1:5180/');
await page.waitForTimeout(1500);
await page.click('[data-tab="sample"]');
await page.waitForTimeout(500);
await page.setInputFiles('#piece-files', ['samples/demo-mask.stl']);
await page.waitForFunction(() => /^(Done|Auto fit failed)/.test(document.querySelector('#fit-msg')?.textContent ?? ''), null, { timeout: 180000 });
console.log('mask fit:', await page.textContent('#fit-msg'));
console.log('mask report:', (await page.textContent('#report')).replace(/\s+/g, ' '));
await page.click('[data-view="front"]');
await page.waitForTimeout(500);
await page.screenshot({ path: 'scripts/out/6-mask-front.png' });
await page.click('[data-view="side"]');
await page.waitForTimeout(500);
await page.screenshot({ path: 'scripts/out/7-mask-side.png' });
// Set eyes by clicking: project the current eye markers to screen and click them, slightly offset.
await page.click('[data-view="front"]');
await page.waitForTimeout(400);
await page.click('#eye-pick');
await page.waitForTimeout(400);
const pts = await page.evaluate(() => {
  const v = window.helmetfit.viewer, c = v.renderer.domElement.getBoundingClientRect();
  return v.eyeMarkers.children.filter((m) => m.geometry.type === 'SphereGeometry').map((m) => {
    const p = m.position.clone(); p.x += 3; p.y += 2; p.project(v.camera);
    return [c.left + (p.x + 1) / 2 * c.width, c.top + (1 - p.y) / 2 * c.height];
  });
});
for (const [x, y] of pts) { await page.mouse.click(x, y); await page.waitForTimeout(300); }
console.log('eye pick:', await page.textContent('#eye-status'), '|', (await page.textContent('#report')).match(/(Eye holes|Visor)[^.]*\./)?.[0]);
console.log('ERRORS:', errors.length ? errors : 'none');
await browser.close();
