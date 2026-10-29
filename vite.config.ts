import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const recentDirs = new Set<string>();
const fileIndexCache = new Map<string, string[]>();

function getSearchRoots(): string[] {
  const home = os.homedir();
  const candidates = [
    ...recentDirs,
    'E:/3D Files',
    'D:/3D Files',
    path.join(home, 'Documents'),
    path.join(home, 'Downloads'),
    path.join(home, 'Desktop'),
    path.resolve(process.cwd(), 'samples'),
  ];
  return [...new Set(candidates.filter((d) => {
    try { return fs.existsSync(d) && fs.statSync(d).isDirectory(); } catch { return false; }
  }))];
}

function findFileOnDisk(fileName: string, hintPath?: string, expectedSize?: number): string | null {
  if (hintPath) {
    try {
      if (fs.existsSync(hintPath) && fs.statSync(hintPath).isFile()) {
        recentDirs.add(path.dirname(hintPath));
        return hintPath.replace(/\\/g, '/');
      }
    } catch { /* ignore */ }
  }

  const targetLower = path.basename(fileName).toLowerCase();
  for (const dir of recentDirs) {
    const direct = path.join(dir, path.basename(fileName));
    try {
      if (fs.existsSync(direct) && fs.statSync(direct).isFile()) {
        return direct.replace(/\\/g, '/');
      }
    } catch { /* ignore */ }
  }

  const cached = fileIndexCache.get(targetLower);
  if (cached?.length) {
    const valid = cached.filter((p) => { try { return fs.existsSync(p); } catch { return false; } });
    if (valid.length) {
      if (expectedSize && expectedSize > 0) {
        const sizeMatch = valid.find((p) => { try { return fs.statSync(p).size === expectedSize; } catch { return false; } });
        if (sizeMatch) return sizeMatch.replace(/\\/g, '/');
      }
      return valid[0].replace(/\\/g, '/');
    }
  }

  const matches: string[] = [];
  const skipDirs = new Set(['node_modules', '.git', 'AppData', 'Windows', 'Program Files', 'Program Files (x86)', '$RECYCLE.BIN', 'System Volume Information']);
  const walk = (dir: string, depth: number) => {
    if (depth > 6 || matches.length >= 10) return;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.isFile() && entry.name.toLowerCase() === targetLower) {
        const full = path.join(dir, entry.name);
        matches.push(full);
      } else if (entry.isDirectory() && !entry.name.startsWith('.') && !skipDirs.has(entry.name)) {
        walk(path.join(dir, entry.name), depth + 1);
      }
    }
  };

  for (const root of getSearchRoots()) {
    walk(root, 0);
    if (matches.length > 0 && expectedSize && matches.some((p) => { try { return fs.statSync(p).size === expectedSize; } catch { return false; } })) {
      break;
    }
  }

  if (!matches.length) return null;
  fileIndexCache.set(targetLower, matches);
  let chosen = matches[0];
  if (expectedSize && expectedSize > 0) {
    const exact = matches.find((p) => { try { return fs.statSync(p).size === expectedSize; } catch { return false; } });
    if (exact) chosen = exact;
  }
  recentDirs.add(path.dirname(chosen));
  return chosen.replace(/\\/g, '/');
}

export default defineConfig({
  server: { host: '127.0.0.1', port: 5180, strictPort: true },
  preview: { host: '127.0.0.1', port: 5181, strictPort: true },
  plugins: [
    {
      name: 'helmetfit-local-mesh-resolver',
      configureServer(server) {
        server.middlewares.use('/api/local-mesh', (req, res) => {
          try {
            const url = new URL(req.url ?? '', 'http://127.0.0.1');
            const name = url.searchParams.get('name') ?? '';
            const hintPath = url.searchParams.get('path') ?? undefined;
            const size = Number(url.searchParams.get('size') ?? 0) || undefined;
            const metaOnly = url.searchParams.get('meta') === '1';
            if (!name && !hintPath) {
              res.statusCode = 400;
              res.end('Missing file name');
              return;
            }
            // Only ever serve mesh files.
            const isMesh = (p: string) => /\.(stl|obj|3mf)$/i.test(p);
            if ((name && !isMesh(name)) || (hintPath && !isMesh(hintPath))) {
              res.statusCode = 403;
              res.end('Only STL / OBJ / 3MF files can be loaded');
              return;
            }
            const found = findFileOnDisk(name || path.basename(hintPath!), hintPath, size);
            if (!found) {
              res.statusCode = 404;
              res.end('Not found');
              return;
            }
            if (metaOnly) {
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ path: found }));
              return;
            }
            const stat = fs.statSync(found);
            res.setHeader('Content-Type', 'application/octet-stream');
            res.setHeader('Content-Length', String(stat.size));
            res.setHeader('X-Resolved-Path', encodeURIComponent(found));
            fs.createReadStream(found).pipe(res);
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      },
    },
  ],
});
