import type { HeadRecord, ProjectRecord } from './types';

// Everything is stored in the browser's IndexedDB on this machine — nothing is uploaded.

const DB_NAME = 'helmetfit';
const DB_VERSION = 2;
type StoreName = 'heads' | 'projects' | 'files';

export interface CachedFileRecord {
  fileName: string;
  filePath?: string;
  data: ArrayBuffer;
  updatedAt: number;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function db(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const d = req.result;
        if (!d.objectStoreNames.contains('heads')) d.createObjectStore('heads', { keyPath: 'id' });
        if (!d.objectStoreNames.contains('projects')) d.createObjectStore('projects', { keyPath: 'id' });
        if (!d.objectStoreNames.contains('files')) d.createObjectStore('files', { keyPath: 'fileName' });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

function tx<T>(store: StoreName, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return db().then((d) => new Promise<T>((resolve, reject) => {
    const t = d.transaction(store, mode);
    const req = fn(t.objectStore(store));
    t.oncomplete = () => resolve(req.result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  }));
}

export function newId(): string {
  return crypto.randomUUID();
}

export const heads = {
  list: () => tx<HeadRecord[]>('heads', 'readonly', (s) => s.getAll() as IDBRequest<HeadRecord[]>)
    .then((r) => r.sort((a, b) => a.name.localeCompare(b.name))),
  get: (id: string) => tx<HeadRecord | undefined>('heads', 'readonly', (s) => s.get(id)),
  put: (rec: HeadRecord) => tx('heads', 'readwrite', (s) => s.put(rec)),
  remove: (id: string) => tx('heads', 'readwrite', (s) => s.delete(id)),
};

export const projects = {
  list: () => tx<ProjectRecord[]>('projects', 'readonly', (s) => s.getAll() as IDBRequest<ProjectRecord[]>)
    .then((r) => r.sort((a, b) => a.name.localeCompare(b.name))),
  get: (id: string) => tx<ProjectRecord | undefined>('projects', 'readonly', (s) => s.get(id)),
  put: (rec: ProjectRecord) => tx('projects', 'readwrite', (s) => s.put(rec)),
  remove: (id: string) => tx('projects', 'readwrite', (s) => s.delete(id)),
};

export const files = {
  get: (fileName: string) => tx<CachedFileRecord | undefined>('files', 'readonly', (s) => s.get(fileName)),
  put: (rec: CachedFileRecord) => tx('files', 'readwrite', (s) => s.put(rec)),
};
