import type { StorageAdapter } from "../core/adapters";

export interface HtmlCacheOptions {
  rootDir: string;
  expiryDays: number;
  storage: StorageAdapter;
}

interface CacheEnvelope {
  schemaVersion: 1;
  cachedAt: string;
  content: string;
}

export class HtmlCache {
  constructor(private opts: HtmlCacheOptions) {}

  async get(key: string, kind: "html" | "abs"): Promise<string | null> {
    const path = this.pathFor(key, kind);
    try {
      if (!(await this.opts.storage.exists(path))) return null;
      const envelope = parseEnvelope(await this.opts.storage.readText(path));
      if (!envelope || isExpired(envelope.cachedAt, this.opts.expiryDays)) {
        await this.opts.storage.remove(path).catch(() => {});
        return null;
      }
      return envelope.content;
    } catch {
      return null;
    }
  }

  async set(key: string, kind: "html" | "abs", content: string): Promise<void> {
    const path = this.pathFor(key, kind);
    await ensureDirDeep(this.opts.storage, parentDir(path));
    const envelope: CacheEnvelope = {
      schemaVersion: 1,
      cachedAt: new Date().toISOString(),
      content,
    };
    await this.opts.storage.writeText(path, `${JSON.stringify(envelope)}\n`);
  }

  async cleanupExpired(): Promise<number> {
    const storage = this.opts.storage;
    if (!storage.list) return 0;
    let removed = 0;
    for (const kind of ["html", "abs"] as const) {
      const dir = storage.normalizePath(`${this.opts.rootDir}/${kind}`);
      if (!(await storage.exists(dir))) continue;
      for (const entry of await storage.list(dir)) {
        if (entry.type !== "file") continue;
        try {
          const envelope = parseEnvelope(await storage.readText(entry.path));
          if (!envelope || isExpired(envelope.cachedAt, this.opts.expiryDays)) {
            await storage.remove(entry.path);
            removed += 1;
          }
        } catch {
          await storage.remove(entry.path).catch(() => {});
          removed += 1;
        }
      }
    }
    return removed;
  }

  private pathFor(key: string, kind: "html" | "abs"): string {
    return this.opts.storage.normalizePath(
      `${this.opts.rootDir}/${kind}/${stableHash(key)}.json`,
    );
  }
}

function stableHash(value: string): string {
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    first = Math.imul(first ^ code, 0x01000193) >>> 0;
    second = Math.imul(second ^ code, 0x85ebca6b) >>> 0;
  }
  return `${first.toString(16).padStart(8, "0")}${second.toString(16).padStart(8, "0")}`;
}

function parseEnvelope(raw: string): CacheEnvelope | null {
  try {
    const parsed = JSON.parse(raw) as Partial<CacheEnvelope>;
    if (
      parsed.schemaVersion !== 1 ||
      typeof parsed.cachedAt !== "string" ||
      typeof parsed.content !== "string"
    ) return null;
    return parsed as CacheEnvelope;
  } catch {
    return null;
  }
}

function isExpired(cachedAt: string, expiryDays: number): boolean {
  const timestamp = Date.parse(cachedAt);
  if (!Number.isFinite(timestamp)) return true;
  return (Date.now() - timestamp) / 86_400_000 > expiryDays;
}

function parentDir(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length <= 1 ? "" : parts.slice(0, -1).join("/");
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  if (!dir) return;
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) await storage.mkdir(current);
  }
}
