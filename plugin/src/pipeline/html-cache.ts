import * as fs from "node:fs/promises";
import * as path from "node:path";
import { createHash } from "node:crypto";
import type { StorageAdapter } from "../core/adapters";

export interface HtmlCacheOptions {
  rootDir: string;
  expiryDays: number;
  storage?: StorageAdapter;
}

interface CacheEnvelope {
  schemaVersion: 1;
  cachedAt: string;
  content: string;
}

export class HtmlCache {
  constructor(private opts: HtmlCacheOptions) {}

  async get(key: string, kind: "html" | "abs"): Promise<string | null> {
    if (this.opts.storage) return this.getFromStorage(key, kind);

    const p = this.pathFor(key, kind);
    try {
      const stat = await fs.stat(p);
      const ageDays = (Date.now() - stat.mtimeMs) / 86_400_000;
      if (ageDays > this.opts.expiryDays) {
        await fs.unlink(p).catch(() => {});
        return null;
      }
      return await fs.readFile(p, "utf8");
    } catch {
      return null;
    }
  }

  async set(key: string, kind: "html" | "abs", content: string): Promise<void> {
    if (this.opts.storage) {
      await this.setInStorage(key, kind, content);
      return;
    }

    const p = this.pathFor(key, kind);
    await fs.mkdir(path.dirname(p), { recursive: true });
    await fs.writeFile(p, content, "utf8");
  }

  async cleanupExpired(): Promise<number> {
    if (this.opts.storage) return this.cleanupStorage();
    return this.cleanupFs();
  }

  private pathFor(key: string, kind: "html" | "abs"): string {
    const safe = createHash("sha1").update(key).digest("hex").slice(0, 24);
    if (this.opts.storage) {
      return this.opts.storage.normalizePath(
        `${this.opts.rootDir}/${kind}/${safe}.json`,
      );
    }
    return path.join(this.opts.rootDir, kind, `${safe}.html`);
  }

  private async getFromStorage(
    key: string,
    kind: "html" | "abs",
  ): Promise<string | null> {
    const storage = this.opts.storage;
    if (!storage) return null;
    const p = this.pathFor(key, kind);
    try {
      if (!(await storage.exists(p))) return null;
      const envelope = parseEnvelope(await storage.readText(p));
      if (!envelope) {
        await storage.remove(p).catch(() => {});
        return null;
      }
      if (isExpired(envelope.cachedAt, this.opts.expiryDays)) {
        await storage.remove(p).catch(() => {});
        return null;
      }
      return envelope.content;
    } catch {
      return null;
    }
  }

  private async setInStorage(
    key: string,
    kind: "html" | "abs",
    content: string,
  ): Promise<void> {
    const storage = this.opts.storage;
    if (!storage) return;
    const p = this.pathFor(key, kind);
    await ensureDirDeep(storage, parentDir(p));
    const envelope: CacheEnvelope = {
      schemaVersion: 1,
      cachedAt: new Date().toISOString(),
      content,
    };
    await storage.writeText(p, `${JSON.stringify(envelope)}\n`);
  }

  private async cleanupStorage(): Promise<number> {
    const storage = this.opts.storage;
    if (!storage?.list) return 0;
    let removed = 0;
    for (const kind of ["html", "abs"] as const) {
      const dir = storage.normalizePath(`${this.opts.rootDir}/${kind}`);
      if (!(await storage.exists(dir))) continue;
      const entries = await storage.list(dir);
      for (const entry of entries) {
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

  private async cleanupFs(): Promise<number> {
    let removed = 0;
    for (const kind of ["html", "abs"] as const) {
      const dir = path.join(this.opts.rootDir, kind);
      let entries: string[];
      try {
        entries = await fs.readdir(dir);
      } catch {
        continue;
      }
      for (const entry of entries) {
        const p = path.join(dir, entry);
        try {
          const stat = await fs.stat(p);
          const ageDays = (Date.now() - stat.mtimeMs) / 86_400_000;
          if (ageDays <= this.opts.expiryDays) continue;
          await fs.unlink(p);
          removed += 1;
        } catch {
          // Ignore cache cleanup races.
        }
      }
    }
    return removed;
  }
}

function parseEnvelope(raw: string): CacheEnvelope | null {
  try {
    const parsed = JSON.parse(raw) as Partial<CacheEnvelope>;
    if (
      parsed?.schemaVersion !== 1 ||
      typeof parsed.cachedAt !== "string" ||
      typeof parsed.content !== "string"
    ) {
      return null;
    }
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

async function ensureDirDeep(
  storage: StorageAdapter,
  dir: string,
): Promise<void> {
  if (!dir) return;
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur = cur ? `${cur}/${part}` : part;
    if (!(await storage.exists(cur))) await storage.mkdir(cur);
  }
}
