import type { StorageAdapter } from "../core/adapters";
import type { AtomPaperMeta } from "./atom-parser";
import { modernArxivResources } from "../utils/arxiv";

export interface AtomMetadataCacheOptions {
  rootDir: string;
  expiryDays: number;
  storage: StorageAdapter;
  now?: () => Date;
}

interface AtomMetadataCacheEnvelope {
  schemaVersion: 1;
  cachedAt: string;
  paper: AtomPaperMeta;
}

export class AtomMetadataCache {
  constructor(private opts: AtomMetadataCacheOptions) {}

  async get(id: string): Promise<AtomPaperMeta | null> {
    const canonicalId = canonicalArxivId(id);
    if (!canonicalId) return null;
    const path = this.pathFor(canonicalId);
    try {
      if (!(await this.opts.storage.exists(path))) return null;
      const envelope = parseEnvelope(await this.opts.storage.readText(path));
      if (
        !envelope ||
        envelope.paper.id !== canonicalId ||
        isExpired(envelope.cachedAt, this.opts.expiryDays, this.now())
      ) {
        await this.removeBestEffort(path);
        return null;
      }
      return envelope.paper;
    } catch {
      return null;
    }
  }

  async set(id: string, paper: AtomPaperMeta): Promise<void> {
    const canonicalId = canonicalArxivId(id);
    if (!canonicalId || paper.id !== canonicalId || !isAtomPaperMeta(paper)) {
      throw new Error(`invalid Atom metadata cache entry for ${id}`);
    }
    const path = this.pathFor(canonicalId);
    await ensureDirDeep(this.opts.storage, parentDir(path));
    const envelope: AtomMetadataCacheEnvelope = {
      schemaVersion: 1,
      cachedAt: this.now().toISOString(),
      paper,
    };
    const content = `${JSON.stringify(envelope)}\n`;
    if (this.opts.storage.writeTextAtomic) {
      await this.opts.storage.writeTextAtomic(path, content);
    } else {
      await this.opts.storage.writeText(path, content);
    }
  }

  async cleanupExpired(): Promise<number> {
    const storage = this.opts.storage;
    if (!storage.list) return 0;
    const dir = this.cacheDir();
    try {
      if (!(await storage.exists(dir))) return 0;
      let removed = 0;
      for (const entry of await storage.list(dir)) {
        if (entry.type !== "file" || !entry.path.endsWith(".json")) continue;
        try {
          const envelope = parseEnvelope(await storage.readText(entry.path));
          const filenameId = decodeFilenameId(entry.path);
          if (
            !envelope ||
            !filenameId ||
            envelope.paper.id !== filenameId ||
            isExpired(envelope.cachedAt, this.opts.expiryDays, this.now())
          ) {
            await storage.remove(entry.path).catch(() => {});
            removed += 1;
          }
        } catch {
          await storage.remove(entry.path).catch(() => {});
          removed += 1;
        }
      }
      return removed;
    } catch {
      return 0;
    }
  }

  private pathFor(canonicalId: string): string {
    return this.opts.storage.normalizePath(
      `${this.cacheDir()}/${encodeURIComponent(canonicalId)}.json`,
    );
  }

  private cacheDir(): string {
    return this.opts.storage.normalizePath(`${this.opts.rootDir}/atom-metadata`);
  }

  private now(): Date {
    return this.opts.now?.() ?? new Date();
  }

  private async removeBestEffort(path: string): Promise<void> {
    await this.opts.storage.remove(path).catch(() => {});
  }
}

export function isAtomPaperMeta(value: unknown): value is AtomPaperMeta {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const paper = value as Partial<AtomPaperMeta>;
  const canonicalId = typeof paper.id === "string" ? canonicalArxivId(paper.id) : null;
  return (
    canonicalId === paper.id &&
    isNonEmptyString(paper.title) &&
    isNonEmptyString(paper.authors) &&
    isNonEmptyString(paper.abstract) &&
    isNonEmptyString(paper.published) &&
    isNonEmptyString(paper.updated) &&
    isNonEmptyString(paper.primaryCategory) &&
    Array.isArray(paper.categories) &&
    paper.categories.length > 0 &&
    paper.categories.every(isNonEmptyString) &&
    paper.categories.includes(paper.primaryCategory)
  );
}

function parseEnvelope(raw: string): AtomMetadataCacheEnvelope | null {
  try {
    const parsed = JSON.parse(raw) as Partial<AtomMetadataCacheEnvelope>;
    if (
      parsed.schemaVersion !== 1 ||
      typeof parsed.cachedAt !== "string" ||
      !Number.isFinite(Date.parse(parsed.cachedAt)) ||
      !isAtomPaperMeta(parsed.paper)
    ) return null;
    return parsed as AtomMetadataCacheEnvelope;
  } catch {
    return null;
  }
}

function canonicalArxivId(id: string): string | null {
  return modernArxivResources(id)?.id ?? null;
}

function isExpired(cachedAt: string, expiryDays: number, now: Date): boolean {
  const timestamp = Date.parse(cachedAt);
  if (!Number.isFinite(timestamp)) return true;
  return now.getTime() - timestamp > expiryDays * 86_400_000;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function decodeFilenameId(path: string): string | null {
  const filename = path.split("/").pop();
  if (!filename?.endsWith(".json")) return null;
  try {
    const id = decodeURIComponent(filename.slice(0, -5));
    return canonicalArxivId(id) === id ? id : null;
  } catch {
    return null;
  }
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
