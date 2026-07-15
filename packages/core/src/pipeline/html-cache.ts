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
      if (await this.opts.storage.exists(path)) {
        const envelope = parseEnvelope(await this.opts.storage.readText(path));
        if (!envelope || isExpired(envelope.cachedAt, this.opts.expiryDays)) {
          await this.opts.storage.remove(path).catch(() => {});
        } else {
          return envelope.content;
        }
      }

      const legacyPath = this.legacyPathFor(key, kind);
      if (!(await this.opts.storage.exists(legacyPath))) return null;
      const content = await this.opts.storage.readText(legacyPath);
      await this.set(key, kind, content).catch(() => {});
      return content;
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
        if (entry.type !== "file" || !entry.path.endsWith(".json")) continue;
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

  private legacyPathFor(key: string, kind: "html" | "abs"): string {
    return this.opts.storage.normalizePath(
      `${this.opts.rootDir}/${kind}/${stableHash(key)}.html`,
    );
  }
}

function stableHash(value: string): string {
  const bytes = new TextEncoder().encode(value);
  const words = new Uint32Array(80);
  const bitLength = bytes.length * 8;
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLength - 4, bitLength >>> 0, false);
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x1_0000_0000), false);

  let h0 = 0x67452301;
  let h1 = 0xefcdab89;
  let h2 = 0x98badcfe;
  let h3 = 0x10325476;
  let h4 = 0xc3d2e1f0;

  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(offset + index * 4, false);
    }
    for (let index = 16; index < 80; index += 1) {
      words[index] = rotateLeft(
        words[index - 3]! ^ words[index - 8]! ^ words[index - 14]! ^ words[index - 16]!,
        1,
      );
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    for (let index = 0; index < 80; index += 1) {
      let f: number;
      let k: number;
      if (index < 20) {
        f = (b & c) | (~b & d);
        k = 0x5a827999;
      } else if (index < 40) {
        f = b ^ c ^ d;
        k = 0x6ed9eba1;
      } else if (index < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8f1bbcdc;
      } else {
        f = b ^ c ^ d;
        k = 0xca62c1d6;
      }
      const next = (rotateLeft(a, 5) + f + e + k + words[index]!) >>> 0;
      e = d;
      d = c;
      c = rotateLeft(b, 30);
      b = a;
      a = next;
    }
    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
  }

  return [h0, h1, h2, h3, h4]
    .map((word) => word.toString(16).padStart(8, "0"))
    .join("")
    .slice(0, 24);
}

function rotateLeft(value: number, bits: number): number {
  return ((value << bits) | (value >>> (32 - bits))) >>> 0;
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
