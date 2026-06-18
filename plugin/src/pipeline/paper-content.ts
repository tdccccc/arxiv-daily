import type { ArxivFetcher } from "./arxiv-fetcher";
import type { HtmlCache } from "./html-cache";
import type { Logger } from "../services/logger";
import type { StorageAdapter } from "../core/adapters";
import {
  extractAbstractConclusion,
  extractSections,
  type ExtractSectionsOpts,
} from "./section-extractor";
import { extractLatexSource } from "./source-extractor";

export interface PaperContent {
  abstractConclusion: string;
  fullSections: string | null;
  fullTextSource?: "arxiv-html" | "arxiv-source";
  fullTextFailure?: string;
}

export interface PaperContentOpts {
  isDetail: boolean;
  sectionCharLimit: number;
  paperCharLimit: number;
}

export interface PaperContentSourceCache {
  storage: StorageAdapter;
  cacheDir: string;
  expiryDays: number;
}

interface SourceCacheMeta {
  schemaVersion: 1;
  cachedAt: string;
}

export class PaperContentFetcher {
  constructor(
    private fetcher: ArxivFetcher,
    private cache: HtmlCache,
    private logger: Logger,
    private sourceCache?: PaperContentSourceCache,
  ) {}

  async fetch(arxivId: string, opts: PaperContentOpts): Promise<PaperContent> {
    // 1. Try the rendered HTML version (cached on hit)
    const htmlKey = `html/${arxivId}`;
    let html = await this.cache.get(htmlKey, "html");
    if (!html) {
      const res = await this.fetcher.fetchPaperHtml(arxivId);
      if (res.ok) {
        html = res.body;
        await this.cache.set(htmlKey, "html", html);
      }
    }

    let htmlContent: PaperContent | null = null;
    if (html) {
      const ac = extractAbstractConclusion(html, {
        sectionCharLimit: opts.sectionCharLimit,
      });
      const sectionsOpts: ExtractSectionsOpts = {
        sectionCharLimit: opts.sectionCharLimit,
        paperCharLimit: opts.paperCharLimit,
      };
      const fs = opts.isDetail ? extractSections(html, sectionsOpts) : null;
      if (ac && (!opts.isDetail || fs)) {
        return {
          abstractConclusion: ac,
          fullSections: fs,
          fullTextSource: fs ? "arxiv-html" : undefined,
        };
      }
      // Fallback: strip tags from full HTML if section extraction missed
      const plain = html
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .slice(0, opts.paperCharLimit);
      htmlContent = {
        abstractConclusion: ac ?? plain,
        fullSections: fs,
        fullTextSource: fs ? "arxiv-html" : undefined,
      };
      if (!opts.isDetail || fs) return htmlContent;
    }

    // 2. Fallback to arXiv source when full text sections are needed.
    let sourceFailure: string | undefined;
    if (opts.isDetail) {
      const source = await this.fetchSourceContent(arxivId, opts);
      if (source.fullSections) {
        return {
          abstractConclusion:
            source.abstractConclusion ??
            htmlContent?.abstractConclusion ??
            (await this.fetchAbsAbstract(arxivId)),
          fullSections: source.fullSections,
          fullTextSource: "arxiv-source",
        };
      }
      sourceFailure = source.reason;
    }

    // 3. Fallback to /abs page for daily summaries. Manual detail summaries
    // still reject this because it is not full text.
    return {
      abstractConclusion:
        htmlContent?.abstractConclusion ?? (await this.fetchAbsAbstract(arxivId)),
      fullSections: null,
      fullTextFailure:
        sourceFailure ??
        `no rendered HTML or extractable arXiv source for ${arxivId}`,
    };
  }

  private async fetchAbsAbstract(arxivId: string): Promise<string> {
    const absKey = `abs/${arxivId}`;
    let abs = await this.cache.get(absKey, "abs");
    if (!abs) {
      try {
        abs = await this.fetcher.fetchPaperAbsPage(arxivId);
        await this.cache.set(absKey, "abs", abs);
      } catch (e) {
        this.logger.error(`paper-content: abs fetch failed ${arxivId}`, e);
        return `[获取失败] arXiv ID: ${arxivId}`;
      }
    }
    const doc = new DOMParser().parseFromString(abs, "text/html");
    const bq = doc.querySelector("blockquote.abstract");
    const text =
      (bq?.textContent ?? "").replace(/^\s*Abstract:?\s*/, "").trim() || "N/A";
    return `## Abstract\n${text}`;
  }

  private async fetchSourceContent(
    arxivId: string,
    opts: PaperContentOpts,
  ): Promise<{
    abstractConclusion: string | null;
    fullSections: string | null;
    reason?: string;
  }> {
    let source: ArrayBuffer | null = null;
    try {
      source = await this.readCachedSource(arxivId);
      if (!source) {
        const fetched = await this.fetcher.fetchSource(arxivId);
        if (!fetched.ok) {
          return {
            abstractConclusion: null,
            fullSections: null,
            reason: `no arXiv source for ${arxivId} (HTTP ${fetched.status})`,
          };
        }
        source = fetched.body;
        await this.writeCachedSource(arxivId, source);
      }
    } catch (e) {
      this.logger.warn(`paper-content: source fetch failed ${arxivId}`, e);
      return {
        abstractConclusion: null,
        fullSections: null,
        reason: `arXiv source fetch failed for ${arxivId}: ${(e as Error).message}`,
      };
    }

    try {
      const extracted = extractLatexSource(source, opts);
      if (!extracted.fullSections) {
        return {
          abstractConclusion: extracted.abstractConclusion,
          fullSections: null,
          reason: `arXiv source for ${arxivId} did not contain extractable TeX sections`,
        };
      }
      this.logger.info(
        `paper-content: extracted ${arxivId} from source ${extracted.mainFile ?? "unknown"}`,
      );
      return extracted;
    } catch (e) {
      this.logger.warn(`paper-content: source extraction failed ${arxivId}`, e);
      return {
        abstractConclusion: null,
        fullSections: null,
        reason: `arXiv source extraction failed for ${arxivId}: ${(e as Error).message}`,
      };
    }
  }

  private async readCachedSource(arxivId: string): Promise<ArrayBuffer | null> {
    const sourceCache = this.sourceCache;
    if (!sourceCache) return null;
    const storage = sourceCache?.storage;
    if (!storage?.readBinary) return null;
    const path = this.sourceCachePath(arxivId);
    const meta = await this.readSourceMeta(arxivId);
    if (!meta || isExpired(meta.cachedAt, sourceCache.expiryDays)) {
      await this.removeCachedSource(arxivId);
      return null;
    }
    if (!(await storage.exists(path))) return null;
    return storage.readBinary(path);
  }

  private async writeCachedSource(
    arxivId: string,
    source: ArrayBuffer,
  ): Promise<void> {
    const storage = this.sourceCache?.storage;
    if (!storage?.writeBinary) return;
    const path = this.sourceCachePath(arxivId);
    await ensureDirDeep(storage, parentDir(path));
    await storage.writeBinary(path, source);
    await storage.writeText(
      this.sourceMetaPath(arxivId),
      `${JSON.stringify({
        schemaVersion: 1,
        cachedAt: new Date().toISOString(),
      } satisfies SourceCacheMeta)}\n`,
    );
  }

  private sourceCachePath(arxivId: string): string {
    if (!this.sourceCache) return "";
    return this.sourceCache.storage.normalizePath(
      `${this.sourceCache.cacheDir}/${arxivId}/source`,
    );
  }

  private sourceMetaPath(arxivId: string): string {
    if (!this.sourceCache) return "";
    return this.sourceCache.storage.normalizePath(
      `${this.sourceCache.cacheDir}/${arxivId}/source.meta.json`,
    );
  }

  private async readSourceMeta(arxivId: string): Promise<SourceCacheMeta | null> {
    const storage = this.sourceCache?.storage;
    if (!storage) return null;
    const path = this.sourceMetaPath(arxivId);
    try {
      if (!(await storage.exists(path))) return null;
      return parseSourceMeta(await storage.readText(path));
    } catch {
      return null;
    }
  }

  private async removeCachedSource(arxivId: string): Promise<void> {
    const storage = this.sourceCache?.storage;
    if (!storage) return;
    for (const path of [this.sourceCachePath(arxivId), this.sourceMetaPath(arxivId)]) {
      try {
        if (await storage.exists(path)) await storage.remove(path);
      } catch {
        // Ignore cache cleanup races.
      }
    }
  }
}

export async function cleanupSourceCache(
  cache: PaperContentSourceCache,
): Promise<number> {
  const storage = cache.storage;
  if (!storage.list) return 0;
  const root = storage.normalizePath(cache.cacheDir);
  if (!(await storage.exists(root))) return 0;

  let removed = 0;
  const entries = await storage.list(root);
  for (const entry of entries) {
    if (entry.type !== "folder") continue;
    const metaPath = storage.normalizePath(`${entry.path}/source.meta.json`);
    const sourcePath = storage.normalizePath(`${entry.path}/source`);
    let expired = false;
    try {
      const meta = (await storage.exists(metaPath))
        ? parseSourceMeta(await storage.readText(metaPath))
        : null;
      expired = !meta || isExpired(meta.cachedAt, cache.expiryDays);
    } catch {
      expired = true;
    }
    if (!expired) continue;
    for (const path of [sourcePath, metaPath]) {
      try {
        if (await storage.exists(path)) {
          await storage.remove(path);
          removed += 1;
        }
      } catch {
        // Ignore cache cleanup races.
      }
    }
    try {
      if (await storage.exists(entry.path)) await storage.remove(entry.path);
    } catch {
      // Ignore cache cleanup races.
    }
  }
  return removed;
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

function parseSourceMeta(raw: string): SourceCacheMeta | null {
  try {
    const parsed = JSON.parse(raw) as Partial<SourceCacheMeta>;
    if (parsed.schemaVersion !== 1 || typeof parsed.cachedAt !== "string") {
      return null;
    }
    return parsed as SourceCacheMeta;
  } catch {
    return null;
  }
}

function isExpired(cachedAt: string, expiryDays: number): boolean {
  const timestamp = Date.parse(cachedAt);
  if (!Number.isFinite(timestamp)) return true;
  return (Date.now() - timestamp) / 86_400_000 > expiryDays;
}
