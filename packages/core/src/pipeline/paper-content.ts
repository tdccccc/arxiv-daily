import type { ArxivFetcher } from "./arxiv-fetcher";
import type { HtmlCache } from "./html-cache";
import type { Logger } from "../services/logger";
import type { MarkupParser, StorageAdapter } from "../core/adapters";
import {
  extractAbsPageAbstract,
  extractAbstractConclusion,
  extractSections,
  preserveMathText,
  type ExtractSectionsOpts,
} from "./section-extractor";
import { extractLatexSource, SOURCE_EXTRACTION_LIMITS } from "./source-extractor";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";

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
    private markupParser: MarkupParser,
    private sourceCache?: PaperContentSourceCache,
  ) {}

  async fetch(arxivId: string, opts: PaperContentOpts, signal?: AbortSignal): Promise<PaperContent> {
    throwIfCancelled(signal);
    // 1. Try the rendered HTML version (cached on hit). Any non-cancellation
    // cache, transport, or extraction exception is a fallback signal.
    const htmlKey = `html/${arxivId}`;
    const failures: string[] = [];
    let htmlContent: PaperContent | null = null;
    try {
      let html = await this.cache.get(htmlKey, "html");
      if (!html) {
        const res = await this.fetcher.fetchPaperHtml(arxivId, signal);
        if (res.ok) {
          html = res.body;
          try {
            await this.cache.set(htmlKey, "html", html);
          } catch (error) {
            if (isCancellationError(error)) throw error;
            this.logger.warn(`paper-content: html cache write failed for ${arxivId}`, error);
          }
        } else {
          failures.push(`rendered HTML unavailable for ${arxivId} (HTTP ${res.status})`);
        }
      } else {
        this.logger.debug(`paper-content: html cache hit for ${arxivId}`);
      }

      if (html) {
        const ac = extractAbstractConclusion(
          html,
          { sectionCharLimit: opts.sectionCharLimit },
          this.markupParser,
        );
        const sectionsOpts: ExtractSectionsOpts = {
          sectionCharLimit: opts.sectionCharLimit,
          paperCharLimit: opts.paperCharLimit,
        };
        const fs = opts.isDetail
          ? extractSections(html, sectionsOpts, this.markupParser)
          : null;
        if (ac && (!opts.isDetail || fs)) {
          this.logger.info(
            `paper-content: extracted ${arxivId} from rendered HTML` +
              (fs ? ` (${fs.length} chars sections)` : ""),
          );
          return {
            abstractConclusion: ac,
            fullSections: fs,
            fullTextSource: fs ? "arxiv-html" : undefined,
          };
        }
        const fallbackDoc = this.markupParser.parseFromString(html, "text/html");
        preserveMathText(fallbackDoc);
        const plain = textWithElementSpacing(fallbackDoc)
          .replace(/\s+/g, " ")
          .slice(0, opts.paperCharLimit);
        htmlContent = {
          abstractConclusion: ac ?? plain,
          fullSections: fs,
          fullTextSource: fs ? "arxiv-html" : undefined,
        };
        if (!opts.isDetail || fs) return htmlContent;
        failures.push(`rendered HTML sections empty for ${arxivId}`);
        this.logger.warn(`paper-content: html sections empty for ${arxivId}, falling back to source`);
      }
    } catch (e) {
      if (isCancellationError(e)) throw e;
      const reason = `rendered HTML/cache failed for ${arxivId}: ${errorMessage(e)}`;
      failures.push(reason);
      this.logger.warn(`paper-content: ${reason}, falling back`, e);
    }

    // 2. Fallback to arXiv source when full text sections are needed.
    let sourceFailure: string | undefined;
    if (opts.isDetail) {
      const source = await this.fetchSourceContent(arxivId, opts, signal);
      if (source.fullSections) {
        return {
          abstractConclusion:
            source.abstractConclusion ??
            htmlContent?.abstractConclusion ??
            (await this.fetchAbsAbstract(arxivId, signal)).abstract,
          fullSections: source.fullSections,
          fullTextSource: "arxiv-source",
        };
      }
      sourceFailure = source.reason;
      if (sourceFailure) failures.push(sourceFailure);
    }

    // 3. Fallback to /abs page for daily summaries. Manual detail summaries
    // still reject this because it is not full text.
    this.logger.info(
      `paper-content: fallback to abs page for ${arxivId} (sourceFailure=${!!sourceFailure})`,
    );
    let abstractConclusion = htmlContent?.abstractConclusion;
    if (!abstractConclusion) {
      const abs = await this.fetchAbsAbstract(arxivId, signal);
      abstractConclusion = abs.abstract;
      if (abs.failure) failures.push(abs.failure);
    }
    return {
      abstractConclusion,
      fullSections: null,
      fullTextFailure:
        failures.length > 0
          ? failures.join("; ")
          : `no rendered HTML or extractable arXiv source for ${arxivId}`,
    };
  }

  /**
   * Abstract text from the arxiv.org/abs page for filtering a listing entry:
   * served from the shared `abs/<id>` cache when present, otherwise fetched and
   * cached (a cache write failure is ignored). The text has no "Abstract:"
   * label and is empty when the page carries no abstract block (still a
   * successful fetch). `fromNetwork` tells the caller whether this call hit
   * arxiv.org so it can budget requests. Fetch failures and cancellation are
   * thrown; this method does not log them.
   */
  async fetchListingAbstract(
    arxivId: string,
    signal?: AbortSignal,
  ): Promise<{ abstract: string; fromNetwork: boolean }> {
    throwIfCancelled(signal);
    const page = await this.loadAbsPage(arxivId, signal, { tolerateCacheWriteFailure: true });
    return {
      abstract: extractAbsPageAbstract(page.html, this.markupParser),
      fromNetwork: page.fromNetwork,
    };
  }

  private async loadAbsPage(
    arxivId: string,
    signal: AbortSignal | undefined,
    opts: { tolerateCacheWriteFailure: boolean },
  ): Promise<{ html: string; fromNetwork: boolean }> {
    const absKey = `abs/${arxivId}`;
    const cached = await this.cache.get(absKey, "abs");
    if (cached) return { html: cached, fromNetwork: false };
    const html = await this.fetcher.fetchPaperAbsPage(arxivId, signal);
    try {
      await this.cache.set(absKey, "abs", html);
    } catch (e) {
      if (!opts.tolerateCacheWriteFailure) throw e;
    }
    return { html, fromNetwork: true };
  }

  private async fetchAbsAbstract(
    arxivId: string,
    signal?: AbortSignal,
  ): Promise<{ abstract: string; failure?: string }> {
    try {
      const page = await this.loadAbsPage(arxivId, signal, { tolerateCacheWriteFailure: false });
      const text = extractAbsPageAbstract(page.html, this.markupParser) || "N/A";
      return { abstract: `## Abstract\n${text}` };
    } catch (e) {
      if (isCancellationError(e)) throw e;
      const failure = `abs/cache failed for ${arxivId}: ${errorMessage(e)}`;
      this.logger.error(`paper-content: ${failure}`, e);
      return { abstract: `[获取失败] arXiv ID: ${arxivId}`, failure };
    }
  }

  private async fetchSourceContent(
    arxivId: string,
    opts: PaperContentOpts,
    signal?: AbortSignal,
  ): Promise<{
    abstractConclusion: string | null;
    fullSections: string | null;
    reason?: string;
  }> {
    let source: ArrayBuffer | null = null;
    try {
      source = await this.readCachedSource(arxivId);
      if (!source) {
        const fetched = await this.fetcher.fetchSource(arxivId, signal);
        if (!fetched.ok) {
          return {
            abstractConclusion: null,
            fullSections: null,
            reason: `no arXiv source for ${arxivId} (HTTP ${fetched.status})`,
          };
        }
        source = fetched.body;
        // An oversized package is never cached or parsed: it would be re-read
        // (and re-rejected) on every retry.
        if (source.byteLength > SOURCE_EXTRACTION_LIMITS.maxArchiveBytes) {
          return {
            abstractConclusion: null,
            fullSections: null,
            reason: `arXiv source for ${arxivId} is too large to process (${source.byteLength} bytes)`,
          };
        }
        await this.writeCachedSource(arxivId, source);
      }
    } catch (e) {
      if (isCancellationError(e)) throw e;
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

function textWithElementSpacing(node: Node): string {
  if (node.nodeType === 3 || node.nodeType === 4) return node.textContent ?? "";
  if (node.nodeType !== 1 && node.nodeType !== 9) return "";
  // Match the fallback's separation between HTML elements without treating
  // comparison operators inside attributes or mathematical text as markup.
  return " " + Array.from(node.childNodes, textWithElementSpacing).join("") + " ";
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

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message ? error.message : String(error);
}
