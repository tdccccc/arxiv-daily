import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import { throwIfCancelled } from "../services/cancellation";
import { parseAtomPapers, type AtomPaperMeta } from "./atom-parser";
import type { HttpClient } from "../core/adapters";
import type { PaperMeta } from "./arxiv-parser";

export interface ArxivFetcherOptions {
  category?: string;
  categories?: string[];
  http: HttpClient;
  logger: Logger;
  requestDelayMs: number;
}

interface HttpStatusError extends Error {
  status?: number;
  headers?: Record<string, string>;
}

let sharedLastRequestAt = 0;
let sharedDelayQueue: Promise<void> = Promise.resolve();

export class ArxivFetcher {
  constructor(private opts: ArxivFetcherOptions) {}

  /** Fetch the /list/<cat>/recent page with show=2000 to capture all 5 days in one shot. */
  async fetchRecent(
    category = this.primaryCategory(),
    signal?: AbortSignal,
  ): Promise<string> {
    const url = `https://arxiv.org/list/${category}/recent?skip=0&show=2000`;
    return this.fetchHtml(url, { allow404: false }, signal);
  }

  /**
   * Bulk-fetch abstracts via arXiv's Atom API.
   *
   * Returns a Map keyed by base arXiv id (e.g. "2605.08080" with version stripped).
   * Papers not found in the response are omitted from the map; callers should
   * fall back to an empty abstract for those.
   *
   * arXiv recommends batches of <=300; we conservatively cap at 200.
   */
  async fetchAbstractsByIds(
    ids: string[],
    signal?: AbortSignal,
  ): Promise<Map<string, string>> {
    const metadata = await this.fetchMetadataByIds(ids, signal);
    const out = new Map<string, string>();
    for (const [id, paper] of metadata) {
      if (paper.abstract) out.set(id, paper.abstract);
    }
    return out;
  }

  async fetchMetadataByIds(
    ids: string[],
    signal?: AbortSignal,
  ): Promise<Map<string, AtomPaperMeta>> {
    const out = new Map<string, AtomPaperMeta>();
    const BATCH = 200;
    for (let i = 0; i < ids.length; i += BATCH) {
      const batch = ids.slice(i, i + BATCH);
      if (batch.length === 0) continue;
      const url = `https://export.arxiv.org/api/query?id_list=${batch.join(",")}&max_results=${batch.length}`;
      const xml = await this.fetchHtml(url, { allow404: false }, signal);
      for (const paper of parseAtomPapers(xml)) out.set(paper.id, paper);
    }
    return out;
  }

  async fetchBySubmittedDate(
    category: string,
    dateStr: string,
  ): Promise<PaperMeta[]> {
    const day = dateStr.replace(/-/g, "");
    const query = `cat:${category} AND submittedDate:[${day}0000 TO ${day}2359]`;
    const url =
      `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(query)}` +
      `&start=0&max_results=2000&sortBy=submittedDate&sortOrder=ascending`;
    const xml = await this.fetchHtml(url, { allow404: false });
    return parseAtomPapers(xml);
  }

  /** Fetch /html/<id> for full paper rendering. Returns ok:false on 404. */
  async fetchPaperHtml(
    arxivId: string,
  ): Promise<{ ok: true; body: string } | { ok: false; status: number }> {
    const url = `https://arxiv.org/html/${arxivId}`;
    try {
      const body = await this.fetchHtml(url, { allow404: true });
      return { ok: true, body };
    } catch (err: any) {
      if (err?.status === 404) return { ok: false, status: 404 };
      throw err;
    }
  }

  async fetchPaperAbsPage(arxivId: string): Promise<string> {
    const url = `https://arxiv.org/abs/${arxivId}`;
    return this.fetchHtml(url, { allow404: false });
  }

  async fetchPdf(arxivId: string): Promise<ArrayBuffer> {
    const url = `https://arxiv.org/pdf/${arxivId}`;
    return this.fetchBinary(url);
  }

  async fetchSource(
    arxivId: string,
  ): Promise<{ ok: true; body: ArrayBuffer } | { ok: false; status: number }> {
    const url = `https://arxiv.org/e-print/${arxivId}`;
    try {
      const body = await this.fetchBinary(url, { allow404: true });
      return { ok: true, body };
    } catch (err: any) {
      if (err?.status === 404) return { ok: false, status: 404 };
      throw err;
    }
  }

  /** Fetch the raw Atom XML for a single id (for manual lookup with full metadata). */
  async fetchAtomEntry(arxivId: string): Promise<string> {
    const url = `https://export.arxiv.org/api/query?id_list=${arxivId}&max_results=1`;
    return this.fetchHtml(url, { allow404: false });
  }

  private async fetchHtml(
    url: string,
    opts: { allow404: boolean },
    signal?: AbortSignal,
  ): Promise<string> {
    await this.respectDelay(signal);
    this.opts.logger.debug(`fetchHtml: GET ${url}`);
    return retry(
      async () => {
        const res = await this.opts.http.request({
          url,
          method: "GET",
          headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
          signal,
        });
        if (res.status >= 200 && res.status < 300) {
          this.opts.logger.debug(`fetchHtml: ${url} → ${res.status} (${(res.bodyText ?? "").length} bytes)`);
          return res.bodyText;
        }
        if (opts.allow404 && res.status === 404) {
          throw httpStatusError(res.status, url, res.headers);
        }
        throw httpStatusError(res.status, url, res.headers);
      },
      {
        maxAttempts: 3,
        baseDelayMs: 2000,
        shouldRetry: (err: any) => err?.status !== 404,
        delayMs: arxivRetryDelayMs,
        signal,
        onRetry: (err, attempt, wait) =>
          this.opts.logger.warn(
            `fetch retry #${attempt} after ${wait}ms: ${url}: ${(err as Error).message}`,
          ),
      },
    );
  }

  private async fetchBinary(
    url: string,
    opts: { allow404?: boolean } = {},
  ): Promise<ArrayBuffer> {
    await this.respectDelay();
    this.opts.logger.debug(`fetchBinary: GET ${url}`);
    return retry(
      async () => {
        const res = await this.opts.http.request({
          url,
          method: "GET",
          headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
          responseType: "arrayBuffer",
        });
        if (res.status < 200 || res.status >= 300) {
          throw httpStatusError(res.status, url, res.headers);
        }
        if (!res.bodyBuffer) {
          throw new Error(`empty binary response: ${url}`);
        }
        this.opts.logger.debug(`fetchBinary: ${url} → ${res.status} (${res.bodyBuffer.byteLength} bytes)`);
        return res.bodyBuffer;
      },
      {
        maxAttempts: 3,
        baseDelayMs: 2000,
        shouldRetry: (err: any) => !(opts.allow404 && err?.status === 404),
        delayMs: arxivRetryDelayMs,
        onRetry: (err, attempt, wait) =>
          this.opts.logger.warn(
            `fetch retry #${attempt} after ${wait}ms: ${url}: ${(err as Error).message}`,
          ),
      },
    );
  }

  private async respectDelay(signal?: AbortSignal) {
    const delayMs = Math.max(0, this.opts.requestDelayMs);
    const next = sharedDelayQueue
      .catch(() => undefined)
      .then(async () => {
        throwIfCancelled(signal);
        if (delayMs > 0) {
          if (sharedLastRequestAt > Date.now()) sharedLastRequestAt = 0;
          const elapsed = Math.max(0, Date.now() - sharedLastRequestAt);
          const wait = delayMs - elapsed;
          if (wait > 0) await abortableDelay(wait, signal);
        }
        throwIfCancelled(signal);
        sharedLastRequestAt = Date.now();
      });
    sharedDelayQueue = next.catch(() => undefined);
    await next;
  }

  private primaryCategory(): string {
    return this.opts.categories?.[0] ?? this.opts.category ?? "astro-ph";
  }
}

function abortableDelay(ms: number, signal?: AbortSignal): Promise<void> {
  throwIfCancelled(signal);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(done, ms);
    const onAbort = () => {
      clearTimeout(timeout);
      cleanup();
      try {
        throwIfCancelled(signal);
      } catch (e) {
        reject(e);
      }
    };
    function done() {
      cleanup();
      resolve();
    }
    function cleanup() {
      signal?.removeEventListener("abort", onAbort);
    }
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
  });
}

function httpStatusError(
  status: number,
  url: string,
  headers: Record<string, string>,
): HttpStatusError {
  const error = new Error(`HTTP ${status}: ${url}`) as HttpStatusError;
  error.status = status;
  error.headers = headers;
  return error;
}

function arxivRetryDelayMs(
  err: unknown,
  _attempt: number,
  defaultWaitMs: number,
): number {
  const status = (err as HttpStatusError | undefined)?.status;
  const headers = (err as HttpStatusError | undefined)?.headers;
  const retryAfterMs =
    status === 429 && headers ? parseRetryAfterMs(headers) : null;
  return jitterDelayMs(retryAfterMs ?? defaultWaitMs);
}

function parseRetryAfterMs(headers: Record<string, string>): number | null {
  const value = headerValue(headers, "retry-after")?.trim();
  if (!value) return null;
  if (/^\d+$/.test(value)) return Number(value) * 1000;
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return null;
  return Math.max(0, timestamp - Date.now());
}

function headerValue(
  headers: Record<string, string>,
  name: string,
): string | undefined {
  const lower = name.toLowerCase();
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() === lower) return value;
  }
  return undefined;
}

function jitterDelayMs(delayMs: number): number {
  const factor = 0.75 + Math.random() * 0.5;
  return Math.max(0, Math.round(delayMs * factor));
}
