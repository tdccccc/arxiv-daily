import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import { throwIfCancelled } from "../services/cancellation";
import { parseAtomPapers, type AtomPaperMeta } from "./atom-parser";
import type { HttpClient, MarkupParser } from "../core/adapters";
import type { PaperMeta } from "./arxiv-parser";
import { modernArxivResources } from "../utils/arxiv";
import { isAtomPaperMeta, type AtomMetadataCache } from "./atom-metadata-cache";

export interface ArxivFetcherOptions {
  category?: string;
  categories?: string[];
  http: HttpClient;
  markupParser: MarkupParser;
  logger: Logger;
  requestDelayMs: number;
  metadataCache?: AtomMetadataCache;
}

const MIN_ARXIV_REQUEST_DELAY_MS = 3_000;
// Keep server-directed waits bounded to the same practical retry horizon.
const MAX_ARXIV_RETRY_AFTER_MS = 30 * 60 * 1000;
const ARXIV_HTTP_ERROR_NAME = "ArxivHttpError";

export class ArxivHttpError extends Error {
  readonly status: number;
  readonly url: string;
  readonly headers: Record<string, string>;

  constructor(status: number, url: string, headers: Record<string, string> = {}) {
    super(`HTTP ${status}: ${url}`);
    this.name = ARXIV_HTTP_ERROR_NAME;
    this.status = status;
    this.url = url;
    this.headers = headers;
  }
}

export function isArxivHttpError(error: unknown): error is ArxivHttpError {
  if (!error || typeof error !== "object") return false;
  const candidate = error as Partial<ArxivHttpError>;
  return (
    candidate.name === ARXIV_HTTP_ERROR_NAME &&
    typeof candidate.message === "string" &&
    typeof candidate.status === "number" &&
    Number.isInteger(candidate.status) &&
    typeof candidate.url === "string" &&
    isStringRecord(candidate.headers)
  );
}

export function formatArxivHttpError(error: unknown): string {
  if (!isArxivHttpError(error)) {
    return error instanceof Error ? error.message : String(error);
  }
  if (error.status === 429) {
    return "arXiv is rate-limiting requests (HTTP 429). Please wait and try again.";
  }
  if (error.status === 503) {
    return "arXiv is temporarily unavailable (HTTP 503). Please try again later.";
  }
  if (error.status === 408) {
    return "The arXiv request timed out (HTTP 408). Please try again.";
  }
  if (error.status >= 500) {
    return `arXiv is temporarily unavailable (HTTP ${error.status}). Please try again later.`;
  }
  if (error.status >= 400) {
    return `arXiv rejected the request (HTTP ${error.status}). Check the arXiv ID or request and try again.`;
  }
  return error.message;
}

let sharedLastAttemptStartedAt = Number.NEGATIVE_INFINITY;
let sharedCooldownUntil = 0;
let sharedAttemptQueue: Promise<void> = Promise.resolve();

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
    const canonicalIds = Array.from(new Set(ids.flatMap((id) => {
      const canonical = modernArxivResources(id)?.id;
      return canonical ? [canonical] : [];
    })));
    const out = new Map<string, AtomPaperMeta>();
    const misses: string[] = [];
    for (const id of canonicalIds) {
      throwIfCancelled(signal);
      const cached = await this.opts.metadataCache?.get(id);
      if (cached) out.set(id, cached);
      else misses.push(id);
    }

    const BATCH = 200;
    for (let i = 0; i < misses.length; i += BATCH) {
      const batch = misses.slice(i, i + BATCH);
      const requested = new Set(batch);
      const url = `https://export.arxiv.org/api/query?id_list=${batch.join(",")}&max_results=${batch.length}`;
      const xml = await this.fetchHtml(url, { allow404: false }, signal);
      for (const paper of parseAtomPapers(xml, this.opts.markupParser)) {
        if (!requested.has(paper.id)) continue;
        const normalized = normalizeAtomPaperMeta(paper);
        if (!normalized) continue;
        out.set(normalized.id, normalized);
        await this.opts.metadataCache?.set(normalized.id, normalized).catch((error) =>
          this.opts.logger.warn(`Atom metadata cache write failed for ${normalized.id}`, error),
        );
      }
    }
    return out;
  }

  async fetchBySubmittedDate(
    category: string,
    dateStr: string,
    signal?: AbortSignal,
  ): Promise<PaperMeta[]> {
    const day = dateStr.replace(/-/g, "");
    const query = `cat:${category} AND submittedDate:[${day}0000 TO ${day}2359]`;
    const url =
      `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(query)}` +
      `&start=0&max_results=2000&sortBy=submittedDate&sortOrder=ascending`;
    const xml = await this.fetchHtml(url, { allow404: false }, signal);
    return parseAtomPapers(xml, this.opts.markupParser);
  }

  /** Fetch /html/<id> for full paper rendering. Returns ok:false on 404. */
  async fetchPaperHtml(
    arxivId: string,
    signal?: AbortSignal,
  ): Promise<{ ok: true; body: string } | { ok: false; status: number }> {
    const url = requireArxivResources(arxivId).htmlUrl;
    try {
      const body = await this.fetchHtml(url, { allow404: true }, signal);
      return { ok: true, body };
    } catch (err: any) {
      if (err?.status === 404) return { ok: false, status: 404 };
      throw err;
    }
  }

  async fetchPaperAbsPage(arxivId: string, signal?: AbortSignal): Promise<string> {
    const url = requireArxivResources(arxivId).absUrl;
    return this.fetchHtml(url, { allow404: false }, signal);
  }

  async fetchPdf(arxivId: string, signal?: AbortSignal): Promise<ArrayBuffer> {
    const url = requireArxivResources(arxivId).pdfUrl;
    return this.fetchBinary(url, {}, signal);
  }

  async fetchSource(
    arxivId: string,
    signal?: AbortSignal,
  ): Promise<{ ok: true; body: ArrayBuffer } | { ok: false; status: number }> {
    const url = requireArxivResources(arxivId).sourceUrl;
    try {
      const body = await this.fetchBinary(url, { allow404: true }, signal);
      return { ok: true, body };
    } catch (err: any) {
      if (err?.status === 404) return { ok: false, status: 404 };
      throw err;
    }
  }

  /** Fetch the raw Atom XML for a single id (for manual lookup with full metadata). */
  async fetchAtomEntry(arxivId: string, signal?: AbortSignal): Promise<string> {
    const url = requireArxivResources(arxivId).atomUrl;
    return this.fetchHtml(url, { allow404: false }, signal);
  }

  private async fetchHtml(
    url: string,
    _opts: { allow404: boolean },
    signal?: AbortSignal,
  ): Promise<string> {
    this.opts.logger.debug(`fetchHtml: GET ${url}`);
    return retry(
      async () => this.coordinateAttempt(async () => {
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
        throw new ArxivHttpError(res.status, url, res.headers);
      }, signal),
      {
        maxAttempts: 3,
        baseDelayMs: 2000,
        shouldRetry: isRetryableArxivError,
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
    _opts: { allow404?: boolean } = {},
    signal?: AbortSignal,
  ): Promise<ArrayBuffer> {
    this.opts.logger.debug(`fetchBinary: GET ${url}`);
    return retry(
      async () => this.coordinateAttempt(async () => {
        const res = await this.opts.http.request({
          url,
          method: "GET",
          headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
          responseType: "arrayBuffer",
          signal,
        });
        if (res.status < 200 || res.status >= 300) {
          throw new ArxivHttpError(res.status, url, res.headers);
        }
        if (!res.bodyBuffer) {
          throw new Error(`empty binary response: ${url}`);
        }
        this.opts.logger.debug(`fetchBinary: ${url} → ${res.status} (${res.bodyBuffer.byteLength} bytes)`);
        return res.bodyBuffer;
      }, signal),
      {
        maxAttempts: 3,
        baseDelayMs: 2000,
        shouldRetry: isRetryableArxivError,
        delayMs: arxivRetryDelayMs,
        signal,
        onRetry: (err, attempt, wait) =>
          this.opts.logger.warn(
            `fetch retry #${attempt} after ${wait}ms: ${url}: ${(err as Error).message}`,
          ),
      },
    );
  }

  private coordinateAttempt<T>(
    attempt: () => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    const configuredDelay = Number.isFinite(this.opts.requestDelayMs)
      ? this.opts.requestDelayMs
      : MIN_ARXIV_REQUEST_DELAY_MS;
    const delayMs = Math.max(MIN_ARXIV_REQUEST_DELAY_MS, configuredDelay);
    const coordinated = sharedAttemptQueue
      .catch(() => undefined)
      .then(async () => {
        throwIfCancelled(signal);
        const now = Date.now();
        const spacingUntil = sharedLastAttemptStartedAt + delayMs;
        const waitUntil = Math.max(spacingUntil, sharedCooldownUntil);
        if (waitUntil > now) await abortableDelay(waitUntil - now, signal);
        throwIfCancelled(signal);
        sharedLastAttemptStartedAt = Date.now();
        try {
          return await attempt();
        } catch (error) {
          recordSharedCooldown(error);
          throw error;
        }
      });
    sharedAttemptQueue = coordinated.then(
      () => undefined,
      () => undefined,
    );
    return abortablePromise(coordinated, signal);
  }

  private primaryCategory(): string {
    return this.opts.categories?.[0] ?? this.opts.category ?? "astro-ph";
  }
}

function abortablePromise<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  throwIfCancelled(signal);
  if (!signal) return promise;
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      try {
        throwIfCancelled(signal);
      } catch (error) {
        reject(error);
      }
    };
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) {
      onAbort();
      return;
    }
    promise.then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      },
    );
  });
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

export function isRetryableArxivError(error: unknown): boolean {
  if (!isArxivHttpError(error)) return true;
  return (
    error.status === 408 ||
    error.status === 429 ||
    (error.status >= 500 && error.status <= 599)
  );
}

function recordSharedCooldown(error: unknown): void {
  if (!isArxivHttpError(error)) return;
  if (error.status !== 429 && error.status !== 503) return;
  const retryAfterMs = parseRetryAfterMs(error.headers);
  if (retryAfterMs == null) return;
  sharedCooldownUntil = Math.max(sharedCooldownUntil, Date.now() + retryAfterMs);
}

function arxivRetryDelayMs(
  err: unknown,
  _attempt: number,
  defaultWaitMs: number,
): number {
  const retryAfterMs =
    isArxivHttpError(err) && (err.status === 429 || err.status === 503)
      ? parseRetryAfterMs(err.headers)
      : null;
  return retryAfterMs == null
    ? jitterDelayMs(defaultWaitMs)
    : Math.max(retryAfterMs, jitterDelayMs(defaultWaitMs));
}

function parseRetryAfterMs(headers: Record<string, string>): number | null {
  const value = headerValue(headers, "retry-after")?.trim();
  if (!value) return null;
  if (/^\d+$/.test(value)) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return MAX_ARXIV_RETRY_AFTER_MS;
    return Math.min(MAX_ARXIV_RETRY_AFTER_MS, seconds * 1000);
  }
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return null;
  const delayMs = timestamp - Date.now();
  if (!Number.isFinite(delayMs)) return null;
  return Math.min(MAX_ARXIV_RETRY_AFTER_MS, Math.max(0, delayMs));
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

function isStringRecord(value: unknown): value is Record<string, string> {
  return (
    Boolean(value) &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.values(value as Record<string, unknown>).every(
      (entry) => typeof entry === "string",
    )
  );
}

function normalizeAtomPaperMeta(paper: AtomPaperMeta): AtomPaperMeta | null {
  const primaryCategory = paper.primaryCategory || paper.categories[0] || "";
  const normalized = {
    ...paper,
    primaryCategory,
    categories: primaryCategory && !paper.categories.includes(primaryCategory)
      ? [...paper.categories, primaryCategory]
      : paper.categories,
  };
  return isAtomPaperMeta(normalized) ? normalized : null;
}

/** Reset shared request state between tests. Not intended for runtime use. */
export async function resetArxivRequestCoordinatorForTests(): Promise<void> {
  await sharedAttemptQueue.catch(() => undefined);
  sharedLastAttemptStartedAt = Number.NEGATIVE_INFINITY;
  sharedCooldownUntil = 0;
  sharedAttemptQueue = Promise.resolve();
}

function requireArxivResources(input: string) {
  const resources = modernArxivResources(input);
  if (!resources) throw new Error(`invalid arXiv ID: ${input}`);
  return resources;
}
