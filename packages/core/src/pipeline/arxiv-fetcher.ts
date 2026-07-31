import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import { parseAtomPapers, type AtomPaperMeta } from "./atom-parser";
import {
  HttpTransportError,
  isHttpTransportError,
  type HttpClient,
  type HttpRequest,
  type HttpResponse,
  type MarkupParser,
} from "../core/adapters";
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
  textTimeoutMs?: number;
  binaryTimeoutMs?: number;
  requestCoordinator?: ArxivRequestCoordinator;
}

export interface ArxivCoordinatorClock {
  monotonicNow(): number;
  wallNow(): number;
  sleep(ms: number, signal?: AbortSignal): Promise<void>;
}

export const DEFAULT_ARXIV_TEXT_TIMEOUT_MS = 60_000;
export const DEFAULT_ARXIV_BINARY_TIMEOUT_MS = 180_000;
const MIN_ARXIV_TIMEOUT_MS = 1;
const MAX_ARXIV_TIMEOUT_MS = 30 * 60 * 1000;
const MIN_ARXIV_REQUEST_DELAY_MS = 3_000;
const MAX_INLINE_ARXIV_COOLDOWN_MS = 30 * 60 * 1000;
const MAX_COOLDOWN_CHUNK_MS = 30_000;
const MAX_DATE_TIMESTAMP_MS = 8_640_000_000_000_000;
const ARXIV_HTTP_ERROR_NAME = "ArxivHttpError";
const ARXIV_RETRY_DEFERRED_ERROR_NAME = "ArxivRetryDeferredError";

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

export class ArxivRetryDeferredError extends Error {
  readonly retryAt: Date;
  readonly remainingMs: number;

  constructor(retryAt: Date, remainingMs: number) {
    const safeRemainingMs = Math.max(0, Math.ceil(remainingMs));
    super(
      `arXiv requested a later retry; retry at ${retryAt.toISOString()} ` +
      `(${formatRemaining(safeRemainingMs)} remaining)`,
    );
    this.name = ARXIV_RETRY_DEFERRED_ERROR_NAME;
    this.retryAt = retryAt;
    this.remainingMs = safeRemainingMs;
  }
}

export function isArxivRetryDeferredError(
  error: unknown,
): error is ArxivRetryDeferredError {
  if (!error || typeof error !== "object") return false;
  const candidate = error as Partial<ArxivRetryDeferredError>;
  return (
    candidate.name === ARXIV_RETRY_DEFERRED_ERROR_NAME &&
    typeof candidate.message === "string" &&
    candidate.retryAt instanceof Date &&
    Number.isFinite(candidate.retryAt.getTime()) &&
    typeof candidate.remainingMs === "number" &&
    Number.isFinite(candidate.remainingMs)
  );
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
  if (isArxivRetryDeferredError(error)) return error.message;
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

const systemCoordinatorClock: ArxivCoordinatorClock = {
  monotonicNow: monotonicNowMs,
  wallNow: () => Date.now(),
  sleep: abortableDelay,
};

export class ArxivRequestCoordinator {
  private lastAttemptStartedAt = Number.NEGATIVE_INFINITY;
  private cooldownUntil = 0;
  private cooldownWallDeadline = 0;
  private attemptQueue: Promise<void> = Promise.resolve();

  constructor(private readonly clock: ArxivCoordinatorClock = systemCoordinatorClock) {}

  coordinate<T>(
    attempt: () => Promise<T>,
    delayMs: number,
    signal?: AbortSignal,
  ): Promise<T> {
    const coordinated = this.attemptQueue
      .catch(() => undefined)
      .then(async () => {
        throwIfCancelled(signal);
        await this.waitUntilEligible(delayMs, signal);
        throwIfCancelled(signal);
        this.lastAttemptStartedAt = this.clock.monotonicNow();
        try {
          return await attempt();
        } catch (error) {
          const deferred = this.recordCooldown(error);
          throw deferred ?? error;
        }
      });
    this.attemptQueue = coordinated.then(
      () => undefined,
      () => undefined,
    );
    return abortablePromise(coordinated, signal);
  }

  async reset(): Promise<void> {
    await Promise.allSettled([this.attemptQueue]);
    this.lastAttemptStartedAt = Number.NEGATIVE_INFINITY;
    this.cooldownUntil = 0;
    this.cooldownWallDeadline = 0;
    this.attemptQueue = Promise.resolve();
  }

  private async waitUntilEligible(delayMs: number, signal?: AbortSignal): Promise<void> {
    while (true) {
      throwIfCancelled(signal);
      const now = this.clock.monotonicNow();
      const cooldownRemaining = this.cooldownUntil - now;
      if (cooldownRemaining > MAX_INLINE_ARXIV_COOLDOWN_MS) {
        throw this.deferredError(cooldownRemaining);
      }
      const spacingRemaining = this.lastAttemptStartedAt + delayMs - now;
      const remaining = Math.max(cooldownRemaining, spacingRemaining);
      if (remaining <= 0) return;
      await this.clock.sleep(Math.min(remaining, MAX_COOLDOWN_CHUNK_MS), signal);
    }
  }

  private recordCooldown(error: unknown): ArxivRetryDeferredError | undefined {
    if (!isArxivHttpError(error)) return undefined;
    if (error.status !== 429 && error.status !== 503) return undefined;
    const receivedWallMs = this.clock.wallNow();
    const retryAfterMs = parseRetryAfterMs(error.headers, receivedWallMs);
    if (retryAfterMs == null) return undefined;
    const monotonicDeadline = safeAdd(this.clock.monotonicNow(), retryAfterMs);
    if (monotonicDeadline > this.cooldownUntil) {
      this.cooldownUntil = monotonicDeadline;
      this.cooldownWallDeadline = safeAdd(receivedWallMs, retryAfterMs);
    }
    const remainingMs = this.cooldownUntil - this.clock.monotonicNow();
    return remainingMs > MAX_INLINE_ARXIV_COOLDOWN_MS
      ? this.deferredError(remainingMs)
      : undefined;
  }

  private deferredError(remainingMs: number): ArxivRetryDeferredError {
    const retryAtMs = this.cooldownWallDeadline > 0
      ? this.cooldownWallDeadline
      : safeAdd(this.clock.wallNow(), remainingMs);
    return new ArxivRetryDeferredError(
      new Date(Math.min(MAX_DATE_TIMESTAMP_MS, retryAtMs)),
      remainingMs,
    );
  }
}

const sharedRequestCoordinator = new ArxivRequestCoordinator();

interface MetadataFlight {
  promise: Promise<AtomPaperMeta | null>;
  resolve: (paper: AtomPaperMeta | null) => void;
  reject: (error: unknown) => void;
}

const sharedMetadataFlights = new Map<string, MetadataFlight>();
const sharedMetadataWorkers = new Set<Promise<void>>();

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
    throwIfCancelled(signal);
    const canonicalIds = Array.from(new Set(ids.flatMap((id) => {
      const canonical = modernArxivResources(id)?.id;
      return canonical ? [canonical] : [];
    })));
    const out = new Map<string, AtomPaperMeta>();
    const misses: string[] = [];
    for (const id of canonicalIds) {
      throwIfCancelled(signal);
      const cached = this.opts.metadataCache
        ? await abortablePromise(this.opts.metadataCache.get(id, signal), signal)
        : null;
      if (cached) out.set(id, cached);
      else misses.push(id);
    }

    // Install every missing ID synchronously, before the first await, so overlapping
    // callers can join even when they use another fetcher or no persistent cache.
    const flights = new Map<string, MetadataFlight>();
    const owned: string[] = [];
    for (const id of misses) {
      const existing = sharedMetadataFlights.get(id);
      if (existing) {
        flights.set(id, existing);
        continue;
      }
      const flight = createMetadataFlight();
      sharedMetadataFlights.set(id, flight);
      flights.set(id, flight);
      owned.push(id);
    }

    if (owned.length > 0) {
      const ownedFlights = new Map(owned.map((id) => [id, flights.get(id)!]));
      const onOwnerAbort = () => {
        const error = cancellationError(signal);
        for (const [id, flight] of ownedFlights) {
          rejectMetadataFlight(id, flight, error);
        }
      };
      signal?.addEventListener("abort", onOwnerAbort, { once: true });
      if (signal?.aborted) onOwnerAbort();
      const worker = this.runOwnedMetadataFlights(ownedFlights, signal);
      sharedMetadataWorkers.add(worker);
      void worker.finally(() => {
        signal?.removeEventListener("abort", onOwnerAbort);
        sharedMetadataWorkers.delete(worker);
      }).catch(() => undefined);
    }

    const ownedSet = new Set(owned);
    for (const [id, flight] of flights) {
      const paper = await abortablePromise(flight.promise, signal);
      if (!paper) continue;
      out.set(id, paper);
      if (!ownedSet.has(id)) await this.persistMetadataBestEffort(paper, signal);
    }
    return out;
  }

  private async runOwnedMetadataFlights(
    ownedFlights: Map<string, MetadataFlight>,
    signal?: AbortSignal,
  ): Promise<void> {
    const owned = [...ownedFlights.keys()];
    const unresolved = new Set(owned);
    try {
      const needsHttp: string[] = [];
      for (const id of owned) {
        throwIfCancelled(signal);
        const cacheRead = this.opts.metadataCache?.get(id, signal);
        const cached = cacheRead
          ? await abortablePromise(cacheRead, signal)
          : null;
        if (cached) {
          settleMetadataFlight(id, ownedFlights.get(id), cached, unresolved);
        } else {
          needsHttp.push(id);
        }
      }

      const BATCH = 200;
      for (let i = 0; i < needsHttp.length; i += BATCH) {
        throwIfCancelled(signal);
        const batch = needsHttp.slice(i, i + BATCH).filter((id) => unresolved.has(id));
        if (batch.length === 0) continue;
        const requested = new Set(batch);
        const url = `https://export.arxiv.org/api/query?id_list=${batch.join(",")}&max_results=${batch.length}`;
        const xml = await this.fetchHtml(url, { allow404: false }, signal);
        const positives = new Map<string, AtomPaperMeta>();
        for (const paper of parseAtomPapers(xml, this.opts.markupParser)) {
          if (!requested.has(paper.id)) continue;
          const normalized = normalizeAtomPaperMeta(paper);
          if (normalized) positives.set(normalized.id, normalized);
        }
        for (const id of batch) {
          const paper = positives.get(id) ?? null;
          if (paper) await this.persistMetadataBestEffort(paper, signal);
          settleMetadataFlight(id, ownedFlights.get(id), paper, unresolved);
        }
      }
    } catch (error) {
      for (const id of unresolved) {
        rejectMetadataFlight(id, ownedFlights.get(id), error);
      }
      unresolved.clear();
      throw error;
    }
  }

  private async persistMetadataBestEffort(
    paper: AtomPaperMeta,
    signal?: AbortSignal,
  ): Promise<void> {
    try {
      const cacheWrite = this.opts.metadataCache?.set(paper.id, paper, signal);
      if (cacheWrite) await abortablePromise(cacheWrite, signal);
    } catch (error) {
      if (isCancellationError(error)) throw error;
      this.opts.logger.warn(`Atom metadata cache write failed for ${paper.id}`, error);
    }
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
    const timeoutMs = boundedTimeoutMs(
      this.opts.textTimeoutMs,
      DEFAULT_ARXIV_TEXT_TIMEOUT_MS,
    );
    return retry(
      async () => this.coordinateAttempt(async () => {
        const res = await requestWithWatchdog(this.opts.http, {
          url,
          method: "GET",
          headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
          timeoutMs,
          signal,
        }, timeoutMs);
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
    const timeoutMs = boundedTimeoutMs(
      this.opts.binaryTimeoutMs,
      DEFAULT_ARXIV_BINARY_TIMEOUT_MS,
    );
    return retry(
      async () => this.coordinateAttempt(async () => {
        const res = await requestWithWatchdog(this.opts.http, {
          url,
          method: "GET",
          headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
          responseType: "arrayBuffer",
          timeoutMs,
          signal,
        }, timeoutMs);
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
    return (this.opts.requestCoordinator ?? sharedRequestCoordinator).coordinate(
      attempt,
      delayMs,
      signal,
    );
  }

  private primaryCategory(): string {
    return this.opts.categories?.[0] ?? this.opts.category ?? "astro-ph";
  }
}

function createMetadataFlight(): MetadataFlight {
  let resolve!: (paper: AtomPaperMeta | null) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<AtomPaperMeta | null>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  // A caller consumes flights in ID order, so a later batch may reject before
  // that caller reaches its promise. Mark it handled without changing its result.
  void promise.catch(() => undefined);
  return { promise, resolve, reject };
}

function settleMetadataFlight(
  id: string,
  flight: MetadataFlight | undefined,
  paper: AtomPaperMeta | null,
  unresolved: Set<string>,
): void {
  if (!flight || !unresolved.delete(id)) return;
  if (sharedMetadataFlights.get(id) === flight) sharedMetadataFlights.delete(id);
  flight.resolve(paper);
}

function rejectMetadataFlight(
  id: string,
  flight: MetadataFlight | undefined,
  error: unknown,
): void {
  if (!flight) return;
  if (sharedMetadataFlights.get(id) === flight) sharedMetadataFlights.delete(id);
  flight.reject(error);
}

function cancellationError(signal?: AbortSignal): unknown {
  try {
    throwIfCancelled(signal);
  } catch (error) {
    return error;
  }
  return new Error("cancelled by user");
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
  if (isArxivRetryDeferredError(error)) return false;
  if (isHttpTransportError(error)) return error.retryableAttempt === true;
  if (!isArxivHttpError(error)) return false;
  return (
    error.status === 408 ||
    error.status === 429 ||
    (error.status >= 500 && error.status <= 599)
  );
}

function boundedTimeoutMs(value: number | undefined, fallback: number): number {
  if (value == null || !Number.isFinite(value)) return fallback;
  return Math.min(MAX_ARXIV_TIMEOUT_MS, Math.max(MIN_ARXIV_TIMEOUT_MS, value));
}

function requestWithWatchdog(
  http: HttpClient,
  req: HttpRequest,
  timeoutMs: number,
): Promise<HttpResponse> {
  throwIfCancelled(req.signal);
  let operation: Promise<HttpResponse>;
  try {
    operation = Promise.resolve(http.request(req));
  } catch (error) {
    operation = Promise.reject(error);
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timeout);
      req.signal?.removeEventListener("abort", onAbort);
    };
    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      cleanup();
      fn();
    };
    const onAbort = () => settle(() => {
      try {
        throwIfCancelled(req.signal);
      } catch (error) {
        reject(error);
      }
    });
    const timeout = setTimeout(
      () => settle(() => reject(new HttpTransportError(
        "timeout",
        `HTTP timeout after ${timeoutMs}ms: ${req.url}`,
        // Core cannot know whether an arbitrary adapter stopped physical I/O.
        // Fail closed: the operation remains transient to callers but is not
        // retried immediately unless the host explicitly marks it safe.
        { retryableAttempt: false },
      ))),
      timeoutMs,
    );
    req.signal?.addEventListener("abort", onAbort, { once: true });
    if (req.signal?.aborted) onAbort();
    operation.then(
      (response) => settle(() => resolve(response)),
      (error) => settle(() => reject(error)),
    );
  });
}

function arxivRetryDelayMs(
  _err: unknown,
  _attempt: number,
  defaultWaitMs: number,
): number {
  // The coordinator owns server minimums using the receipt-time wall clock and
  // a monotonic deadline. retry() only adds ordinary client backoff.
  return jitterDelayMs(defaultWaitMs);
}

export function parseRetryAfterMs(
  headers: Record<string, string>,
  receivedWallMs = Date.now(),
): number | null {
  const value = headerValue(headers, "retry-after")?.trim();
  if (!value) return null;
  if (/^\d+$/.test(value)) {
    try {
      const milliseconds = BigInt(value) * 1000n;
      const maxDelay = BigInt(Math.max(0, MAX_DATE_TIMESTAMP_MS - receivedWallMs));
      return Number(milliseconds > maxDelay ? maxDelay : milliseconds);
    } catch {
      return null;
    }
  }
  // RFC 7231 Retry-After dates are IMF-fixdate only. Validate the exact wire
  // grammar before Date.parse so obsolete or implementation-specific forms do
  // not accidentally become server minimums.
  if (!/^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (?:0[1-9]|[12]\d|3[01]) (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4} (?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d GMT$/.test(value)) {
    return null;
  }
  const timestamp = Date.parse(value);
  if (
    !Number.isFinite(timestamp) ||
    new Date(timestamp).toUTCString() !== value ||
    !Number.isFinite(receivedWallMs)
  ) return null;
  const delayMs = timestamp - receivedWallMs;
  if (!Number.isSafeInteger(delayMs)) return null;
  return Math.max(0, delayMs);
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

function monotonicNowMs(): number {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

function safeAdd(left: number, right: number): number {
  const sum = left + right;
  return Number.isFinite(sum) && Math.abs(sum) <= Number.MAX_SAFE_INTEGER
    ? sum
    : Number.MAX_SAFE_INTEGER;
}

function formatRemaining(ms: number): string {
  const seconds = Math.max(0, Math.ceil(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.ceil(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.ceil(minutes / 60);
  return `${hours}h`;
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

/** Reset shared request and metadata-flight state between tests. Not intended for runtime use. */
export async function resetArxivRequestCoordinatorForTests(): Promise<void> {
  await Promise.allSettled([...sharedMetadataWorkers]);
  await sharedRequestCoordinator.reset();
  for (const [id, flight] of sharedMetadataFlights) {
    rejectMetadataFlight(id, flight, new Error("arXiv metadata flight reset"));
  }
  sharedMetadataFlights.clear();
  sharedMetadataWorkers.clear();
}

function requireArxivResources(input: string) {
  const resources = modernArxivResources(input);
  if (!resources) throw new Error(`invalid arXiv ID: ${input}`);
  return resources;
}
