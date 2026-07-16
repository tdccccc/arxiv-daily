import type { MarkupParser } from "../core/adapters";
import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import { parseRecent } from "../pipeline/arxiv-parser";
import { arxivCategories } from "../settings/categories";
import type { PluginSettings } from "../settings/types";
import type { Logger } from "./logger";
import { RunCancelledError, throwIfCancelled } from "./cancellation";

type RecentFetcher = Pick<ArxivFetcher, "fetchRecent">;
type RecentLogger = Pick<Logger, "debug" | "warn">;

export type RecentDatesStatus = "idle" | "ready" | "failed";

export interface RecentDatesSnapshot {
  status: RecentDatesStatus;
  dates: Set<string>;
  refreshedAt: number;
  error?: string;
}

export interface RecentDatesRefreshResult {
  snapshot: RecentDatesSnapshot;
  refresh: Promise<RecentDatesSnapshot>;
  completed: boolean;
  timedOut: boolean;
}

export interface RecentDatesCacheDeps {
  getSettings: () => PluginSettings;
  buildFetcher: () => RecentFetcher;
  markupParser: MarkupParser;
  logger: RecentLogger;
  now?: () => Date;
  ttlMs?: number;
}

const DEFAULT_RECENT_DATES_TTL_MS = 10 * 60_000;

export class RecentDatesCache {
  private state: RecentDatesSnapshot = {
    status: "idle",
    dates: new Set(),
    refreshedAt: 0,
  };
  private inFlight: Promise<RecentDatesSnapshot> | null = null;

  constructor(private readonly deps: RecentDatesCacheDeps) {}

  snapshot(): RecentDatesSnapshot {
    return cloneSnapshot(this.state);
  }

  hasDate(date: string): boolean {
    return this.state.dates.has(date);
  }

  async refresh(signal?: AbortSignal): Promise<RecentDatesSnapshot> {
    const caller = callerWait(this.ensureRefresh(), signal);
    try {
      return await caller.promise;
    } finally {
      caller.dispose();
    }
  }

  async refreshWithin(timeoutMs: number, signal?: AbortSignal): Promise<RecentDatesRefreshResult> {
    const underlyingRefresh = this.ensureRefresh();
    const caller = callerWait(underlyingRefresh, signal);
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<"timeout">((resolve) => {
      timeoutHandle = setTimeout(() => resolve("timeout"), Math.max(0, timeoutMs));
    });
    try {
      const result = await Promise.race([caller.promise, timeout]);

      if (result === "timeout") {
        return {
          snapshot: this.snapshot(),
          refresh: underlyingRefresh,
          completed: false,
          timedOut: true,
        };
      }

      return {
        snapshot: cloneSnapshot(result),
        refresh: underlyingRefresh,
        completed: true,
        timedOut: false,
      };
    } finally {
      caller.dispose();
      if (timeoutHandle !== undefined) clearTimeout(timeoutHandle);
    }
  }

  private ensureRefresh(): Promise<RecentDatesSnapshot> {
    if (this.inFlight) return this.inFlight;
    const now = (this.deps.now ?? (() => new Date()))().getTime();
    const ttlMs = this.deps.ttlMs ?? DEFAULT_RECENT_DATES_TTL_MS;
    if (
      this.state.status === "ready" &&
      now - this.state.refreshedAt < ttlMs
    ) {
      return Promise.resolve(this.snapshot());
    }
    const refresh = this.doRefresh().finally(() => {
      if (this.inFlight === refresh) this.inFlight = null;
    });
    this.inFlight = refresh;
    // A caller may stop waiting after cancellation while the host request keeps
    // running. Keep the shared refresh observed until another caller joins it.
    void refresh.catch(() => undefined);
    return refresh;
  }

  private async doRefresh(): Promise<RecentDatesSnapshot> {
    const settings = this.deps.getSettings();
    const categories = arxivCategories(settings.arxiv);
    const fetcher = this.deps.buildFetcher();
    const dates = new Set<string>();
    const errors: string[] = [];

    for (const category of categories) {
      try {
        // Obsidian requestUrl cannot be interrupted. The shared refresh remains
        // independent from any one caller and may complete after callers cancel.
        const html = await fetcher.fetchRecent(category);
        const buckets = parseRecent(html, this.deps.markupParser);
        for (const bucket of buckets) dates.add(bucket.announceDate);
      } catch (e) {
        const message = (e as Error).message;
        errors.push(`${category}: ${message}`);
        this.deps.logger.warn(
          `recent dates refresh failed for ${category}: ${message}`,
        );
      }
    }

    const refreshedAt = (this.deps.now ?? (() => new Date()))().getTime();
    this.state =
      dates.size > 0
        ? {
            status: "ready",
            dates,
            refreshedAt,
            error: errors.length > 0 ? errors.join("; ") : undefined,
          }
        : {
            status: "failed",
            dates: new Set(),
            refreshedAt,
            error: errors.join("; ") || "no /recent dates found",
          };

    this.deps.logger.debug(
      `recent dates refreshed: ${this.state.status}, ${this.state.dates.size} dates`,
    );
    return this.snapshot();
  }
}

function callerWait<T>(
  shared: Promise<T>,
  signal?: AbortSignal,
): { promise: Promise<T>; dispose(): void } {
  throwIfCancelled(signal);
  if (!signal) return { promise: shared, dispose: () => undefined };
  let settled = false;
  let rejectWait: (error: unknown) => void = () => undefined;
  const cleanup = () => signal.removeEventListener("abort", onAbort);
  const onAbort = () => {
    if (settled) return;
    settled = true;
    cleanup();
    const reason = (signal as AbortSignal & { reason?: unknown }).reason;
    rejectWait(
      new RunCancelledError(
        typeof reason === "string" && reason ? reason : "cancelled by user",
      ),
    );
  };
  const promise = new Promise<T>((resolve, reject) => {
    rejectWait = reject;
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) {
      onAbort();
      return;
    }
    shared.then(
      (value) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(value);
      },
      (error) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(error);
      },
    );
  });
  return {
    promise,
    dispose: () => {
      if (settled) return;
      settled = true;
      cleanup();
    },
  };
}

function cloneSnapshot(snapshot: RecentDatesSnapshot): RecentDatesSnapshot {
  return {
    status: snapshot.status,
    dates: new Set(snapshot.dates),
    refreshedAt: snapshot.refreshedAt,
    error: snapshot.error,
  };
}
