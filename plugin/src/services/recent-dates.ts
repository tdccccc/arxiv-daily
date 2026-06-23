import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import { parseRecent } from "../pipeline/arxiv-parser";
import { arxivCategories } from "../settings/categories";
import type { PluginSettings } from "../settings/types";
import type { Logger } from "./logger";

type RecentFetcher = Pick<ArxivFetcher, "fetchRecent">;
type RecentLogger = Pick<Logger, "debug" | "warn">;

export type RecentDatesStatus = "idle" | "ready" | "failed";

export interface RecentDatesSnapshot {
  status: RecentDatesStatus;
  dates: Set<string>;
  refreshedAt: number;
  error?: string;
}

export interface RecentDatesCacheDeps {
  getSettings: () => PluginSettings;
  buildFetcher: () => RecentFetcher;
  logger: RecentLogger;
  now?: () => Date;
}

export class RecentDatesCache {
  private state: RecentDatesSnapshot = {
    status: "idle",
    dates: new Set(),
    refreshedAt: 0,
  };

  constructor(private readonly deps: RecentDatesCacheDeps) {}

  snapshot(): RecentDatesSnapshot {
    return cloneSnapshot(this.state);
  }

  hasDate(date: string): boolean {
    return this.state.dates.has(date);
  }

  async refresh(): Promise<RecentDatesSnapshot> {
    const settings = this.deps.getSettings();
    const categories = arxivCategories(settings.arxiv);
    const fetcher = this.deps.buildFetcher();
    const dates = new Set<string>();
    const errors: string[] = [];

    for (const category of categories) {
      try {
        const html = await fetcher.fetchRecent(category);
        const buckets = parseRecent(html);
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

function cloneSnapshot(snapshot: RecentDatesSnapshot): RecentDatesSnapshot {
  return {
    status: snapshot.status,
    dates: new Set(snapshot.dates),
    refreshedAt: snapshot.refreshedAt,
    error: snapshot.error,
  };
}
