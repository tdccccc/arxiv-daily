import { ArxivFetcher, Logger, RecentDatesCache, arxivCategories } from "@arxiv-daily/core";
import { LinkedomMarkupParser, NodeHttpClient } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "../config";

/** How long a click waits for arXiv's listings before the first report falls back to today. */
const LISTING_TIMEOUT_MS = 10_000;

/** The Obsidian first report's source: the days that arXiv's /recent listings name. */
export function latestAnnouncedDayFrom(getConfig: () => CliRuntimeConfig | undefined): (onOrBefore: string, signal: AbortSignal) => Promise<string | null> {
  return async (onOrBefore, signal) => {
    const config = getConfig();
    if (!config) return null;
    const logger = new Logger("error");
    const markupParser = new LinkedomMarkupParser();
    const recent = new RecentDatesCache({
      getSettings: () => config.settings,
      buildFetcher: () => new ArxivFetcher({
        category: config.settings.arxiv.category, categories: arxivCategories(config.settings.arxiv),
        http: new NodeHttpClient(), markupParser, logger, requestDelayMs: config.settings.advanced.requestDelayMs,
      }),
      markupParser, logger,
    });
    const { snapshot } = await recent.refreshWithin(LISTING_TIMEOUT_MS, signal);
    return [...snapshot.dates].filter(date => date <= onOrBefore).sort().at(-1) ?? null;
  };
}
