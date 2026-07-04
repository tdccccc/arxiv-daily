import type { HostAdapters } from "../core/adapters";
import { buildNodeHostAdapters } from "../hosts/node";
import { LlmClient } from "../llm/client";
import { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import { HtmlCache } from "../pipeline/html-cache";
import { MarkdownWriter } from "../pipeline/markdown-writer";
import {
  cleanupSourceCache,
  PaperContentFetcher,
} from "../pipeline/paper-content";
import { ArxivPipeline } from "../pipeline/pipeline";
import { arxivCategories } from "../settings/categories";
import { RunCancellationService } from "../services/cancellation";
import { Logger } from "../services/logger";
import { ManualFetchService } from "../services/manual-fetch";
import { PaperIndexStore } from "../services/paper-index";
import { RunHistoryStore } from "../services/run-history";
import { RunLock } from "../services/run-lock";
import { SchedulerService } from "../services/scheduler";
import {
  createStorageStateStore,
  type StateStore,
} from "../services/state-store";
import type { CliRuntimeConfig } from "./config";

export interface CliRuntime {
  host: HostAdapters;
  logger: Logger;
  fetcher: ArxivFetcher;
  paperFetcher: PaperContentFetcher;
  writer: MarkdownWriter;
  paperIndex: PaperIndexStore;
  stateStore: StateStore;
  runHistoryStore: RunHistoryStore;
  scheduler: SchedulerService;
  llm: LlmClient;
  pipeline: ArxivPipeline;
  manualFetch: ManualFetchService;
}

export interface BuildCliRuntimeOptions {
  host?: HostAdapters;
  logger?: Logger;
}

export async function buildCliRuntime(
  config: CliRuntimeConfig,
  opts: BuildCliRuntimeOptions = {},
): Promise<CliRuntime> {
  const host = opts.host ?? buildNodeHostAdapters({ rootDir: config.vaultRoot });
  const logger =
    opts.logger ?? new Logger(config.settings.advanced.logLevel, undefined, config.settings.arxiv.timezone);
  const llm = new LlmClient(config.settings.llm, logger, host.http);
  const fetcher = new ArxivFetcher({
    category: config.settings.arxiv.category,
    categories: arxivCategories(config.settings.arxiv),
    http: host.http,
    logger,
    requestDelayMs: config.settings.advanced.requestDelayMs,
  });
  const cache = new HtmlCache({
    rootDir: config.cacheDir,
    expiryDays: config.settings.advanced.cacheExpiryDays,
  });
  await cache.cleanupExpired().catch((e) =>
    logger.warn(`cache cleanup failed: ${(e as Error).message}`),
  );
  const paperFetcher = new PaperContentFetcher(fetcher, cache, logger, {
    storage: host.storage,
    cacheDir: ".arxiv-daily/cache/source",
    expiryDays: config.settings.advanced.cacheExpiryDays,
  });
  await cleanupSourceCache({
    storage: host.storage,
    cacheDir: ".arxiv-daily/cache/source",
    expiryDays: config.settings.advanced.cacheExpiryDays,
  }).catch((e) =>
    logger.warn(`source cache cleanup failed: ${(e as Error).message}`),
  );
  const writer = new MarkdownWriter({
    storage: host.storage,
    logger,
    arxiv: config.settings.arxiv,
    output: config.settings.output,
  });
  await writer.cleanupTemporaryFiles().catch((e) =>
    logger.warn(`markdown temp cleanup failed: ${(e as Error).message}`),
  );
  const paperIndex = new PaperIndexStore(host.storage, config.settings.output);
  const stateStore = createStorageStateStore(
    host.storage,
    config.settings.output,
  );
  const runHistoryStore = RunHistoryStore.fromStorage(
    host.storage,
    config.settings.output,
    logger,
  );
  await stateStore.load();
  const pipeline = new ArxivPipeline({
    fetcher,
    paperFetcher,
    writer,
    paperIndex,
    llm,
    logger,
    arxiv: config.settings.arxiv,
    advanced: config.settings.advanced,
    output: config.settings.output,
    llmSettings: config.settings.llm,
    progress: host.progress,
  });
  const manualFetch = new ManualFetchService({
    storage: host.storage,
    fetcher,
    paperFetcher,
    writer,
    paperIndex,
    llm,
    logger,
    arxiv: config.settings.arxiv,
    advanced: config.settings.advanced,
    output: config.settings.output,
    llmSettings: config.settings.llm,
  });
  const scheduler = new SchedulerService({
    getSettings: () => config.settings,
    store: stateStore,
    lock: new RunLock(),
    runForDate: (date, signal) => pipeline.runForDate(date, signal),
    logger,
    progress: host.progress,
    cancellation: new RunCancellationService(),
    runHistory: runHistoryStore,
    dailyPathForDate: (date) => writer.dailyPath(date),
  });

  return {
    host,
    logger,
    fetcher,
    paperFetcher,
    writer,
    paperIndex,
    stateStore,
    runHistoryStore,
    scheduler,
    llm,
    pipeline,
    manualFetch,
  };
}
