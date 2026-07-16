import type { HostAdapters } from "@arxiv-daily/core";
import { buildNodeHostAdapters, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { LlmClient } from "@arxiv-daily/core";
import { ArxivFetcher } from "@arxiv-daily/core";
import { HtmlCache } from "@arxiv-daily/core";
import { MarkdownWriter } from "@arxiv-daily/core";
import {
  cleanupSourceCache,
  PaperContentFetcher,
} from "@arxiv-daily/core";
import { ArxivPipeline } from "@arxiv-daily/core";
import { arxivCategories } from "@arxiv-daily/core";
import { OperationRegistry, RunCancellationService } from "@arxiv-daily/core";
import { Logger } from "@arxiv-daily/core";
import { ManualFetchService } from "@arxiv-daily/core";
import { PaperIndexStore } from "@arxiv-daily/core";
import { RunHistoryStore } from "@arxiv-daily/core";
import { RunLock } from "@arxiv-daily/core";
import { SchedulerService } from "@arxiv-daily/core";
import {
  createStorageStateStore,
  type StateStore,
} from "@arxiv-daily/core";
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
  operations: OperationRegistry;
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
  logger.setSensitiveValues([config.settings.llm.apiKey]);
  const llm = new LlmClient(config.settings.llm, logger, host.http);
  const fetcher = new ArxivFetcher({
    category: config.settings.arxiv.category,
    categories: arxivCategories(config.settings.arxiv),
    http: host.http,
    markupParser: host.markupParser,
    logger,
    requestDelayMs: config.settings.advanced.requestDelayMs,
  });
  const cache = new HtmlCache({
    rootDir: "",
    expiryDays: config.settings.advanced.cacheExpiryDays,
    storage: new NodeStorageAdapter(config.cacheDir),
  });
  await cache.cleanupExpired().catch((e) =>
    logger.warn(`cache cleanup failed: ${(e as Error).message}`),
  );
  const paperFetcher = new PaperContentFetcher(fetcher, cache, logger, host.markupParser, {
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
    markupParser: host.markupParser,
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
    markupParser: host.markupParser,
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
  const operations = new OperationRegistry();
  const scheduler = new SchedulerService({
    getSettings: () => config.settings,
    store: stateStore,
    lock: new RunLock(),
    runForDate: (date, signal) => pipeline.runForDate(date, signal),
    logger,
    progress: host.progress,
    cancellation: new RunCancellationService(operations),
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
    operations,
  };
}
