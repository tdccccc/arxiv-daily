import type { HostAdapters } from "../core/adapters";
import { buildNodeHostAdapters } from "../hosts/node";
import { LlmClient } from "../llm/client";
import { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import { HtmlCache } from "../pipeline/html-cache";
import { MarkdownWriter } from "../pipeline/markdown-writer";
import { PaperContentFetcher } from "../pipeline/paper-content";
import { ArxivPipeline } from "../pipeline/pipeline";
import { arxivCategories } from "../settings/categories";
import { Logger } from "../services/logger";
import { ManualFetchService } from "../services/manual-fetch";
import { PaperIndexStore } from "../services/paper-index";
import type { CliRuntimeConfig } from "./config";

export interface CliRuntime {
  host: HostAdapters;
  logger: Logger;
  fetcher: ArxivFetcher;
  paperFetcher: PaperContentFetcher;
  writer: MarkdownWriter;
  paperIndex: PaperIndexStore;
  llm: LlmClient;
  pipeline: ArxivPipeline;
  manualFetch: ManualFetchService;
}

export interface BuildCliRuntimeOptions {
  host?: HostAdapters;
  logger?: Logger;
}

export function buildCliRuntime(
  config: CliRuntimeConfig,
  opts: BuildCliRuntimeOptions = {},
): CliRuntime {
  const host = opts.host ?? buildNodeHostAdapters({ rootDir: config.vaultRoot });
  const logger =
    opts.logger ?? new Logger(config.settings.advanced.logLevel);
  const llm = new LlmClient(config.settings.llm, logger);
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
  const paperFetcher = new PaperContentFetcher(fetcher, cache, logger);
  const writer = new MarkdownWriter({
    storage: host.storage,
    logger,
    arxiv: config.settings.arxiv,
    output: config.settings.output,
  });
  const paperIndex = new PaperIndexStore(host.storage, config.settings.output);
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

  return {
    host,
    logger,
    fetcher,
    paperFetcher,
    writer,
    paperIndex,
    llm,
    pipeline,
    manualFetch,
  };
}
