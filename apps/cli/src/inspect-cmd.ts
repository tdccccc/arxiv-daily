import * as path from "node:path";
import {
  PaperIndexStore, PaperSearchIndex, arxivCategories, createStorageStateStore,
  validateFilterConfig, validateLlmConfig,
} from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "./config";

/** Read product state without constructing a generation runtime or returning secrets. */
export async function inspectProduct(config: CliRuntimeConfig) {
  const storage = new NodeStorageAdapter(config.vaultRoot);
  const index = new PaperIndexStore(storage, config.settings.output);
  const state = createStorageStateStore(storage, config.settings.output);
  const [inspection] = await Promise.all([index.inspect(), state.load()]);
  return {
    configPath: config.configPath,
    vaultRoot: config.vaultRoot,
    output: {
      dailyDirectory: path.join(config.vaultRoot, config.settings.output.dailyDir),
      papersDirectory: path.join(config.vaultRoot, config.settings.output.papersDir),
      summaryLanguage: config.settings.output.summaryLanguage,
      linkStyle: config.linkStyle,
    },
    llm: {
      provider: config.settings.llm.provider,
      model: config.settings.llm.model,
      keyConfigured: Boolean(config.settings.llm.apiKey.trim()),
      ready: validateLlmConfig(config.settings).ok,
    },
    dailyReady: validateFilterConfig(config.settings).ok,
    categories: arxivCategories(config.settings.arxiv),
    topics: config.settings.arxiv.topics.map(({ name, tag, description, detail }) => ({ name, tag, description, detail })),
    emailEnabled: config.settings.email.enabled,
    scheduleIntent: config.scheduleIntent,
    paperCount: Object.keys(inspection.inbox.papers).length,
    indexRecoveredFromBackup: inspection.recoveredFromBackup,
    recentRuns: Object.entries(state.snapshot()).sort(([a], [b]) => b.localeCompare(a)).slice(0, 20)
      .map(([date, entry]) => ({ date, ...entry })),
  };
}

/** Query the same Paper Index and lexical ranking used by the reading Dashboard. */
export async function inspectPapers(config: CliRuntimeConfig, query: string, offset: number, limit: number) {
  const storage = new NodeStorageAdapter(config.vaultRoot);
  const { inbox, recoveredFromBackup } = await new PaperIndexStore(storage, config.settings.output).inspect();
  const entries = Object.values(inbox.papers);
  const matches = query.trim()
    ? new PaperSearchIndex(entries).search(query).map(result => result.entry)
    : entries.sort((a, b) => b.updated.localeCompare(a.updated) || a.paperKey.localeCompare(b.paperKey));
  return {
    vaultRoot: config.vaultRoot,
    query, total: matches.length, offset, limit,
    nextOffset: offset + limit < matches.length ? offset + limit : null,
    recoveredFromBackup,
    papers: matches.slice(offset, offset + limit),
  };
}
