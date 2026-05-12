import { Plugin } from "obsidian";
import * as path from "node:path";
import * as os from "node:os";
import { DEFAULT_SETTINGS } from "./src/settings/defaults";
import type { PluginSettings, RunState } from "./src/settings/types";
import { ArxivDailySettingTab } from "./src/settings/tab";
import { Logger } from "./src/services/logger";
import { StateStore } from "./src/services/state-store";
import { RunLock } from "./src/services/run-lock";
import { SchedulerService } from "./src/services/scheduler";
import { LlmClient } from "./src/llm/client";
import { ArxivFetcher } from "./src/pipeline/arxiv-fetcher";
import { HtmlCache } from "./src/pipeline/html-cache";
import { PaperContentFetcher } from "./src/pipeline/paper-content";
import { MarkdownWriter } from "./src/pipeline/markdown-writer";
import { ArxivPipeline } from "./src/pipeline/pipeline";
import { ManualFetchService } from "./src/services/manual-fetch";
import { registerCommands } from "./src/commands";

interface PersistedData {
  settings: PluginSettings;
  runState: RunState;
}

export default class ArxivDailyPlugin extends Plugin {
  settings!: PluginSettings;
  logger!: Logger;
  stateStore!: StateStore;
  scheduler!: SchedulerService;
  manualFetch!: { fetchAndSummarize: ManualFetchService["fetchAndSummarize"] };
  private runLock = new RunLock();

  async onload() {
    await this.loadSettingsAndState();
    this.logger = new Logger(this.settings.advanced.logLevel);

    this.stateStore = new StateStore(
      async () => {
        const data = (await this.loadData()) as PersistedData | null;
        return { runState: data?.runState ?? {} };
      },
      async ({ runState }) => {
        await this.persistAll(runState);
      },
    );
    await this.stateStore.load();

    this.scheduler = new SchedulerService({
      getSettings: () => this.settings,
      store: this.stateStore,
      lock: this.runLock,
      logger: this.logger,
      runForDate: (date) => this.buildPipeline().runForDate(date),
    });

    // Wrap in an object that rebuilds dependencies on every call so settings
    // changes (model, key, paths) always take effect without needing to reload.
    this.manualFetch = {
      fetchAndSummarize: (raw: string, date: string) =>
        this.buildManualFetch().fetchAndSummarize(raw, date),
    };

    this.addSettingTab(new ArxivDailySettingTab(this.app, this));
    registerCommands(this);

    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  onunload() {
    this.scheduler?.stop();
  }

  async saveSettings(): Promise<void> {
    await this.persistAll(this.stateStore?.snapshot() ?? {});
  }

  restartScheduler(): void {
    this.scheduler.stop();
    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  private async loadSettingsAndState(): Promise<void> {
    const data = ((await this.loadData()) as PersistedData | null) ?? {
      settings: DEFAULT_SETTINGS,
      runState: {},
    };
    this.settings = mergeSettings(DEFAULT_SETTINGS, data.settings ?? ({} as PluginSettings));
  }

  private async persistAll(runState: RunState): Promise<void> {
    const data: PersistedData = { settings: this.settings, runState };
    await this.saveData(data);
  }

  private buildPipeline(): ArxivPipeline {
    const { llm, fetcher, paperFetcher, writer } = this.buildSharedDeps();
    return new ArxivPipeline({
      fetcher,
      paperFetcher,
      writer,
      llm,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      advanced: this.settings.advanced,
      output: this.settings.output,
      llmSettings: this.settings.llm,
    });
  }

  private buildManualFetch(): ManualFetchService {
    const { llm, fetcher, paperFetcher, writer } = this.buildSharedDeps();
    return new ManualFetchService({
      vault: this.app.vault,
      fetcher,
      paperFetcher,
      writer,
      llm,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      advanced: this.settings.advanced,
      output: this.settings.output,
      llmSettings: this.settings.llm,
    });
  }

  private buildSharedDeps() {
    const llm = new LlmClient(this.settings.llm, this.logger);
    const fetcher = new ArxivFetcher({
      category: this.settings.arxiv.category,
      logger: this.logger,
      requestDelayMs: this.settings.advanced.requestDelayMs,
    });
    const cache = new HtmlCache({
      rootDir: this.resolveCacheDir(),
      expiryDays: this.settings.advanced.cacheExpiryDays,
    });
    const paperFetcher = new PaperContentFetcher(fetcher, cache, this.logger);
    const writer = new MarkdownWriter({
      vault: this.app.vault,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      output: this.settings.output,
    });
    return { llm, fetcher, paperFetcher, writer };
  }

  private resolveCacheDir(): string {
    let base: string | null = null;
    try {
      const electron = (globalThis as any).require?.("electron");
      if (electron) {
        base =
          (electron.remote ? electron.remote.app : electron.app)?.getPath?.(
            "userData",
          ) ?? null;
      }
    } catch {
      base = null;
    }
    if (!base) base = os.homedir();
    return path.join(base, "arxiv-daily-cache");
  }
}

function mergeSettings(
  defaults: PluginSettings,
  partial: Partial<PluginSettings>,
): PluginSettings {
  return {
    llm: { ...defaults.llm, ...(partial.llm ?? {}) },
    arxiv: { ...defaults.arxiv, ...(partial.arxiv ?? {}) },
    output: { ...defaults.output, ...(partial.output ?? {}) },
    schedule: { ...defaults.schedule, ...(partial.schedule ?? {}) },
    advanced: { ...defaults.advanced, ...(partial.advanced ?? {}) },
  };
}
