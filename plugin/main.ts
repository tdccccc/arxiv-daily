import { Notice, Plugin } from "obsidian";
import * as path from "node:path";
import * as os from "node:os";
import { DEFAULT_SETTINGS } from "./src/settings/defaults";
import type { PluginSettings, RunState } from "./src/settings/types";
import { ArxivDailySettingTab } from "./src/settings/tab";
import { migrateArxivSettings } from "./src/settings/migration";
import { validateFilterConfig } from "./src/settings/validation";
import { Logger } from "./src/services/logger";
import { StateStore } from "./src/services/state-store";
import { RunLock } from "./src/services/run-lock";
import { RunCancellationService } from "./src/services/cancellation";
import { SchedulerService } from "./src/services/scheduler";
import { StatusBarController } from "./src/services/status-bar";
import { NoopProgressReporter, type ProgressReporter } from "./src/services/progress";
import { chooseModal } from "./src/services/modal";
import { LlmClient } from "./src/llm/client";
import { ArxivFetcher } from "./src/pipeline/arxiv-fetcher";
import { HtmlCache } from "./src/pipeline/html-cache";
import { PaperContentFetcher } from "./src/pipeline/paper-content";
import { MarkdownWriter } from "./src/pipeline/markdown-writer";
import { ArxivPipeline } from "./src/pipeline/pipeline";
import { ManualFetchService } from "./src/services/manual-fetch";
import { registerCommands } from "./src/commands";
import { todayInTz, formatDate } from "./src/utils/time";
import { PaperIndexStore } from "./src/services/paper-index";

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
  progress!: ProgressReporter;
  private runLock = new RunLock();
  private runCancellation = new RunCancellationService();

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

    try {
      this.progress = new StatusBarController(
        this.addStatusBarItem(),
        this.stateStore,
        { initiallyEnabled: this.settings.schedule.enabled },
      );
    } catch (e) {
      this.logger.warn("status bar unavailable, using noop", e);
      this.progress = new NoopProgressReporter();
    }

    this.scheduler = new SchedulerService({
      getSettings: () => this.settings,
      store: this.stateStore,
      lock: this.runLock,
      logger: this.logger,
      runForDate: (date, signal) => this.buildPipeline().runForDate(date, signal),
      progress: this.progress,
      cancellation: this.runCancellation,
    });

    // Wrap in an object that rebuilds dependencies on every call so settings
    // changes (model, key, paths) always take effect without needing to reload.
    this.manualFetch = {
      fetchAndSummarize: (raw: string, date: string) =>
        this.buildManualFetch().fetchAndSummarize(raw, date),
    };

    this.addSettingTab(new ArxivDailySettingTab(this.app, this));
    registerCommands(this);

    if (this.settings.schedule.enabled) {
      this.scheduler.start();
      this.scheduler
        .tickTodayScheduled()
        .catch((e) =>
          this.logger.error("scheduler initial tickTodayScheduled failed", e),
        );
    }
  }

  onunload() {
    this.scheduler?.cancelCurrentRun("plugin unloaded");
    this.scheduler?.stop();
  }

  async saveSettings(): Promise<void> {
    await this.persistAll(this.stateStore?.snapshot() ?? {});
  }

  restartScheduler(): void {
    this.scheduler.stop();
    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  async setScheduleEnabled(enabled: boolean): Promise<boolean> {
    if (this.settings.schedule.enabled === enabled) return true;

    if (enabled) {
      const v = validateFilterConfig(this.settings);
      if (!v.ok) {
        new Notice(`Cannot enable arXiv Daily:\n${v.reasons.map((r) => "• " + r).join("\n")}`, 10_000);
        return false;
      }
      const choice = await chooseModal(
        this.app,
        "Enable arXiv Daily",
        "Scheduler will check for new papers daily. Run today's summary right now?",
        [
          { label: "Cancel", value: "cancel" },
          { label: "Skip today", value: "skip" },
          { label: "Run today", value: "run", cta: true },
        ],
      );
      if (choice === null || choice === "cancel") return false;

      this.settings.schedule.enabled = true;
      await this.saveSettings();
      this.scheduler.start();
      if (choice === "skip") {
        const today = formatDate(todayInTz(new Date(), this.settings.arxiv.timezone));
        await this.stateStore.setSkipped(today, "user opted out at enable time");
        this.logger.notice("arXiv Daily: enabled. Today skipped — will run on next workday.");
      } else {
        const result = await this.scheduler.tickToday();
        if (result && (result as any).kind === "skipped" && (result as any).reason === "weekend") {
          this.logger.notice("arXiv Daily: weekend, no update — will check next workday");
        }
      }
      return true;
    }

    this.settings.schedule.enabled = false;
    await this.saveSettings();
    this.scheduler.stop();
    this.progress.setDisabled();
    return true;
  }

  private async loadSettingsAndState(): Promise<void> {
    const data = ((await this.loadData()) as PersistedData | null) ?? {
      settings: DEFAULT_SETTINGS,
      runState: {},
    };
    const merged = mergeSettings(DEFAULT_SETTINGS, data.settings ?? ({} as PluginSettings));
    merged.arxiv = migrateArxivSettings((data.settings as any)?.arxiv);
    this.settings = merged;
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
      paperIndex: this.buildPaperIndex(),
      llm,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      advanced: this.settings.advanced,
      output: this.settings.output,
      llmSettings: this.settings.llm,
      progress: this.progress,
    });
  }

  private buildManualFetch(): ManualFetchService {
    const { llm, fetcher, paperFetcher, writer } = this.buildSharedDeps();
    return new ManualFetchService({
      vault: this.app.vault,
      fetcher,
      paperFetcher,
      writer,
      paperIndex: this.buildPaperIndex(),
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

  buildPaperIndex(): PaperIndexStore {
    return new PaperIndexStore(this.app.vault, this.settings.output);
  }

  buildMarkdownWriter(): MarkdownWriter {
    return new MarkdownWriter({
      vault: this.app.vault,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      output: this.settings.output,
    });
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
