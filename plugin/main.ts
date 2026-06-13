import { Notice, Plugin } from "obsidian";
import * as path from "node:path";
import * as os from "node:os";
import { DEFAULT_SETTINGS } from "./src/settings/defaults";
import type { PluginSettings, RunState } from "./src/settings/types";
import { ArxivDailySettingTab } from "./src/settings/tab";
import { migrateArxivSettings } from "./src/settings/migration";
import { validateFilterConfig } from "./src/settings/validation";
import { Logger } from "./src/services/logger";
import { createStorageStateStore, type StateStore } from "./src/services/state-store";
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
import { DailySelectionSyncService } from "./src/services/daily-selection";
import { BibtexService } from "./src/services/bibtex";
import { PdfService } from "./src/services/pdf";
import { ProjectNotesService } from "./src/services/project-notes";
import { arxivCategories } from "./src/settings/categories";
import type { HostAdapters } from "./src/core/adapters";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  registerDashboardView,
} from "./src/dashboard/view";
import { buildObsidianHostAdapters } from "./src/hosts/obsidian";

interface PersistedData {
  settings: PluginSettings;
  runState?: RunState;
}

export default class ArxivDailyPlugin extends Plugin {
  settings!: PluginSettings;
  logger!: Logger;
  stateStore!: StateStore;
  scheduler!: SchedulerService;
  manualFetch!: { fetchAndSummarize: ManualFetchService["fetchAndSummarize"] };
  progress!: ProgressReporter;
  private dailySelectionSync!: DailySelectionSyncService;
  private runLock = new RunLock();
  private runCancellation = new RunCancellationService();
  private legacyRunState: RunState = {};
  private host!: HostAdapters;

  async onload() {
    await this.loadSettingsAndState();
    this.logger = new Logger(
      this.settings.advanced.logLevel,
      (message, timeoutMs) => new Notice(message, timeoutMs),
    );
    this.host = buildObsidianHostAdapters({
      app: this.app,
      getSettings: () => this.settings,
      persistSettings: () => this.persistSettings(),
    });

    this.stateStore = createStorageStateStore(
      this.host.storage,
      this.settings.output,
    );
    await this.stateStore.load();
    if (
      Object.keys(this.stateStore.snapshot()).length === 0 &&
      Object.keys(this.legacyRunState).length > 0
    ) {
      await this.stateStore.replaceAll(this.legacyRunState);
    }

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
    this.host.progress = this.progress;

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
    registerDashboardView(this);
    registerCommands(this);
    this.dailySelectionSync = new DailySelectionSyncService({
      storage: this.host.storage,
      getOutput: () => this.settings.output,
      getLookbackDays: () => this.settings.schedule.lookbackDays,
      getTimezone: () => this.settings.arxiv.timezone,
      buildPaperIndex: () => this.buildPaperIndex(),
      logger: this.logger,
    });
    this.registerEvent(
      this.app.vault.on("modify", (file) => this.dailySelectionSync.schedule(file)),
    );
    this.app.workspace.onLayoutReady(() => {
      this.dailySelectionSync
        .syncRecentDailyFiles()
        .catch((e) =>
          this.logger.error("daily-selection: startup sync failed", e),
        );
    });

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
    void this.app.workspace.detachLeavesOfType(ARXIV_DAILY_DASHBOARD_VIEW);
    this.dailySelectionSync?.clear();
    this.scheduler?.cancelCurrentRun("plugin unloaded");
    this.scheduler?.stop();
  }

  async saveSettings(): Promise<void> {
    await this.persistSettings();
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
    this.legacyRunState = data.runState ?? {};
    const merged = mergeSettings(DEFAULT_SETTINGS, data.settings ?? ({} as PluginSettings));
    merged.arxiv = migrateArxivSettings((data.settings as any)?.arxiv);
    this.settings = merged;
  }

  private async persistSettings(): Promise<void> {
    const data: PersistedData = { settings: this.settings };
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
      storage: this.host.storage,
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
      categories: arxivCategories(this.settings.arxiv),
      http: this.host.http,
      logger: this.logger,
      requestDelayMs: this.settings.advanced.requestDelayMs,
    });
    const cache = new HtmlCache({
      rootDir: this.resolveCacheDir(),
      expiryDays: this.settings.advanced.cacheExpiryDays,
    });
    const paperFetcher = new PaperContentFetcher(fetcher, cache, this.logger);
    const writer = new MarkdownWriter({
      storage: this.host.storage,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      output: this.settings.output,
    });
    return { llm, fetcher, paperFetcher, writer };
  }

  buildPaperIndex(): PaperIndexStore {
    return new PaperIndexStore(
      this.host.storage,
      this.settings.output,
    );
  }

  buildMarkdownWriter(): MarkdownWriter {
    return new MarkdownWriter({
      storage: this.host.storage,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      output: this.settings.output,
    });
  }

  buildBibtexService(): BibtexService {
    const { fetcher } = this.buildSharedDeps();
    return new BibtexService({
      fetcher,
      paperIndex: this.buildPaperIndex(),
      storage: this.host.storage,
      output: this.settings.output,
      logger: this.logger,
    });
  }

  buildPdfService(): PdfService {
    const { fetcher } = this.buildSharedDeps();
    return new PdfService({
      fetcher,
      storage: this.host.storage,
      paperIndex: this.buildPaperIndex(),
      output: this.settings.output,
      logger: this.logger,
    });
  }

  buildProjectNotesService(): ProjectNotesService {
    return new ProjectNotesService({
      storage: this.host.storage,
      paperIndex: this.buildPaperIndex(),
      output: this.settings.output,
      logger: this.logger,
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
