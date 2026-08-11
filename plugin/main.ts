import { Notice, Plugin } from "obsidian";
import type { PipelineResult, PluginSettings, RunState } from "@arxiv-daily/core";
import { ArxivDailySettingTab } from "./src/settings/tab";
import { settingsAndStateFromPersistedData } from "./src/settings/load";
import { sanitizeDetailSelection, validateSchedulerConfig } from "@arxiv-daily/core";
import { Logger } from "@arxiv-daily/core";
import { createStorageStateStore, type StateStore } from "@arxiv-daily/core";
import { RunHistoryStore } from "@arxiv-daily/core";
import { RunLock } from "@arxiv-daily/core";
import {
  OperationRegistry,
  RunCancellationService,
  normalizeArxivId,
  type OperationHandle,
  type OperationKind,
} from "@arxiv-daily/core";
import { SchedulerService } from "@arxiv-daily/core";
import { StatusBarController } from "./src/services/status-bar";
import { NoopProgressReporter, type ProgressReporter } from "@arxiv-daily/core";
import { chooseModal } from "./src/services/modal";
import { LlmClient } from "@arxiv-daily/core";
import { ArxivFetcher } from "@arxiv-daily/core";
import { AtomMetadataCache, HtmlCache } from "@arxiv-daily/core";
import {
  cleanupSourceCache,
  PaperContentFetcher,
} from "@arxiv-daily/core";
import { MarkdownWriter } from "@arxiv-daily/core";
import { ArxivPipeline } from "@arxiv-daily/core";
import { ManualFetchService } from "@arxiv-daily/core";
import { registerCommands } from "./src/commands";
import { todayInTz, formatDate } from "@arxiv-daily/core";
import { PaperIndexStore } from "@arxiv-daily/core";
import {
  DailyFilterCheckpointStore,
  DailySummaryCheckpointStore,
} from "@arxiv-daily/core";
import { PdfService } from "@arxiv-daily/core";
import { ProjectNotesService } from "@arxiv-daily/core";
import { RecentDatesCache } from "@arxiv-daily/core";
import { arxivCategories } from "@arxiv-daily/core";
import type { HostAdapters, HttpClient } from "@arxiv-daily/core";
import {
  deliverDailyEmailIfEnabled,
  resolveResendApiKey,
  sampleDailyDigest,
  startHostedEmailVerification,
} from "@arxiv-daily/core";
import { registerDashboardView } from "./src/dashboard/view";
import { buildObsidianHostAdapters } from "./src/hosts/obsidian";
import {
  SettingsChangeService,
  type PreparedOutputStores,
} from "./src/settings/change-service";

interface PersistedData {
  settings: PluginSettings;
  runState?: RunState;
}

class SettingsOperationRegistry extends OperationRegistry {
  private outputTransitionActive = false;

  override begin(
    kind: OperationKind,
    label: string,
    key?: string,
  ): OperationHandle {
    if (this.outputTransitionActive) {
      throw new Error(
        "Cannot start an operation while output directories are changing",
      );
    }
    return super.begin(kind, label, key);
  }

  beginOutputTransition(): () => void {
    if (this.outputTransitionActive || this.snapshot().length > 0) {
      throw new Error(
        "Output directories cannot change while operations or runs are active",
      );
    }
    this.outputTransitionActive = true;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.outputTransitionActive = false;
    };
  }
}

let lastCacheCleanupDate: string | null = null;

export function cacheCleanupDateKey(now: Date, timezone: string): string {
  return formatDate(todayInTz(now, timezone));
}

export function shouldRunCacheCleanup(
  lastCleanupDate: string | null | undefined,
  now: Date,
  timezone: string,
): boolean {
  return lastCleanupDate !== cacheCleanupDateKey(now, timezone);
}

export function resolvePluginDir(
  manifestDir: string | undefined,
  configDir: string,
  pluginId: string,
): string {
  return manifestDir ?? `${configDir}/plugins/${pluginId}`;
}

export default class ArxivDailyPlugin extends Plugin {
  declare settings: PluginSettings;
  logger!: Logger;
  stateStore!: StateStore;
  runHistoryStore!: RunHistoryStore;
  scheduler!: SchedulerService;
  settingsChanges!: SettingsChangeService;
  recentDates!: RecentDatesCache;
  manualFetch!: { fetchAndSummarize: ManualFetchService["fetchAndSummarize"] };
  progress!: ProgressReporter;
  readonly operations = new SettingsOperationRegistry();
  private runLock = new RunLock();
  private runCancellation = new RunCancellationService(this.operations);
  private unloading = false;
  private unsubscribeOperations?: () => void;
  private legacyRunState: RunState = {};
  private scheduleIntentRevision = 0;
  private scheduleIntentQueue: Promise<void> = Promise.resolve();
  private host!: HostAdapters;

  getHttpClient(): HttpClient {
    if (!this.host) throw new Error("Obsidian host adapters are not initialized");
    return this.host.http;
  }

  async onload() {
    const settingsWarnings = await this.loadSettingsAndState();
    this.logger = new Logger(
      this.settings.advanced.logLevel,
      (message, timeoutMs) => new Notice(message, timeoutMs),
      this.settings.arxiv.timezone,
    );
    this.refreshSensitiveValues();
    for (const warning of settingsWarnings) {
      this.logger.warn(`settings: ${warning}`);
    }
    this.host = buildObsidianHostAdapters({
      app: this.app,
      getSettings: () => this.settings,
      changeSettingValue: (key, value) =>
        this.settingsChanges.changeValue(key, value),
    });
    this.recentDates = new RecentDatesCache({
      getSettings: () => this.settings,
      buildFetcher: () => this.buildArxivFetcher(),
      markupParser: this.host.markupParser,
      logger: this.logger,
    });

    this.stateStore = createStorageStateStore(
      this.host.storage,
      this.settings.output,
      this.logger,
    );
    this.runHistoryStore = RunHistoryStore.fromStorage(
      this.host.storage,
      this.settings.output,
      this.logger,
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
    this.unsubscribeOperations = this.operations.subscribe((active) => {
      if (this.unloading || !(this.progress instanceof StatusBarController)) return;
      if (active.length > 0 && active.every((operation) => operation.cancellationRequested)) {
        this.progress.setTask("Cancelling active tasks", `${active.length} unwinding`);
      }
    });
    await this.buildMarkdownWriter().cleanupTemporaryFiles().catch((e) =>
      this.logger.warn("markdown temp cleanup failed", e),
    );

    this.scheduler = new SchedulerService({
      getSettings: () => this.settings,
      store: this.stateStore,
      lock: this.runLock,
      logger: this.logger,
      runForDate: (date, signal) => this.buildPipeline().runForDate(date, signal),
      progress: this.progress,
      cancellation: this.runCancellation,
      recentDates: this.recentDates,
      runHistory: this.runHistoryStore,
      dailyPathForDate: (date) => this.buildMarkdownWriter().dailyPath(date),
      onDailyCompleted: (date, result) => this.deliverCompletedDigest(date, result),
    });
    this.settingsChanges = new SettingsChangeService({
      settings: this.settings,
      persistSettings: (candidate) => this.persistSettings(candidate),
      prepareOutputStores: (candidate) => this.prepareOutputStores(candidate),
      installOutputStores: (prepared) => this.installOutputStores(prepared),
      hasActiveOutputWork: () => this.hasActiveOutputWork(),
      beginOutputTransition: () => this.beginOutputTransition(),
      reportPostCommitError: (action, error) =>
        this.logger.error(`settings: failed to ${action} after persistence`, error),
      setLoggerLevel: (level) => this.logger.setLevel(level),
      setLoggerTimezone: (timezone) => this.logger.setTimezone(timezone),
      restartScheduler: () => this.restartScheduler(),
      setScheduleEnabled: (enabled) => this.applyScheduleEnabledRuntime(enabled),
      refreshSensitiveValues: () => this.refreshSensitiveValues(),
    });

    // Wrap in an object that rebuilds dependencies on every call so settings
    // changes (model, key, paths) always take effect without needing to reload.
    this.manualFetch = {
      fetchAndSummarize: async (raw: string, date: string) => {
        const id = normalizeArxivId(raw);
        const key = id ?? raw.trim();
        if (this.operations.find("detail-summary", key)) {
          return { kind: "error", reason: `detail summary already active for ${key}` };
        }
        const operation = this.operations.begin("detail-summary", `Detail summary: ${key}`, key);
        try {
          return await this.buildManualFetch().fetchAndSummarize(raw, date, operation.signal);
        } finally {
          operation.finish();
        }
      },
    };
    this.cleanupCachesIfDue();

    this.addSettingTab(new ArxivDailySettingTab(this.app, this));
    registerDashboardView(this);
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
    this.unloading = true;
    this.scheduler?.stop();
    this.operations.cancelAll("plugin unloaded");
    this.unsubscribeOperations?.();
    this.unsubscribeOperations = undefined;
    if (this.progress instanceof StatusBarController) this.progress.dispose();
  }

  isUnloading(): boolean {
    return this.unloading;
  }

  async saveSettings(): Promise<void> {
    if (this.settingsChanges) {
      await this.settingsChanges.persistCurrent();
      return;
    }
    Object.assign(
      this.settings.detailSelection,
      sanitizeDetailSelection(this.settings.detailSelection),
    );
    await this.persistSettings(this.settings);
    this.refreshSensitiveValues();
  }

  restartScheduler(): void {
    this.scheduler.stop();
    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  private async prepareOutputStores(
    candidate: PluginSettings,
  ): Promise<PreparedOutputStores> {
    const stateStore = createStorageStateStore(
      this.host.storage,
      candidate.output,
      this.logger,
    );
    await stateStore.load();
    const runHistoryStore = RunHistoryStore.fromStorage(
      this.host.storage,
      candidate.output,
      this.logger,
    );
    await runHistoryStore.readLatest(1);
    return { stateStore, runHistoryStore };
  }

  private installOutputStores(prepared: PreparedOutputStores): void {
    // Scheduler validates both references first, then publishes one immutable
    // pair. A failure therefore leaves every old reference installed without a
    // rollback path that could itself throw and split state from history.
    this.scheduler.replacePersistenceStores(
      prepared.stateStore,
      prepared.runHistoryStore,
    );
    this.stateStore = prepared.stateStore;
    this.runHistoryStore = prepared.runHistoryStore;
    if (this.settings.schedule.enabled) {
      this.progress.setIdle(latestCompletedDate(prepared.stateStore));
    } else {
      this.progress.setDisabled();
    }
  }

  hasActiveOutputWork(): boolean {
    return this.operations.snapshot().length > 0 || this.scheduler.activeRuns().length > 0;
  }

  async withOutputOperation<T>(
    kind: "paper-index" | "paper-note",
    label: string,
    key: string,
    operation: () => Promise<T>,
  ): Promise<T> {
    const handle = this.operations.begin(kind, label, key);
    try {
      return await operation();
    } finally {
      handle.finish();
    }
  }

  private beginOutputTransition(): () => void {
    if (this.scheduler.activeRuns().length > 0) {
      throw new Error(
        "Output directories cannot change while operations or runs are active",
      );
    }
    return this.operations.beginOutputTransition();
  }

  /** Kept for callers outside Settings; new path changes use settingsChanges. */
  async reloadStateStoreForOutputPaths(): Promise<void> {
    this.installOutputStores(await this.prepareOutputStores(this.settings));
  }

  async setScheduleEnabled(enabled: boolean): Promise<boolean> {
    const revision = (this.scheduleIntentRevision ?? 0) + 1;
    this.scheduleIntentRevision = revision;
    let choice: "skip" | "run" | "none" | null = enabled ? "none" : null;
    if (enabled && !this.settings.schedule.enabled) {
      const candidate: PluginSettings = {
        ...this.settings,
        schedule: { ...this.settings.schedule, enabled: true },
      };
      const validation = validateSchedulerConfig(candidate);
      if (!validation.ok) {
        if (revision === this.scheduleIntentRevision) {
          new Notice(
            `Cannot enable arXiv Daily:\n${validation.reasons.map((reason) => "• " + reason).join("\n")}`,
            10_000,
          );
        }
        return false;
      }
      const selected = await this.chooseScheduleEnableAction();
      if (revision !== this.scheduleIntentRevision) return false;
      if (selected === null || selected === "cancel") return false;
      choice = selected === "run" ? "run" : "skip";
    }

    return this.enqueueScheduleIntent(async () => {
      if (revision !== this.scheduleIntentRevision) return false;
      if (this.settings.schedule.enabled !== enabled) {
        await this.settingsChanges.changeValue("schedule.enabled", enabled);
      }
      if (revision !== this.scheduleIntentRevision) return false;
      if (!enabled || choice === "none") return true;

      if (choice === "skip") {
        const today = formatDate(todayInTz(new Date(), this.settings.arxiv.timezone));
        await this.stateStore.setSkipped(today, "user opted out at enable time");
        if (revision === this.scheduleIntentRevision) {
          this.logger.notice("arXiv Daily: enabled. Today skipped — will run on next workday.");
        }
      } else {
        const result = await this.scheduler.tickToday();
        if (
          revision === this.scheduleIntentRevision &&
          result?.kind === "skipped" &&
          result.reason === "weekend"
        ) {
          this.logger.notice("arXiv Daily: weekend, no update — will check next workday");
        }
      }
      return revision === this.scheduleIntentRevision;
    });
  }

  private chooseScheduleEnableAction(): Promise<string | null> {
    return chooseModal(
      this.app,
      "Enable arXiv Daily",
      "Scheduler will check for new papers daily. Run today's summary right now?",
      [
        { label: "Cancel", value: "cancel" },
        { label: "Skip today", value: "skip" },
        { label: "Run today", value: "run", cta: true },
      ],
    );
  }

  private enqueueScheduleIntent(
    operation: () => Promise<boolean>,
  ): Promise<boolean> {
    const queued = (this.scheduleIntentQueue ?? Promise.resolve()).then(operation);
    this.scheduleIntentQueue = queued.then(() => undefined, () => undefined);
    return queued;
  }

  private applyScheduleEnabledRuntime(enabled: boolean): void {
    if (enabled) {
      this.scheduler.start();
    } else {
      this.scheduler.stop();
      this.progress.setDisabled();
    }
  }

  private async loadSettingsAndState(): Promise<string[]> {
    const loaded = settingsAndStateFromPersistedData(await this.loadData());
    this.legacyRunState = loaded.runState;
    this.settings = loaded.settings;
    return loaded.warnings;
  }

  private async persistSettings(settings: PluginSettings): Promise<void> {
    const data: PersistedData = { settings };
    await this.saveData(data);
  }

  refreshSensitiveValues(): void {
    this.logger?.setSensitiveValues(
      [
        this.settings.llm.apiKey,
        this.settings.email?.apiKey ?? "",
        this.settings.email?.hostedToken ?? "",
      ].filter(Boolean),
    );
  }

  async deliverCompletedDigest(
    date: string,
    result: Extract<PipelineResult, { kind: "completed" }>,
  ): Promise<void> {
    if (!result.digest) {
      this.logger.debug(`email: no digest for ${date}; skip auto-send (repair path)`);
      return;
    }
    await deliverDailyEmailIfEnabled(result.digest, {
      storage: this.host.storage,
      http: this.host.http,
      output: this.settings.output,
      email: this.settings.email,
      apiKey: resolveResendApiKey(this.settings.email),
      logger: this.logger,
    });
  }

  async sendTestEmail(date?: string): Promise<string> {
    const day =
      date ??
      formatDate(todayInTz(new Date(), this.settings.arxiv.timezone));
    const digest = sampleDailyDigest({
      date: day,
      language: this.settings.output.summaryLanguage,
      categories: arxivCategories(this.settings.arxiv).join(", "),
      dailyPath: `${this.settings.output.dailyDir}/${day}.md`,
    });
    const email = { ...this.settings.email, enabled: true };
    const result = await deliverDailyEmailIfEnabled(digest, {
      storage: this.host.storage,
      http: this.host.http,
      output: this.settings.output,
      email,
      apiKey: resolveResendApiKey(this.settings.email),
      logger: this.logger,
      force: true,
    });
    if (result.kind === "delivered") {
      return `Test email delivered to ${email.to}` +
        (result.providerMessageId ? ` (${result.providerMessageId})` : "");
    }
    throw new Error(`${result.kind}: ${result.reason}`);
  }

  async sendHostedVerificationEmail(): Promise<string> {
    const to = this.settings.email.to?.trim() ?? "";
    if (!to) throw new Error("Enter your email before sending a verification message");
    await startHostedEmailVerification({
      http: this.host.http,
      baseUrl: this.settings.email.hostedBaseUrl,
      email: to,
    });
    return `Verification email sent to ${to}. Open the link, then paste the code from that page into Verification code.`;
  }

  private buildPipeline(): ArxivPipeline {
    const { llm, fetcher, paperFetcher, writer } = this.buildSharedDeps();
    const checkpointStoreOptions = {
      onWarning: (message: string, error?: unknown) =>
        this.logger.warn(message, error),
    };
    return new ArxivPipeline({
      fetcher,
      markupParser: this.host.markupParser,
      paperFetcher,
      writer,
      paperIndex: this.buildPaperIndex(),
      checkpointStores: {
        filter: new DailyFilterCheckpointStore(
          this.host.storage,
          this.settings.output,
          checkpointStoreOptions,
        ),
        summary: new DailySummaryCheckpointStore(
          this.host.storage,
          this.settings.output,
          checkpointStoreOptions,
        ),
      },
      llm,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      advanced: this.settings.advanced,
      output: this.settings.output,
      llmSettings: this.settings.llm,
      detailSelection: this.settings.detailSelection,
      progress: this.progress,
    });
  }

  private buildManualFetch(): ManualFetchService {
    const { llm, fetcher, paperFetcher, writer } = this.buildSharedDeps();
    return new ManualFetchService({
      storage: this.host.storage,
      markupParser: this.host.markupParser,
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

  buildArxivFetcher(): ArxivFetcher {
    return new ArxivFetcher({
      category: this.settings.arxiv.category,
      categories: arxivCategories(this.settings.arxiv),
      http: this.host.http,
      markupParser: this.host.markupParser,
      logger: this.logger,
      requestDelayMs: this.settings.advanced.requestDelayMs,
      metadataCache: new AtomMetadataCache({
        rootDir: this.pluginCacheDir(),
        expiryDays: this.settings.advanced.cacheExpiryDays,
        storage: this.host.storage,
      }),
    });
  }

  private buildSharedDeps() {
    const llm = new LlmClient(this.settings.llm, this.logger, this.host.http);
    const fetcher = this.buildArxivFetcher();
    const cache = new HtmlCache({
      rootDir: this.pluginCacheDir(),
      expiryDays: this.settings.advanced.cacheExpiryDays,
      storage: this.host.storage,
    });
    const paperFetcher = new PaperContentFetcher(fetcher, cache, this.logger, this.host.markupParser, {
      storage: this.host.storage,
      cacheDir: `${this.pluginDir()}/.cache/source`,
      expiryDays: this.settings.advanced.cacheExpiryDays,
    });
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

  async downloadPdf(entry: Parameters<PdfService["downloadForEntry"]>[0]) {
    const key = entry.arxivId;
    if (this.operations.find("pdf-download", key)) {
      return { kind: "fetch_error" as const, reason: `PDF download already active for ${key}` };
    }
    const operation = this.operations.begin("pdf-download", `PDF download: ${key}`, key);
    try {
      return await this.buildPdfService().downloadForEntry(entry, operation.signal);
    } finally {
      operation.finish();
    }
  }

  buildProjectNotesService(): ProjectNotesService {
    return new ProjectNotesService({
      storage: this.host.storage,
      paperIndex: this.buildPaperIndex(),
      output: this.settings.output,
      logger: this.logger,
    });
  }

  private pluginDir(): string {
    return resolvePluginDir(
      this.manifest.dir,
      this.app.vault.configDir,
      this.manifest.id,
    );
  }

  private pluginCacheDir(): string {
    return `${this.pluginDir()}/.cache`;
  }

  private cleanupCachesIfDue(now = new Date()): void {
    if (
      !shouldRunCacheCleanup(
        lastCacheCleanupDate,
        now,
        this.settings.arxiv.timezone,
      )
    ) {
      return;
    }
    lastCacheCleanupDate = cacheCleanupDateKey(
      now,
      this.settings.arxiv.timezone,
    );
    this.cleanupCaches().catch((e) =>
      this.logger.warn("cache cleanup failed", e),
    );
  }

  private async cleanupCaches(): Promise<void> {
    const cache = new HtmlCache({
      rootDir: this.pluginCacheDir(),
      expiryDays: this.settings.advanced.cacheExpiryDays,
      storage: this.host.storage,
    });
    const textRemoved = await cache.cleanupExpired();
    const metadataRemoved = await new AtomMetadataCache({
      rootDir: this.pluginCacheDir(),
      expiryDays: this.settings.advanced.cacheExpiryDays,
      storage: this.host.storage,
    }).cleanupExpired();
    const sourceRemoved = await cleanupSourceCache({
      storage: this.host.storage,
      cacheDir: `${this.pluginDir()}/.cache/source`,
      expiryDays: this.settings.advanced.cacheExpiryDays,
    });
    if (textRemoved || metadataRemoved || sourceRemoved) {
      this.logger.info(
        `cache cleanup: removed ${textRemoved} html/abs files, ${metadataRemoved} Atom metadata files, and ${sourceRemoved} source files`,
      );
    }
  }
}

function latestCompletedDate(store: StateStore): string | undefined {
  const completed = Object.entries(store.snapshot())
    .filter(([, entry]) => entry.status === "completed")
    .map(([date]) => date)
    .sort();
  return completed[completed.length - 1];
}
