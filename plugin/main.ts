import { Notice, Plugin } from "obsidian";
import type {
  LibraryInventory,
  PipelineResult,
  PluginSettings,
  RunState,
} from "@arxiv-daily/core";
import type { OpenedScopedLibrarySource } from "@arxiv-daily/node-runtime/scoped-library-source";
import { ArxivDailySettingTab } from "./src/settings/tab";
import { settingsAndStateFromPersistedData } from "./src/settings/load";
import { sanitizeDetailSelection, validateSchedulerConfig } from "@arxiv-daily/core";
import { Logger } from "@arxiv-daily/core";
import { createStorageStateStore, type StateStore } from "@arxiv-daily/core";
import { RunHistoryStore } from "@arxiv-daily/core";
import { RunLock } from "@arxiv-daily/core";
import { OperationRegistry, RunCancellationService, normalizeArxivId } from "@arxiv-daily/core";
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
import {
  buildObsidianHostAdapters,
  ObsidianLibraryDirectoryPicker,
  openObsidianLibrarySource,
} from "./src/hosts/obsidian";
import {
  authorizeLibraryConnection,
  buildLibraryInventoryPreview,
  createLibraryConnection,
  decodeLibraryConnection,
  libraryAuthorizationDisclosure,
  libraryConnectionStatus,
  revokeLibraryConnection,
  type LibraryAuthorizationDisclosure,
  type LibraryConnectionStatus,
  type LibraryInventoryPreview,
  type PersistedLibraryConnection,
} from "./src/library/connection";

interface PersistedData {
  settings: PluginSettings;
  runState?: RunState;
  libraryConnection?: PersistedLibraryConnection;
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
  recentDates!: RecentDatesCache;
  manualFetch!: { fetchAndSummarize: ManualFetchService["fetchAndSummarize"] };
  progress!: ProgressReporter;
  readonly operations = new OperationRegistry();
  private runLock = new RunLock();
  private runCancellation = new RunCancellationService(this.operations);
  private unloading = false;
  private unsubscribeOperations?: () => void;
  private legacyRunState: RunState = {};
  private host!: HostAdapters;
  private libraryConnection?: PersistedLibraryConnection;
  private librarySource?: OpenedScopedLibrarySource;
  private libraryDirectoryPicker = new ObsidianLibraryDirectoryPicker();
  private openLibrarySource: (selectedRoot: string) => Promise<OpenedScopedLibrarySource>
    = openObsidianLibrarySource;
  private libraryInventoryController?: AbortController;
  private libraryMutationQueue: Promise<void> = Promise.resolve();
  private librarySelectionRevision = 0;

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
      persistSettings: () => this.enqueueLibraryMutation(() => this.persistSettings()),
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
    this.libraryInventoryController?.abort("plugin unloaded");
    this.unsubscribeOperations?.();
    this.unsubscribeOperations = undefined;
    if (this.progress instanceof StatusBarController) this.progress.dispose();
  }

  isUnloading(): boolean {
    return this.unloading;
  }

  async saveSettings(): Promise<void> {
    this.settings.detailSelection = sanitizeDetailSelection(
      this.settings.detailSelection,
    );
    this.refreshSensitiveValues();
    await this.enqueueLibraryMutation(() => this.persistSettings());
  }

  getLibraryConnectionStatus(): LibraryConnectionStatus {
    return libraryConnectionStatus(
      this.libraryConnection,
      this.settings.llm.baseUrl,
    );
  }

  getLibraryAuthorizationDisclosure(): LibraryAuthorizationDisclosure | null {
    if (!this.libraryConnection) return null;
    return libraryAuthorizationDisclosure(
      this.libraryConnection,
      this.settings.llm.baseUrl,
    );
  }

  async selectLibraryRoot(): Promise<"selected" | "cancelled" | "unsupported"> {
    const revision = ++this.librarySelectionRevision;
    const selection = await this.libraryDirectoryPicker.select();
    if (selection.kind !== "selected") return selection.kind;
    const source = await this.openLibrarySource(selection.path);
    return await this.enqueueLibraryMutation(async () => {
      if (revision !== this.librarySelectionRevision) return "cancelled" as const;
      const previousConnection = this.libraryConnection;
      const previousSource = this.librarySource;
      this.libraryInventoryController?.abort("library folder changed");
      this.libraryConnection = createLibraryConnection(source.canonicalRoot, source.rootIdentity);
      this.librarySource = source;
      this.refreshSensitiveValues();
      try {
        await this.persistSettings();
        return "selected" as const;
      } catch (error) {
        this.libraryConnection = previousConnection;
        this.librarySource = previousSource;
        this.refreshSensitiveValues();
        throw error;
      }
    });
  }

  async authorizeLibraryProcessing(expectedFingerprint?: string): Promise<void> {
    await this.enqueueLibraryMutation(async () => {
      if (!this.libraryConnection) throw new Error("Choose a personal library first");
      const disclosure = libraryAuthorizationDisclosure(
        this.libraryConnection,
        this.settings.llm.baseUrl,
      );
      if (
        expectedFingerprint
        && disclosure.authorizationFingerprint !== expectedFingerprint
      ) {
        throw new Error("Library authorization terms changed; review them again");
      }
      const previous = this.libraryConnection;
      this.libraryConnection = authorizeLibraryConnection(
        previous,
        this.settings.llm.baseUrl,
      );
      try {
        await this.persistSettings();
      } catch (error) {
        this.libraryConnection = previous;
        throw error;
      }
    });
  }

  async revokeLibraryProcessing(): Promise<void> {
    await this.enqueueLibraryMutation(async () => {
      if (!this.libraryConnection) return;
      const previous = this.libraryConnection;
      this.libraryConnection = revokeLibraryConnection(previous);
      try {
        await this.persistSettings();
      } catch (error) {
        this.libraryConnection = previous;
        throw error;
      }
    });
  }

  async previewLibraryInventory(signal?: AbortSignal): Promise<LibraryInventoryPreview> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    this.libraryInventoryController?.abort("superseded by a new preview");
    const controller = new AbortController();
    this.libraryInventoryController = controller;
    const onAbort = () => controller.abort(signal?.reason);
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
    try {
      controller.signal.throwIfAborted();
      const source = this.librarySource
        ?? await this.openLibrarySource(connection.selectedRoot);
      controller.signal.throwIfAborted();
      if (
        source.canonicalRoot !== connection.selectedRoot
        || source.rootIdentity !== connection.rootIdentity
      ) {
        throw new Error("Library folder identity changed; choose it again");
      }
      if (this.libraryConnection !== connection) {
        throw new Error("Library connection changed while building the preview");
      }
      this.librarySource = source;
      const inventory: LibraryInventory = await source.inventory({
        signal: controller.signal,
      });
      controller.signal.throwIfAborted();
      if (this.libraryConnection !== connection) {
        throw new Error("Library connection changed while building the preview");
      }
      return buildLibraryInventoryPreview(
        inventory,
        connection.eligibleExtensions,
      );
    } finally {
      signal?.removeEventListener("abort", onAbort);
      if (this.libraryInventoryController === controller) {
        this.libraryInventoryController = undefined;
      }
    }
  }

  restartScheduler(): void {
    this.scheduler.stop();
    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  async reloadStateStoreForOutputPaths(): Promise<void> {
    const nextStore = createStorageStateStore(
      this.host.storage,
      this.settings.output,
      this.logger,
    );
    await nextStore.load();
    this.stateStore = nextStore;
    this.scheduler.replaceStore(nextStore);
    this.runHistoryStore = RunHistoryStore.fromStorage(
      this.host.storage,
      this.settings.output,
      this.logger,
    );
    this.scheduler.replaceRunHistory(this.runHistoryStore);
    if (this.settings.schedule.enabled) {
      this.progress.setIdle(latestCompletedDate(nextStore));
    } else {
      this.progress.setDisabled();
    }
  }

  async setScheduleEnabled(enabled: boolean): Promise<boolean> {
    if (this.settings.schedule.enabled === enabled) return true;

    if (enabled) {
      const v = validateSchedulerConfig(this.settings);
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
        if (result?.kind === "skipped" && result.reason === "weekend") {
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

  private async loadSettingsAndState(): Promise<string[]> {
    const raw: unknown = await this.loadData();
    const loaded = settingsAndStateFromPersistedData(raw);
    this.legacyRunState = loaded.runState;
    this.settings = loaded.settings;
    const persisted = raw && typeof raw === "object"
      ? raw as Record<string, unknown>
      : {};
    const persistedLibraryConnection = persisted.libraryConnection;
    this.libraryConnection = decodeLibraryConnection(
      persistedLibraryConnection,
    );
    if (
      persistedLibraryConnection !== undefined
      && !this.libraryConnection
    ) {
      loaded.warnings.push("ignored invalid personal library connection metadata");
    }
    return loaded.warnings;
  }

  private enqueueLibraryMutation<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.libraryMutationQueue ?? Promise.resolve();
    const result = previous.then(operation, operation);
    this.libraryMutationQueue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  private async persistSettings(): Promise<void> {
    const data: PersistedData = {
      settings: this.settings,
      ...(this.libraryConnection
        ? { libraryConnection: this.libraryConnection }
        : {}),
    };
    await this.saveData(data);
  }

  refreshSensitiveValues(): void {
    this.logger?.setSensitiveValues(
      [
        this.settings.llm.apiKey,
        this.settings.email?.apiKey ?? "",
        this.settings.email?.hostedToken ?? "",
        this.libraryConnection?.selectedRoot ?? "",
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
