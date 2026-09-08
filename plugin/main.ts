import { Notice, Plugin, loadPdfJs } from "obsidian";
import type {
  LibraryInventory,
  PersonalLibraryCatalog,
  PersonalLibraryDirectionProposal,
  PersonalLibraryDirectionTextPatch,
  PersonalLibraryReviewedDirectionDraft,
  PipelineResult,
  PluginSettings,
  RunState,
  FullTextIndexRunSummary,
  FullTextLegacyMigrationLease,
  FullTextGenerationMaintenanceReport,
  KnowledgeBaseChunkHit,
  DirectionDiffSuggestion,
  IncrementalSuggestionsDocument,
  ClusteringInputPaper,
  ProposalAcceptanceReceipt,
  Topic,
  AcceptProposedTopicsResult,
  LibraryDirectionPreview,
  LibraryPreviewPaper,
} from "@arxiv-daily/core";
import type { OpenedScopedLibrarySource } from "@arxiv-daily/node-runtime/scoped-library-source";
import { ArxivDailySettingTab } from "./src/settings/tab";
import { settingsAndStateFromPersistedData } from "./src/settings/load";
import { sanitizeDetailSelection, validateSchedulerConfig } from "@arxiv-daily/core";
import { Logger } from "@arxiv-daily/core";
import { createStorageStateStore, type StateStore } from "@arxiv-daily/core";
import { RunHistoryStore } from "@arxiv-daily/core";
import { RunLock } from "@arxiv-daily/core";
import {
ArxivLibraryMetadataResolver,
extractPdfIdentificationEvidence,
searchArxivTitle,
createPersonalLibraryIdentificationFingerprint,
createPersonalLibraryScopeFingerprint,
OperationRegistry,
PersonalLibraryCatalogStore,
PersonalLibraryDirectionProposalStore,
buildChatCompletionsUrl,
createPersonalLibraryCatalogInputFingerprint,
type DirectionProposalProgress,
proposeClusteredPersonalLibraryDirections,
previewPersonalLibraryDirection,
PERSONAL_LIBRARY_SIMILARITY_QUANTILE,
mergePersonalLibraryDirectionCandidates,
movePersonalLibraryDirectionCandidate,
removePersonalLibraryDirectionCandidate,
renamePersonalLibraryProposedTopic,
acceptProposedTopics,
matchingProposalAcceptance,
topicNameKey,
updatePersonalLibraryDirectionCandidate,
selectPersonalLibraryDirectionPapers,
reconcilePersonalLibraryCatalog,
RunCancellationService,
normalizeArxivId,
FullTextKnowledgeBaseFileStore,
FullTextGenerationIndexStore,
indexPersonalLibraryFullText as indexFullTextKnowledgeBase,
searchFullTextKnowledgeBase as searchFullTextKnowledgeBaseCore,
preflightFullTextGenerationSynchronization,
synchronizeFullTextGenerationIndex,
IncrementalSuggestionsStore,
centerCorpusChunks,
loadClusteringInput,
PDF_IDENTIFICATION_EVIDENCE_VERSION,
sha256Hex,
type OperationHandle,
type OperationKind,
} from "@arxiv-daily/core";
import { SchedulerService } from "@arxiv-daily/core";
import { StatusBarController } from "./src/services/status-bar";
import { NoopProgressReporter, type ProgressReporter } from "@arxiv-daily/core";
import { chooseModal } from "./src/services/modal";
import {
  openPersonalLibraryInterestProfileModal,
  type InterestProfileReviewController,
} from "./src/library/interest-profile-modal";
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
  ObsidianPdfTextExtractor,
  createTransformersEmbeddingModel,
  describeRuntimeProbe,
  inspectTransformersEnv,
  openObsidianLibrarySource,
} from "./src/hosts/obsidian";
import {
  createRemoteEmbeddingModel,
  validateEmbeddingConfig,
  type EmbeddingModel,
} from "@arxiv-daily/core";
import {
  describeDiagnosticsError,
  type EmbeddingDiagnostics,
  type FullTextRuntimeDiagnostics,
  type LibraryDiagnostics,
  type PdfJsDiagnostics,
  type PdfJsSmokeDiagnostics,
} from "./src/services/fulltext-runtime-diagnostics";
import {
  authorizeLibraryConnection,
  type LibraryAuthorizationScope,
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
import { projectLibraryFullTextMatches } from "./src/library/fulltext-results";
import { confirmLibraryAuthorization } from "./src/library/modal";
import {
  SettingsChangeService,
  type PreparedOutputStores,
} from "./src/settings/change-service";
import {
  openLibraryPdfAtPage,
  resolveLibraryPdfOpenTarget,
} from "./src/library/pdf-opener";
import { LibraryIndexStatusStore } from "./src/library/index-status";
import { decodeProposalAcceptanceReceipts } from "./src/library/proposal-acceptance-state";

interface PersistedData {
  settings: PluginSettings;
  runState?: RunState;
  libraryConnection?: PersistedLibraryConnection;
  libraryProposalAcceptances?: unknown;
}

export interface PersonalLibraryReviewLoadError {
  kind: "catalog" | "proposal" | "profile" | "suggestions";
  code: string;
  message: string;
}

/** One line for the status bar, for the command-palette path that has no modal. */
function describeProposalProgress(progress: DirectionProposalProgress): string {
  if (progress.phase === "reading") return "reading the index";
  if (progress.phase === "grouping") return "grouping papers";
  return "organizing topics and directions";
}

/** An error the review page can recognize and phrase for the researcher. */
class CodedError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "CodedError";
  }
}

/** One indexed paper the catalog has no record of, named by the index. */
export interface PersonalLibraryIndexedPaper {
  paperKey: string;
  title: string;
}

export interface PersonalLibraryProfileSnapshot {
  catalog: PersonalLibraryCatalog | null;
  /** Fallback-indexed papers, empty when every indexed paper is in the catalog. */
  indexedPapers: PersonalLibraryIndexedPaper[];
  proposal: PersonalLibraryDirectionProposal | null;
  suggestions: IncrementalSuggestionsDocument | null;
  authorization: LibraryConnectionStatus;
  catalogLoadError: PersonalLibraryReviewLoadError | null;
  proposalLoadError: PersonalLibraryReviewLoadError | null;
  suggestionsLoadError: PersonalLibraryReviewLoadError | null;
  /**
   * Names of topics already in `settings.arxiv.topics`. The review page uses
   * this to mark proposed topics that a prior accept already added, since the
   * proposal is kept around for the rest to be accepted later and would
   * otherwise look identical whether or not it was accepted already.
   */
  settingsTopicNames: string[];
  settingsTopics?: Topic[];
  arxivCategories?: string[];
  proposalAcceptance?: ProposalAcceptanceReceipt | null;
  acceptanceLoadError?: string | null;
}

/**
 * One immutable daily discovery snapshot captured from the currently
 * authorized eligibility join: the personalized discovery input plus, when
 * novelty preparation succeeds, the library-derived novelty representative
 * evidence and the direction→representative mapping. Discovery and novelty
 * share the same gate and lifecycle guards; per-run daily paper evidence for
 * novelty is derived inside the pipeline from the fetched source papers, never
 * here. A novelty-preparation failure degrades to a discovery-only snapshot
 * and never drops personalized discovery.
 */
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
    return this.beginExclusiveTransition("Output directories cannot change while operations are active");
  }

  beginFullTextMaintenanceTransition(): () => void {
    return this.beginExclusiveTransition("Full-text generation maintenance cannot start while operations are active");
  }

  private beginExclusiveTransition(message: string): () => void {
    if (this.outputTransitionActive || this.snapshot().length > 0) {
      throw new Error(message);
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

export const IDENTIFICATION_HEAD_BYTES = 4 * 1024 * 1024;
const IDENTIFICATION_TAIL_BYTES = 1024 * 1024;

/**
 * Minimum buffer-pool size before the incremental direction update runs the
 * low-frequency recluster + LLM diff pass; below it only deterministic
 * placement attach suggestions are recorded.
 */
export const INCREMENTAL_BUFFER_TRIGGER = 3 as const;

/** Fixed reason attached to deterministic placement attach suggestions. */

function cacheCleanupDateKey(now: Date, timezone: string): string {
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
  private settingsTab?: ArxivDailySettingTab;
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
  private libraryConnection?: PersistedLibraryConnection;
  /**
   * Full-text indexing as the settings page can see it. Public because the
   * Library row subscribes to it: the run reports here, the row renders it, and
   * neither has to know whether the other exists.
   */
  readonly libraryIndexStatus = new LibraryIndexStatusStore();
  private librarySource?: OpenedScopedLibrarySource;
  private libraryDirectoryPicker = new ObsidianLibraryDirectoryPicker();
  private openLibrarySource: (selectedRoot: string) => Promise<OpenedScopedLibrarySource>
    = openObsidianLibrarySource;
  private libraryInventoryController?: AbortController;
  private libraryCatalog: PersonalLibraryCatalog | null = null;
  private libraryCatalogLoadError: PersonalLibraryReviewLoadError | null = null;
  /**
   * Papers the index holds that the catalog cannot name: files the scan could
   * not identify. The review page needs them to know there is anything to
   * propose from, and to show a title instead of a bare content hash.
   */
  private libraryIndexedPapers: PersonalLibraryIndexedPaper[] = [];
  private libraryMutationQueue: Promise<void> = Promise.resolve();
  private librarySelectionRevision = 0;
  private libraryConnectionRevision = 0;
  private libraryOutputRevision = 0;
  /**
   * Bumped by every change that supersedes an in-flight library operation: a
   * new folder, a revoked authorization, a changed model endpoint, a reloaded
   * catalog or review document, changed output paths. `authorizeLibraryProcessing`
   * captures it before showing the disclosure and refuses the confirmation if it
   * moved underneath.
   *
   * It used to double as the gate for the library-profile daily discovery path,
   * which is why every mutation announced itself as "discovery unavailable".
   * That gate went with the profile document (ADR 0012 / ADR 0014); the
   * supersession guard is the only remaining reader, so there is nothing to
   * restore on rollback — the counter only ever moves forward.
   */
  private libraryMutationRevision = 0;
  private libraryProposal: PersonalLibraryDirectionProposal | null = null;
  private libraryProposalAcceptances: ProposalAcceptanceReceipt[] = [];
  private libraryProposalAcceptanceRaw: unknown;
  private libraryProposalAcceptanceLoadError: string | null = null;
  private librarySuggestions: IncrementalSuggestionsDocument | null = null;
  private libraryProposalLoadError: PersonalLibraryReviewLoadError | null = null;
  private librarySuggestionsLoadError: PersonalLibraryReviewLoadError | null = null;

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
      persistSettings: () =>
        this.enqueueLibraryMutation(() => this.persistSettings()),
      changeSettingValue: (key, value) =>
        this.settingsChanges.changeValue(key, value),
    });
    if (!this.host.storage.writeTextAtomic) {
      throw new Error("Obsidian storage does not support atomic personal library catalog writes");
    }
    if (this.libraryConnection) {
      await this.reloadPersonalLibraryCatalog().catch((error) => {
        this.libraryCatalog = null;
        this.libraryCatalogLoadError = this.safeProfileLoadError("catalog", error);
        this.logger.error("personal library catalog load failed", error);
        new Notice(`arXiv Daily: personal library catalog could not be loaded: ${error instanceof Error ? error.message : String(error)}`, 10_000);
      });
      await this.reloadPersonalLibraryProfileDocuments();
      // So a settings page opened before anything else happens can already say
      // when the library was last indexed.
      await this.refreshLibraryIndexTrace();
    }
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
      runForDate: async (date, signal) => {
        return await this.buildPipeline().runForDate(date, signal);
      },
      progress: this.progress,
      cancellation: this.runCancellation,
      recentDates: this.recentDates,
      runHistory: this.runHistoryStore,
      dailyPathForDate: (date) => this.buildMarkdownWriter().dailyPath(date),
      onDailyCompleted: (date, result) => this.deliverCompletedDigest(date, result),
    });
    this.settingsChanges = new SettingsChangeService({
      settings: this.settings,
      persistSettings: (candidate) =>
        this.enqueueLibraryMutation(() => this.persistSettings(candidate)),
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
      prepareCandidateChange: (previous, candidate, changedKeys) => {
        let rollback: (() => void) | undefined;
        if (changedKeys.includes("llm.baseUrl")) {
          const previousEndpoint = this.effectiveLlmEndpoint(previous.llm.baseUrl);
          const nextEndpoint = this.effectiveLlmEndpoint(candidate.llm.baseUrl);
          if (previousEndpoint !== nextEndpoint) {
            this.beginLibraryMutation();
            this.cancelPersonalLibraryDirectionGeneration("model endpoint changed");
          }
        }
        return rollback;
      },
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

    this.settingsTab = new ArxivDailySettingTab(this.app, this);
    this.addSettingTab(this.settingsTab);
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
    if (this.settingsChanges) {
      await this.settingsChanges.persistCurrent();
      return;
    }
    Object.assign(
      this.settings.detailSelection,
      sanitizeDetailSelection(this.settings.detailSelection),
    );
    this.refreshSensitiveValues();
    await this.enqueueLibraryMutation(() => this.persistSettings());
  }

  getLibraryConnectionStatus(): LibraryConnectionStatus {
    return libraryConnectionStatus(
      this.libraryConnection,
      this.libraryAuthorizationScope(),
    );
  }

  /**
   * The scope a grant would cover. `embeddingMode` may name a mode that is not
   * applied yet, so consent for switching to remote embedding can be disclosed
   * and decided before anything changes (ADR 0008).
   */
  getLibraryAuthorizationDisclosure(
    options?: { embeddingMode?: "local" | "remote" },
  ): LibraryAuthorizationDisclosure | null {
    if (!this.libraryConnection) return null;
    return libraryAuthorizationDisclosure(
      this.libraryConnection,
      this.libraryAuthorizationScope(options?.embeddingMode),
    );
  }

  /** Authorization scope: the LLM endpoint plus the embedding endpoint when remote embedding is enabled. */
  private libraryAuthorizationScope(
    embeddingMode: "local" | "remote" = this.settings.embedding.mode,
  ): LibraryAuthorizationScope {
    return {
      llmBaseUrl: this.settings.llm.baseUrl,
      ...(embeddingMode === "remote" && this.settings.embedding.baseUrl.trim()
        ? { embeddingEndpoint: { baseUrl: this.settings.embedding.baseUrl } }
        : {}),
    };
  }

  async selectLibraryRoot(): Promise<"selected" | "cancelled" | "unsupported"> {
    const revision = ++this.librarySelectionRevision;
    const selection = await this.libraryDirectoryPicker.select();
    if (selection.kind !== "selected") return selection.kind;
    const source = await this.openLibrarySource(selection.path);
    if (revision !== this.librarySelectionRevision) return "cancelled";
    const mutationRevision = this.beginLibraryMutation();
    this.cancelPersonalLibraryOperations("library folder changed");
    return await this.enqueueLibraryMutation(async () => {
      if (revision !== this.librarySelectionRevision) return "cancelled" as const;
      const previousConnection = this.libraryConnection;
      const previousSource = this.librarySource;
      const previousCatalog = this.libraryCatalog;
      const previousProposal = this.libraryProposal;
      const previousProposalError = this.libraryProposalLoadError;
      const previousConnectionRevision = this.libraryConnectionRevision;
      this.libraryInventoryController?.abort("library folder changed");
      this.libraryConnectionRevision += 1;
      this.libraryConnection = createLibraryConnection(source.canonicalRoot, source.rootIdentity);
      this.librarySource = source;
      this.resetPersonalLibraryProfileState();
      this.refreshSensitiveValues();
      try {
        await this.persistSettings();
      } catch (error) {
        this.libraryConnection = previousConnection;
        this.librarySource = previousSource;
        this.libraryCatalog = previousCatalog;
        this.libraryProposal = previousProposal;
        this.libraryProposalLoadError = previousProposalError;
        this.libraryConnectionRevision = previousConnectionRevision;
        this.refreshSensitiveValues();
        throw error;
      }
      await this.reloadPersonalLibraryCatalog(mutationRevision).catch((error) => {
        this.libraryCatalog = null;
        this.logger.error?.("personal library catalog load failed after folder selection", error);
      });
      await this.reloadPersonalLibraryProfileDocuments(mutationRevision);
      // A new folder has its own knowledge base, so the previous folder's "last
      // indexed" sentence must not survive the switch.
      await this.refreshLibraryIndexTrace();
      return "selected" as const;
    });
  }

  async authorizeLibraryProcessing(expectedFingerprint?: string): Promise<void> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    const connectionRevision = this.libraryConnectionRevision;
    const outputRevision = this.libraryOutputRevision;
    const mutationRevision = this.libraryMutationRevision;
    const endpoint = this.effectiveLlmEndpoint(this.settings.llm.baseUrl);
    const disclosure = libraryAuthorizationDisclosure(connection, this.libraryAuthorizationScope());
    if (expectedFingerprint
      && disclosure.authorizationFingerprint !== expectedFingerprint) {
      throw new Error("Library authorization terms changed; review them again");
    }
    await this.enqueueLibraryMutation(async () => {
      if (this.libraryConnection !== connection
        || this.libraryConnectionRevision !== connectionRevision
        || this.libraryOutputRevision !== outputRevision
        || this.libraryMutationRevision !== mutationRevision
        || this.effectiveLlmEndpoint(this.settings.llm.baseUrl) !== endpoint) {
        throw new Error("Library authorization was superseded by a newer library change");
      }
      const authorized = authorizeLibraryConnection(connection, this.libraryAuthorizationScope());
      this.libraryConnection = authorized;
      try {
        await this.persistSettings();
      } catch (error) {
        if (this.libraryConnection === authorized) this.libraryConnection = connection;
        throw error;
      }
    });
  }

  async revokeLibraryProcessing(): Promise<void> {
    const previous = this.libraryConnection;
    if (!previous) return;
    const revoked = revokeLibraryConnection(previous);
    this.beginLibraryMutation();
    this.cancelPersonalLibraryDirectionGeneration("library processing authorization revoked");
    // Consent becomes ineffective synchronously at invocation, before waiting for
    // an earlier library mutation to finish.
    this.libraryConnection = revoked;
    await this.enqueueLibraryMutation(async () => {
      this.cancelPersonalLibraryDirectionGeneration("library processing authorization revoked");
      if (this.libraryConnection !== revoked) return;
      try {
        await this.persistSettings();
      } catch (error) {
        if (this.libraryConnection === revoked) {
          this.libraryConnection = previous;
        }
        throw error;
      }
    });
  }

  async setLlmBaseUrl(next: string): Promise<void> {
    const normalized = next.trim();
    const requestedEndpointChanged = this.effectiveLlmEndpoint(this.settings.llm.baseUrl)
      !== this.effectiveLlmEndpoint(normalized);
    if (requestedEndpointChanged) {
      this.beginLibraryMutation();
      this.cancelPersonalLibraryDirectionGeneration("model endpoint changed");
    }
    await this.enqueueLibraryMutation(async () => {
      const previous = this.settings.llm.baseUrl;
      this.settings.llm.baseUrl = normalized;
      this.settings.detailSelection = sanitizeDetailSelection(this.settings.detailSelection);
      this.refreshSensitiveValues();
      try {
        await this.persistSettings();
      } catch (error) {
        this.settings.llm.baseUrl = previous;
        this.refreshSensitiveValues();
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

  getPersonalLibraryCatalog(): PersonalLibraryCatalog | null {
    return this.libraryCatalog
      ? structuredClone(this.libraryCatalog)
      : null;
  }

  openPersonalLibraryDirectionReview(): void {
    openPersonalLibraryInterestProfileModal(this.app, this.personalLibraryReviewController());
  }

  private personalLibraryReviewController(): InterestProfileReviewController {
    return {
      snapshot: () => this.getPersonalLibraryProfileSnapshot(),
      reload: async () => {
        await this.reloadPersonalLibraryCatalog().catch((error) => {
          this.libraryCatalog = null;
          this.libraryCatalogLoadError = this.safeProfileLoadError("catalog", error);
          this.logger.error("personal library catalog reload failed", error);
        });
        // A library of unidentified files has an empty catalog, so without this
        // the review page would believe there is nothing to propose from.
        await this.refreshLibraryIndexTrace();
        return this.reloadPersonalLibraryProfileDocuments();
      },
      authorize: () => this.confirmPersonalLibraryDirectionAuthorization(),
      logError: (action, error) => this.logger.error(`library review: ${action} failed`, error),
      generate: (onProgress) => this.generatePersonalLibraryDirections(onProgress),
      updateProposal: (input) => this.updatePersonalLibraryProposalCandidate(input),
      discardProposal: (candidateId) => this.removePersonalLibraryProposalCandidate(candidateId),
      renameTopic: (input) => this.renamePersonalLibraryProposedTopic(input),
      moveDirection: (input) => this.movePersonalLibraryProposalCandidate(input),
      acceptTopics: (topicIds, candidateIds) => this.acceptPersonalLibraryProposedTopics(topicIds, candidateIds),
      previewDirection: (input) => this.previewPersonalLibraryDirection(input),
    };
  }

  /**
   * Where one generation's progress goes. The status bar serves the
   * command-palette path; a caller that passed its own reporter has a surface
   * of its own and covers the bar with it, so writing both left two
   * differently-worded counters on screen for the same run.
   */
  proposalProgressReporter(
    caller?: (progress: DirectionProposalProgress) => void,
  ): (progress: DirectionProposalProgress) => void {
    return (progress) => {
      if (!caller) {
        this.progress?.setTask("Generating personal library directions", describeProposalProgress(progress));
      }
      caller?.(progress);
    };
  }

  /**
   * Disclose what generating directions would send, and record the grant if
   * the researcher agrees. The scope is the live one — with local embedding
   * that is the chat endpoint and titles-and-abstracts depth, which the
   * remote-embedding consent flow in settings never asks about, leaving this
   * as the only path to the grant that generation requires.
   */
  async confirmPersonalLibraryDirectionAuthorization(): Promise<boolean> {
    const disclosure = this.getLibraryAuthorizationDisclosure();
    if (!disclosure) return false;
    if (!await confirmLibraryAuthorization(this.app, disclosure)) return false;
    try {
      await this.authorizeLibraryProcessing(disclosure.authorizationFingerprint);
    } catch (error) {
      // These two are the guards around the grant, and they are the difference
      // between "review it again" and "something is wrong"; the review page can
      // only tell them apart if they arrive carrying a code.
      const message = error instanceof Error ? error.message : "";
      if (message.includes("terms changed")) throw new CodedError("authorization-terms-changed", message);
      if (message.includes("superseded")) throw new CodedError("authorization-superseded", message);
      throw error;
    }
    const status = this.getLibraryConnectionStatus();
    if (status.kind !== "authorized") {
      // The grant was written but reading it back does not say authorized: the
      // fingerprint the status recomputes disagrees with the one just stored.
      throw new CodedError(
        "authorization-not-recorded",
        `authorization stored but the connection reads back as ${status.kind}`,
      );
    }
    return true;
  }

  /** Renames one proposed topic before its tag is derived (ADR 0014 §1). */
  async renamePersonalLibraryProposedTopic(input: {
    topicId: string;
    suggestedName: string;
  }): Promise<PersonalLibraryProfileSnapshot> {
    const candidateIds = this.libraryProposal?.topics.find(({ id }) => id === input.topicId)?.directions.map(({ id }) => id) ?? [];
    this.assertProposalCandidatesUnprocessed(candidateIds);
    return this.mutatePersonalLibraryProposal((proposal) => {
      this.assertProposalCandidatesUnprocessed(candidateIds, proposal);
      return renamePersonalLibraryProposedTopic({ proposal, topicId: input.topicId, suggestedName: input.suggestedName });
    });
  }

  /**
   * Accepts kept proposed topics into `settings.topics` and persists settings.
   * The proposal is left alone: it stays the durable record of what was
   * proposed, and the researcher can accept the rest later.
   */
  async acceptPersonalLibraryProposedTopics(
    topicIds: readonly string[],
    candidateIds?: readonly string[],
  ): Promise<PersonalLibraryProfileSnapshot> {
    const proposal = this.libraryProposal;
    if (!proposal) throw new Error("Generate a direction proposal first");
    if (topicIds.length === 0) throw new Error("Select at least one proposed topic to accept");
    const connection = this.libraryConnection;
    const connectionRevision = this.libraryConnectionRevision;
    const outputRevision = this.libraryOutputRevision;
    const proposalRevision = proposal.revision;
    const assertCurrent = () => {
      if (this.libraryProposalAcceptanceLoadError) {
        throw new CodedError("acceptance-unavailable", "Library review state could not be loaded. Restore it and reload before accepting directions.");
      }
      if (this.libraryProposal !== proposal || proposal.revision !== proposalRevision
        || this.libraryConnection !== connection || this.libraryConnectionRevision !== connectionRevision
        || this.libraryOutputRevision !== outputRevision) {
        throw new CodedError("conflict", "The proposal or library changed. Refresh the review before accepting.");
      }
    };
    let accepted: AcceptProposedTopicsResult | undefined;
    // Lock order is settings transaction → library storage queue. Calculation
    // happens inside the transaction, after any earlier acceptance committed.
    await this.settingsChanges.changeComputed((current) => {
      assertCurrent();
      const selected = candidateIds === undefined ? null : new Set(candidateIds);
      const kept = topicIds.map((id) => {
        const topic = proposal.topics.find((item) => item.id === id);
        if (!topic) throw new CodedError("not-found", "The proposed topic no longer exists.");
        return { ...topic, directions: selected ? topic.directions.filter(({ id }) => selected.has(id)) : topic.directions };
      }).filter(({ directions }) => directions.length > 0);
      if (kept.length === 0) throw new CodedError("invalid-input", "Select at least one proposed direction to accept");
      accepted = acceptProposedTopics({
        proposalId: proposal.proposalId, scopeFingerprint: proposal.scopeFingerprint,
        topics: kept, existingTopics: current.arxiv.topics,
        acceptance: this.currentProposalAcceptance(proposal),
      });
      const receipts = [
        ...(this.libraryProposalAcceptances ?? []).filter((item) => item.scopeFingerprint !== proposal.scopeFingerprint),
        accepted.acceptance,
      ].sort((left, right) => left.scopeFingerprint.localeCompare(right.scopeFingerprint));
      const receiptChanged = JSON.stringify(receipts) !== JSON.stringify(this.libraryProposalAcceptances ?? []);
      return {
        changes: [{ key: "arxiv.topics", value: accepted.topics }],
        forcePersist: receiptChanged,
        persist: (candidate) => this.enqueueLibraryMutation(async () => {
          assertCurrent();
          await this.persistSettings(candidate, receipts);
          // This reflects the durable write even if a later view refresh fails.
          this.libraryProposalAcceptances = receipts;
        }),
      };
    });
    if (!accepted) throw new Error("Proposal acceptance did not produce a result");
    if (accepted.addedDirectionCount === 0) {
      new Notice("The selected directions have already been applied to research settings.");
      return this.getPersonalLibraryProfileSnapshot();
    }
    let refreshFailed = false;
    try {
      await this.settingsTab?.refreshAfterTopicChanges();
    } catch (error) {
      refreshFailed = true;
      this.logger.error("settings: accepted directions were saved but the view could not refresh", error);
    }
    const newTopics = accepted.addedTopicCount > 0
      ? ` Created ${accepted.addedTopicCount} ${accepted.addedTopicCount === 1 ? "topic" : "topics"}.` : "";
    new Notice(`Added ${accepted.addedDirectionCount} ${accepted.addedDirectionCount === 1 ? "direction" : "directions"} to research settings.${newTopics}${refreshFailed ? " Reopen settings to refresh the list." : ""}`);
    return this.getPersonalLibraryProfileSnapshot();
  }

  private currentProposalAcceptance(proposal = this.libraryProposal): ProposalAcceptanceReceipt | null {
    if (!proposal) return null;
    return matchingProposalAcceptance(proposal,
      (this.libraryProposalAcceptances ?? []).find((item) => item.scopeFingerprint === proposal.scopeFingerprint));
  }

  async reloadPersonalLibraryCatalog(
    mutationRevision?: number,
  ): Promise<PersonalLibraryCatalog | null> {
    // A caller that already began the mutation passes its revision so one
    // logical change bumps the supersession counter exactly once.
    if (mutationRevision === undefined) this.beginLibraryMutation();
    const connection = this.libraryConnection;
    if (!connection) {
      this.libraryCatalog = null;
      this.libraryCatalogLoadError = null;
      return null;
    }
    const revision = this.libraryConnectionRevision;
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    const catalog = await this.buildPersonalLibraryCatalogStore().load(
      scopeFingerprint,
      identificationFingerprint,
    );
    this.assertLibraryConnectionCurrent(connection, revision);
    this.libraryCatalog = catalog;
    this.libraryCatalogLoadError = null;
    return structuredClone(catalog);
  }

  getPersonalLibraryProfileSnapshot(): PersonalLibraryProfileSnapshot {
    return structuredClone({
      catalog: this.libraryCatalog,
      indexedPapers: this.libraryIndexedPapers,
      proposal: this.libraryProposal,
      suggestions: this.librarySuggestions,
      authorization: this.getLibraryConnectionStatus(),
      catalogLoadError: this.libraryCatalogLoadError,
      proposalLoadError: this.libraryProposalLoadError,
      suggestionsLoadError: this.librarySuggestionsLoadError,
      settingsTopicNames: this.settings.arxiv.topics.map(({ name }) => name),
      settingsTopics: this.settings.arxiv.topics,
      arxivCategories: arxivCategories(this.settings.arxiv),
      proposalAcceptance: this.currentProposalAcceptance(),
      acceptanceLoadError: this.libraryProposalAcceptanceLoadError ?? null,
    });
  }

  getPersonalLibraryDirectionProposal(): PersonalLibraryDirectionProposal | null {
    return this.libraryProposal ? structuredClone(this.libraryProposal) : null;
  }

  async reloadPersonalLibraryProfileDocuments(
    mutationRevision?: number,
  ): Promise<PersonalLibraryProfileSnapshot> {
    // A caller that already began the mutation passes its revision so one
    // logical change bumps the supersession counter exactly once.
    if (mutationRevision === undefined) this.beginLibraryMutation();
    const connection = this.libraryConnection;
    if (!connection) {
      this.libraryProposal = null;
      this.librarySuggestions = null;
      this.libraryProposalLoadError = null;
      this.librarySuggestionsLoadError = null;
      return this.getPersonalLibraryProfileSnapshot();
    }
    const connectionRevision = this.libraryConnectionRevision;
    const outputRevision = this.libraryOutputRevision;
    const stores = this.buildPersonalLibraryProfileStores(connection);
    await Promise.all([
      stores.proposal.load().then((proposal) => {
        this.assertPersonalLibraryDocumentLoadCurrent(connection, connectionRevision, outputRevision);
        this.libraryProposal = proposal;
        this.libraryProposalLoadError = null;
      }).catch((error) => {
        this.assertPersonalLibraryDocumentLoadCurrent(connection, connectionRevision, outputRevision);
        this.libraryProposal = null;
        this.libraryProposalLoadError = this.safeProfileLoadError("proposal", error);
        this.logger?.error("personal library direction proposal load failed", error);
      }),
      stores.suggestions.load().then((suggestions) => {
        this.assertPersonalLibraryDocumentLoadCurrent(connection, connectionRevision, outputRevision);
        this.librarySuggestions = suggestions;
        this.librarySuggestionsLoadError = null;
      }).catch((error) => {
        this.assertPersonalLibraryDocumentLoadCurrent(connection, connectionRevision, outputRevision);
        this.librarySuggestions = null;
        this.librarySuggestionsLoadError = this.safeProfileLoadError("suggestions", error);
        this.logger?.error("incremental suggestions load failed", error);
      }),
    ]);
    return this.getPersonalLibraryProfileSnapshot();
  }

  async previewPersonalLibraryDirection(input: { candidateId: string; text: string }): Promise<LibraryDirectionPreview> {
    const connection = this.libraryConnection;
    const proposal = this.libraryProposal;
    if (!connection || this.getLibraryConnectionStatus().kind !== "authorized") {
      throw new Error("Authorize personal library model processing first");
    }
    const candidate = proposal?.topics.flatMap(({ directions }) => directions).find(({ id }) => id === input.candidateId);
    if (!candidate) throw new CodedError("not-found", "The proposed direction is no longer available");
    const revision = this.libraryConnectionRevision;
    const outputRevision = this.libraryOutputRevision;
    const settings = structuredClone(this.settings);
    const catalog = this.libraryCatalog ? structuredClone(this.libraryCatalog) : null;
    const scope = this.libraryFingerprints(connection).scopeFingerprint;
    if (this.operations.find("personal-library-direction-generation", scope)) {
      throw new CodedError("conflict", "Library direction processing is already running");
    }
    const operation = this.operations.begin("personal-library-direction-generation", "Preview direction matches", scope);
    const assertCurrent = () => {
      operation.signal.throwIfAborted();
      this.assertLibraryConnectionCurrent(connection, revision);
      if (this.libraryProposal !== proposal || this.libraryOutputRevision !== outputRevision
        || this.getLibraryConnectionStatus().kind !== "authorized"
        || JSON.stringify(this.settings.llm) !== JSON.stringify(settings.llm)
        || JSON.stringify(arxivCategories(this.settings.arxiv)) !== JSON.stringify(arxivCategories(settings.arxiv))) {
        throw new CodedError("conflict", "The library review changed during preview");
      }
    };
    try {
      const manifest = await this.buildFullTextKnowledgeBaseStore(connection).loadManifest();
      assertCurrent();
      const keys = [...new Set([
        ...candidate.representatives.map(({ paperKey }) => paperKey),
        ...Object.keys(manifest.papers).sort(),
      ])];
      const papers: LibraryPreviewPaper[] = [];
      for (const paperKey of keys) {
        const indexed = manifest.papers[paperKey];
        if (!indexed || indexed.status !== "ready") continue;
        const record = catalog?.papers[paperKey];
        const title = record?.title ?? indexed.title;
        if (!title) continue;
        papers.push({ paperKey, title, abstract: record?.abstract ?? indexed.abstract ?? "", categories: record?.categories ?? [] });
        if (papers.length === 20) break;
      }
      const result = await previewPersonalLibraryDirection({
        text: input.text, papers, categories: arxivCategories(settings.arxiv),
        llm: new LlmClient(settings.llm, this.logger, this.host.http), signal: operation.signal,
      });
      assertCurrent();
      return result;
    } finally {
      operation.finish();
    }
  }

  async generatePersonalLibraryDirections(
    onProgress?: (progress: DirectionProposalProgress) => void,
  ): Promise<PersonalLibraryDirectionProposal> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    if (this.getLibraryConnectionStatus().kind !== "authorized") {
      throw new Error("Authorize personal library model processing first");
    }
    const catalog = this.libraryCatalog;
    if (!catalog) throw new Error("Scan and load the personal library catalog first");
    const { scopeFingerprint } = this.libraryFingerprints(connection);
    if (this.operations.find("personal-library-direction-generation", scopeFingerprint)) {
      throw new Error("Personal library direction generation is already active");
    }
    const connectionRevision = this.libraryConnectionRevision;
    const outputRevision = this.libraryOutputRevision;
    const authorizationFingerprint = connection.authorization?.fingerprint;
    const selectedInputFingerprint = this.selectedCatalogFingerprint(catalog);
    const existingTopics = this.libraryExistingTopics();
    const existingTopicsFingerprint = JSON.stringify(existingTopics);
    const expectedProposalRevision = this.libraryProposal?.revision ?? null;
    const llmSettings = structuredClone(this.settings.llm);
    const store = this.buildPersonalLibraryProfileStores(connection).proposal;
    const operation = this.operations.begin(
      "personal-library-direction-generation",
      "Personal library direction generation",
      scopeFingerprint,
    );
    try {
      this.assertPersonalLibraryGenerationCurrent({
        connection, connectionRevision, outputRevision, authorizationFingerprint,
        catalog, selectedInputFingerprint, expectedProposalRevision, existingTopicsFingerprint,
      });
      const proposal = await proposeClusteredPersonalLibraryDirections({
        catalog: structuredClone(catalog),
        existingTopics,
        knowledgeBase: this.buildFullTextKnowledgeBaseStore(connection),
        // Tight groups provide evidence; the model organizes their topic scope.
        clustering: { similarityQuantile: PERSONAL_LIBRARY_SIMILARITY_QUANTILE },
        llm: new LlmClient(llmSettings, this.logger, this.host.http),
        signal: operation.signal,
        createId: () => crypto.randomUUID(),
        onProgress: this.proposalProgressReporter(onProgress),
      });
      operation.signal.throwIfAborted();
      this.assertPersonalLibraryGenerationCurrent({
        connection, connectionRevision, outputRevision, authorizationFingerprint,
        catalog, selectedInputFingerprint, expectedProposalRevision, existingTopicsFingerprint,
      });
      const result = await this.enqueueLibraryMutation(async () => {
        operation.signal.throwIfAborted();
        this.assertPersonalLibraryGenerationCurrent({
          connection, connectionRevision, outputRevision, authorizationFingerprint,
          catalog, selectedInputFingerprint, expectedProposalRevision, existingTopicsFingerprint,
        });
        const saved = await store.replace(proposal, expectedProposalRevision);
        this.libraryProposal = saved;
        this.libraryProposalLoadError = null;
        return structuredClone(saved);
      });
      // Whoever wrote a task into the status bar has to take it back out. This
      // reported progress but never an ending, so a failed or finished run left
      // the bar showing "naming directions (5/9)" until the next task replaced
      // it — scanning and indexing both close theirs the same way.
      this.progress?.setComplete("Personal library directions generated");
      return result;
    } catch (error) {
      if (!operation.signal.aborted) {
        this.progress?.setError("Personal library direction generation failed");
      }
      if (this.isReviewPersistenceConflict(error)) await this.reloadPersonalLibraryProfileDocuments();
      throw error;
    } finally {
      operation.finish();
    }
  }

  async updatePersonalLibraryProposalCandidate(input: {
    candidateId: string;
    patch: PersonalLibraryDirectionTextPatch;
    representativePaperKeys?: string[];
  }): Promise<PersonalLibraryProfileSnapshot> {
    this.assertProposalCandidatesUnprocessed([input.candidateId]);
    return this.mutatePersonalLibraryProposal((proposal, catalog) => {
      this.assertProposalCandidatesUnprocessed([input.candidateId], proposal);
      return updatePersonalLibraryDirectionCandidate({
        proposal,
        candidateId: input.candidateId,
        patch: input.patch,
        ...(input.representativePaperKeys === undefined ? {} : {
          representativePaperKeys: input.representativePaperKeys,
          catalog,
        }),
      });
    });
  }

  async mergePersonalLibraryProposalCandidates(input: {
    sourceCandidateIds: string[];
    draft: PersonalLibraryReviewedDirectionDraft;
    candidateId?: string;
  }): Promise<PersonalLibraryProfileSnapshot> {
    this.assertProposalCandidatesUnprocessed(input.sourceCandidateIds);
    return this.mutatePersonalLibraryProposal((proposal, catalog) => {
      this.assertProposalCandidatesUnprocessed(input.sourceCandidateIds, proposal);
      return mergePersonalLibraryDirectionCandidates({
        proposal,
        sourceCandidateIds: input.sourceCandidateIds,
        candidateId: input.candidateId ?? crypto.randomUUID(),
        draft: input.draft,
        catalog,
      });
    });
  }

  async removePersonalLibraryProposalCandidate(candidateId: string): Promise<PersonalLibraryProfileSnapshot> {
    this.assertProposalCandidatesUnprocessed([candidateId]);
    return this.mutatePersonalLibraryProposal((proposal) => {
      this.assertProposalCandidatesUnprocessed([candidateId], proposal);
      return removePersonalLibraryDirectionCandidate({ proposal, candidateId });
    });
  }

  async movePersonalLibraryProposalCandidate(input: {
    candidateId: string;
    targetTopicId: string | null;
    suggestedName?: string;
  }): Promise<PersonalLibraryProfileSnapshot> {
    this.assertProposalCandidatesUnprocessed([input.candidateId]);
    const destinationName = (): string => {
      if (input.targetTopicId !== null) {
        const target = this.settings.arxiv.topics.find(({ id }) => id === input.targetTopicId);
        if (!target) throw new CodedError("target-missing", "The destination topic was removed. Choose a destination again.");
        return target.name;
      }
      const name = input.suggestedName?.trim();
      if (!name || this.settings.arxiv.topics.some((topic) => topicNameKey(topic.name) === topicNameKey(name))) {
        throw new CodedError("invalid-input", "Enter a distinct new topic name, or choose an existing destination.");
      }
      return name;
    };
    destinationName();
    return this.mutatePersonalLibraryProposal((proposal) => {
      this.assertProposalCandidatesUnprocessed([input.candidateId], proposal);
      return movePersonalLibraryDirectionCandidate({
        proposal, candidateId: input.candidateId, targetTopicId: input.targetTopicId,
        suggestedName: destinationName(), topicId: crypto.randomUUID(),
      });
    });
  }

  private assertProposalCandidatesUnprocessed(
    candidateIds: readonly string[],
    proposal = this.libraryProposal,
  ): void {
    const processed = new Set(this.currentProposalAcceptance(proposal)?.processedCandidateIds ?? []);
    if (candidateIds.some((id) => processed.has(id))) {
      throw new CodedError("already-accepted", "This direction was already applied. Edit it in research settings.");
    }
  }

  getIncrementalSuggestions(): IncrementalSuggestionsDocument | null {
    return this.librarySuggestions ? structuredClone(this.librarySuggestions) : null;
  }

  async scanPersonalLibrary(): Promise<PersonalLibraryCatalog> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    const revision = this.libraryConnectionRevision;
    const store = this.buildPersonalLibraryCatalogStore();
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    if (this.operations.find("personal-library-scan", scopeFingerprint)) {
      throw new Error("Personal library scan is already active");
    }
    const operation = this.operations.begin(
      "personal-library-scan",
      "Personal library scan",
      scopeFingerprint,
    );
    const updateProgress = this.operations.snapshot().length === 1;
    if (updateProgress) this.progress?.setTask("Scanning personal library", "Identifying library papers");
    try {
      operation.signal.throwIfAborted();
      const source = this.librarySource
        ?? await this.openLibrarySource(connection.selectedRoot);
      operation.signal.throwIfAborted();
      this.assertLibraryConnectionCurrent(connection, revision);
      if (
        source.canonicalRoot !== connection.selectedRoot
        || source.rootIdentity !== connection.rootIdentity
      ) {
        throw new Error("Library folder identity changed; choose it again");
      }
      this.librarySource = source;
      const inventory = await source.inventory({ signal: operation.signal });
      operation.signal.throwIfAborted();
      this.assertLibraryConnectionCurrent(connection, revision);
      const current = await store.load(scopeFingerprint, identificationFingerprint);
      operation.signal.throwIfAborted();
      this.assertLibraryConnectionCurrent(connection, revision);
      const reconciled = await reconcilePersonalLibraryCatalog({
        current,
        inventory,
        eligibleExtensions: connection.eligibleExtensions,
        resolver: new ArxivLibraryMetadataResolver(this.buildArxivFetcher()),
        // Content-based identification (strategy v2): files whose names carry
        // no arXiv ID are identified from PDF text evidence, with an arXiv
        // title-search fallback. Failures keep files unresolved.
        identifyFile: {
          version: PDF_IDENTIFICATION_EVIDENCE_VERSION,
          // Identification reads bounded ranges only (header + tail), never
          // the whole file: arXiv page headers, XMP, and Info metadata all
          // live there, and full-file reads made scans hang on large PDFs.
          identify: async (logicalPath, signal, size) => {
            const source = this.librarySource;
            if (!source) return null;
            try {
              const [head, tail] = await Promise.all([
                source.readBinary(logicalPath, { signal, start: 0, end: IDENTIFICATION_HEAD_BYTES }),
                size && size > IDENTIFICATION_HEAD_BYTES
                  ? source.readBinary(logicalPath, {
                      signal,
                      start: size - IDENTIFICATION_TAIL_BYTES,
                      end: size,
                    })
                  : Promise.resolve(new ArrayBuffer(0)),
              ]);
              const combined = new Uint8Array(head.byteLength + tail.byteLength);
              combined.set(new Uint8Array(head), 0);
              combined.set(new Uint8Array(tail), head.byteLength);
              const evidence = extractPdfIdentificationEvidence(combined);
              const directId = evidence.arxivId
                ? normalizeArxivId(evidence.arxivId)
                : null;
              if (directId) {
                // The document title is an independent witness: a title
                // search that resolves to a DIFFERENT paper means the direct
                // ID is a reference-list misidentification ("… arXiv:0912.0201
                // …" in the references) — trust the title search. A failed or
                // empty search keeps the direct ID (garbage document titles
                // must not demote real papers).
                if (evidence.title && !/^arxiv:/i.test(evidence.title)) {
                  try {
                    const result = await searchArxivTitle(this.host.http, evidence.title, signal);
                    if (result.arxivId) {
                      const searched = normalizeArxivId(result.arxivId);
                      if (searched && searched !== directId) return searched;
                    }
                  } catch {
                    // Search failure keeps the direct ID.
                  }
                }
                return directId;
              }
              if (evidence.title) {
                try {
                  const result = await searchArxivTitle(this.host.http, evidence.title, signal);
                  return result.arxivId ? normalizeArxivId(result.arxivId) : null;
                } catch {
                  return null;
                }
              }
            } catch {
              return null;
            }
            return null;
          },
        },
        signal: operation.signal,
      });
      operation.signal.throwIfAborted();
      this.assertLibraryConnectionCurrent(connection, revision);
      this.beginLibraryMutation();
      const saved = await this.enqueueLibraryMutation(async () => {
        operation.signal.throwIfAborted();
        this.assertLibraryConnectionCurrent(connection, revision);
        const saved = await store.replace(reconciled.catalog);
        // Atomic promotion cannot be interrupted. Once it succeeds, the scan is
        // committed even if cancellation arrives during that final write.
        this.assertLibraryConnectionCurrent(connection, revision);
        this.libraryCatalog = saved;
        this.libraryCatalogLoadError = null;
        return saved;
      });
      this.libraryCatalog = saved;
      if (updateProgress) this.progress?.setComplete("Personal library scan complete");
      return structuredClone(saved);
    } catch (error) {
      if (updateProgress && !operation.signal.aborted) {
        this.progress?.setError("Personal library scan failed");
      }
      throw error;
    } finally {
      operation.finish();
    }
  }

  private buildFullTextKnowledgeBaseStore(
    connection: PersistedLibraryConnection,
  ): FullTextKnowledgeBaseFileStore {
    if (!this.host?.storage.writeTextAtomic) {
      throw new Error("Personal library full-text index requires atomic storage writes");
    }
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    return new FullTextKnowledgeBaseFileStore(
      this.host.storage,
      this.settings.output,
      scopeFingerprint,
      identificationFingerprint,
      { onWarning: (message, error) => this.logger.warn(message, error) },
    );
  }

  private buildFullTextGenerationIndexStore(
    connection: PersistedLibraryConnection,
  ): FullTextGenerationIndexStore {
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    return new FullTextGenerationIndexStore(
      this.host.storage,
      this.settings.output,
      scopeFingerprint,
      identificationFingerprint,
      { onWarning: (message, error) => this.logger.warn(message, error) },
    );
  }

  /**
   * Embedding backend for the full-text knowledge base (ADR 0008): the
   * bundled local transformers.js model by default, or the remote
   * OpenAI-compatible model when the embedding mode is `remote`.
   */
  private buildEmbeddingModel(): EmbeddingModel {
    if (this.settings.embedding.mode === "remote") {
      return createRemoteEmbeddingModel({
        baseUrl: this.settings.embedding.baseUrl,
        apiKey: this.settings.embedding.apiKey,
        model: this.settings.embedding.model,
        dimension: this.settings.embedding.dimension,
        http: this.host.http,
      });
    }
    return createTransformersEmbeddingModel();
  }

  /**
   * Build the PDF text extractor the index runs on.
   *
   * The index covers each paper's title and abstract (ADR 0013), which need
   * plain text from the leading pages — `headings` and `locator` exist to serve
   * full-text chunking, so the structured-parser sidecar has no subject on this
   * path and is no longer probed. The sidecar settings and client are left in
   * place pending a separate decision on retiring them; indexing simply does
   * not consult them, and says so when one is switched on.
   */
  private buildFullTextExtractor() {
    if (this.settings.pdfParserSidecar.enabled) {
      this.logger.info(
        "fulltext: the local PDF parser sidecar is not used for indexing; "
        + "the index reads titles and abstracts with PDF.js",
      );
    }
    return new ObsidianPdfTextExtractor();
  }

  /**
   * Remote embedding sends full-text chunks to a named endpoint, so it needs
   * a valid remote configuration AND full-text processing authorization
   * (ADR 0008). The local mode needs neither.
   */
  private assertRemoteEmbeddingReady(): void {
    if (this.settings.embedding.mode !== "remote") return;
    const validation = validateEmbeddingConfig(this.settings);
    if (!validation.ok) {
      throw new Error(`Remote embedding configuration incomplete: ${validation.reasons.join("; ")}`);
    }
    if (this.getLibraryConnectionStatus().kind !== "authorized") {
      throw new Error("Remote embedding requires authorizing full-text processing first");
    }
  }

  /**
   * Incrementally index the personal library's full text into the local
   * knowledge base: extract (Obsidian built-in pdf.js) → chunk → embed
   * (multilingual-e5-small q8, or the remote endpoint in remote mode) →
   * store. Unchanged papers are reused via their
   * catalog observation fingerprints; failures are recorded and retried on
   * the next run. Local mode is independent of any model processing
   * authorization; remote mode requires full-text authorization and sends
   * full-text chunks to the configured embedding endpoint.
   */
  async indexPersonalLibraryFullText(): Promise<FullTextIndexRunSummary> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    const revision = this.libraryConnectionRevision;
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    if (this.operations.find("personal-library-fulltext-index", scopeFingerprint)) {
      throw new Error("Personal library full-text indexing is already active");
    }
    const operation = this.operations.begin(
      "personal-library-fulltext-index",
      "Personal library full-text index",
      scopeFingerprint,
    );
    const updateProgress = this.operations.snapshot().length === 1;
    if (updateProgress) {
      this.progress?.setTask("Indexing personal library full text", "Extracting and embedding PDF text");
    }
    // The status bar is gated on being the only operation; the settings row is
    // not. It shows this run and nothing else, and it is the surface that hides
    // the status bar when a person starts the run from it.
    this.libraryIndexStatus.beginRun(operation.id, "reading the library catalog");
    let legacyMigrationLease: FullTextLegacyMigrationLease | undefined;
    try {
      operation.signal.throwIfAborted();
      const catalog = await this.buildPersonalLibraryCatalogStore().load(
        scopeFingerprint,
        identificationFingerprint,
      );
      operation.signal.throwIfAborted();
      this.assertLibraryConnectionCurrent(connection, revision);
      const source = this.librarySource
        ?? await this.openLibrarySource(connection.selectedRoot);
      operation.signal.throwIfAborted();
      this.assertLibraryConnectionCurrent(connection, revision);
      if (
        source.canonicalRoot !== connection.selectedRoot
        || source.rootIdentity !== connection.rootIdentity
      ) {
        throw new Error("Library folder identity changed; choose it again");
      }
      this.librarySource = source;
      // Obsidian's built-in pdf.js becomes reachable via `window.pdfjsLib`
      // after the official loader resolves; the extractor defaults to it.
      await loadPdfJs();
      const extractor = this.buildFullTextExtractor();
      this.assertRemoteEmbeddingReady();
      const embedding = this.buildEmbeddingModel();
      const store = this.buildFullTextKnowledgeBaseStore(connection);
      const generationStore = this.buildFullTextGenerationIndexStore(connection);
      const generationWriterToken = createFullTextGenerationWriterToken();
      const generationMode = await preflightFullTextGenerationSynchronization({
        storage: this.host.storage,
        generationStore,
      });
      if (generationMode === "migration-fallback") {
        legacyMigrationLease = await generationStore.acquireLegacyMigrationLease(
          generationWriterToken,
        );
      }
      const summary = await indexFullTextKnowledgeBase({
        catalog,
        source,
        extractor,
        embedding,
        store,
        logger: this.logger,
        beforeManifestCommit: legacyMigrationLease
          ? () => legacyMigrationLease!.assertOwned()
          : undefined,
        afterManifestCommit: legacyMigrationLease
          ? () => legacyMigrationLease!.assertOwned()
          : undefined,
        onProgress: (detail, progress) => {
          operation.signal.throwIfAborted();
          if (updateProgress) this.progress?.setTask("Indexing personal library full text", detail);
          this.libraryIndexStatus.report({
            phase: progress?.phase === "preparing"
              ? "preparing local documents"
              : "extracting and embedding PDF text",
            ...(progress ? { completed: progress.completed, total: progress.total } : {}),
          });
        },
        signal: operation.signal,
      });
      if (legacyMigrationLease) {
        const lease = legacyMigrationLease;
        legacyMigrationLease = undefined;
        await lease.release();
      }
      operation.signal.throwIfAborted();
      this.assertLibraryConnectionCurrent(connection, revision);
      let generationFailed = false;
      let generationFailure: unknown;
      if (generationMode === "available") {
        try {
          const generation = await synchronizeFullTextGenerationIndex({
            sourceStore: store,
            generationStore,
            storage: this.host.storage,
            output: this.settings.output,
            scopeFingerprint,
            identificationFingerprint,
            writerToken: generationWriterToken,
            signal: operation.signal,
            onProgress: (progress) => {
              const label = progress.phase === "papers"
                ? "Building bounded full-text blocks"
                : progress.phase === "dictionary"
                  ? "Building lexical routes"
                  : "Validating full-text generation";
              this.libraryIndexStatus.report({
                phase: label.charAt(0).toLowerCase() + label.slice(1),
                completed: progress.completed,
                total: progress.total,
              });
              if (!updateProgress) return;
              this.progress?.setTask("Indexing personal library full text", `${label} (${progress.completed}/${progress.total})`);
            },
          });
          this.logger.info(
            `fulltext: generation ${generation.kind} (${generation.generationId}, source revision ${generation.sourceRevision})`,
          );
        } catch (error) {
          if (operation.signal.aborted) throw error;
          generationFailed = true;
          generationFailure = error;
          this.logger.warn(
            "fulltext: generation synchronization failed; running the post-commit direction update before reporting it",
            error,
          );
        }
      } else {
        this.logger.warn(
          "fulltext: immutable generation cutover is unavailable on this host; retaining the legacy migration fallback",
        );
      }
      // The trace comes from the commit that just happened rather than from a
      // fresh manifest read: the summary already names the revision this run
      // wrote, and re-reading would be a second answer to a settled question.
      this.libraryIndexStatus.setLastRun({
        updatedAt: summary.manifestUpdatedAt,
        papers: summary.searchablePapers,
      });
      // ADR 0007: every durable legacy manifest commit remains a trigger even
      // when rebuilding its derived generation fails.
      operation.signal.throwIfAborted();
      operation.signal.throwIfAborted();
      if (generationFailed) throw generationFailure;
      if (updateProgress) {
        const refreshed = summary.titlesRefreshed > 0
          ? `, ${summary.titlesRefreshed} titles refreshed`
          : "";
        this.progress?.setComplete(
          `Full-text index: ${summary.indexed} indexed, ${summary.reused} reused, `
          + `${summary.failed} failed, ${summary.pruned} pruned${refreshed}`,
        );
      }
      return summary;
    } catch (error) {
      if (updateProgress && !operation.signal.aborted) {
        this.progress?.setError("Personal library full-text indexing failed");
      }
      throw error;
    } finally {
      try {
        if (legacyMigrationLease) {
          const lease = legacyMigrationLease;
          legacyMigrationLease = undefined;
          try {
            await lease.release();
          } catch (releaseError) {
            this.logger.warn(
              "fulltext: failed to release the legacy migration lease after indexing failed",
              releaseError,
            );
          }
        }
      } finally {
        operation.finish();
        this.libraryIndexStatus.endRun();
      }
    }
  }

  /**
   * Ask the settings row's in-flight indexing run to stop. Reports whether
   * there was one: the row can be a frame behind the run it is describing.
   */
  cancelPersonalLibraryIndexing(): boolean {
    const activity = this.libraryIndexStatus.snapshot().activity;
    if (!activity) return false;
    if (!this.operations.cancel(activity.operationId, "cancelled from settings")) return false;
    this.libraryIndexStatus.markCancelling();
    return true;
  }

  /**
   * Republish what the knowledge base manifest holds.
   *
   * The manifest is read rather than the last run summary remembered, because
   * only the manifest can say what a search would find right now — a summary
   * would keep claiming an index that a rebuild, a folder change or a deletion
   * has since taken away. Revision 0 is the empty manifest the store invents
   * when nothing was ever committed; its `updatedAt` is the clock, not a run.
   */
  async refreshLibraryIndexTrace(): Promise<void> {
    const connection = this.libraryConnection;
    if (!connection) {
      this.libraryIndexStatus.setLastRun(undefined);
      return;
    }
    try {
      const manifest = await this.buildFullTextKnowledgeBaseStore(connection).loadManifest();
      const readyPapers = Object.values(manifest.papers).filter((paper) => paper.status === "ready");
      this.libraryIndexStatus.setLastRun(
        manifest.revision > 0 ? { updatedAt: manifest.updatedAt, papers: readyPapers.length } : undefined,
      );
      // Only papers the index had to name itself: an arXiv paper's title lives
      // in the catalog and stays the one source for it.
      this.libraryIndexedPapers = readyPapers
        .filter((paper) => paper.title !== undefined && paper.title.length > 0)
        .map((paper) => ({ paperKey: paper.paperKey, title: paper.title! }))
        .sort((left, right) => (left.paperKey < right.paperKey ? -1 : left.paperKey > right.paperKey ? 1 : 0));
    } catch (error) {
      // A row that cannot read the manifest says nothing about past runs; it
      // must not say the index is gone.
      this.logger.warn("fulltext: could not read the index manifest for the settings row", error);
    }
  }

  /**
   * Embed the query locally and return the most similar indexed papers with
   * their best matching passages. Joins catalog titles for display; the
   * similarity evidence (hit chunk text) is explainable end to end.
   */
  async searchPersonalLibraryFullText(
    queryText: string,
    options?: { lexicalQueryText?: string },
  ): Promise<Array<{
    paperKey: string;
    title: string;
    /** Relative library path for fallback-indexed files; arXiv papers leave it unset. */
    filePath?: string;
    score: number;
    scoreKind: "cosine" | "bm25";
    rankingScore: number;
    rankingScoreKind: "cosine" | "bm25" | "rrf";
    hits: readonly KnowledgeBaseChunkHit[];
  }>> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    const catalog = await this.buildPersonalLibraryCatalogStore().load(
      scopeFingerprint,
      identificationFingerprint,
    );
    this.assertRemoteEmbeddingReady();
    // Ranking uses catalog titles only. Fallback titles from this exact source
    // snapshot are display metadata; generation search uses its persisted title.
    const rankingTitles = new Map<string, string>();
    for (const [paperKey, paper] of Object.entries(catalog.papers)) {
      if (paper.title) rankingTitles.set(paperKey, paper.title);
    }
    const store = this.buildFullTextKnowledgeBaseStore(connection);
    const manifest = await store.loadManifest();
    const matches = await searchFullTextKnowledgeBaseCore({
      store,
      sourceManifest: manifest,
      generationStore: this.buildFullTextGenerationIndexStore(connection),
      embedding: this.buildEmbeddingModel(),
      queryText,
      lexicalQueryText: options?.lexicalQueryText,
      titles: rankingTitles,
      logger: this.logger,
    });
    return projectLibraryFullTextMatches({ catalogTitles: rankingTitles, manifest, matches });
  }

  async openPersonalLibraryFullTextEvidence(input: {
    paperKey: string;
    filePath: string;
    page?: number;
  }): Promise<"page-targeted" | "file-fallback"> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    const manifest = await this.buildFullTextKnowledgeBaseStore(connection).loadManifest();
    const record = manifest.papers[input.paperKey];
    if (!record?.filePaths.includes(input.filePath)) {
      throw new Error("The selected evidence PDF is no longer part of this library index");
    }
    const source = this.librarySource ?? await this.openLibrarySource(connection.selectedRoot);
    if (
      source.canonicalRoot !== connection.selectedRoot
      || source.rootIdentity !== connection.rootIdentity
    ) {
      throw new Error("Library folder identity changed; choose it again");
    }
    // Revalidate the selected root, logical path, and no-symlink boundary just
    // before handing the user-selected file to the host opener.
    await source.readBinary(input.filePath, {
      start: 0,
      end: 1,
      maxBytes: Number.MAX_SAFE_INTEGER,
    });
    return await openLibraryPdfAtPage({
      app: this.app,
      target: resolveLibraryPdfOpenTarget({
        canonicalRoot: source.canonicalRoot,
        logicalPath: input.filePath,
        page: input.page,
        vaultRoot: desktopVaultRoot(this.app.vault.adapter),
      }),
    });
  }

  /**
   * Host-authorized quiet-period maintenance for immutable full-text
   * generations. The caller must arrange that other Obsidian/Node processes
   * using the same vault have also stopped admission; Core only tracks this
   * plugin process and never performs online cross-process GC automatically.
   */
  async maintainPersonalLibraryFullTextGenerations(): Promise<FullTextGenerationMaintenanceReport> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    if (this.scheduler.activeRuns().length > 0) {
      throw new Error("Full-text generation maintenance requires the scheduler to be idle");
    }
    const releaseAdmission = this.operations.beginFullTextMaintenanceTransition();
    try {
      this.scheduler.stop();
      const generationStore = this.buildFullTextGenerationIndexStore(connection);
      return await generationStore.runHostAuthorizedMaintenance();
    } finally {
      try {
        if (this.settings.schedule.enabled && !this.unloading) this.scheduler.start();
      } finally {
        releaseAdmission();
      }
    }
  }

  /**
   * Pre-flight diagnostics for the full-text runtime, isolating the two
   * Obsidian-only unknowns that Node-side tests cannot cover: pdf.js
   * availability after `loadPdfJs()` (window.pdfjsLib + a real smoke
   * extraction) and transformers.js model/wasm loading in the renderer.
   * Each part is probed independently; a failure in one does not block the
   * other, and the whole run never throws — problems are reported in the
   * result.
   */
  async diagnoseFullTextRuntime(): Promise<FullTextRuntimeDiagnostics> {
    const updateProgress = this.operations.snapshot().length === 0;
    const library: LibraryDiagnostics = { connected: false };
    const connection = this.libraryConnection;
    if (connection) {
      const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
      library.connected = true;
      library.scopeFingerprint = scopeFingerprint;
      try {
        const catalog = await this.buildPersonalLibraryCatalogStore().load(
          scopeFingerprint,
          identificationFingerprint,
        );
        library.paperCount = Object.keys(catalog.papers).length;
      } catch (error) {
        this.logger.warn("diagnostics: personal library catalog load failed", error);
      }
    }
    if (updateProgress) this.progress?.setTask("Diagnosing full-text runtime", "Checking pdf.js");
    const pdfJs = await this.diagnosePdfJs(connection);
    if (updateProgress) {
      this.progress?.setTask("Diagnosing full-text runtime", "Loading embedding model");
    }
    const embedding = await this.diagnoseEmbedding();
    if (updateProgress) this.progress?.setComplete("Full-text runtime diagnostics complete");
    return { library, pdfJs, embedding };
  }

  private async diagnosePdfJs(
    connection: PersistedLibraryConnection | undefined,
  ): Promise<PdfJsDiagnostics> {
    let loadPdfJsResolved = false;
    let loaderReturnedLib = false;
    let windowPdfJsLibPresent = false;
    let windowPdfJsLibVersion: string | undefined;
    try {
      const returned: unknown = await loadPdfJs();
      loadPdfJsResolved = true;
      loaderReturnedLib =
        returned != null && (typeof returned === "object" || typeof returned === "function");
      const win = window as unknown as { pdfjsLib?: { version?: string } };
      windowPdfJsLibPresent = win.pdfjsLib != null;
      windowPdfJsLibVersion = win.pdfjsLib?.version;
    } catch (error) {
      return {
        status: "fail",
        loadPdfJsResolved,
        loaderReturnedLib,
        windowPdfJsLibPresent,
        error: describeDiagnosticsError(error),
      };
    }
    const smoke = connection
      ? await this.smokeExtractFirstLibraryPdf(connection)
      : {
          status: "skipped" as const,
          error: "no library connection — smoke extraction skipped",
        };
    // The production path reads `window.pdfjsLib`; without it the feature
    // cannot run, so that alone is a failure regardless of smoke availability.
    let status: PdfJsDiagnostics["status"];
    if (!windowPdfJsLibPresent) status = "fail";
    else if (smoke.status === "pass") status = "pass";
    else if (smoke.status === "fail") status = "fail";
    else status = "skipped";
    return {
      status,
      loadPdfJsResolved,
      loaderReturnedLib,
      windowPdfJsLibPresent,
      windowPdfJsLibVersion,
      smoke,
    };
  }

  private async smokeExtractFirstLibraryPdf(
    connection: PersistedLibraryConnection,
  ): Promise<PdfJsSmokeDiagnostics> {
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    try {
      const catalog = await this.buildPersonalLibraryCatalogStore().load(
        scopeFingerprint,
        identificationFingerprint,
      );
      const entry = Object.values(catalog.papers).find((paper) => paper.filePaths.length > 0);
      const filePath = entry?.filePaths[0];
      if (!entry || !filePath) {
        return {
          status: "skipped",
          error: "no library papers with PDF files — smoke extraction skipped",
        };
      }
      const source = this.librarySource ?? await this.openLibrarySource(connection.selectedRoot);
      const bytes = await source.readBinary(filePath);
      // Deliberately the default path: the extractor reads `window.pdfjsLib`,
      // exactly what `index-personal-library-fulltext` runs.
      const extractor = new ObsidianPdfTextExtractor();
      const result = await extractor.extractPdfText(new Uint8Array(bytes));
      const chars = result.pages.reduce((sum, page) => sum + page.length, 0);
      if (result.pages.length === 0 || chars === 0) {
        return {
          status: "fail",
          paperKey: entry.paperKey,
          pages: result.pages.length,
          chars,
          error: "extraction returned no text",
        };
      }
      return { status: "pass", paperKey: entry.paperKey, pages: result.pages.length, chars };
    } catch (error) {
      return { status: "fail", error: describeDiagnosticsError(error) };
    }
  }

  private async diagnoseEmbedding(): Promise<EmbeddingDiagnostics> {
    const embedding = createTransformersEmbeddingModel();
    try {
      const started = Date.now();
      await embedding.embed(["diagnostic probe"]);
      const loadMs = Date.now() - started;
      const facts = await inspectTransformersEnv();
      return {
        status: "pass",
        modelId: embedding.modelId,
        dimension: embedding.dimension,
        remoteHost: facts?.remoteHost,
        wasmPaths: facts?.wasmPaths,
        runtimeProbe: describeRuntimeProbe(),
        loadMs,
      };
    } catch (error) {
      const facts = await inspectTransformersEnv().catch(() => null);
      return {
        status: "fail",
        modelId: embedding.modelId,
        dimension: embedding.dimension,
        remoteHost: facts?.remoteHost,
        wasmPaths: facts?.wasmPaths,
        runtimeProbe: describeRuntimeProbe(),
        error: describeDiagnosticsError(error),
      };
    }
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
    // Hosts assembled before the paired API existed only implement the
    // singular replaceStore / replaceRunHistory — keep that path working.
    const scheduler = this.scheduler as {
      replacePersistenceStores?: (
        stateStore: StateStore,
        runHistoryStore: RunHistoryStore,
      ) => void;
      replaceStore?: (store: StateStore) => void;
      replaceRunHistory?: (runHistory: RunHistoryStore) => void;
    };
    if (scheduler.replacePersistenceStores) {
      scheduler.replacePersistenceStores(
        prepared.stateStore,
        prepared.runHistoryStore,
      );
    } else {
      scheduler.replaceStore?.(prepared.stateStore);
      scheduler.replaceRunHistory?.(prepared.runHistoryStore);
    }
    this.stateStore = prepared.stateStore;
    this.runHistoryStore = prepared.runHistoryStore;
    if (this.settings.schedule.enabled) {
      this.progress.setIdle(latestCompletedDate(prepared.stateStore));
    } else {
      this.progress.setDisabled();
    }
  }

  async reloadStateStoreForOutputPaths(): Promise<void> {
    this.libraryOutputRevision += 1;
    const mutationRevision = this.beginLibraryMutation();
    this.cancelPersonalLibraryOperations("output paths changed");
    await this.enqueueLibraryMutation(async () => {
      this.installOutputStores(await this.prepareOutputStores(this.settings));
      await this.reloadPersonalLibraryCatalog(mutationRevision);
      await this.reloadPersonalLibraryProfileDocuments(mutationRevision);
      if (this.settings.schedule.enabled) {
        this.progress.setIdle(latestCompletedDate(this.stateStore));
      } else {
        this.progress.setDisabled();
      }
    });
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
    const raw: unknown = await this.loadData();
    const loaded = settingsAndStateFromPersistedData(raw);
    this.legacyRunState = loaded.runState;
    this.settings = loaded.settings;
    const persisted = raw && typeof raw === "object"
      ? raw as Record<string, unknown>
      : {};
    this.libraryProposalAcceptanceRaw = persisted.libraryProposalAcceptances;
    const receipts = decodeProposalAcceptanceReceipts(persisted.libraryProposalAcceptances);
    this.libraryProposalAcceptances = receipts ?? [];
    this.libraryProposalAcceptanceLoadError = receipts === null
      ? "Library review state could not be loaded. Restore the saved review state and reload before accepting directions."
      : null;
    if (this.libraryProposalAcceptanceLoadError) loaded.warnings.push(this.libraryProposalAcceptanceLoadError);
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

  private libraryFingerprints(connection: PersistedLibraryConnection): {
    scopeFingerprint: string;
    identificationFingerprint: string;
  } {
    return {
      scopeFingerprint: createPersonalLibraryScopeFingerprint({
        rootIdentity: connection.rootIdentity,
        eligibleExtensions: connection.eligibleExtensions,
      }),
      identificationFingerprint: createPersonalLibraryIdentificationFingerprint(
        connection.eligibleExtensions,
      ),
    };
  }

  private buildPersonalLibraryCatalogStore(): PersonalLibraryCatalogStore {
    if (!this.host?.storage.writeTextAtomic) {
      throw new Error("Personal library catalog requires atomic storage writes");
    }
    return new PersonalLibraryCatalogStore(
      this.host.storage,
      this.settings.output,
      { onWarning: (message, error) => this.logger.warn(message, error) },
    );
  }

  private buildPersonalLibraryProfileStores(connection: PersistedLibraryConnection): {
    proposal: PersonalLibraryDirectionProposalStore;
    suggestions: IncrementalSuggestionsStore;
  } {
    if (!this.host?.storage.writeTextAtomic) {
      throw new Error("Personal library review requires atomic storage writes");
    }
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    const options = { onWarning: (message: string, error?: unknown) => this.logger.warn(message, error) };
    return {
      proposal: new PersonalLibraryDirectionProposalStore(
        this.host.storage, this.settings.output, scopeFingerprint, identificationFingerprint, options,
      ),
      suggestions: new IncrementalSuggestionsStore(
        this.host.storage, this.settings.output, scopeFingerprint, identificationFingerprint, options,
      ),
    };
  }

  private capturePersonalLibraryReviewGuard(): {
    connection: PersistedLibraryConnection;
    connectionRevision: number;
    outputRevision: number;
  } {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    return {
      connection,
      connectionRevision: this.libraryConnectionRevision,
      outputRevision: this.libraryOutputRevision,
    };
  }

  private assertPersonalLibraryReviewGuard(guard: {
    connection: PersistedLibraryConnection;
    connectionRevision: number;
    outputRevision: number;
  }): void {
    this.assertLibraryConnectionCurrent(guard.connection, guard.connectionRevision);
    if (this.libraryOutputRevision !== guard.outputRevision) {
      throw new Error("Output paths changed during personal library review");
    }
  }

  /**
   * Loads the proposal beside the catalog it was generated against. The
   * confirmed-profile arm went with the interest profile document
   * (ADR 0012 / ADR 0014); only the proposal is durable review state now.
   */
  private async loadPersonalLibraryProposalStateDirect(
    guard: { connection: PersistedLibraryConnection; connectionRevision: number; outputRevision: number },
    stores: { proposal: PersonalLibraryDirectionProposalStore },
  ): Promise<{ catalog: PersonalLibraryCatalog; proposal: PersonalLibraryDirectionProposal }> {
    this.assertPersonalLibraryReviewGuard(guard);
    const catalog = this.libraryCatalog;
    if (!catalog) throw new Error("Scan and load the personal library catalog first");
    const loaded = await this.loadPersonalLibraryProposalDocumentDirect(guard, stores);
    if (loaded.status === "rejected" || !loaded.value) {
      throw new Error("Personal library direction proposal is unavailable");
    }
    this.assertPersonalLibraryReviewGuard(guard);
    return { catalog: structuredClone(catalog), proposal: structuredClone(loaded.value) };
  }

  private async loadPersonalLibraryProposalDocumentDirect(
    guard: { connection: PersistedLibraryConnection; connectionRevision: number; outputRevision: number },
    stores: { proposal: PersonalLibraryDirectionProposalStore },
  ): Promise<PromiseSettledResult<PersonalLibraryDirectionProposal | null>> {
    const [proposal] = await Promise.allSettled([stores.proposal.load()]);
    this.assertPersonalLibraryReviewGuard(guard);
    return proposal;
  }

  private async mutatePersonalLibraryProposal(
    mutation: (
      proposal: PersonalLibraryDirectionProposal,
      catalog: PersonalLibraryCatalog,
    ) => PersonalLibraryDirectionProposal,
  ): Promise<PersonalLibraryProfileSnapshot> {
    const guard = this.capturePersonalLibraryReviewGuard();
    return this.enqueueLibraryMutation(async () => {
      this.assertPersonalLibraryReviewGuard(guard);
      const stores = this.buildPersonalLibraryProfileStores(guard.connection);
      const current = await this.loadPersonalLibraryProposalStateDirect(guard, stores);
      try {
        const saved = await stores.proposal.replace(
          mutation(current.proposal, current.catalog),
          current.proposal.revision,
        );
        this.assertPersonalLibraryReviewGuard(guard);
        this.libraryProposal = saved;
        this.libraryProposalLoadError = null;
        return this.getPersonalLibraryProfileSnapshot();
      } catch (error) {
        if (this.isReviewPersistenceConflict(error)) {
          await this.loadPersonalLibraryProposalDocumentDirect(guard, stores);
        }
        throw error;
      }
    });
  }

  /**
   * Identity of the evidence one generation runs on, for the guard that
   * refuses to finish if that evidence changed underneath it.
   *
   * Two things make this more than a call to the catalog fingerprint. An empty
   * catalog selection is a real state — a library whose files carry no arXiv
   * identity has one — and the proposal fingerprint refuses to describe an
   * empty selection, which turned every such generation into a TypeError
   * before it began. And the input is no longer the catalog alone: papers only
   * the index can name are part of it, so the guard has to cover them or it
   * would sleep through exactly the evidence this kind of library runs on.
   */
  private selectedCatalogFingerprint(catalog: PersonalLibraryCatalog): string {
    const papers = selectPersonalLibraryDirectionPapers(catalog);
    const catalogPart = papers.length === 0
      ? "none"
      : createPersonalLibraryCatalogInputFingerprint({
        scopeFingerprint: catalog.scopeFingerprint,
        identificationFingerprint: catalog.identificationFingerprint,
        papers,
      });
    const indexedPart = sha256Hex(this.libraryIndexedPapers
      .map(({ paperKey, title }) => `${paperKey} ${title}`)
      .join(""));
    return `sha256:${sha256Hex(JSON.stringify({
      version: 2,
      scopeFingerprint: catalog.scopeFingerprint,
      identificationFingerprint: catalog.identificationFingerprint,
      catalogPart,
      indexedPart,
    }))}`;
  }

  private libraryExistingTopics() {
    return this.settings.arxiv.topics.map(({ id, name, directions }) => ({
      id, name, directions: directions.map(({ id, text }) => ({ id, text })),
    }));
  }

  private assertPersonalLibraryGenerationCurrent(input: {
    connection: PersistedLibraryConnection;
    connectionRevision: number;
    outputRevision: number;
    authorizationFingerprint?: string;
    catalog: PersonalLibraryCatalog;
    selectedInputFingerprint: string;
    expectedProposalRevision: number | null;
    existingTopicsFingerprint: string;
  }): void {
    this.assertLibraryConnectionCurrent(input.connection, input.connectionRevision);
    if (JSON.stringify(this.libraryExistingTopics()) !== input.existingTopicsFingerprint) {
      throw new CodedError("conflict", "Research topics changed during direction generation. Generate a fresh proposal.");
    }
    if (this.libraryOutputRevision !== input.outputRevision) {
      throw new Error("Output paths changed during personal library direction generation");
    }
    if (this.getLibraryConnectionStatus().kind !== "authorized"
      || this.libraryConnection?.authorization?.fingerprint !== input.authorizationFingerprint) {
      throw new Error("Personal library model authorization changed during generation");
    }
    const currentCatalog = this.libraryCatalog;
    if (!currentCatalog
      || currentCatalog.scopeFingerprint !== input.catalog.scopeFingerprint
      || currentCatalog.identificationFingerprint !== input.catalog.identificationFingerprint
      || this.selectedCatalogFingerprint(currentCatalog) !== input.selectedInputFingerprint) {
      throw new Error("Selected personal library catalog evidence changed during generation");
    }
    if ((this.libraryProposal?.revision ?? null) !== input.expectedProposalRevision) {
      throw new Error("Personal library direction proposal changed during generation");
    }
  }

  private buildIncrementalSuggestionsStore(connection: PersistedLibraryConnection): IncrementalSuggestionsStore {
    if (!this.host?.storage.writeTextAtomic) {
      throw new Error("Incremental suggestions require atomic storage writes");
    }
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    return new IncrementalSuggestionsStore(
      this.host.storage,
      this.settings.output,
      scopeFingerprint,
      identificationFingerprint,
      { onWarning: (message, error) => this.logger.warn(message, error) },
    );
  }

  /**
   * Load the ready papers and apply the corpus-centered chunk transform —
   * the same transform the placement pass applies internally. Core exports
   * the transform (`centerCorpusChunks`) so the recluster pass cannot drift
   * from the clustering implementation.
   */
  private async loadCenteredClusteringInput(
    knowledgeBase: FullTextKnowledgeBaseFileStore,
    signal?: AbortSignal,
  ): Promise<ClusteringInputPaper[]> {
    const papers = await loadClusteringInput(knowledgeBase, signal);
    return centerCorpusChunks(papers);
  }

  private assertPersonalLibraryDocumentLoadCurrent(
    connection: PersistedLibraryConnection,
    connectionRevision: number,
    outputRevision: number,
  ): void {
    this.assertLibraryConnectionCurrent(connection, connectionRevision);
    if (this.libraryOutputRevision !== outputRevision) {
      throw new Error("Output paths changed while loading personal library review state");
    }
  }

  private resetPersonalLibraryProfileState(): void {
    this.libraryCatalog = null;
    this.libraryCatalogLoadError = null;
    this.libraryProposal = null;
    this.librarySuggestions = null;
    this.libraryProposalLoadError = null;
    this.librarySuggestionsLoadError = null;
  }

  private safeProfileLoadError(
    kind: "catalog" | "proposal" | "profile" | "suggestions",
    error: unknown,
  ): PersonalLibraryReviewLoadError {
    const code = typeof error === "object" && error !== null && "code" in error
      && typeof (error as { code?: unknown }).code === "string"
      ? (error as { code: string }).code
      : "load-failed";
    const label = kind === "catalog"
      ? "catalog"
      : kind === "proposal" ? "direction proposal"
        : kind === "suggestions" ? "incremental suggestions"
        : "confirmed profile";
    return { kind, code, message: `Personal library ${label} could not be loaded (${code}).` };
  }

  private isReviewPersistenceConflict(error: unknown): boolean {
    if (typeof error !== "object" || error === null) return false;
    const code = (error as { code?: unknown }).code;
    return code === "stale" || code === "partial-confirmation-conflict";
  }

  private effectiveLlmEndpoint(baseUrl: string): string {
    try {
      return buildChatCompletionsUrl(baseUrl.trim());
    } catch {
      return baseUrl.trim();
    }
  }

  private assertLibraryConnectionCurrent(
    connection: PersistedLibraryConnection,
    revision: number,
  ): void {
    if (
      this.libraryConnection !== connection
      || this.libraryConnectionRevision !== revision
    ) {
      throw new Error("Library connection changed during personal library operation");
    }
  }

  private cancelPersonalLibraryScans(reason: string): void {
    this.cancelPersonalLibraryOperationKinds(reason, ["personal-library-scan"]);
  }

  private cancelPersonalLibraryDirectionGeneration(reason: string): void {
    this.cancelPersonalLibraryOperationKinds(reason, ["personal-library-direction-generation"]);
  }


  private cancelPersonalLibraryOperations(reason: string): void {
    this.cancelPersonalLibraryOperationKinds(reason, [
      "personal-library-scan",
      "personal-library-direction-generation",
      "personal-library-fulltext-index",
    ]);
  }

  private cancelPersonalLibraryOperationKinds(
    reason: string,
    kinds: Array<
      | "personal-library-scan"
      | "personal-library-direction-generation"
      | "personal-library-fulltext-index"
    >,
  ): void {
    const registry = this.operations as (OperationRegistry & {
      snapshot?: OperationRegistry["snapshot"];
      cancel?: OperationRegistry["cancel"];
    }) | undefined;
    if (!registry || !registry.snapshot || !registry.cancel) return;
    for (const operation of registry.snapshot()) {
      if (kinds.includes(operation.kind as typeof kinds[number])) {
        registry.cancel(operation.id, reason);
      }
    }
  }

  private beginLibraryMutation(): number {
    // `?? 0` because the plugin instance is also built from the prototype in
    // tests, where class field initializers never run.
    this.libraryMutationRevision = (this.libraryMutationRevision ?? 0) + 1;
    return this.libraryMutationRevision;
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

  private async persistSettings(
    settings?: PluginSettings,
    receipts?: ProposalAcceptanceReceipt[],
  ): Promise<void> {
    const reviewState = receipts ?? (this.libraryProposalAcceptanceLoadError
      ? this.libraryProposalAcceptanceRaw : this.libraryProposalAcceptances ?? []);
    const data: PersistedData = {
      settings: settings ?? this.settings,
      ...(this.libraryProposalAcceptanceLoadError || (Array.isArray(reviewState) && reviewState.length > 0)
        ? { libraryProposalAcceptances: reviewState } : {}),
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
        this.settings.embedding?.apiKey ?? "",
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
    if (
      result.kind === "delivered" ||
      result.kind === "delivered_unrecorded"
    ) {
      return "Test email delivered" +
        (result.kind === "delivered_unrecorded"
          ? `; delivery record unavailable: ${result.reason}`
          : "");
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
    return "Verification email sent. Open the link, then paste the code from that page into Verification code.";
  }

  private buildPipeline(): ArxivPipeline {
    const settings = structuredClone(this.settings);
    const { llm, fetcher, paperFetcher, writer } = this.buildSharedDeps(settings);
    const checkpointStoreOptions = {
      onWarning: (message: string, error?: unknown) =>
        this.logger.warn(message, error),
    };
    const pipeline = new ArxivPipeline({
      fetcher,
      markupParser: this.host.markupParser,
      paperFetcher,
      writer,
      paperIndex: this.buildPaperIndex(settings.output),
      checkpointStores: {
        filter: new DailyFilterCheckpointStore(
          this.host.storage,
          settings.output,
          checkpointStoreOptions,
        ),
        summary: new DailySummaryCheckpointStore(
          this.host.storage,
          settings.output,
          checkpointStoreOptions,
        ),
      },
      llm,
      logger: this.logger,
      arxiv: settings.arxiv,
      advanced: settings.advanced,
      output: settings.output,
      llmSettings: settings.llm,
      detailSelection: settings.detailSelection,
      progress: this.progress,
    });
    return pipeline;
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

  buildArxivFetcher(settings: PluginSettings = this.settings): ArxivFetcher {
    return new ArxivFetcher({
      category: settings.arxiv.category,
      categories: arxivCategories(settings.arxiv),
      http: this.host.http,
      markupParser: this.host.markupParser,
      logger: this.logger,
      requestDelayMs: settings.advanced.requestDelayMs,
      metadataCache: new AtomMetadataCache({
        rootDir: this.pluginCacheDir(),
        expiryDays: settings.advanced.cacheExpiryDays,
        storage: this.host.storage,
      }),
    });
  }

  private buildSharedDeps(settings: PluginSettings = this.settings) {
    const llm = new LlmClient(settings.llm, this.logger, this.host.http);
    const fetcher = this.buildArxivFetcher(settings);
    const cache = new HtmlCache({
      rootDir: this.pluginCacheDir(),
      expiryDays: settings.advanced.cacheExpiryDays,
      storage: this.host.storage,
    });
    const paperFetcher = new PaperContentFetcher(fetcher, cache, this.logger, this.host.markupParser, {
      storage: this.host.storage,
      cacheDir: `${this.pluginDir()}/.cache/source`,
      expiryDays: settings.advanced.cacheExpiryDays,
    });
    const writer = new MarkdownWriter({
      storage: this.host.storage,
      logger: this.logger,
      arxiv: settings.arxiv,
      output: settings.output,
    });
    return { llm, fetcher, paperFetcher, writer };
  }

  buildPaperIndex(output: PluginSettings["output"] = this.settings.output): PaperIndexStore {
    return new PaperIndexStore(
      this.host.storage,
      output,
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

// ---------------------------------------------------------------------------
// Incremental direction update helpers (plugin-internal).
// ---------------------------------------------------------------------------

function createFullTextGenerationWriterToken(): string {
  return `writer-${crypto.randomUUID().replaceAll("-", "")}`;
}

function desktopVaultRoot(adapter: unknown): string | undefined {
  if (!adapter || typeof adapter !== "object" || !("getBasePath" in adapter)) return undefined;
  const getBasePath = (adapter as { getBasePath?: () => unknown }).getBasePath;
  if (typeof getBasePath !== "function") return undefined;
  const root = getBasePath.call(adapter);
  return typeof root === "string" && root ? root : undefined;
}
