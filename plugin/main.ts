import { Notice, Plugin, loadPdfJs } from "obsidian";
import type {
  LibraryInventory,
  PersonalLibraryCatalog,
  PersonalLibraryDirectionProposal,
  PersonalLibraryInterestEligibility,
  PersonalLibraryInterestProfile,
  PersonalizedDiscoveryInput,
  PersonalNoveltyMatchInput,
  PersonalizedNoveltyRepresentativesInput,
  NoveltyRepresentativePaper,
  PersonalLibraryReviewedDirectionDraft,
  PersonalLibraryDirectionTextPatch,
  PipelineResult,
  PluginSettings,
  RunState,
  FullTextIndexRunSummary,
  KnowledgeBaseChunkHit,
  DirectionDiffSuggestion,
  IncrementalSuggestionsDocument,
  PersonalLibraryDirectionCandidate,
  PersonalLibraryRepresentativeEvidence,
  ClusteringInputPaper,
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
PersonalLibraryInterestProfileStore,
confirmPersonalLibraryDirectionWithStores,
buildChatCompletionsUrl,
createPersonalLibraryCatalogInputFingerprint,
disablePersonalLibraryConfirmedDirection,
enablePersonalLibraryConfirmedDirection,
evaluatePersonalLibraryInterestEligibility,
mergePersonalLibraryConfirmedDirections,
mergePersonalLibraryDirectionCandidates,
proposeClusteredPersonalLibraryDirections,
preparePersonalizedDiscoveryInput,
preparePersonalizedNoveltyRepresentatives,
preparePersonalNoveltyMatches,
PERSONAL_NOVELTY_MAX_AUTHORS,
PERSONAL_NOVELTY_MAX_CATEGORIES,
PERSONAL_NOVELTY_MAX_ABSTRACT_CODE_UNITS,
removePersonalLibraryConfirmedDirection,
removePersonalLibraryDirectionCandidate,
selectPersonalLibraryDirectionPapers,
updatePersonalLibraryConfirmedDirection,
updatePersonalLibraryDirectionCandidate,
reconcilePersonalLibraryCatalog,
RunCancellationService,
normalizeArxivId,
FullTextKnowledgeBaseFileStore,
indexPersonalLibraryFullText as indexFullTextKnowledgeBase,
searchFullTextKnowledgeBase as searchFullTextKnowledgeBaseCore,
IncrementalSuggestionsStore,
ReadingCandidatesStore,
applyAttachSuggestion,
applyMergeSuggestion,
applySplitSuggestion,
buildNewDirectionDraft,
createEmptyIncrementalSuggestionsDocument,
createPersonalLibraryCatalogInputManifestFingerprint,
createPersonalLibraryGenerationContractFingerprint,
createPersonalLibraryPaperEvidenceFingerprint,
createPersonalLibraryRepresentativeSetFingerprint,
centerCorpusChunks,
loadClusteringInput,
lockPersonalLibraryConfirmedDirection,
unlockPersonalLibraryConfirmedDirection,
PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS,
PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH,
PERSONAL_LIBRARY_MAX_REPRESENTATIVES,
PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
PDF_IDENTIFICATION_EVIDENCE_VERSION,
reclusterPool,
suggestDirectionDiff,
upsertReadingCandidate,
decideReadingCandidate,
removeReadingCandidate,
readingCandidateFromRowSnapshot,
type ReadingCandidateDecisionKind,
type ReadingCandidateRowSnapshot,
type ReadingCandidatesDocument,
suggestIncrementalPlacement,
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
  ObsidianPdfDocumentParser,
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
import { ReadingCandidatesModal } from "./src/library/reading-candidates-modal";
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
import {
  SettingsChangeService,
  type PreparedOutputStores,
} from "./src/settings/change-service";

interface PersistedData {
  settings: PluginSettings;
  runState?: RunState;
  libraryConnection?: PersistedLibraryConnection;
}

export interface PersonalLibraryReviewLoadError {
  kind: "catalog" | "proposal" | "profile" | "suggestions" | "reading-candidates";
  code: string;
  message: string;
}

export interface PersonalLibraryProfileSnapshot {
  catalog: PersonalLibraryCatalog | null;
  proposal: PersonalLibraryDirectionProposal | null;
  profile: PersonalLibraryInterestProfile | null;
  suggestions: IncrementalSuggestionsDocument | null;
  readingCandidates: ReadingCandidatesDocument | null;
  eligibility: PersonalLibraryInterestEligibility;
  authorization: LibraryConnectionStatus;
  catalogLoadError: PersonalLibraryReviewLoadError | null;
  proposalLoadError: PersonalLibraryReviewLoadError | null;
  profileLoadError: PersonalLibraryReviewLoadError | null;
  suggestionsLoadError: PersonalLibraryReviewLoadError | null;
  readingCandidatesLoadError: PersonalLibraryReviewLoadError | null;
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
interface PersonalizedDailyDiscoverySnapshot {
  personalizedDiscovery: PersonalizedDiscoveryInput;
  personalizedNoveltyRepresentatives?: PersonalizedNoveltyRepresentativesInput;
  personalizedNoveltyMatches?: PersonalNoveltyMatchInput;
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

export const IDENTIFICATION_HEAD_BYTES = 4 * 1024 * 1024;
const IDENTIFICATION_TAIL_BYTES = 1024 * 1024;

/**
 * Minimum buffer-pool size before the incremental direction update runs the
 * low-frequency recluster + LLM diff pass; below it only deterministic
 * placement attach suggestions are recorded.
 */
export const INCREMENTAL_BUFFER_TRIGGER = 3 as const;

/** Fixed reason attached to deterministic placement attach suggestions. */
const INCREMENTAL_ATTACH_REASON = "Newly indexed paper matches this confirmed direction." as const;

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
  private librarySource?: OpenedScopedLibrarySource;
  private libraryDirectoryPicker = new ObsidianLibraryDirectoryPicker();
  private openLibrarySource: (selectedRoot: string) => Promise<OpenedScopedLibrarySource>
    = openObsidianLibrarySource;
  private libraryInventoryController?: AbortController;
  private libraryCatalog: PersonalLibraryCatalog | null = null;
  private libraryCatalogLoadError: PersonalLibraryReviewLoadError | null = null;
  private libraryMutationQueue: Promise<void> = Promise.resolve();
  private librarySelectionRevision = 0;
  private libraryConnectionRevision = 0;
  private libraryOutputRevision = 0;
  private libraryProposal: PersonalLibraryDirectionProposal | null = null;
  private libraryProfile: PersonalLibraryInterestProfile | null = null;
  private librarySuggestions: IncrementalSuggestionsDocument | null = null;
  private libraryReadingCandidates: ReadingCandidatesDocument | null = null;
  private libraryProposalLoadError: PersonalLibraryReviewLoadError | null = null;
  private libraryProfileLoadError: PersonalLibraryReviewLoadError | null = null;
  private librarySuggestionsLoadError: PersonalLibraryReviewLoadError | null = null;
  private libraryReadingCandidatesLoadError: PersonalLibraryReviewLoadError | null = null;
  private personalizedDailyDiscoveryAvailable = true;
  private personalizedDailyDiscoveryRevision = 0;
  private personalizedDailyRunControllers = new Map<ArxivPipeline, AbortController>();

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
        const pipeline = this.buildPipeline();
        try {
          return await pipeline.runForDate(date, signal);
        } finally {
          this.releasePersonalizedDailyPipeline(pipeline);
        }
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
        if (!changedKeys.includes("llm.baseUrl")) return undefined;
        const previousEndpoint = this.effectiveLlmEndpoint(previous.llm.baseUrl);
        const nextEndpoint = this.effectiveLlmEndpoint(candidate.llm.baseUrl);
        if (previousEndpoint === nextEndpoint) return undefined;
        const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable(
          "model endpoint changed",
        );
        this.cancelPersonalLibraryDirectionGeneration("model endpoint changed");
        return () => {
          if (discoveryRevision !== undefined) {
            this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision);
          }
        };
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

  getLibraryAuthorizationDisclosure(): LibraryAuthorizationDisclosure | null {
    if (!this.libraryConnection) return null;
    return libraryAuthorizationDisclosure(
      this.libraryConnection,
      this.libraryAuthorizationScope(),
    );
  }

  /** Authorization scope: the LLM endpoint plus the embedding endpoint when remote embedding is enabled. */
  private libraryAuthorizationScope(): LibraryAuthorizationScope {
    return {
      llmBaseUrl: this.settings.llm.baseUrl,
      ...(this.settings.embedding.mode === "remote" && this.settings.embedding.baseUrl.trim()
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
    const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable("library folder changed");
    this.cancelPersonalLibraryOperations("library folder changed");
    return await this.enqueueLibraryMutation(async () => {
      if (revision !== this.librarySelectionRevision) return "cancelled" as const;
      const previousConnection = this.libraryConnection;
      const previousSource = this.librarySource;
      const previousCatalog = this.libraryCatalog;
      const previousProposal = this.libraryProposal;
      const previousProfile = this.libraryProfile;
      const previousProposalError = this.libraryProposalLoadError;
      const previousProfileError = this.libraryProfileLoadError;
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
        this.libraryProfile = previousProfile;
        this.libraryProposalLoadError = previousProposalError;
        this.libraryProfileLoadError = previousProfileError;
        this.libraryConnectionRevision = previousConnectionRevision;
        this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision);
        this.refreshSensitiveValues();
        throw error;
      }
      await this.reloadPersonalLibraryCatalog(discoveryRevision).catch((error) => {
        this.libraryCatalog = null;
        this.logger.error?.("personal library catalog load failed after folder selection", error);
      });
      await this.reloadPersonalLibraryProfileDocuments(discoveryRevision);
      this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision);
      return "selected" as const;
    });
  }

  async authorizeLibraryProcessing(expectedFingerprint?: string): Promise<void> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    const connectionRevision = this.libraryConnectionRevision;
    const outputRevision = this.libraryOutputRevision;
    const discoveryRevision = this.personalizedDailyDiscoveryRevision;
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
        || this.personalizedDailyDiscoveryRevision !== discoveryRevision
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
      const identity = this.capturePersonalizedDiscoveryIdentity(authorized);
      if (identity) {
        this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
      }
    });
  }

  async revokeLibraryProcessing(): Promise<void> {
    const previous = this.libraryConnection;
    if (!previous) return;
    const revoked = revokeLibraryConnection(previous);
    const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable("library processing authorization revoked");
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
          this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision);
        }
        throw error;
      }
    });
  }

  async setLlmBaseUrl(next: string): Promise<void> {
    const normalized = next.trim();
    const requestedEndpointChanged = this.effectiveLlmEndpoint(this.settings.llm.baseUrl)
      !== this.effectiveLlmEndpoint(normalized);
    const discoveryRevision = requestedEndpointChanged
      ? this.markPersonalizedDailyDiscoveryUnavailable("model endpoint changed")
      : undefined;
    if (requestedEndpointChanged) {
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
        if (discoveryRevision !== undefined) {
          this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision);
        }
        this.refreshSensitiveValues();
        throw error;
      }
      if (discoveryRevision !== undefined) {
        this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision);
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

  openReadingCandidatesReview(): void {
    if (!this.libraryConnection) {
      new Notice("arXiv Daily: Choose a personal library first", 10_000);
      return;
    }
    new ReadingCandidatesModal(this.app, {
      getCandidates: () => this.getReadingCandidates(),
      decide: (paperKey, kind) => this.decideReadingCandidateForReview(paperKey, kind),
      remove: (paperKey) => this.removeReadingCandidateForReview(paperKey),
      onError: (action, error) => {
        this.logger.warn(`reading candidates: ${action} failed`, error);
        new Notice(`arXiv Daily: ${action} failed. Try again.`, 10_000);
      },
    }).open();
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
        return this.reloadPersonalLibraryProfileDocuments();
      },
      generate: () => this.generatePersonalLibraryDirections(),
      updateProposal: (input) => this.updatePersonalLibraryProposalCandidate(input),
      mergeProposals: (input) => this.mergePersonalLibraryProposalCandidates(input),
      discardProposal: (candidateId) => this.removePersonalLibraryProposalCandidate(candidateId),
      confirmProposal: (input) => this.confirmPersonalLibraryProposalCandidate(input),
      updateConfirmed: (input) => this.updatePersonalLibraryConfirmedDirection(input),
      mergeConfirmed: (input) => this.mergePersonalLibraryConfirmedDirections(input),
      enable: (directionId) => this.enablePersonalLibraryConfirmedDirection(directionId),
      disable: (directionId) => this.disablePersonalLibraryConfirmedDirection(directionId),
      remove: (input) => this.removePersonalLibraryConfirmedDirection(input),
      applySuggestion: (key) => this.applyIncrementalSuggestion(key),
      dismissSuggestion: (key) => this.dismissIncrementalSuggestion(key),
      lock: (directionId) => this.lockPersonalLibraryConfirmedDirection(directionId),
      unlock: (directionId) => this.unlockPersonalLibraryConfirmedDirection(directionId),
    };
  }

  async reloadPersonalLibraryCatalog(
    transitionRevision?: number,
  ): Promise<PersonalLibraryCatalog | null> {
    const ownsTransition = transitionRevision === undefined;
    const discoveryRevision = transitionRevision
      ?? this.markPersonalizedDailyDiscoveryUnavailable("personal library catalog reloaded");
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
    const identity = this.capturePersonalizedDiscoveryIdentity(connection);
    if (ownsTransition && identity) {
      this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
    }
    return structuredClone(catalog);
  }

  getPersonalLibraryProfileSnapshot(): PersonalLibraryProfileSnapshot {
    return structuredClone({
      catalog: this.libraryCatalog,
      proposal: this.libraryProposal,
      profile: this.libraryProfile,
      suggestions: this.librarySuggestions,
      readingCandidates: this.libraryReadingCandidates,
      eligibility: evaluatePersonalLibraryInterestEligibility(this.libraryProfile, this.libraryCatalog),
      authorization: this.getLibraryConnectionStatus(),
      catalogLoadError: this.libraryCatalogLoadError,
      proposalLoadError: this.libraryProposalLoadError,
      profileLoadError: this.libraryProfileLoadError,
      suggestionsLoadError: this.librarySuggestionsLoadError,
      readingCandidatesLoadError: this.libraryReadingCandidatesLoadError,
    });
  }

  getPersonalLibraryDirectionProposal(): PersonalLibraryDirectionProposal | null {
    return this.libraryProposal ? structuredClone(this.libraryProposal) : null;
  }

  getPersonalLibraryInterestProfile(): PersonalLibraryInterestProfile | null {
    return this.libraryProfile ? structuredClone(this.libraryProfile) : null;
  }

  async reloadPersonalLibraryProfileDocuments(
    transitionRevision?: number,
  ): Promise<PersonalLibraryProfileSnapshot> {
    const ownsTransition = transitionRevision === undefined;
    const discoveryRevision = transitionRevision
      ?? this.markPersonalizedDailyDiscoveryUnavailable("personal library profile reloaded");
    const connection = this.libraryConnection;
    if (!connection) {
      this.libraryProposal = null;
      this.libraryProfile = null;
      this.librarySuggestions = null;
      this.libraryReadingCandidates = null;
      this.libraryProposalLoadError = null;
      this.libraryProfileLoadError = null;
      this.librarySuggestionsLoadError = null;
      this.libraryReadingCandidatesLoadError = null;
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
      stores.profile.load().then((profile) => {
        this.assertPersonalLibraryDocumentLoadCurrent(connection, connectionRevision, outputRevision);
        this.libraryProfile = profile;
        this.libraryProfileLoadError = null;
      }).catch((error) => {
        this.assertPersonalLibraryDocumentLoadCurrent(connection, connectionRevision, outputRevision);
        this.libraryProfile = null;
        this.libraryProfileLoadError = this.safeProfileLoadError("profile", error);
        this.logger?.error("personal library interest profile load failed", error);
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
      this.buildReadingCandidatesStore(connection).load().then((candidates) => {
        this.assertPersonalLibraryDocumentLoadCurrent(connection, connectionRevision, outputRevision);
        this.libraryReadingCandidates = candidates;
        this.libraryReadingCandidatesLoadError = null;
      }).catch((error) => {
        this.assertPersonalLibraryDocumentLoadCurrent(connection, connectionRevision, outputRevision);
        this.libraryReadingCandidates = null;
        this.libraryReadingCandidatesLoadError = this.safeProfileLoadError("reading-candidates", error);
        this.logger?.error("reading candidates load failed", error);
      }),
    ]);
    const identity = this.capturePersonalizedDiscoveryIdentity(connection);
    if (ownsTransition && identity) {
      this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
    }
    return this.getPersonalLibraryProfileSnapshot();
  }

  async generatePersonalLibraryDirections(): Promise<PersonalLibraryDirectionProposal> {
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
        catalog, selectedInputFingerprint, expectedProposalRevision,
      });
      const proposal = await proposeClusteredPersonalLibraryDirections({
        catalog: structuredClone(catalog),
        knowledgeBase: this.buildFullTextKnowledgeBaseStore(connection),
        llm: new LlmClient(llmSettings, this.logger, this.host.http),
        signal: operation.signal,
        createId: () => crypto.randomUUID(),
      });
      operation.signal.throwIfAborted();
      this.assertPersonalLibraryGenerationCurrent({
        connection, connectionRevision, outputRevision, authorizationFingerprint,
        catalog, selectedInputFingerprint, expectedProposalRevision,
      });
      return await this.enqueueLibraryMutation(async () => {
        operation.signal.throwIfAborted();
        this.assertPersonalLibraryGenerationCurrent({
          connection, connectionRevision, outputRevision, authorizationFingerprint,
          catalog, selectedInputFingerprint, expectedProposalRevision,
        });
        const saved = await store.replace(proposal, expectedProposalRevision);
        this.libraryProposal = saved;
        this.libraryProposalLoadError = null;
        return structuredClone(saved);
      });
    } catch (error) {
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
    return this.mutatePersonalLibraryProposal((proposal, catalog) =>
      updatePersonalLibraryDirectionCandidate({
        proposal,
        candidateId: input.candidateId,
        patch: input.patch,
        ...(input.representativePaperKeys === undefined ? {} : {
          representativePaperKeys: input.representativePaperKeys,
          catalog,
        }),
      }));
  }

  async mergePersonalLibraryProposalCandidates(input: {
    sourceCandidateIds: string[];
    draft: PersonalLibraryReviewedDirectionDraft;
    candidateId?: string;
  }): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProposal((proposal, catalog) =>
      mergePersonalLibraryDirectionCandidates({
        proposal,
        sourceCandidateIds: input.sourceCandidateIds,
        candidateId: input.candidateId ?? crypto.randomUUID(),
        draft: input.draft,
        catalog,
      }));
  }

  async removePersonalLibraryProposalCandidate(candidateId: string): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProposal((proposal) =>
      removePersonalLibraryDirectionCandidate({ proposal, candidateId }));
  }

  async confirmPersonalLibraryProposalCandidate(input: {
    candidateId: string;
    draft: PersonalLibraryReviewedDirectionDraft;
    status: "active" | "disabled";
    directionId?: string;
    now?: Date;
  }): Promise<PersonalLibraryProfileSnapshot> {
    const guard = this.capturePersonalLibraryReviewGuard();
    const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable(
      "confirmed personal library direction changed",
    );
    return this.enqueueLibraryMutation(async () => {
      this.assertPersonalLibraryReviewGuard(guard);
      const stores = this.buildPersonalLibraryProfileStores(guard.connection);
      const current = await this.loadPersonalLibraryReviewStateDirect(guard, stores, true);
      try {
        this.assertPersonalLibraryReviewGuard(guard);
        const saved = await confirmPersonalLibraryDirectionWithStores({
          proposalStore: stores.proposal,
          profileStore: stores.profile,
          proposal: current.proposal!,
          profile: current.profile,
          catalog: current.catalog,
          candidateId: input.candidateId,
          directionId: input.directionId ?? crypto.randomUUID(),
          status: input.status,
          draft: input.draft,
          now: input.now ?? new Date(),
          expectedProposalRevision: current.proposal!.revision,
          expectedProfileRevision: current.profile.revision,
        });
        this.assertPersonalLibraryReviewGuard(guard);
        this.libraryProposal = saved.proposal;
        this.libraryProfile = saved.profile;
        this.libraryProposalLoadError = null;
        this.libraryProfileLoadError = null;
        const identity = this.capturePersonalizedDiscoveryIdentity(guard.connection);
        if (identity) this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
        return this.getPersonalLibraryProfileSnapshot();
      } catch (error) {
        if (this.isReviewPersistenceConflict(error)) {
          await this.loadPersonalLibraryReviewDocumentsDirect(guard, stores, discoveryRevision);
        }
        throw error;
      }
    });
  }

  async updatePersonalLibraryConfirmedDirection(input: {
    directionId: string;
    patch: PersonalLibraryDirectionTextPatch;
    representativePaperKeys?: string[];
    now?: Date;
  }): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProfile((profile, catalog) =>
      updatePersonalLibraryConfirmedDirection({
        profile,
        directionId: input.directionId,
        patch: input.patch,
        ...(input.representativePaperKeys === undefined ? {} : {
          representativePaperKeys: input.representativePaperKeys,
          catalog,
        }),
        now: input.now ?? new Date(),
      }));
  }

  async disablePersonalLibraryConfirmedDirection(directionId: string, now = new Date()): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProfile((profile) =>
      disablePersonalLibraryConfirmedDirection({ profile, directionId, now }));
  }

  async enablePersonalLibraryConfirmedDirection(directionId: string, now = new Date()): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProfile((profile, catalog) =>
      enablePersonalLibraryConfirmedDirection({ profile, directionId, catalog, now }));
  }

  async mergePersonalLibraryConfirmedDirections(input: {
    sourceDirectionIds: string[];
    draft: PersonalLibraryReviewedDirectionDraft;
    status: "active" | "disabled";
    directionId?: string;
    now?: Date;
  }): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProfile((profile, catalog) =>
      mergePersonalLibraryConfirmedDirections({
        profile,
        sourceDirectionIds: input.sourceDirectionIds,
        directionId: input.directionId ?? crypto.randomUUID(),
        status: input.status,
        draft: input.draft,
        catalog,
        now: input.now ?? new Date(),
      }));
  }

  async removePersonalLibraryConfirmedDirection(input: {
    directionId: string;
    mode: "restrict" | "cascade";
  }): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProfile((profile) =>
      removePersonalLibraryConfirmedDirection({
        profile,
        directionId: input.directionId,
        mode: input.mode,
      }));
  }

  getIncrementalSuggestions(): IncrementalSuggestionsDocument | null {
    return this.librarySuggestions ? structuredClone(this.librarySuggestions) : null;
  }

  /**
   * Incremental direction update: place every indexed paper not yet covered by
   * a confirmed direction (deterministic attach vs. buffer), then — only when
   * the buffer pool reached INCREMENTAL_BUFFER_TRIGGER papers — recluster the
   * pool and ask the LLM for direction-diff suggestions. The merged suggestion
   * set replaces the persisted suggestions document (CAS); pending suggestions
   * from an earlier run are superseded by the newest evidence.
   *
   * Consent gate (ADR 0007): placement is local embedding similarity and runs
   * without model-processing consent; the recluster + LLM diff stage requires
   * it. Without consent the LLM stage is skipped and the document records
   * `pendingAuthorization` for the buffered papers.
   */
  async runIncrementalDirectionUpdate(): Promise<{
    suggestions: number;
    attachments: number;
    buffered: number;
    pendingAuthorizationBuffered: number;
    /** Un-reviewed suggestions from the previous run that this run replaced. */
    superseded: number;
  }> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    const profile = this.libraryProfile;
    if (!profile) throw new Error("Load the confirmed personal library profile first");
    const { scopeFingerprint } = this.libraryFingerprints(connection);
    // The operation reuses the direction-generation kind: the closed core
    // OperationKind union has no incremental kind, and the two flows share the
    // authorization gate and cancellation scope (revocation cancels both).
    if (this.operations.find("personal-library-direction-generation", scopeFingerprint)) {
      throw new Error("Incremental direction update is already active");
    }
    const connectionRevision = this.libraryConnectionRevision;
    const outputRevision = this.libraryOutputRevision;
    const authorizationFingerprint = connection.authorization?.fingerprint;
    const llmSettings = structuredClone(this.settings.llm);
    const knowledgeBase = this.buildFullTextKnowledgeBaseStore(connection);
    const operation = this.operations.begin(
      "personal-library-direction-generation",
      "Incremental direction update",
      scopeFingerprint,
    );
    try {
      operation.signal.throwIfAborted();
      this.assertIncrementalUpdateCurrent({
        connection, connectionRevision, outputRevision, authorizationFingerprint, profile,
      }, false);
      const placement = await suggestIncrementalPlacement({
        profile: { directions: profile.directions },
        knowledgeBase,
        signal: operation.signal,
      });
      operation.signal.throwIfAborted();
      const attachSuggestions: DirectionDiffSuggestion[] = [];
      const bufferPaperKeys: string[] = [];
      for (const paperKey of Object.keys(placement.placements).sort()) {
        const decision = placement.placements[paperKey]!;
        if (decision.kind === "attach") {
          attachSuggestions.push({
            kind: "attach",
            directionId: decision.directionId,
            paperKeys: [paperKey],
            reason: INCREMENTAL_ATTACH_REASON,
          });
        } else {
          bufferPaperKeys.push(paperKey);
        }
      }
      // Reclustering + LLM diff are the low-frequency path. The placement pass
      // already loaded and centered its own corpus copy, but the core API does
      // not expose the centered papers (nor the centering transform), so the
      // pool pass reloads and re-centers the corpus — a second load is fine
      // here. LLM suggestions can only reference cluster papers, so the union
      // with the per-paper placement attaches is always conflict-free.
      // The LLM stage requires model-processing consent (ADR 0007); without
      // it the buffered papers are recorded as pending authorization.
      let llmSuggestions: DirectionDiffSuggestion[] = [];
      let pendingAuthorizationBuffered = 0;
      const llmAuthorized = this.getLibraryConnectionStatus().kind === "authorized"
        && this.libraryConnection?.authorization?.fingerprint === authorizationFingerprint;
      if (bufferPaperKeys.length >= INCREMENTAL_BUFFER_TRIGGER) {
        if (!llmAuthorized) {
          pendingAuthorizationBuffered = bufferPaperKeys.length;
        } else {
          const centered = await this.loadCenteredClusteringInput(knowledgeBase, operation.signal);
          operation.signal.throwIfAborted();
          const pooled = reclusterPool(centered, {
            poolPaperKeys: bufferPaperKeys,
            directions: profile.directions,
          });
          llmSuggestions = await suggestDirectionDiff({
            directions: profile.directions,
            clusters: pooled.candidates,
            llm: new LlmClient(llmSettings, this.logger, this.host.http),
            signal: operation.signal,
          });
        }
      }
      operation.signal.throwIfAborted();
      const suggestions = mergeIncrementalSuggestions(attachSuggestions, llmSuggestions);
      const nextDocument = emptyIncrementalSuggestionsDocument(connection, suggestions, new Date());
      if (pendingAuthorizationBuffered > 0) {
        nextDocument.pendingAuthorization = {
          bufferedPaperCount: pendingAuthorizationBuffered,
          updatedAt: nextDocument.updatedAt,
        };
      }
      return await this.enqueueLibraryMutation(async () => {
        operation.signal.throwIfAborted();
        this.assertIncrementalUpdateCurrent({
          connection, connectionRevision, outputRevision, authorizationFingerprint, profile,
        }, llmAuthorized);
        const store = this.buildIncrementalSuggestionsStore(connection);
        const current = await store.load();
        // Whole-document replace (ADR 0007): un-reviewed suggestions from the
        // previous run are superseded by the newest evidence. Count them so
        // the notice can make the replacement visible.
        const superseded = current.suggestions.length > 0
          && JSON.stringify(current.suggestions) !== JSON.stringify(nextDocument.suggestions)
          ? current.suggestions.length
          : 0;
        const saved = await store.replace(nextDocument, current.revision);
        this.assertIncrementalUpdateCurrent({
          connection, connectionRevision, outputRevision, authorizationFingerprint, profile,
        }, llmAuthorized);
        this.librarySuggestions = saved;
        this.librarySuggestionsLoadError = null;
        return {
          suggestions: saved.suggestions.length,
          attachments: saved.suggestions.filter((entry) => entry.kind === "attach").length,
          buffered: bufferPaperKeys.length,
          pendingAuthorizationBuffered,
          superseded,
        };
      });
    } catch (error) {
      if (this.isReviewPersistenceConflict(error)) await this.reloadPersonalLibraryProfileDocuments();
      throw error;
    } finally {
      operation.finish();
    }
  }

  /**
   * Apply one persisted incremental suggestion. attach/split/merge mutate the
   * confirmed profile through the same review persistence path; "new" becomes
   * a review candidate in the proposal store (the existing confirmation flow
   * then decides the direction — the caller surfaces the proposal). The
   * suggestion is removed from the suggestions document (CAS) after the
   * mutation commits. Local operation: no model authorization required.
   */
  async applyIncrementalSuggestion(key: string): Promise<PersonalLibraryProfileSnapshot> {
    const guard = this.capturePersonalLibraryReviewGuard();
    const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable(
      "incremental direction suggestion applied",
    );
    return this.enqueueLibraryMutation(async () => {
      this.assertPersonalLibraryReviewGuard(guard);
      const stores = this.buildPersonalLibraryProfileStores(guard.connection);
      const current = await this.loadPersonalLibraryReviewStateDirect(guard, stores, false);
      const suggestions = await stores.suggestions.load();
      const suggestion = findIncrementalSuggestionByKey(suggestions.suggestions, key);
      if (!suggestion) {
        throw new Error("Incremental suggestion no longer exists. Refresh and try again.");
      }
      const now = new Date();
      let profile = current.profile;
      let proposal = current.proposal;
      const expectedProposalRevision = proposal?.revision ?? null;
      if (suggestion.kind === "new") {
        proposal = this.attachNewSuggestionToProposal(suggestion, proposal, current.catalog, now);
      } else {
        profile = applySuggestionToProfile(profile, suggestion, now);
      }
      try {
        if (suggestion.kind === "new") {
          const savedProposal = await stores.proposal.replace(proposal!, expectedProposalRevision);
          this.assertPersonalLibraryReviewGuard(guard);
          this.libraryProposal = savedProposal;
          this.libraryProposalLoadError = null;
        } else {
          const savedProfile = await stores.profile.replace(profile, profile.revision);
          this.assertPersonalLibraryReviewGuard(guard);
          this.libraryProfile = savedProfile;
          this.libraryProfileLoadError = null;
        }
        const remaining = suggestions.suggestions.filter((entry) => !sameIncrementalSuggestion(entry, suggestion));
        const nextDocument = suggestionsDocumentWithout(
          suggestions,
          remaining,
          now,
        );
        const savedSuggestions = await stores.suggestions.replace(nextDocument, suggestions.revision);
        this.assertPersonalLibraryReviewGuard(guard);
        this.librarySuggestions = savedSuggestions;
        this.librarySuggestionsLoadError = null;
      } catch (error) {
        if (this.isReviewPersistenceConflict(error)) {
          await this.loadPersonalLibraryReviewDocumentsDirect(guard, stores, discoveryRevision);
        }
        throw error;
      }
      const identity = this.capturePersonalizedDiscoveryIdentity(guard.connection);
      if (identity) this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
      return this.getPersonalLibraryProfileSnapshot();
    });
  }

  /** Remove one persisted incremental suggestion without applying it. */
  async dismissIncrementalSuggestion(key: string): Promise<PersonalLibraryProfileSnapshot> {
    const guard = this.capturePersonalLibraryReviewGuard();
    return this.enqueueLibraryMutation(async () => {
      this.assertPersonalLibraryReviewGuard(guard);
      const stores = this.buildPersonalLibraryProfileStores(guard.connection);
      const suggestions = await stores.suggestions.load();
      const suggestion = findIncrementalSuggestionByKey(suggestions.suggestions, key);
      if (!suggestion) {
        throw new Error("Incremental suggestion no longer exists. Refresh and try again.");
      }
      const remaining = suggestions.suggestions.filter((entry) => !sameIncrementalSuggestion(entry, suggestion));
      const saved = await stores.suggestions.replace(
        suggestionsDocumentWithout(suggestions, remaining, new Date()),
        suggestions.revision,
      );
      this.assertPersonalLibraryReviewGuard(guard);
      this.librarySuggestions = saved;
      this.librarySuggestionsLoadError = null;
      return this.getPersonalLibraryProfileSnapshot();
    });
  }

  async lockPersonalLibraryConfirmedDirection(directionId: string, now = new Date()): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProfile((profile) =>
      lockPersonalLibraryConfirmedDirection({ profile, directionId, now }));
  }

  async unlockPersonalLibraryConfirmedDirection(directionId: string, now = new Date()): Promise<PersonalLibraryProfileSnapshot> {
    return this.mutatePersonalLibraryProfile((profile) =>
      unlockPersonalLibraryConfirmedDirection({ profile, directionId, now }));
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
      const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable(
        "personal library catalog evidence changed",
      );
      const saved = await this.enqueueLibraryMutation(async () => {
        operation.signal.throwIfAborted();
        this.assertLibraryConnectionCurrent(connection, revision);
        const saved = await store.replace(reconciled.catalog);
        // Atomic promotion cannot be interrupted. Once it succeeds, the scan is
        // committed even if cancellation arrives during that final write.
        this.assertLibraryConnectionCurrent(connection, revision);
        this.libraryCatalog = saved;
        this.libraryCatalogLoadError = null;
        const identity = this.capturePersonalizedDiscoveryIdentity(connection);
        if (identity) this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
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
      const parser = new ObsidianPdfDocumentParser();
      this.assertRemoteEmbeddingReady();
      const embedding = this.buildEmbeddingModel();
      const store = this.buildFullTextKnowledgeBaseStore(connection);
      const summary = await indexFullTextKnowledgeBase({
        catalog,
        source,
        parser,
        embedding,
        store,
        logger: this.logger,
        onProgress: (detail) => {
          operation.signal.throwIfAborted();
          if (updateProgress) this.progress?.setTask("Indexing personal library full text", detail);
        },
        signal: operation.signal,
      });
      operation.signal.throwIfAborted();
      // ADR 0007: new or changed papers trigger the incremental direction
      // update automatically (placement always; LLM diff only with consent).
      // Failures never fail the index command. The completion notice is
      // deferred until the update finishes, so the progress view does not
      // read "done" while the update still holds the command open.
      await this.runIncrementalDirectionUpdateAfterIndex(summary, updateProgress);
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
      operation.finish();
    }
  }

  /**
   * Embed the query locally and return the most similar indexed papers with
   * their best matching passages. Joins catalog titles for display; the
   * similarity evidence (hit chunk text) is explainable end to end.
   */
  async searchPersonalLibraryFullText(
    queryText: string,
  ): Promise<Array<{
    paperKey: string;
    title: string;
    /** Relative library path for fallback-indexed files; arXiv papers leave it unset. */
    filePath?: string;
    score: number;
    hits: KnowledgeBaseChunkHit[];
  }>> {
    const connection = this.libraryConnection;
    if (!connection) throw new Error("Choose a personal library first");
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    const catalog = await this.buildPersonalLibraryCatalogStore().load(
      scopeFingerprint,
      identificationFingerprint,
    );
    this.assertRemoteEmbeddingReady();
    // Title fusion: catalog titles for arXiv papers, extracted first-page
    // titles from the knowledge-base manifest for fallback-indexed files
    // (see `title-similarity.ts`).
    const titles = new Map<string, string>();
    for (const [paperKey, paper] of Object.entries(catalog.papers)) {
      if (paper.title) titles.set(paperKey, paper.title);
    }
    const store = this.buildFullTextKnowledgeBaseStore(connection);
    const manifest = await store.loadManifest();
    const fallbackPaths = new Map<string, string>();
    for (const [paperKey, record] of Object.entries(manifest.papers)) {
      if (record.title && !titles.has(paperKey)) titles.set(paperKey, record.title);
      if (paperKey.startsWith("file:") && record.filePaths[0]) {
        fallbackPaths.set(paperKey, record.filePaths[0]);
      }
    }
    const matches = await searchFullTextKnowledgeBaseCore({
      store,
      embedding: this.buildEmbeddingModel(),
      queryText,
      titles,
      logger: this.logger,
    });
    return matches.map((match) => ({
      paperKey: match.paperKey,
      title: titles.get(match.paperKey) ?? match.paperKey,
      filePath: fallbackPaths.get(match.paperKey),
      score: match.score,
      hits: match.hits,
    }));
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

  /**
   * ADR 0007 auto-trigger after an index run: only new or changed papers
   * (indexed > 0) trigger the incremental update. Runs the same update as
   * the manual command and surfaces the result with a Notice; any failure
   * (already active, no profile, transient LLM error) is logged and
   * swallowed so the index command stays successful. While it runs, the
   * progress text moves to the direction update so the post-index wait
   * (clustering, possibly an LLM diff) reads as work, not a stall.
   */
  private async runIncrementalDirectionUpdateAfterIndex(
    summary: FullTextIndexRunSummary,
    updateProgress: boolean,
  ): Promise<void> {
    if (summary.indexed <= 0) return;
    if (updateProgress) {
      this.progress?.setTask("Updating paper directions", "Placing new papers");
    }
    try {
      const update = await this.runIncrementalDirectionUpdate();
      const pending = update.pendingAuthorizationBuffered > 0
        ? `, ${update.pendingAuthorizationBuffered} buffered awaiting model authorization`
        : "";
      const superseded = update.superseded > 0
        ? `, ${update.superseded} un-reviewed suggestion(s) superseded by new evidence`
        : "";
      new Notice(
        `arXiv Daily: incremental update — ${update.suggestions} suggestions `
        + `(${update.attachments} attaches), ${update.buffered} buffered${pending}${superseded}`,
        10_000,
      );
    } catch (error) {
      this.logger.warn("incremental direction update after indexing failed", error);
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
    const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable("output paths changed");
    this.cancelPersonalLibraryOperations("output paths changed");
    await this.enqueueLibraryMutation(async () => {
      this.installOutputStores(await this.prepareOutputStores(this.settings));
      await this.reloadPersonalLibraryCatalog(discoveryRevision);
      await this.reloadPersonalLibraryProfileDocuments(discoveryRevision);
      const identity = this.capturePersonalizedDiscoveryIdentity();
      if (identity) this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
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
    profile: PersonalLibraryInterestProfileStore;
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
      profile: new PersonalLibraryInterestProfileStore(
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

  private async loadPersonalLibraryReviewStateDirect(
    guard: { connection: PersistedLibraryConnection; connectionRevision: number; outputRevision: number },
    stores: { proposal: PersonalLibraryDirectionProposalStore; profile: PersonalLibraryInterestProfileStore },
    requireProposal: boolean,
  ): Promise<{
    catalog: PersonalLibraryCatalog;
    proposal: PersonalLibraryDirectionProposal | null;
    profile: PersonalLibraryInterestProfile;
  }> {
    this.assertPersonalLibraryReviewGuard(guard);
    const catalog = this.libraryCatalog;
    if (!catalog) throw new Error("Scan and load the personal library catalog first");
    const loaded = await this.loadPersonalLibraryReviewDocumentsDirect(guard, stores);
    if (loaded.profile.status === "rejected") {
      throw new Error("Personal library confirmed profile is unavailable");
    }
    if (requireProposal && loaded.proposal.status === "rejected") {
      throw new Error("Personal library direction proposal is unavailable");
    }
    this.assertPersonalLibraryReviewGuard(guard);
    return {
      catalog: structuredClone(catalog),
      proposal: loaded.proposal.status === "fulfilled" && loaded.proposal.value
        ? structuredClone(loaded.proposal.value)
        : null,
      profile: structuredClone(loaded.profile.value),
    };
  }

  private async loadPersonalLibraryReviewDocumentsDirect(
    guard: { connection: PersistedLibraryConnection; connectionRevision: number; outputRevision: number },
    stores: { proposal: PersonalLibraryDirectionProposalStore; profile: PersonalLibraryInterestProfileStore },
    discoveryRevision?: number,
  ): Promise<{
    proposal: PromiseSettledResult<PersonalLibraryDirectionProposal | null>;
    profile: PromiseSettledResult<PersonalLibraryInterestProfile>;
  }> {
    const [proposal, profile] = await Promise.allSettled([stores.proposal.load(), stores.profile.load()]);
    this.assertPersonalLibraryReviewGuard(guard);
    if (discoveryRevision !== undefined) {
      if (proposal.status === "fulfilled") {
        this.libraryProposal = proposal.value;
        this.libraryProposalLoadError = null;
      } else {
        this.libraryProposal = null;
        this.libraryProposalLoadError = this.safeProfileLoadError("proposal", proposal.reason);
        this.logger?.error("personal library direction proposal load failed", proposal.reason);
      }
      if (profile.status === "fulfilled") {
        this.libraryProfile = profile.value;
        this.libraryProfileLoadError = null;
      } else {
        this.libraryProfile = null;
        this.libraryProfileLoadError = this.safeProfileLoadError("profile", profile.reason);
        this.logger?.error("personal library interest profile load failed", profile.reason);
      }
      const identity = this.capturePersonalizedDiscoveryIdentity(guard.connection);
      if (identity) this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
    }
    return { proposal, profile };
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
      const current = await this.loadPersonalLibraryReviewStateDirect(guard, stores, true);
      try {
        const saved = await stores.proposal.replace(
          mutation(current.proposal!, current.catalog),
          current.proposal!.revision,
        );
        this.assertPersonalLibraryReviewGuard(guard);
        this.libraryProposal = saved;
        this.libraryProposalLoadError = null;
        return this.getPersonalLibraryProfileSnapshot();
      } catch (error) {
        if (this.isReviewPersistenceConflict(error)) {
          await this.loadPersonalLibraryReviewDocumentsDirect(guard, stores);
        }
        throw error;
      }
    });
  }

  private async mutatePersonalLibraryProfile(
    mutation: (
      profile: PersonalLibraryInterestProfile,
      catalog: PersonalLibraryCatalog,
    ) => PersonalLibraryInterestProfile,
  ): Promise<PersonalLibraryProfileSnapshot> {
    const guard = this.capturePersonalLibraryReviewGuard();
    const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable(
      "confirmed personal library direction changed",
    );
    return this.enqueueLibraryMutation(async () => {
      this.assertPersonalLibraryReviewGuard(guard);
      const stores = this.buildPersonalLibraryProfileStores(guard.connection);
      const current = await this.loadPersonalLibraryReviewStateDirect(guard, stores, false);
      try {
        const saved = await stores.profile.replace(
          mutation(current.profile, current.catalog),
          current.profile.revision,
        );
        this.assertPersonalLibraryReviewGuard(guard);
        this.libraryProfile = saved;
        this.libraryProfileLoadError = null;
        const identity = this.capturePersonalizedDiscoveryIdentity(guard.connection);
        if (identity) this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
        return this.getPersonalLibraryProfileSnapshot();
      } catch (error) {
        if (this.isReviewPersistenceConflict(error)) {
          await this.loadPersonalLibraryReviewDocumentsDirect(guard, stores, discoveryRevision);
        }
        throw error;
      }
    });
  }

  private selectedCatalogFingerprint(catalog: PersonalLibraryCatalog): string {
    return createPersonalLibraryCatalogInputFingerprint({
      scopeFingerprint: catalog.scopeFingerprint,
      identificationFingerprint: catalog.identificationFingerprint,
      papers: selectPersonalLibraryDirectionPapers(catalog),
    });
  }

  private assertPersonalLibraryGenerationCurrent(input: {
    connection: PersistedLibraryConnection;
    connectionRevision: number;
    outputRevision: number;
    authorizationFingerprint?: string;
    catalog: PersonalLibraryCatalog;
    selectedInputFingerprint: string;
    expectedProposalRevision: number | null;
  }): void {
    this.assertLibraryConnectionCurrent(input.connection, input.connectionRevision);
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

  /**
   * Guards the incremental update run: connection/output/authorization are the
   * same gates as direction generation; the confirmed profile is captured by
   * reference so any reload supersedes the run.
   */
  private assertIncrementalUpdateCurrent(
    input: {
      connection: PersistedLibraryConnection;
      connectionRevision: number;
      outputRevision: number;
      authorizationFingerprint?: string;
      profile: PersonalLibraryInterestProfile;
    },
    requireAuthorization: boolean,
  ): void {
    this.assertLibraryConnectionCurrent(input.connection, input.connectionRevision);
    if (this.libraryOutputRevision !== input.outputRevision) {
      throw new Error("Output paths changed during incremental direction update");
    }
    if (requireAuthorization
      && (this.getLibraryConnectionStatus().kind !== "authorized"
        || this.libraryConnection?.authorization?.fingerprint !== input.authorizationFingerprint)) {
      throw new Error("Personal library model authorization changed during incremental update");
    }
    if (this.libraryProfile !== input.profile) {
      throw new Error("Personal library confirmed profile changed during incremental update");
    }
  }

  private buildReadingCandidatesStore(connection: PersistedLibraryConnection): ReadingCandidatesStore {
    if (!this.host?.storage.writeTextAtomic) {
      throw new Error("Reading candidates require atomic storage writes");
    }
    const { scopeFingerprint, identificationFingerprint } = this.libraryFingerprints(connection);
    return new ReadingCandidatesStore(
      this.host.storage,
      this.settings.output,
      scopeFingerprint,
      identificationFingerprint,
      { onWarning: (message, error) => this.logger.warn(message, error) },
    );
  }

  /**
   * Save a dashboard row as a reading candidate. The row must carry discovery
   * provenance (the source that brought the paper in); rows without it cannot
   * be saved.
   */
  async saveReadingCandidateForRow(
    snapshot: ReadingCandidateRowSnapshot,
  ): Promise<"saved" | "missing-source" | "unavailable"> {
    const connection = this.libraryConnection;
    if (!connection) return "unavailable";
    const nowIso = new Date().toISOString();
    const record = readingCandidateFromRowSnapshot(snapshot, nowIso);
    if (!record) return "missing-source";
    return this.enqueueLibraryMutation(async () => {
      const store = this.buildReadingCandidatesStore(connection);
      const current = await store.load();
      const next = upsertReadingCandidate(current, record, nowIso);
      if (!next.changed) return "saved" as const;
      const saved = await store.replace(next.document, current.revision);
      this.libraryReadingCandidates = saved;
      this.libraryReadingCandidatesLoadError = null;
      if (next.evicted.length > 0) {
        this.logger.info(`reading candidates: evicted ${next.evicted.length} oldest undecided`);
      }
      return "saved" as const;
    });
  }

  async decideReadingCandidateForReview(
    paperKey: string,
    kind: ReadingCandidateDecisionKind,
    note?: string,
  ): Promise<boolean> {
    const connection = this.libraryConnection;
    if (!connection) return false;
    return this.enqueueLibraryMutation(async () => {
      const store = this.buildReadingCandidatesStore(connection);
      const current = await store.load();
      const next = decideReadingCandidate(current, paperKey, kind, new Date().toISOString(), note);
      if (!next.changed) return true;
      const saved = await store.replace(next.document, current.revision);
      this.libraryReadingCandidates = saved;
      this.libraryReadingCandidatesLoadError = null;
      return true;
    });
  }

  async removeReadingCandidateForReview(paperKey: string): Promise<boolean> {
    const connection = this.libraryConnection;
    if (!connection) return false;
    return this.enqueueLibraryMutation(async () => {
      const store = this.buildReadingCandidatesStore(connection);
      const current = await store.load();
      const next = removeReadingCandidate(current, paperKey, new Date().toISOString());
      if (!next.changed) return true;
      const saved = await store.replace(next.document, current.revision);
      this.libraryReadingCandidates = saved;
      this.libraryReadingCandidatesLoadError = null;
      return true;
    });
  }

  getReadingCandidates(): ReadingCandidatesDocument | null {
    return this.libraryReadingCandidates ? structuredClone(this.libraryReadingCandidates) : null;
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

  /**
   * Convert a "new" suggestion into a review candidate and append it to the
   * proposal document (creating one when none exists yet). The candidate
   * carries the suggestion's cluster members so the review shows them; its
   * representatives are resolved against the current catalog evidence so the
   * existing confirmation flow can verify and confirm it.
   */
  private attachNewSuggestionToProposal(
    suggestion: Extract<DirectionDiffSuggestion, { kind: "new" }>,
    proposal: PersonalLibraryDirectionProposal | null,
    catalog: PersonalLibraryCatalog,
    now: Date,
  ): PersonalLibraryDirectionProposal {
    const draft = buildNewDirectionDraft(suggestion);
    const candidateId = crypto.randomUUID();
    const representativePaperKeys = draft.representativePaperKeys
      .slice(0, PERSONAL_LIBRARY_MAX_REPRESENTATIVES);
    const representatives = representativePaperKeys.map((paperKey) => {
      const paper = catalog.papers[paperKey];
      if (!paper) {
        throw new Error(`Suggestion paper is missing from the current catalog: ${paperKey}`);
      }
      return { paperKey, evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(paper) };
    });
    // Discovery cues: representative paper titles (deduped, code-unit sorted
    // — the strict candidate decoder requires >=1 cue, strictly ordered and
    // unique). Titles are natural cues for what the new direction is about;
    // the reason falls back when no titles are available.
    const titleCues = [...new Set(
      representativePaperKeys
        .map((paperKey) => catalog.papers[paperKey]?.title?.trim() ?? "")
        .filter((title) => title.length > 0)
        .map((title) => title.slice(0, PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH)),
    )].sort(codeUnitCompare);
    const candidate: PersonalLibraryDirectionCandidate = {
      id: candidateId,
      name: draft.name,
      description: draft.description,
      discoveryCues: titleCues.length > 0
        ? titleCues
        : [draft.description.slice(0, PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH)],
      representatives,
      representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
      lineage: { candidateIds: [candidateId] },
      clusterMembers: draft.clusterMembers.slice(0, PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS),
    };
    const inputPapers = mergeRepresentativeEvidence(
      proposal?.catalogInputPapers ?? [],
      representatives,
    );
    const scopeFingerprint = catalog.scopeFingerprint;
    const identificationFingerprint = catalog.identificationFingerprint;
    return {
      schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
      revision: proposal?.revision ?? 0,
      proposalId: proposal?.proposalId ?? crypto.randomUUID(),
      scopeFingerprint,
      identificationFingerprint,
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint,
        identificationFingerprint,
        catalogInputPapers: inputPapers,
      }),
      catalogInputPapers: inputPapers,
      generationContractFingerprint: proposal?.generationContractFingerprint
        ?? createPersonalLibraryGenerationContractFingerprint("incremental-new-suggestion"),
      generatedAt: proposal?.generatedAt ?? now.toISOString(),
      candidates: [...(proposal?.candidates ?? []), candidate].sort(byOpaqueId),
    };
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
    this.libraryProfile = null;
    this.librarySuggestions = null;
    this.libraryReadingCandidates = null;
    this.libraryProposalLoadError = null;
    this.libraryProfileLoadError = null;
    this.librarySuggestionsLoadError = null;
    this.libraryReadingCandidatesLoadError = null;
  }

  private safeProfileLoadError(
    kind: "catalog" | "proposal" | "profile" | "suggestions" | "reading-candidates",
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
        : kind === "reading-candidates" ? "reading candidates" : "confirmed profile";
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

  private markPersonalizedDailyDiscoveryUnavailable(reason: string): number {
    this.personalizedDailyDiscoveryAvailable = false;
    const revision = (this.personalizedDailyDiscoveryRevision ?? 0) + 1;
    this.personalizedDailyDiscoveryRevision = revision;
    for (const controller of this.personalizedDailyRunControllers?.values() ?? []) {
      if (!controller.signal.aborted) controller.abort(reason);
    }
    return revision;
  }

  private restorePersonalizedDailyDiscoveryAvailability(
    revision: number,
    expected?: {
      connection: PersistedLibraryConnection;
      connectionRevision: number;
      outputRevision: number;
      endpoint: string;
    },
  ): void {
    if (this.personalizedDailyDiscoveryRevision !== revision) return;
    if (expected) {
      if (this.libraryConnection !== expected.connection
        || this.libraryConnectionRevision !== expected.connectionRevision
        || this.libraryOutputRevision !== expected.outputRevision
        || this.effectiveLlmEndpoint(this.settings.llm.baseUrl) !== expected.endpoint
        || this.getLibraryConnectionStatus().kind !== "authorized") {
        return;
      }
    }
    this.personalizedDailyDiscoveryAvailable = true;
  }

  private capturePersonalizedDiscoveryIdentity(connection = this.libraryConnection): {
    connection: PersistedLibraryConnection;
    connectionRevision: number;
    outputRevision: number;
    endpoint: string;
  } | undefined {
    if (!connection) return undefined;
    return {
      connection,
      connectionRevision: this.libraryConnectionRevision,
      outputRevision: this.libraryOutputRevision,
      endpoint: this.effectiveLlmEndpoint(this.settings.llm.baseUrl),
    };
  }

  private releasePersonalizedDailyPipeline(pipeline: ArxivPipeline): void {
    this.personalizedDailyRunControllers?.delete(pipeline);
  }

  private buildPersonalizedDailyDiscoverySnapshot(): PersonalizedDailyDiscoverySnapshot | undefined {
    if (this.personalizedDailyDiscoveryAvailable === false
      || this.libraryCatalogLoadError
      || this.libraryProfileLoadError
      || this.getLibraryConnectionStatus().kind !== "authorized") {
      return undefined;
    }
    const catalog = this.libraryCatalog;
    const profile = this.libraryProfile;
    if (!catalog || !profile) return undefined;
    const eligibility = evaluatePersonalLibraryInterestEligibility(profile, catalog);
    if (eligibility.documentDiagnostics.length > 0
      || eligibility.eligibleDirections.length === 0) {
      return undefined;
    }
    // Discovery is prepared first, exactly as before: a discovery-preparation
    // failure still degrades to manual-only with the original warning.
    let personalizedDiscovery: PersonalizedDiscoveryInput;
    try {
      personalizedDiscovery = preparePersonalizedDiscoveryInput({
        directions: eligibility.eligibleDirections.map((direction) => ({
          id: direction.id,
          name: direction.name,
          description: direction.description,
          discoveryCues: [...direction.discoveryCues],
          representatives: direction.representatives.map((representative) => {
            const paper = catalog.papers[representative.paperKey];
            if (!paper) throw new Error("eligible representative is missing from catalog");
            return {
              paperKey: representative.paperKey,
              title: paper.title,
              evidenceDepth: paper.evidenceDepth,
            };
          }),
        })),
      });
    } catch (error) {
      this.logger.warn("personal library discovery snapshot invalid; using manual-only", error);
      return undefined;
    }
    // Novelty representative evidence and the direction→representative mapping
    // derive from the same eligibility join, but novelty preparation is
    // decoupled from the discovery gate: a failure degrades to a
    // discovery-only snapshot with a fixed-text warning and never drops
    // personalized discovery. Evidence is metadata and abstracts within the
    // authorized processing depth: no paths, PDF bytes, fingerprints,
    // authorization state, credentials, or unrelated catalog records ever
    // enter the snapshot.
    try {
      const representativeByKey = new Map<string, NoveltyRepresentativePaper>();
      for (const direction of eligibility.eligibleDirections) {
        for (const representative of direction.representatives) {
          if (representativeByKey.has(representative.paperKey)) continue;
          const paper = catalog.papers[representative.paperKey];
          if (!paper) throw new Error("eligible representative is missing from catalog");
          // Display-metadata/evidence caps at the join boundary: titles and
          // abstracts are trimmed, and authors/categories/abstract are sliced
          // to the strict DTO bounds. Basis membership is the set of
          // representative paperKeys carried by matches.directionRepresentatives
          // — these caps bound only the rendered metadata/abstract evidence
          // and never truncate the comparison basis.
          representativeByKey.set(representative.paperKey, {
            paperKey: representative.paperKey,
            title: paper.title.trim(),
            authors: paper.authors.slice(0, PERSONAL_NOVELTY_MAX_AUTHORS),
            abstract: paper.abstract.trim().slice(0, PERSONAL_NOVELTY_MAX_ABSTRACT_CODE_UNITS),
            published: paper.published,
            categories: paper.categories.slice(0, PERSONAL_NOVELTY_MAX_CATEGORIES),
          });
        }
      }
      const representatives = [...representativeByKey.values()]
        .sort((left, right) =>
          left.paperKey < right.paperKey ? -1 : left.paperKey > right.paperKey ? 1 : 0,
        );
      const directionRepresentatives = eligibility.eligibleDirections
        .map((direction) => ({
          directionId: direction.id,
          representativePaperKeys: direction.representatives
            .map(({ paperKey }) => paperKey)
            .sort((left, right) => left < right ? -1 : left > right ? 1 : 0),
        }))
        .sort((left, right) =>
          left.directionId < right.directionId ? -1 : left.directionId > right.directionId ? 1 : 0,
        );
      return {
        personalizedDiscovery,
        personalizedNoveltyRepresentatives: preparePersonalizedNoveltyRepresentatives({
          representatives,
        }),
        personalizedNoveltyMatches: preparePersonalNoveltyMatches({
          paperMatches: [],
          directionRepresentatives,
        }),
      };
    } catch (error) {
      this.logger.warn("personal novelty snapshot invalid; continuing with discovery-only", error);
      return { personalizedDiscovery };
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

  private enqueueLibraryMutation<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.libraryMutationQueue ?? Promise.resolve();
    const result = previous.then(operation, operation);
    this.libraryMutationQueue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  private async persistSettings(settings?: PluginSettings): Promise<void> {
    const data: PersistedData = {
      settings: settings ?? this.settings,
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
    const personalizedSnapshot = this.buildPersonalizedDailyDiscoverySnapshot();
    const personalizedDiscovery = personalizedSnapshot?.personalizedDiscovery;
    const personalizedController = personalizedDiscovery ? new AbortController() : undefined;
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
      personalizedDiscovery,
      personalizedDiscoverySignal: personalizedController?.signal,
      personalizedNoveltyRepresentatives: personalizedSnapshot?.personalizedNoveltyRepresentatives,
      personalizedNoveltyMatches: personalizedSnapshot?.personalizedNoveltyMatches,
      progress: this.progress,
    });
    if (personalizedController) {
      (this.personalizedDailyRunControllers ??= new Map()).set(pipeline, personalizedController);
    }
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

/**
 * Content key of one persisted suggestion, used by the review UI and the
 * plugin methods to address a single suggestion. The key is never parsed —
 * lookups recompute it for every suggestion in the loaded document — so the
 * colons inside paper keys (e.g. "arxiv:2608.00001") are harmless.
 */
function incrementalSuggestionKey(suggestion: DirectionDiffSuggestion): string {
  switch (suggestion.kind) {
    case "attach":
      return `attach:${suggestion.directionId}:${suggestion.paperKeys[0]}`;
    case "new":
      return `new::${suggestion.paperKeys[0]}`;
    case "split":
      return `split:${suggestion.directionId}:${suggestion.paperKeys[0]}`;
    case "merge":
      return `merge:${suggestion.directionIds[0]}:${suggestion.directionIds[1]}`;
  }
}

function findIncrementalSuggestionByKey(
  suggestions: readonly DirectionDiffSuggestion[],
  key: string,
): DirectionDiffSuggestion | undefined {
  return suggestions.find((suggestion) => incrementalSuggestionKey(suggestion) === key);
}

function sameIncrementalSuggestion(
  left: DirectionDiffSuggestion,
  right: DirectionDiffSuggestion,
): boolean {
  return incrementalSuggestionKey(left) === incrementalSuggestionKey(right);
}

function suggestionsDocumentWithout(
  document: IncrementalSuggestionsDocument,
  suggestions: DirectionDiffSuggestion[],
  now: Date,
): IncrementalSuggestionsDocument {
  const empty = createEmptyIncrementalSuggestionsDocument(
    document.scopeFingerprint,
    document.identificationFingerprint,
    now,
  );
  return {
    ...empty,
    suggestions,
    // Applying or dismissing a suggestion does not change the buffer pool,
    // so a pending-authorization note from the last run stays visible.
    ...(document.pendingAuthorization
      ? { pendingAuthorization: document.pendingAuthorization }
      : {}),
  };
}

function emptyIncrementalSuggestionsDocument(
  connection: PersistedLibraryConnection,
  suggestions: DirectionDiffSuggestion[],
  now: Date,
): IncrementalSuggestionsDocument {
  const { scopeFingerprint, identificationFingerprint } = libraryFingerprints(connection);
  const empty = createEmptyIncrementalSuggestionsDocument(
    scopeFingerprint,
    identificationFingerprint,
    now,
  );
  return { ...empty, suggestions };
}

/**
 * Canonical union of placement attaches (deterministic) and LLM diff
 * suggestions (cluster-bound): papers never overlap between the two sources,
 * so the merged list is conflict-free once sorted into the store's canonical
 * code-unit order.
 */
function mergeIncrementalSuggestions(
  attach: readonly DirectionDiffSuggestion[],
  llm: readonly DirectionDiffSuggestion[],
): DirectionDiffSuggestion[] {
  return [...attach, ...llm].sort(compareIncrementalSuggestions);
}

const SUGGESTION_KIND_ORDER: Readonly<Record<DirectionDiffSuggestion["kind"], number>> = {
  attach: 0,
  merge: 1,
  new: 2,
  split: 3,
};

function compareIncrementalSuggestions(left: DirectionDiffSuggestion, right: DirectionDiffSuggestion): number {
  const leftKey = incrementalSuggestionSortKey(left);
  const rightKey = incrementalSuggestionSortKey(right);
  for (let index = 0; index < leftKey.length; index += 1) {
    const diff = codeUnitCompare(leftKey[index]!, rightKey[index]!);
    if (diff !== 0) return diff;
  }
  return 0;
}

function incrementalSuggestionSortKey(suggestion: DirectionDiffSuggestion): string[] {
  switch (suggestion.kind) {
    case "attach":
      return [String(SUGGESTION_KIND_ORDER.attach), suggestion.directionId, suggestion.paperKeys[0] ?? ""];
    case "merge":
      return [String(SUGGESTION_KIND_ORDER.merge), suggestion.directionIds[0], suggestion.directionIds[1]];
    case "new":
      return [String(SUGGESTION_KIND_ORDER.new), suggestion.paperKeys[0] ?? "", ""];
    case "split":
      return [String(SUGGESTION_KIND_ORDER.split), suggestion.directionId, suggestion.paperKeys[0] ?? ""];
  }
}

function applySuggestionToProfile(
  profile: PersonalLibraryInterestProfile,
  suggestion: DirectionDiffSuggestion,
  now: Date,
): PersonalLibraryInterestProfile {
  switch (suggestion.kind) {
    case "attach":
      return applyAttachSuggestion({ profile, suggestion, now });
    case "split":
      return applySplitSuggestion({
        profile,
        suggestion,
        createId: () => crypto.randomUUID(),
        now,
      }).profile;
    case "merge":
      return applyMergeSuggestion({
        profile,
        suggestion,
        createId: () => crypto.randomUUID(),
        now,
      });
    case "new":
      throw new Error("new suggestions are converted to review candidates, not applied to the profile");
  }
}

/** Merge representative evidence by paperKey, keeping the first occurrence. */
function mergeRepresentativeEvidence(
  current: readonly PersonalLibraryRepresentativeEvidence[],
  additional: readonly PersonalLibraryRepresentativeEvidence[],
): PersonalLibraryRepresentativeEvidence[] {
  const byKey = new Map(current.map((entry) => [entry.paperKey, entry]));
  for (const entry of additional) {
    if (!byKey.has(entry.paperKey)) byKey.set(entry.paperKey, entry);
  }
  return [...byKey.values()].sort((left, right) => codeUnitCompare(left.paperKey, right.paperKey));
}

function byOpaqueId(left: { id: string }, right: { id: string }): number {
  return codeUnitCompare(left.id, right.id);
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function libraryFingerprints(connection: PersistedLibraryConnection): {
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
