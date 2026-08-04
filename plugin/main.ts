import { Notice, Plugin } from "obsidian";
import type {
  LibraryInventory,
  PersonalLibraryCatalog,
  PersonalLibraryDirectionProposal,
  PersonalLibraryInterestEligibility,
  PersonalLibraryInterestProfile,
  PersonalizedDiscoveryInput,
  PersonalLibraryReviewedDirectionDraft,
  PersonalLibraryDirectionTextPatch,
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
import {
  ArxivLibraryMetadataResolver,
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
  proposePersonalLibraryDirections,
  preparePersonalizedDiscoveryInput,
  removePersonalLibraryConfirmedDirection,
  removePersonalLibraryDirectionCandidate,
  selectPersonalLibraryDirectionPapers,
  updatePersonalLibraryConfirmedDirection,
  updatePersonalLibraryDirectionCandidate,
  reconcilePersonalLibraryCatalog,
  RunCancellationService,
  normalizeArxivId,
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

export interface PersonalLibraryReviewLoadError {
  kind: "catalog" | "proposal" | "profile";
  code: string;
  message: string;
}

export interface PersonalLibraryProfileSnapshot {
  catalog: PersonalLibraryCatalog | null;
  proposal: PersonalLibraryDirectionProposal | null;
  profile: PersonalLibraryInterestProfile | null;
  eligibility: PersonalLibraryInterestEligibility;
  authorization: LibraryConnectionStatus;
  catalogLoadError: PersonalLibraryReviewLoadError | null;
  proposalLoadError: PersonalLibraryReviewLoadError | null;
  profileLoadError: PersonalLibraryReviewLoadError | null;
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
  private libraryCatalog: PersonalLibraryCatalog | null = null;
  private libraryCatalogLoadError: PersonalLibraryReviewLoadError | null = null;
  private libraryMutationQueue: Promise<void> = Promise.resolve();
  private librarySelectionRevision = 0;
  private libraryConnectionRevision = 0;
  private libraryOutputRevision = 0;
  private libraryProposal: PersonalLibraryDirectionProposal | null = null;
  private libraryProfile: PersonalLibraryInterestProfile | null = null;
  private libraryProposalLoadError: PersonalLibraryReviewLoadError | null = null;
  private libraryProfileLoadError: PersonalLibraryReviewLoadError | null = null;
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
      persistSettings: () => this.enqueueLibraryMutation(() => this.persistSettings()),
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
    const disclosure = libraryAuthorizationDisclosure(connection, this.settings.llm.baseUrl);
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
      const authorized = authorizeLibraryConnection(connection, this.settings.llm.baseUrl);
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
      eligibility: evaluatePersonalLibraryInterestEligibility(this.libraryProfile, this.libraryCatalog),
      authorization: this.getLibraryConnectionStatus(),
      catalogLoadError: this.libraryCatalogLoadError,
      proposalLoadError: this.libraryProposalLoadError,
      profileLoadError: this.libraryProfileLoadError,
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
      this.libraryProposalLoadError = null;
      this.libraryProfileLoadError = null;
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
      const proposal = await proposePersonalLibraryDirections({
        catalog: structuredClone(catalog),
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
    if (updateProgress) this.progress?.setTask("Scanning personal library", "Inventorying eligible files");
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

  restartScheduler(): void {
    this.scheduler.stop();
    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  async reloadStateStoreForOutputPaths(): Promise<void> {
    this.libraryOutputRevision += 1;
    const discoveryRevision = this.markPersonalizedDailyDiscoveryUnavailable("output paths changed");
    this.cancelPersonalLibraryOperations("output paths changed");
    await this.enqueueLibraryMutation(async () => {
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
      await this.reloadPersonalLibraryCatalog(discoveryRevision);
      await this.reloadPersonalLibraryProfileDocuments(discoveryRevision);
      const identity = this.capturePersonalizedDiscoveryIdentity();
      if (identity) this.restorePersonalizedDailyDiscoveryAvailability(discoveryRevision, identity);
      if (this.settings.schedule.enabled) {
        this.progress.setIdle(latestCompletedDate(nextStore));
      } else {
        this.progress.setDisabled();
      }
    });
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
    this.libraryProposalLoadError = null;
    this.libraryProfileLoadError = null;
  }

  private safeProfileLoadError(
    kind: "catalog" | "proposal" | "profile",
    error: unknown,
  ): PersonalLibraryReviewLoadError {
    const code = typeof error === "object" && error !== null && "code" in error
      && typeof (error as { code?: unknown }).code === "string"
      ? (error as { code: string }).code
      : "load-failed";
    const label = kind === "catalog"
      ? "catalog"
      : kind === "proposal" ? "direction proposal" : "confirmed profile";
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

  private buildPersonalizedDailyDiscoverySnapshot(): PersonalizedDiscoveryInput | undefined {
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
    try {
      return preparePersonalizedDiscoveryInput({
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
    ]);
  }

  private cancelPersonalLibraryOperationKinds(
    reason: string,
    kinds: Array<"personal-library-scan" | "personal-library-direction-generation">,
  ): void {
    const registry = this.operations as OperationRegistry & {
      snapshot?: OperationRegistry["snapshot"];
      cancel?: OperationRegistry["cancel"];
    };
    if (!registry.snapshot || !registry.cancel) return;
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
    const settings = structuredClone(this.settings);
    const personalizedDiscovery = this.buildPersonalizedDailyDiscoverySnapshot();
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
