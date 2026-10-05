import type { HttpClient, StorageAdapter } from "../core/adapters";
import type { DocumentParser, DocumentParserSelector } from "../documents/parsed-document";
import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import type { Logger } from "../services/logger";
import type { OutputSettings } from "../settings/types";
import type { PersistedLibraryConnection } from "./library-connection";
import type { ScopedLibrarySource } from "./scoped-library-source";
import type { EmbeddingModel } from "./fulltext/ports";
import type { FullTextIndexProgressReporter, FullTextIndexRunSummary } from "./fulltext/index-orchestration";
import type { PersonalLibraryDirectionLlmPort } from "./personal-library-direction-proposer";
import type { PersonalLibraryCatalog } from "./personal-library-catalog";
import type { PersonalLibraryDirectionProposal } from "./personal-library-interest-profile";
import type { PersonalLibraryReviewedDirectionDraft, PersonalLibraryDirectionTextPatch } from "./personal-library-proposal-review";
import type { Topic } from "../settings/types";
import type { PdfTextExtractor } from "./fulltext/ports";
import { parsedDocumentToPdfExtractionResult } from "./fulltext/pdf-text-compat";
import { throwIfCancelled } from "../services/cancellation";
import { sha256Hex } from "../utils/digest";
import { createPersonalLibraryScopeFingerprint, createPersonalLibraryIdentificationFingerprint, PersonalLibraryCatalogStore } from "./personal-library-catalog";
import { reconcilePersonalLibraryCatalog } from "./personal-library-reconciliation";
import { ArxivLibraryMetadataResolver } from "./arxiv-library-metadata-resolver";
import { createPdfLibraryFileIdentifier } from "./pdf-library-file-identifier";
import { PersonalLibraryDirectionProposalStore, PersonalLibraryDirectionProposalStoreError } from "./personal-library-proposal-store";
import { FullTextKnowledgeBaseFileStore } from "./fulltext/knowledge-base-store";
import { FullTextGenerationIndexStore, type FullTextLegacyMigrationLease } from "./fulltext/generation-index-store";
import { indexPersonalLibraryFullText } from "./fulltext/index-orchestration";
import { preflightFullTextGenerationSynchronization, synchronizeFullTextGenerationIndex } from "./fulltext/generation-index-orchestration";
import { proposeClusteredPersonalLibraryDirections } from "./personal-library-direction-proposer";
import { updatePersonalLibraryDirectionCandidate, renamePersonalLibraryProposedTopic, movePersonalLibraryDirectionCandidate, removePersonalLibraryDirectionCandidate } from "./personal-library-proposal-review";
import { previewPersonalLibraryDirection, type LibraryDirectionPreview, type LibraryPreviewPaper } from "./personal-library-direction-preview";
import { acceptProposedTopics, type ProposalAcceptanceReceipt } from "../settings/accept-proposed-topics";

export interface LibraryTopicSettingsSnapshot {
  topics: Topic[];
  acceptances: ProposalAcceptanceReceipt[];
}
/** The host commits topics and receipts together under its settings transaction lock. */
export interface LibraryTopicSettingsPort {
  read(): Promise<LibraryTopicSettingsSnapshot>;
  change(compute: (current: LibraryTopicSettingsSnapshot) => Promise<LibraryTopicSettingsSnapshot>): Promise<void>;
}
export interface LibraryWorkflowOptions {
  storage: StorageAdapter;
  output: OutputSettings;
  connection: PersistedLibraryConnection;
  source: ScopedLibrarySource;
  http: HttpClient;
  arxivFetcher: Pick<ArxivFetcher, "fetchMetadataByIds">;
  createParser: (signal?: AbortSignal) => Promise<{ parser?: DocumentParser; parserSelector?: DocumentParserSelector }>;
  createEmbedding: (signal?: AbortSignal) => Promise<EmbeddingModel>;
  llm: PersonalLibraryDirectionLlmPort;
  assertCurrent: () => void | Promise<void>;
  assertAuthorized: (purpose: "directions" | "embedding") => void | Promise<void>;
  embeddingRequiresAuthorization?: boolean;
  topicSettings?: LibraryTopicSettingsPort;
  createId?: () => string;
  now?: () => Date;
  logger?: Logger;
}
export interface LibraryReviewSnapshot extends LibraryTopicSettingsSnapshot {
  catalog: PersonalLibraryCatalog;
  proposal: PersonalLibraryDirectionProposal | null;
  indexedPapers: LibraryPreviewPaper[];
}
export interface LibraryWorkflowRunOptions { signal?: AbortSignal }
export interface LibraryReviewVersions { expectedProposalRevision?: number | null }
export interface LibraryAcceptTopicsInput extends LibraryWorkflowRunOptions, LibraryReviewVersions {
  topicIds: readonly string[];
  candidateIds?: readonly string[];
}
export interface LibraryConfirmInput extends LibraryWorkflowRunOptions, LibraryReviewVersions {
  candidateId: string;
  draft: PersonalLibraryReviewedDirectionDraft;
  status?: "active" | "disabled";
}
export interface LibraryCandidateUpdateInput extends LibraryWorkflowRunOptions, LibraryReviewVersions {
  candidateId: string;
  patch: PersonalLibraryDirectionTextPatch;
  representativePaperKeys?: string[];
}

export interface LibraryRenameTopicInput extends LibraryWorkflowRunOptions, LibraryReviewVersions {
  topicId: string;
  suggestedName: string;
}
export interface LibraryRemoveCandidateInput extends LibraryWorkflowRunOptions, LibraryReviewVersions {
  candidateId: string;
}
export interface LibraryMoveDirectionInput extends LibraryRemoveCandidateInput {
  targetTopicId: string | null;
  suggestedName: string;
}
export interface LibraryPreviewDirectionInput extends LibraryRemoveCandidateInput {
  paperKeys?: string[];
  categories: string[];
}

/** Shared library application service; accepted directions live in normal topic settings. */
export class LibraryWorkflow {
  private readonly scopeFingerprint: string;
  private readonly identificationFingerprint: string;
  private readonly catalogStore: PersonalLibraryCatalogStore;
  private readonly proposalStore: PersonalLibraryDirectionProposalStore;
  private readonly knowledgeBase: FullTextKnowledgeBaseFileStore;
  private readonly generationStore: FullTextGenerationIndexStore;
  private readonly now: () => Date;
  private readonly createId: () => string;
  private busy = false;

  constructor(private readonly options: LibraryWorkflowOptions) {
    this.scopeFingerprint = createPersonalLibraryScopeFingerprint(options.connection);
    this.identificationFingerprint = createPersonalLibraryIdentificationFingerprint(options.connection.eligibleExtensions);
    this.now = options.now ?? (() => new Date());
    this.createId = options.createId ?? (() => crypto.randomUUID());
    const storeOptions = { now: this.now, onWarning: (message: string, error?: unknown) => options.logger?.warn(message, error) };
    const storeArgs = [options.storage, options.output, this.scopeFingerprint, this.identificationFingerprint, storeOptions] as const;
    this.catalogStore = new PersonalLibraryCatalogStore(options.storage, options.output, storeOptions);
    this.proposalStore = new PersonalLibraryDirectionProposalStore(...storeArgs);
    this.knowledgeBase = new FullTextKnowledgeBaseFileStore(...storeArgs);
    this.generationStore = new FullTextGenerationIndexStore(...storeArgs);
  }

  async scan(run: LibraryWorkflowRunOptions = {}): Promise<PersonalLibraryCatalog> {
    return this.mutate(async () => {
      await this.check(run.signal);
      const inventory = await this.options.source.inventory({ signal: run.signal });
      await this.check(run.signal);
      const current = await this.loadCatalog();
      const reconciled = await reconcilePersonalLibraryCatalog({
        current, inventory, eligibleExtensions: this.options.connection.eligibleExtensions,
        resolver: new ArxivLibraryMetadataResolver(this.options.arxivFetcher),
        identifyFile: createPdfLibraryFileIdentifier({ source: this.options.source, http: this.options.http }),
        now: this.now(), signal: run.signal,
      });
      await this.check(run.signal);
      const saved = await this.catalogStore.replace(reconciled.catalog);
      // Cancellation after the atomic promotion does not undo a committed scan.
      await this.options.assertCurrent();
      return saved;
    });
  }

  async index(run: LibraryWorkflowRunOptions & { onProgress?: FullTextIndexProgressReporter } = {}): Promise<FullTextIndexRunSummary> {
    return this.mutate(async () => {
      await this.check(run.signal);
      const catalog = await this.loadCatalog();
      await this.checkEmbedding(run.signal);
      const { parser } = await this.options.createParser(run.signal);
      if (!parser) throw new Error("Library indexing requires a page-text PDF parser");
      const extractor: PdfTextExtractor = {
        provenance: parser.provenance,
        extractPdfText: async (bytes, options) => parsedDocumentToPdfExtractionResult(await parser.parse(bytes, options), parser.capabilities),
      };
      await this.checkEmbedding(run.signal);
      const embedding = await this.options.createEmbedding(run.signal);
      const guardedEmbedding: EmbeddingModel = {
        modelId: embedding.modelId, dimension: embedding.dimension, prefixPolicy: embedding.prefixPolicy,
        embed: async (texts, opts) => {
          await this.checkEmbedding(opts?.signal);
          return embedding.embed(texts, opts);
        },
      };
      const mode = await preflightFullTextGenerationSynchronization({ storage: this.options.storage, generationStore: this.generationStore });
      const writerToken = `writer-${sha256Hex(this.createId())}`;
      let lease: FullTextLegacyMigrationLease | undefined;
      try {
        if (mode === "migration-fallback") lease = await this.generationStore.acquireLegacyMigrationLease(writerToken);
        const commitGuard = async () => {
          await this.checkEmbedding(run.signal);
          await lease?.assertOwned();
        };
        const summary = await indexPersonalLibraryFullText({
          catalog, source: this.options.source, extractor, embedding: guardedEmbedding, store: this.knowledgeBase,
          logger: this.options.logger, now: this.now, signal: run.signal, onProgress: run.onProgress,
          beforeManifestCommit: commitGuard, afterManifestCommit: commitGuard,
        });
        if (lease) {
          const held = lease;
          lease = undefined;
          await held.release();
        }
        await this.check(run.signal);
        let generationFailure: unknown;
        let generationFailed = false;
        if (mode === "available") {
          try {
            await synchronizeFullTextGenerationIndex({
              sourceStore: this.knowledgeBase, generationStore: this.generationStore,
              storage: this.options.storage, output: this.options.output,
              scopeFingerprint: this.scopeFingerprint, identificationFingerprint: this.identificationFingerprint,
              writerToken, signal: run.signal,
            });
          } catch (error) {
            throwIfCancelled(run.signal);
            generationFailed = true;
            generationFailure = error;
            this.options.logger?.warn("fulltext: generation synchronization failed; preserving the committed index update trigger", error);
          }
        }
        await this.check(run.signal);
        if (generationFailed) throw generationFailure;
        return summary;
      } finally {
        if (lease) {
          try { await lease.release(); }
          catch (error) { this.options.logger?.warn("fulltext: failed to release legacy migration lease", error); }
        }
      }
    });
  }

  async propose(run: LibraryWorkflowRunOptions = {}): Promise<PersonalLibraryDirectionProposal> {
    return this.mutate(async () => {
      await this.check(run.signal);
      await this.options.assertAuthorized("directions");
      const catalog = await this.loadCatalog();
      const previous = await this.proposalStore.load();
      const settings = await this.readTopicSettings();
      const proposal = await proposeClusteredPersonalLibraryDirections({
        catalog, knowledgeBase: this.knowledgeBase, llm: this.guardedLlm(run.signal),
        existingTopics: settings.topics, now: this.now, createId: this.createId, signal: run.signal,
      });
      await this.check(run.signal);
      await this.options.assertAuthorized("directions");
      if ((await this.loadCatalog()).revision !== catalog.revision) throw new Error("Personal library catalog changed during direction generation");
      if (JSON.stringify((await this.readTopicSettings()).topics) !== JSON.stringify(settings.topics)) throw new Error("Research topics changed during direction generation");
      const saved = await this.proposalStore.replace(proposal, previous?.revision ?? null);
      await this.options.assertCurrent();
      return saved;
    });
  }

  async review(): Promise<LibraryReviewSnapshot> {
    await this.check();
    const [catalog, proposal, settings] = await Promise.all([this.loadCatalog(), this.proposalStore.load(), this.readTopicSettings()]);
    await this.check();
    const manifest = await this.knowledgeBase.loadManifest();
    const indexedPapers = Object.values(manifest.papers).flatMap(record => {
      if (record.status !== "ready") return [];
      const paper = catalog.papers[record.paperKey];
      const title = paper?.title ?? record.title;
      if (!title) return [];
      return [{ paperKey: record.paperKey, title, abstract: paper?.abstract ?? record.abstract ?? "", categories: [...(paper?.categories ?? [])] }];
    }).sort((left, right) => left.paperKey.localeCompare(right.paperKey));
    await this.check();
    return { catalog, proposal, indexedPapers, ...settings };
  }

  async updateCandidate(input: LibraryCandidateUpdateInput): Promise<LibraryReviewSnapshot> {
    return this.mutate(async () => {
      await this.check(input.signal);
      const current = await this.review();
      this.assertProposalRevision(current.proposal, input);
      if (!current.proposal) throw new Error("Generate personal library direction proposals first");
      this.assertUnprocessed(current, input.candidateId);
      const proposal = updatePersonalLibraryDirectionCandidate({
        proposal: current.proposal, candidateId: input.candidateId, patch: input.patch,
        ...(input.representativePaperKeys === undefined ? {} : { representativePaperKeys: input.representativePaperKeys, catalog: current.catalog }),
      });
      await this.check(input.signal);
      await this.proposalStore.replace(proposal, current.proposal.revision);
      return this.review();
    });
  }

  async renameTopic(input: LibraryRenameTopicInput): Promise<LibraryReviewSnapshot> {
    return this.editProposal(input, current => {
      const topic = current.proposal.topics.find(item => item.id === input.topicId);
      if (!topic) throw new Error("Proposed topic was not found; refresh the review");
      for (const candidate of topic.directions) this.assertUnprocessed(current, candidate.id);
      if (topic.targetTopicId) throw new Error("Edit an existing research topic in settings");
      return renamePersonalLibraryProposedTopic({ proposal: current.proposal, topicId: input.topicId, suggestedName: input.suggestedName });
    });
  }

  async moveDirection(input: LibraryMoveDirectionInput): Promise<LibraryReviewSnapshot> {
    return this.editProposal(input, current => {
      this.assertUnprocessed(current, input.candidateId);
      if (input.targetTopicId !== null && !current.topics.some(topic => topic.id === input.targetTopicId)) {
        throw new Error("Destination research topic no longer exists; refresh the review");
      }
      return movePersonalLibraryDirectionCandidate({ proposal: current.proposal, candidateId: input.candidateId,
        targetTopicId: input.targetTopicId, suggestedName: input.suggestedName, topicId: this.createId() });
    });
  }

  async removeCandidate(input: LibraryRemoveCandidateInput): Promise<LibraryReviewSnapshot> {
    return this.editProposal(input, current => {
      this.assertUnprocessed(current, input.candidateId);
      return removePersonalLibraryDirectionCandidate({ proposal: current.proposal, candidateId: input.candidateId });
    });
  }

  async previewDirection(input: LibraryPreviewDirectionInput): Promise<LibraryDirectionPreview> {
    return this.mutate(async () => {
      await this.check(input.signal);
      const current = await this.review();
      this.assertProposalRevision(current.proposal, input);
      const candidate = current.proposal?.topics.flatMap(topic => topic.directions).find(item => item.id === input.candidateId);
      if (!candidate) throw new Error("Direction candidate was not found; refresh the review");
      await this.options.assertAuthorized("directions");
      const keys = input.paperKeys ?? candidate.representatives.map(paper => paper.paperKey);
      if (keys.length === 0 || keys.length > 20 || new Set(keys).size !== keys.length) throw new Error("Select 1 to 20 distinct indexed evidence papers");
      const papers = keys.map(key => {
        const paper = current.indexedPapers.find(item => item.paperKey === key);
        if (!paper) throw new Error("Selected indexed evidence is no longer available; refresh the review");
        return paper;
      });
      const result = await previewPersonalLibraryDirection({ text: candidate.text, papers, categories: input.categories,
        llm: { call: async (messages, options) => {
          const raw = await this.guardedLlm(input.signal).call(messages, options);
          await this.check(input.signal);
          await this.options.assertAuthorized("directions");
          return raw;
        } }, signal: input.signal });
      await this.check(input.signal);
      const latest = await this.review();
      this.assertProposalRevision(latest.proposal, { expectedProposalRevision: current.proposal!.revision });
      if (latest.catalog.revision !== current.catalog.revision || JSON.stringify(latest.indexedPapers) !== JSON.stringify(current.indexedPapers)) {
        throw new Error("Library evidence changed during preview; refresh the review");
      }
      return result;
    });
  }

  private async editProposal(input: LibraryWorkflowRunOptions & LibraryReviewVersions,
    edit: (snapshot: LibraryReviewSnapshot & { proposal: PersonalLibraryDirectionProposal }) => PersonalLibraryDirectionProposal,
  ): Promise<LibraryReviewSnapshot> {
    return this.mutate(async () => {
      await this.check(input.signal);
      const current = await this.review();
      this.assertProposalRevision(current.proposal, input);
      if (!current.proposal) throw new Error("Generate personal library direction proposals first");
      const proposal = edit({ ...current, proposal: current.proposal });
      await this.check(input.signal);
      await this.proposalStore.replace(proposal, current.proposal.revision);
      return this.review();
    });
  }

  private assertUnprocessed(snapshot: LibraryReviewSnapshot, candidateId: string): void {
    const proposal = snapshot.proposal;
    if (!proposal) throw new Error("Generate personal library direction proposals first");
    const receipt = snapshot.acceptances.find(item => item.proposalId === proposal.proposalId && item.scopeFingerprint === proposal.scopeFingerprint);
    const candidate = proposal.topics.flatMap(topic => topic.directions).find(item => item.id === candidateId);
    if (!candidate) throw new Error("Direction candidate was not found; refresh the review");
    if (receipt?.processedCandidateIds.some(id => id === candidateId || candidate.lineage.candidateIds.includes(id))) {
      throw new Error("This direction was already accepted; edit research topics in settings");
    }
  }

  async acceptTopics(input: LibraryAcceptTopicsInput): Promise<LibraryReviewSnapshot> {
    return this.mutate(() => this.acceptTopicsDirect(input));
  }

  /** Compatibility entry: a reviewed candidate is accepted into its proposed topic. */
  async confirm(input: LibraryConfirmInput): Promise<LibraryReviewSnapshot> {
    return this.mutate(async () => {
      if (input.status === "disabled") throw new Error("Disabled library profiles are retired; keep the direction as an unaccepted proposal");
      if (!this.options.topicSettings) throw new Error("This host must configure transactional topic settings before accepting directions");
      await this.check(input.signal);
      const current = await this.review();
      this.assertProposalRevision(current.proposal, input);
      if (!current.proposal) throw new Error("Generate personal library direction proposals first");
      this.assertUnprocessed(current, input.candidateId);
      const topic = current.proposal.topics.find(item => item.directions.some(candidate => candidate.id === input.candidateId));
      if (!topic) throw new Error("Direction candidate was not found; refresh the review");
      const proposal = updatePersonalLibraryDirectionCandidate({
        proposal: current.proposal, candidateId: input.candidateId,
        patch: { text: input.draft.text, discoveryCues: input.draft.discoveryCues },
        representativePaperKeys: input.draft.representativePaperKeys, catalog: current.catalog,
      });
      await this.check(input.signal);
      const saved = await this.proposalStore.replace(proposal, current.proposal.revision);
      return this.acceptTopicsDirect({ topicIds: [topic.id], candidateIds: [input.candidateId], expectedProposalRevision: saved.revision, signal: input.signal });
    });
  }

  private async acceptTopicsDirect(input: LibraryAcceptTopicsInput): Promise<LibraryReviewSnapshot> {
    const port = this.options.topicSettings;
    if (!port) throw new Error("This host must configure transactional topic settings before accepting directions");
    await this.check(input.signal);
    const proposal = await this.proposalStore.load();
    this.assertProposalRevision(proposal, input);
    if (!proposal) throw new Error("Generate personal library direction proposals first");
    if (input.topicIds.length === 0) throw new Error("Select at least one proposed topic to accept");
    const selected = input.candidateIds === undefined ? null : new Set(input.candidateIds);
    const kept = [...new Set(input.topicIds)].map(id => {
      const topic = proposal.topics.find(item => item.id === id);
      if (!topic) throw new Error("The proposed topic no longer exists; refresh the review");
      return { ...topic, directions: selected ? topic.directions.filter(item => selected.has(item.id)) : topic.directions };
    }).filter(topic => topic.directions.length > 0);
    if (kept.length === 0) throw new Error("Select at least one proposed direction to accept");
    await port.change(async current => {
      await this.check(input.signal);
      this.assertProposalRevision(await this.proposalStore.load(), { expectedProposalRevision: proposal.revision });
      const accepted = acceptProposedTopics({
        proposalId: proposal.proposalId, scopeFingerprint: proposal.scopeFingerprint,
        topics: kept, existingTopics: current.topics,
        acceptance: current.acceptances.find(receipt => receipt.scopeFingerprint === proposal.scopeFingerprint && receipt.proposalId === proposal.proposalId),
      });
      await this.check(input.signal);
      return { topics: accepted.topics, acceptances: [
        ...current.acceptances.filter(receipt => receipt.scopeFingerprint !== proposal.scopeFingerprint), accepted.acceptance,
      ].sort((left, right) => left.scopeFingerprint.localeCompare(right.scopeFingerprint)) };
    });
    return this.review();
  }

  private assertProposalRevision(proposal: PersonalLibraryDirectionProposal | null, expected: LibraryReviewVersions): void {
    if (expected.expectedProposalRevision !== undefined && expected.expectedProposalRevision !== (proposal?.revision ?? null)) {
      throw new PersonalLibraryDirectionProposalStoreError("Personal library proposal revision changed; refresh before reviewing", "stale", {
        expectedRevision: expected.expectedProposalRevision, currentRevision: proposal?.revision ?? null,
      });
    }
  }
  private readTopicSettings(): Promise<LibraryTopicSettingsSnapshot> {
    return this.options.topicSettings?.read() ?? Promise.resolve({ topics: [], acceptances: [] });
  }
  private loadCatalog(): Promise<PersonalLibraryCatalog> {
    return this.catalogStore.load(this.scopeFingerprint, this.identificationFingerprint);
  }
  private async check(signal?: AbortSignal): Promise<void> {
    throwIfCancelled(signal);
    await this.options.assertCurrent();
    throwIfCancelled(signal);
  }
  private async checkEmbedding(signal?: AbortSignal): Promise<void> {
    await this.check(signal);
    if (this.options.embeddingRequiresAuthorization) await this.options.assertAuthorized("embedding");
  }
  private guardedLlm(signal?: AbortSignal): PersonalLibraryDirectionLlmPort {
    return { call: async (messages, options) => {
      await this.check(signal);
      await this.options.assertAuthorized("directions");
      return this.options.llm.call(messages, options);
    } };
  }
  private async mutate<T>(operation: () => Promise<T>): Promise<T> {
    if (this.busy) throw new Error("A personal library workflow operation is already active");
    this.busy = true;
    try { return await operation(); } finally { this.busy = false; }
  }
}
