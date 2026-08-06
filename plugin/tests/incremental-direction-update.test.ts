import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  IncrementalSuggestionsStore,
  LlmClient,
  OperationRegistry,
  PersonalLibraryDirectionProposalStore,
  PersonalLibraryInterestProfileStore,
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryPaperEvidenceFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  createPersonalLibraryScopeFingerprint,
  type PersonalLibraryCatalog,
  type PersonalLibraryInterestProfile,
  type StorageAdapter,
} from "@arxiv-daily/core";
import ArxivDailyPlugin, { INCREMENTAL_BUFFER_TRIGGER } from "../main.ts";
import { authorizeLibraryConnection, createLibraryConnection } from "../src/library/connection";

const PAPER_VECTORS = {
  // Anchor/representative theme (covered by the confirmed direction).
  "arxiv:2608.00001": [1, 0, 0, 0],
  // Same theme: deterministic attach.
  "arxiv:2608.00002": [1, 0, 0, 0],
  // Distinct themes: buffer pool; two of them cluster into one new group.
  "arxiv:2608.00003": [0, 1, 0, 0],
  "arxiv:2608.00004": [0, 1, 0, 0],
  "arxiv:2608.00005": [0, 0, 1, 0],
  "arxiv:2608.00006": [0, 0, 0, 1],
} as const;

function makeStorage() {
  const files = new Map<string, string>();
  const directories = new Set<string>();
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/"),
    exists: async (path) => files.has(path) || directories.has(path),
    readText: async (path) => {
      const value = files.get(path);
      if (value === undefined) throw new Error("missing");
      return value;
    },
    writeText: async (path, value) => { files.set(path, value); },
    writeTextAtomic: async (path, value) => { files.set(path, value); },
    mkdir: async (path) => { directories.add(path); },
    rename: async (from, to) => { files.set(to, files.get(from)!); files.delete(from); },
    remove: async (path) => { files.delete(path); },
  };
  return { storage, files };
}

function makeKnowledgeBase(
  scopeFingerprint: string,
  identificationFingerprint: string,
  papers: ReadonlyArray<{ paperKey: string; vectors: readonly number[] }>,
) {
  const documents = new Map<string, unknown>();
  const manifest: Record<string, unknown> = {};
  for (const [index, paper] of papers.entries()) {
    const seed = String.fromCharCode(97 + (index % 26)).repeat(64);
    const paperKey = paper.paperKey;
    documents.set(paperKey, {
      schemaVersion: 1,
      paperKey,
      modelId: "fake",
      dimension: paper.vectors.length,
      textHash: `sha256:${"d".repeat(64)}`,
      filePaths: [`papers/${paperKey.replace("arxiv:", "")}.pdf`],
      observationFingerprints: [`sha256:${seed}`],
      chunks: [{ index: 0, page: 1, text: `paper ${paperKey}` }],
      vectors: new Float32Array(paper.vectors),
      updatedAt: "2026-08-03T00:00:00.000Z",
    });
    manifest[paperKey] = {
      paperKey,
      status: "ready",
      modelId: "fake",
      dimension: paper.vectors.length,
      textHash: `sha256:${"d".repeat(64)}`,
      filePaths: [`papers/${paperKey.replace("arxiv:", "")}.pdf`],
      observationFingerprints: [`sha256:${seed}`],
      chunkCount: 1,
      updatedAt: "2026-08-03T00:00:00.000Z",
    };
  }
  return {
    paths: { directory: "kb", manifest: { directory: "kb", documentPath: "kb/manifest.json", backupPath: "kb/m.json.backup" }, papersDirectory: "kb/papers" },
    loadManifest: async () => ({
      schemaVersion: 1,
      revision: 1,
      scopeFingerprint,
      identificationFingerprint,
      modelId: "fake",
      dimension: 4,
      updatedAt: "2026-08-03T00:00:00.000Z",
      papers: manifest,
    }),
    loadPaper: async (paperKey: string) => documents.get(paperKey) ?? null,
    replaceManifest: async () => { throw new Error("not used"); },
    savePaper: async () => {},
    removePaper: async () => {},
    removeAll: async () => {},
  };
}

function paperRecord(paperKey: string, title: string) {
  return {
    paperKey,
    source: "arxiv",
    externalId: paperKey.slice("arxiv:".length),
    title,
    authors: ["Ada"],
    abstract: "Evidence",
    published: "2026-08-01T00:00:00.000Z",
    updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI",
    categories: ["cs.AI"],
    evidenceDepth: "metadata-and-abstract",
    filePaths: [`papers/${paperKey.slice("arxiv:".length)}.pdf`],
  };
}

function fixture(papers: ReadonlyArray<{ paperKey: string; vectors: readonly number[] }>) {
  const { storage, files } = makeStorage();
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.llm.apiKey = "secret";
  settings.llm.model = "test-model";
  settings.llm.baseUrl = "https://model.example/v1";
  const rawConnection = createLibraryConnection("/private/library", "1:2");
  const connection = authorizeLibraryConnection(rawConnection, settings.llm.baseUrl, new Date("2026-08-03T00:00:00Z"));
  const scopeFingerprint = createPersonalLibraryScopeFingerprint({
    rootIdentity: connection.rootIdentity,
    eligibleExtensions: connection.eligibleExtensions,
  });
  const identificationFingerprint = createPersonalLibraryIdentificationFingerprint(connection.eligibleExtensions);
  const catalog = createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint, new Date("2026-08-03T00:00:00Z"));
  catalog.revision = 1;
  for (const paper of papers) {
    const arxivId = paper.paperKey.slice("arxiv:".length);
    catalog.files[`papers/${arxivId}.pdf`] = {
      path: `papers/${arxivId}.pdf`,
      status: "ready",
      observationFingerprint: `sha256:${"c".repeat(64)}`,
      paperKey: paper.paperKey,
      arxivId,
      updatedAt: "2026-08-03T00:00:00.000Z",
    };
    catalog.papers[paper.paperKey] = paperRecord(paper.paperKey, `Paper ${arxivId}`);
  }
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  Object.assign(plugin, {
    settings,
    logger: { warn: vi.fn(), error: vi.fn(), setSensitiveValues: vi.fn() },
    host: { storage, http: {} },
    progress: {},
    operations: new OperationRegistry(),
    libraryConnection: connection,
    libraryCatalog: catalog,
    libraryMutationQueue: Promise.resolve(),
    libraryConnectionRevision: 0,
    libraryOutputRevision: 0,
    librarySelectionRevision: 0,
    saveData: vi.fn().mockResolvedValue(undefined),
    buildFullTextKnowledgeBaseStore: () => makeKnowledgeBase(scopeFingerprint, identificationFingerprint, papers),
  });
  return { plugin, internals: plugin as any, storage, files, catalog, connection, scopeFingerprint, identificationFingerprint };
}

function confirmedProfile(catalog: PersonalLibraryCatalog): PersonalLibraryInterestProfile {
  // The anchor paper may be absent from the catalog (empty-KB fixtures still
  // need a confirmed direction); synthesize the record then.
  const anchorKey = "arxiv:2608.00001";
  if (!catalog.papers[anchorKey]) {
    catalog.papers[anchorKey] = paperRecord(anchorKey, "Anchor paper");
  }
  const paper = catalog.papers[anchorKey]!;
  const representatives = [{
    paperKey: paper.paperKey,
    evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(paper),
  }];
  return {
    schemaVersion: 3,
    revision: 1,
    scopeFingerprint: catalog.scopeFingerprint,
    identificationFingerprint: catalog.identificationFingerprint,
    updatedAt: "2026-08-03T01:00:00.000Z",
    directions: [{
      id: "direction-confirmed",
      status: "active",
      name: "Confirmed reliable agents",
      description: "Researcher-reviewed reliable agent methods.",
      discoveryCues: ["agent evaluation"],
      representatives,
      representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
      clusterMembers: [],
      timeline: [{ kind: "created", at: "2026-08-03T01:00:00.000Z" }],
      lineage: { proposalIds: ["proposal.1"], candidateIds: ["candidate.1"], directionIds: [] },
      createdAt: "2026-08-03T01:00:00.000Z",
      updatedAt: "2026-08-03T01:00:00.000Z",
    }],
  };
}

/** Suggestion content key, mirroring the plugin's private scheme. */
function keyOf(kind: string, directionId: string | null, firstPaperKey: string): string {
  return `${kind}:${directionId ?? ""}:${firstPaperKey}`;
}

function suggestionsStore(
  storage: StorageAdapter,
  plugin: ArxivDailyPlugin,
  scopeFingerprint: string,
  identificationFingerprint: string,
): IncrementalSuggestionsStore {
  return new IncrementalSuggestionsStore(
    storage,
    plugin.settings.output,
    scopeFingerprint,
    identificationFingerprint,
  );
}

const SIX_PAPERS = Object.entries(PAPER_VECTORS).map(([paperKey, vectors]) => ({ paperKey, vectors }));

describe("incremental direction update", () => {
  it("rejects the update run without model authorization and writes nothing", async () => {
    const { plugin, internals, files } = fixture(SIX_PAPERS);
    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    const call = vi.spyOn(LlmClient.prototype, "call");
    await expect(plugin.runIncrementalDirectionUpdate()).rejects.toThrow("Authorize");
    expect(call).not.toHaveBeenCalled();
    expect([...files.keys()].some((path) => path.includes("incremental-suggestions"))).toBe(false);
  });

  it("produces an empty suggestions document for an empty knowledge base without an LLM call", async () => {
    const { plugin, internals, storage, scopeFingerprint, identificationFingerprint } = fixture([]);
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    const call = vi.spyOn(LlmClient.prototype, "call");
    const summary = await plugin.runIncrementalDirectionUpdate();
    expect(summary).toEqual({ suggestions: 0, attachments: 0, buffered: 0 });
    expect(call).not.toHaveBeenCalled();
    const doc = await suggestionsStore(storage, plugin, scopeFingerprint, identificationFingerprint).load();
    expect(doc.suggestions).toEqual([]);
    expect(plugin.getIncrementalSuggestions()?.suggestions).toEqual([]);
  });

  it("stores deterministic attach suggestions and skips reclustering below the buffer trigger", async () => {
    const { plugin, internals, storage, scopeFingerprint, identificationFingerprint } = fixture([
      { paperKey: "arxiv:2608.00001", vectors: [1, 0, 0, 0] },
      { paperKey: "arxiv:2608.00002", vectors: [1, 0, 0, 0] },
      { paperKey: "arxiv:2608.00003", vectors: [0, 1, 0, 0] },
    ]);
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    const call = vi.spyOn(LlmClient.prototype, "call");
    const summary = await plugin.runIncrementalDirectionUpdate();
    // One attach suggestion; one buffered paper stays below the trigger.
    expect(summary).toEqual({ suggestions: 1, attachments: 1, buffered: 1 });
    expect(call).not.toHaveBeenCalled();
    const doc = await suggestionsStore(storage, plugin, scopeFingerprint, identificationFingerprint).load();
    expect(doc.suggestions).toEqual([{
      kind: "attach",
      directionId: "direction-confirmed",
      paperKeys: ["arxiv:2608.00002"],
      reason: "Newly indexed paper matches this confirmed direction.",
    }]);
  });

  it("reclusters the buffer pool at the trigger and merges LLM diff suggestions into the store", async () => {
    const { plugin, internals, storage, scopeFingerprint, identificationFingerprint } = fixture(SIX_PAPERS);
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    const call = vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(JSON.stringify({
      suggestions: [{
        kind: "attach",
        directionId: "direction-confirmed",
        paperKeys: ["arxiv:2608.00003", "arxiv:2608.00004"],
        reason: "Papers form the same evaluation theme.",
      }],
    }));
    const summary = await plugin.runIncrementalDirectionUpdate();
    expect(summary).toEqual({ suggestions: 2, attachments: 2, buffered: 4 });
    expect(call).toHaveBeenCalledTimes(1);
    const doc = await suggestionsStore(storage, plugin, scopeFingerprint, identificationFingerprint).load();
    expect(doc.suggestions.map((suggestion) => suggestion.paperKeys)).toEqual([
      ["arxiv:2608.00002"],
      ["arxiv:2608.00003", "arxiv:2608.00004"],
    ]);
    expect(doc.suggestions.every((suggestion) => suggestion.kind === "attach")).toBe(true);
  });

  it("applies an attach suggestion: profile cluster members update and the suggestion leaves the document", async () => {
    const { plugin, internals, storage, scopeFingerprint, identificationFingerprint } = fixture([
      { paperKey: "arxiv:2608.00001", vectors: [1, 0, 0, 0] },
      { paperKey: "arxiv:2608.00002", vectors: [1, 0, 0, 0] },
      { paperKey: "arxiv:2608.00003", vectors: [0, 1, 0, 0] },
    ]);
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    await new PersonalLibraryInterestProfileStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    ).replace(internals.libraryProfile, 0);
    vi.spyOn(LlmClient.prototype, "call");
    await plugin.runIncrementalDirectionUpdate();
    const before = await suggestionsStore(storage, plugin, scopeFingerprint, identificationFingerprint).load();
    expect(before.suggestions).toHaveLength(1);

    const snapshot = await plugin.applyIncrementalSuggestion(
      keyOf("attach", "direction-confirmed", "arxiv:2608.00002"),
    );
    expect(snapshot.profile?.directions[0]?.clusterMembers).toEqual([
      { paperKey: "arxiv:2608.00002", confidence: 0.9 },
    ]);
    expect(snapshot.profile?.directions[0]?.timeline.map(({ kind }) => kind))
      .toEqual(["created", "members-updated"]);
    expect(snapshot.suggestions?.suggestions).toEqual([]);

    const doc = await suggestionsStore(storage, plugin, scopeFingerprint, identificationFingerprint).load();
    expect(doc.suggestions).toEqual([]);
    const durable = await new PersonalLibraryInterestProfileStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    ).load();
    expect(durable.directions[0]?.clusterMembers).toEqual([
      { paperKey: "arxiv:2608.00002", confidence: 0.9 },
    ]);
  });

  it("dismisses a suggestion without touching the profile", async () => {
    const { plugin, internals, storage, scopeFingerprint, identificationFingerprint } = fixture([
      { paperKey: "arxiv:2608.00001", vectors: [1, 0, 0, 0] },
      { paperKey: "arxiv:2608.00002", vectors: [1, 0, 0, 0] },
      { paperKey: "arxiv:2608.00003", vectors: [0, 1, 0, 0] },
    ]);
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    await new PersonalLibraryInterestProfileStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    ).replace(internals.libraryProfile, 0);
    vi.spyOn(LlmClient.prototype, "call");
    await plugin.runIncrementalDirectionUpdate();

    const snapshot = await plugin.dismissIncrementalSuggestion(
      keyOf("attach", "direction-confirmed", "arxiv:2608.00002"),
    );
    expect(snapshot.suggestions?.suggestions).toEqual([]);
    expect(snapshot.profile?.directions[0]?.clusterMembers).toEqual([]);
    expect(await suggestionsStore(storage, plugin, scopeFingerprint, identificationFingerprint).load())
      .toEqual(expect.objectContaining({ suggestions: [] }));
  });

  it("locks and unlocks a confirmed direction through the profile persistence path", async () => {
    const { plugin, internals, storage, scopeFingerprint, identificationFingerprint } = fixture([]);
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    await new PersonalLibraryInterestProfileStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    ).replace(internals.libraryProfile, 0);

    const locked = await plugin.lockPersonalLibraryConfirmedDirection(
      "direction-confirmed", new Date("2026-08-04T00:00:00Z"),
    );
    expect(locked.profile?.directions[0]?.lockedAt).toBe("2026-08-04T00:00:00.000Z");
    expect(locked.profile?.directions[0]?.timeline.map(({ kind }) => kind)).toEqual(["created", "locked"]);

    const unlocked = await plugin.unlockPersonalLibraryConfirmedDirection(
      "direction-confirmed", new Date("2026-08-05T00:00:00Z"),
    );
    expect(unlocked.profile?.directions[0]?.lockedAt).toBeUndefined();
    expect(unlocked.profile?.directions[0]?.timeline.map(({ kind }) => kind)).toEqual(["created", "locked", "unlocked"]);

    const durable = await new PersonalLibraryInterestProfileStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    ).load();
    expect(durable.directions[0]?.lockedAt).toBeUndefined();
    expect(durable.directions[0]?.timeline.at(-1)?.kind).toBe("unlocked");
  });

  it("converts a new suggestion into a review candidate in the proposal store and removes it from the document", async () => {
    const { plugin, internals, storage, scopeFingerprint, identificationFingerprint } = fixture(SIX_PAPERS);
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(JSON.stringify({
      suggestions: [{
        kind: "new",
        paperKeys: ["arxiv:2608.00003", "arxiv:2608.00004"],
        reason: "New evaluation theme for evaluation methods.",
      }],
    }));
    await plugin.runIncrementalDirectionUpdate();

    const snapshot = await plugin.applyIncrementalSuggestion(keyOf("new", null, "arxiv:2608.00003"));
    const candidate = snapshot.proposal?.candidates[0];
    expect(candidate?.name).toBe("New evaluation theme for evaluation methods.");
    expect(candidate?.clusterMembers).toEqual([
      { paperKey: "arxiv:2608.00003", confidence: 0.9 },
      { paperKey: "arxiv:2608.00004", confidence: 0.9 },
    ]);
    expect(candidate?.representatives.map(({ paperKey }) => paperKey)).toEqual([
      "arxiv:2608.00003", "arxiv:2608.00004",
    ]);
    // The placement attach remains; only the converted suggestion leaves.
    expect(snapshot.suggestions?.suggestions.map((suggestion) => suggestion.paperKeys)).toEqual([
      ["arxiv:2608.00002"],
    ]);

    const durable = await new PersonalLibraryDirectionProposalStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    ).load();
    expect(durable).not.toBeNull();
    expect(durable!.candidates).toHaveLength(1);
    expect(durable!.candidates[0]!.clusterMembers).toEqual(candidate?.clusterMembers);
  });

  it("rejects applying a suggestion that is no longer in the document", async () => {
    const { plugin, internals, storage, scopeFingerprint, identificationFingerprint } = fixture([
      { paperKey: "arxiv:2608.00001", vectors: [1, 0, 0, 0] },
      { paperKey: "arxiv:2608.00002", vectors: [1, 0, 0, 0] },
      { paperKey: "arxiv:2608.00003", vectors: [0, 1, 0, 0] },
    ]);
    internals.libraryProfile = confirmedProfile(internals.libraryCatalog);
    await new PersonalLibraryInterestProfileStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    ).replace(internals.libraryProfile, 0);
    vi.spyOn(LlmClient.prototype, "call");
    await plugin.runIncrementalDirectionUpdate();
    await plugin.dismissIncrementalSuggestion(keyOf("attach", "direction-confirmed", "arxiv:2608.00002"));
    await expect(plugin.applyIncrementalSuggestion(
      keyOf("attach", "direction-confirmed", "arxiv:2608.00002"),
    )).rejects.toThrow("no longer exists");
  });

  it("exposes the buffer trigger constant used by the update", () => {
    expect(INCREMENTAL_BUFFER_TRIGGER).toBe(3);
  });
});
