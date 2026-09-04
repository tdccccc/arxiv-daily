import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  LlmClient,
  OperationRegistry,
  PersonalLibraryCatalogStore,
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
import ArxivDailyPlugin from "../main.ts";
import { authorizeLibraryConnection, createLibraryConnection } from "../src/library/connection";

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

function fixture() {
  const { storage, files } = makeStorage();
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.llm.apiKey = "secret";
  settings.llm.model = "test-model";
  settings.llm.baseUrl = "https://model.example/v1";
  const rawConnection = createLibraryConnection("/private/library", "1:2");
  const connection = authorizeLibraryConnection(rawConnection, { llmBaseUrl: settings.llm.baseUrl }, new Date("2026-08-03T00:00:00Z"));
  const scopeFingerprint = createPersonalLibraryScopeFingerprint({
    rootIdentity: connection.rootIdentity,
    eligibleExtensions: connection.eligibleExtensions,
  });
  const identificationFingerprint = createPersonalLibraryIdentificationFingerprint(connection.eligibleExtensions);
  const catalog = createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint, new Date("2026-08-03T00:00:00Z"));
  catalog.revision = 1;
  catalog.files["papers/2608.00001.pdf"] = {
    path: "papers/2608.00001.pdf",
    status: "ready",
    observationFingerprint: `sha256:${"c".repeat(64)}`,
    paperKey: "arxiv:2608.00001",
    arxivId: "2608.00001",
    updatedAt: "2026-08-03T00:00:00.000Z",
  };
  catalog.papers["arxiv:2608.00001"] = {
    paperKey: "arxiv:2608.00001", source: "arxiv", externalId: "2608.00001",
    title: "Reliable agents", authors: ["Ada"], abstract: "Evidence", published: "2026-08-01T00:00:00.000Z",
    updated: "2026-08-02T00:00:00.000Z", primaryCategory: "cs.AI", categories: ["cs.AI"],
    evidenceDepth: "metadata-and-abstract", filePaths: ["papers/2608.00001.pdf"],
  };
  catalog.files["papers/2608.00002.pdf"] = {
    path: "papers/2608.00002.pdf",
    status: "ready",
    observationFingerprint: `sha256:${"e".repeat(64)}`,
    paperKey: "arxiv:2608.00002",
    arxivId: "2608.00002",
    updatedAt: "2026-08-03T00:00:00.000Z",
  };
  catalog.papers["arxiv:2608.00002"] = {
    paperKey: "arxiv:2608.00002", source: "arxiv", externalId: "2608.00002",
    title: "Agent evaluation methods", authors: ["Bob"], abstract: "Evidence", published: "2026-08-01T00:00:00.000Z",
    updated: "2026-08-02T00:00:00.000Z", primaryCategory: "cs.AI", categories: ["cs.AI"],
    evidenceDepth: "metadata-and-abstract", filePaths: ["papers/2608.00002.pdf"],
  };
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  Object.assign(plugin, {
    settings,
    logger: { warn: vi.fn(), error: vi.fn(), setSensitiveValues: vi.fn() },
    host: { storage, http: {} },
    // Direction generation reports each cluster to the status bar.
    progress: { setTask: vi.fn(), setIdle: vi.fn() },
    operations: new OperationRegistry(),
    libraryConnection: connection,
    libraryCatalog: catalog,
    libraryMutationQueue: Promise.resolve(),
    libraryConnectionRevision: 0,
    libraryOutputRevision: 0,
    librarySelectionRevision: 0,
    saveData: vi.fn().mockResolvedValue(undefined),
    // Clustered generation reads the full-text knowledge base for its input;
    // an empty KB would fail with no-evidence before any LLM call, so stub a
    // populated in-memory KB whose two papers cluster into one theme.
    buildFullTextKnowledgeBaseStore: () => makeFullTextKnowledgeBase(scopeFingerprint, identificationFingerprint),
  });
  return { plugin, internals: plugin as any, storage, files, catalog, connection, scopeFingerprint, identificationFingerprint };
}

function confirmedProfile(catalog: PersonalLibraryCatalog): PersonalLibraryInterestProfile {
  const paper = catalog.papers["arxiv:2608.00001"]!;
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

function modelResult() {
  return JSON.stringify({ candidates: [
    {
      name: "Reliable agents", description: "Methods for reliable systems.",
      discoveryCues: ["agent evaluation"], representativePaperKeys: ["arxiv:2608.00001"],
    },
    {
      name: "Agent verification", description: "Verification for agentic systems.",
      discoveryCues: ["agent verification"], representativePaperKeys: ["arxiv:2608.00001"],
    },
    {
      name: "Agent evaluation", description: "Evaluation methods for agentic systems.",
      discoveryCues: ["agent benchmarks"], representativePaperKeys: ["arxiv:2608.00001"],
    },
  ] });
}

describe("personal library profile lifecycle", () => {
  it("does not reopen authorization across a newer queued root or output transition", async () => {
    const { plugin, internals } = fixture();
    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    let releaseQueue!: () => void;
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });
    const authorize = plugin.authorizeLibraryProcessing();
    internals.beginLibraryMutation();
    const newerRevision = internals.libraryMutationRevision;
    releaseQueue();
    await expect(authorize).rejects.toThrow("superseded");
    // The counter only moves forward: a rejected authorization must not roll it
    // back, or the next queued change would look like no change at all.
    expect(internals.libraryMutationRevision).toBe(newerRevision);

    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    await plugin.authorizeLibraryProcessing();
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });
    const second = plugin.authorizeLibraryProcessing();
    internals.beginLibraryMutation();
    releaseQueue();
    await expect(second).rejects.toThrow("superseded");
  });

  it("makes queued revoke win over an earlier queued authorize and keeps later daily builds manual-only", async () => {
    const { plugin, internals, catalog } = fixture();
    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    internals.libraryProfile = confirmedProfile(catalog);
    let releaseQueue!: () => void;
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });

    const authorize = plugin.authorizeLibraryProcessing();
    const revoke = plugin.revokeLibraryProcessing();
    expect(plugin.getLibraryConnectionStatus().kind).not.toBe("authorized");
    releaseQueue();

    await expect(authorize).rejects.toThrow("superseded");
    await revoke;
    expect(plugin.getLibraryConnectionStatus().kind).not.toBe("authorized");
    const persisted = vi.mocked(internals.saveData).mock.calls.at(-1)?.[0];
    expect(persisted.libraryConnection.authorization).toBeUndefined();

    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    const directionCall = vi.spyOn(LlmClient.prototype, "call");
    const pipeline = internals.buildPipeline();
    expect(pipeline.deps.personalizedDiscovery).toBeUndefined();
    expect(directionCall).not.toHaveBeenCalled();
  });

  it("rejects queued authorization after invocation-time connection replacement or disconnect", async () => {
    const { plugin, internals } = fixture();
    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    let releaseQueue!: () => void;
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });
    const rootChanged = plugin.authorizeLibraryProcessing();
    internals.libraryConnection = createLibraryConnection("/next", "1:3");
    internals.libraryConnectionRevision += 1;
    releaseQueue();
    await expect(rootChanged).rejects.toThrow("superseded");
    expect(internals.libraryConnection.selectedRoot).toBe("/next");

    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });
    const disconnected = plugin.authorizeLibraryProcessing();
    internals.libraryConnection = undefined;
    internals.libraryConnectionRevision += 1;
    releaseQueue();
    await expect(disconnected).rejects.toThrow("superseded");
    expect(plugin.getLibraryConnectionStatus().kind).toBe("disconnected");
  });

  it("gates consent before calls, generates with dedicated LLM calls, rejects duplicates, and avoids daily dependencies", async () => {
    const { plugin, internals } = fixture();
    const call = vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    internals.buildSharedDeps = vi.fn(() => { throw new Error("daily deps must not be built"); });
    internals.buildPipeline = vi.fn(() => { throw new Error("pipeline must not be built"); });

    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    await expect(plugin.generatePersonalLibraryDirections()).rejects.toThrow("Authorize");
    expect(call).not.toHaveBeenCalled();
    internals.libraryConnection = fixture().connection;

    let release!: () => void;
    call.mockImplementationOnce(() => new Promise<string>((resolve) => { release = () => resolve(modelResult()); }));
    const first = plugin.generatePersonalLibraryDirections();
    await vi.waitFor(() => expect(call).toHaveBeenCalledTimes(1));
    await expect(plugin.generatePersonalLibraryDirections()).rejects.toThrow("already active");
    release();
    const saved = await first;
    expect(saved.revision).toBe(0);
    // One extraction for the single cluster, then one synthesis across the
    // clusters' combined candidates — and nothing from the daily pipeline.
    expect(call).toHaveBeenCalledTimes(2);
    expect(internals.buildSharedDeps).not.toHaveBeenCalled();
    expect(internals.buildPipeline).not.toHaveBeenCalled();
  });

  it.each([
    ["root", (_plugin: ArxivDailyPlugin, internals: any) => {
      internals.cancelPersonalLibraryOperations("library folder changed");
      internals.libraryConnection = createLibraryConnection("/next", "1:3");
      internals.libraryConnectionRevision += 1;
    }],
    ["output", (_plugin: ArxivDailyPlugin, internals: any) => {
      internals.libraryOutputRevision += 1;
      internals.cancelPersonalLibraryOperations("output paths changed");
    }],
    ["endpoint", (plugin: ArxivDailyPlugin) => plugin.setLlmBaseUrl("https://other.example/v1")],
    ["revocation", (plugin: ArxivDailyPlugin) => plugin.revokeLibraryProcessing()],
  ])("cancels generation on %s changes", async (_name, change) => {
    const { plugin, internals } = fixture();
    let rejectCall!: (reason: unknown) => void;
    vi.spyOn(LlmClient.prototype, "call").mockImplementation((_messages, options) => new Promise((_resolve, reject) => {
      rejectCall = reject;
      options?.signal?.addEventListener("abort", () => reject(options.signal?.reason), { once: true });
    }));
    const generation = plugin.generatePersonalLibraryDirections();
    const rejected = expect(generation).rejects.toBeDefined();
    await vi.waitFor(() => expect(plugin.operations.snapshot()).toHaveLength(1));
    const result = change(plugin, internals);
    if (result instanceof Promise) await result;
    await rejected;
    rejectCall?.("cancelled");
  });

  it("uses selected evidence staleness, tolerating unrelated and path-only catalog changes", async () => {
    const { plugin, internals } = fixture();
    const call = vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    let release!: () => void;
    call.mockImplementationOnce(() => new Promise<string>((resolve) => { release = () => resolve(modelResult()); }));
    const generation = plugin.generatePersonalLibraryDirections();
    await vi.waitFor(() => expect(call).toHaveBeenCalled());
    internals.libraryCatalog.papers["arxiv:2608.00001"].filePaths = ["moved/paper.pdf"];
    const movedFile = internals.libraryCatalog.files["papers/2608.00001.pdf"];
    delete internals.libraryCatalog.files["papers/2608.00001.pdf"];
    movedFile.path = "moved/paper.pdf";
    internals.libraryCatalog.files["moved/paper.pdf"] = movedFile;
    release();
    await expect(generation).resolves.toBeDefined();

    call.mockImplementationOnce(() => new Promise<string>((resolve) => { release = () => resolve(modelResult()); }));
    const stale = plugin.generatePersonalLibraryDirections();
    // The first generation already spent two calls (extraction + synthesis),
    // so the third call is this generation's blocked extraction.
    await vi.waitFor(() => expect(call).toHaveBeenCalledTimes(3));
    internals.libraryCatalog.papers["arxiv:2608.00001"].abstract = "Changed selected evidence";
    release();
    await expect(stale).rejects.toThrow("Selected personal library catalog evidence changed");
  });

  it("makes revocation effective synchronously while the library queue is blocked", async () => {
    const { plugin, internals } = fixture();
    const call = vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    let releaseQueue!: () => void;
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });

    const revoke = plugin.revokeLibraryProcessing();
    expect(plugin.getLibraryConnectionStatus().kind).not.toBe("authorized");
    await expect(plugin.generatePersonalLibraryDirections()).rejects.toThrow("Authorize");
    expect(call).not.toHaveBeenCalled();
    releaseQueue();
    await revoke;
    const persisted = vi.mocked(internals.saveData).mock.calls.at(-1)?.[0];
    expect(persisted.libraryConnection.authorization).toBeUndefined();
  });

  it("restores revoked authorization on persistence failure without clobbering newer state", async () => {
    const { plugin, internals, connection } = fixture();
    internals.saveData.mockRejectedValueOnce(new Error("disk full"));
    await expect(plugin.revokeLibraryProcessing()).rejects.toThrow("disk full");
    expect(internals.libraryConnection).toBe(connection);

    internals.saveData.mockRejectedValueOnce(new Error("disk full again"));
    let releaseQueue!: () => void;
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });
    const revoke = plugin.revokeLibraryProcessing();
    const newer = createLibraryConnection("/newer", "1:9");
    internals.libraryConnection = newer;
    releaseQueue();
    await expect(revoke).resolves.toBeUndefined();
    expect(internals.libraryConnection).toBe(newer);
  });

  it("uses proposal CAS when durable review changes during generation", async () => {
    const { plugin, storage, scopeFingerprint, identificationFingerprint } = fixture();
    vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    const original = await plugin.generatePersonalLibraryDirections();
    let release!: () => void;
    vi.spyOn(LlmClient.prototype, "call").mockImplementationOnce(() =>
      new Promise<string>((resolve) => { release = () => resolve(modelResult()); }));
    const generation = plugin.generatePersonalLibraryDirections();
    await vi.waitFor(() => expect(plugin.operations.snapshot()).toHaveLength(1));
    const store = new PersonalLibraryDirectionProposalStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    );
    const changed = structuredClone(original);
    changed.candidates[0]!.description = "Locally reviewed while generation was active.";
    await store.replace(changed, original.revision);
    release();
    await expect(generation).rejects.toMatchObject({ code: "stale" });
    expect(plugin.getPersonalLibraryDirectionProposal()?.candidates[0]?.description)
      .toBe("Locally reviewed while generation was active.");
  });

  it("serializes two confirmations and preserves both durable results", async () => {
    const { plugin } = fixture();
    vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    const proposal = await plugin.generatePersonalLibraryDirections();
    const [first, second] = proposal.candidates;
    const draft = (candidate: typeof first) => ({
      name: candidate!.name,
      description: candidate!.description,
      discoveryCues: candidate!.discoveryCues,
      representativePaperKeys: candidate!.representatives.map((entry) => entry.paperKey),
    });

    const confirmations = await Promise.all([
      plugin.confirmPersonalLibraryProposalCandidate({
        candidateId: first!.id, directionId: "direction-first", status: "active",
        draft: draft(first), now: new Date("2026-08-03T01:00:00Z"),
      }),
      plugin.confirmPersonalLibraryProposalCandidate({
        candidateId: second!.id, directionId: "direction-second", status: "active",
        draft: draft(second), now: new Date("2026-08-03T01:01:00Z"),
      }),
    ]);

    expect(confirmations[1]!.profile?.directions.map(({ id }) => id)).toEqual([
      "direction-first", "direction-second",
    ]);
    expect(plugin.getPersonalLibraryDirectionProposal()?.candidates).toHaveLength(1);
  });

  it("keeps root transition queued behind an in-flight confirmation", async () => {
    const { plugin, internals, storage } = fixture();
    vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    const proposal = await plugin.generatePersonalLibraryDirections();
    const candidate = proposal.candidates[0]!;
    const atomic = vi.spyOn(storage, "writeTextAtomic");
    let releaseWrite!: () => void;
    atomic.mockImplementationOnce(async () => {
      await new Promise<void>((resolve) => { releaseWrite = resolve; });
    });
    internals.libraryDirectoryPicker = { select: vi.fn().mockResolvedValue({ kind: "selected", path: "/next" }) };
    internals.openLibrarySource = vi.fn().mockResolvedValue({
      canonicalRoot: "/next", rootIdentity: "1:3", inventory: vi.fn(), readBinary: vi.fn(),
    });

    const confirmation = plugin.confirmPersonalLibraryProposalCandidate({
      candidateId: candidate.id, directionId: "direction-before-root", status: "active",
      draft: {
        name: candidate.name, description: candidate.description,
        discoveryCues: candidate.discoveryCues,
        representativePaperKeys: candidate.representatives.map((entry) => entry.paperKey),
      },
    });
    await vi.waitFor(() => expect(atomic).toHaveBeenCalled());
    const rootChange = plugin.selectLibraryRoot();
    expect(internals.libraryConnection.selectedRoot).toBe("/private/library");
    releaseWrite();
    await expect(confirmation).resolves.toBeDefined();
    await expect(rootChange).resolves.toBe("selected");
    expect(internals.libraryConnection.selectedRoot).toBe("/next");
  });

  it("guards confirmation assignment when an output transition is requested", async () => {
    const { plugin, internals, storage } = fixture();
    vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    const proposal = await plugin.generatePersonalLibraryDirections();
    const candidate = proposal.candidates[0]!;
    const atomic = vi.spyOn(storage, "writeTextAtomic");
    let releaseWrite!: () => void;
    atomic.mockImplementationOnce(async () => {
      await new Promise<void>((resolve) => { releaseWrite = resolve; });
    });
    internals.scheduler = { replaceStore: vi.fn(), replaceRunHistory: vi.fn() };
    internals.progress = { setIdle: vi.fn(), setDisabled: vi.fn() };
    const nextOutput = {
      ...structuredClone(plugin.settings.output), dailyDir: "next/daily", papersDir: "next/papers",
    };

    const confirmation = plugin.confirmPersonalLibraryProposalCandidate({
      candidateId: candidate.id, directionId: "direction-old-output", status: "active",
      draft: {
        name: candidate.name, description: candidate.description,
        discoveryCues: candidate.discoveryCues,
        representativePaperKeys: candidate.representatives.map((entry) => entry.paperKey),
      },
    });
    await vi.waitFor(() => expect(atomic).toHaveBeenCalled());
    internals.settings.output = nextOutput;
    const outputChange = plugin.reloadStateStoreForOutputPaths();
    releaseWrite();
    await expect(confirmation).rejects.toThrow("Output paths changed");
    await expect(outputChange).resolves.toBeUndefined();
  });

  it("reloads proposal/profile independently, preserves the healthy document, and allows local review after revoke", async () => {
    const { plugin, internals, files, storage, connection, scopeFingerprint, identificationFingerprint } = fixture();
    vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    const proposal = await plugin.generatePersonalLibraryDirections();
    await plugin.reloadPersonalLibraryProfileDocuments();
    const profileStore = new PersonalLibraryInterestProfileStore(storage, plugin.settings.output, scopeFingerprint, identificationFingerprint);
    const profile = await profileStore.load();
    const proposalStore = new PersonalLibraryDirectionProposalStore(storage, plugin.settings.output, scopeFingerprint, identificationFingerprint);
    files.set(profileStore.paths.documentPath, "{broken");
    files.set(profileStore.paths.backupPath, "{broken");

    const snapshot = await plugin.reloadPersonalLibraryProfileDocuments();
    expect(snapshot.proposal).toEqual(proposal);
    expect(snapshot.profile).toBeNull();
    expect(snapshot.profileLoadError?.message).not.toContain(profileStore.paths.documentPath);
    files.delete(profileStore.paths.documentPath);
    files.delete(profileStore.paths.backupPath);
    await plugin.reloadPersonalLibraryProfileDocuments();
    internals.libraryConnection = connection;
    await plugin.revokeLibraryProcessing();
    const candidate = plugin.getPersonalLibraryDirectionProposal()!.candidates[0]!;
    const confirmed = await plugin.confirmPersonalLibraryProposalCandidate({
      candidateId: candidate.id,
      directionId: "direction-local-review",
      status: "active",
      draft: {
        name: candidate.name, description: candidate.description,
        discoveryCues: candidate.discoveryCues,
        representativePaperKeys: candidate.representatives.map((entry) => entry.paperKey),
      },
      now: new Date("2026-08-03T01:00:00Z"),
    });
    expect(confirmed.authorization.kind).not.toBe("authorized");
    expect(confirmed.profile?.directions[0]?.id).toBe("direction-local-review");
    expect(await proposalStore.load()).toEqual(confirmed.proposal);
    expect(profile.revision).toBe(0);
  });
});

function makeFullTextKnowledgeBase(scopeFingerprint: string, identificationFingerprint: string) {
  const documents = new Map<string, unknown>();
  for (const [index, seed] of [[1, "c"], [2, "e"]] as const) {
    const paperKey = `arxiv:2608.0000${index}`;
    documents.set(paperKey, {
      schemaVersion: 1,
      paperKey,
      modelId: "fake",
      dimension: 8,
      textHash: `sha256:${"d".repeat(64)}`,
      filePaths: [`papers/2608.0000${index}.pdf`],
      observationFingerprints: [`sha256:${seed.repeat(64)}`],
      chunks: [{ index: 0, page: 1, text: `paper ${index}` }],
      // Identical theme chunk: the two papers cluster into one theme.
      vectors: new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]),
      updatedAt: "2026-08-03T00:00:00.000Z",
    });
  }
  return {
    paths: { directory: "kb", manifest: { directory: "kb", documentPath: "kb/manifest.json", backupPath: "kb/m.json.backup" }, papersDirectory: "kb/papers" },
    loadManifest: async () => ({
      schemaVersion: 1,
      revision: 1,
      scopeFingerprint,
      identificationFingerprint,
      modelId: "fake",
      dimension: 8,
      updatedAt: "2026-08-03T00:00:00.000Z",
      papers: {
        "arxiv:2608.00001": { paperKey: "arxiv:2608.00001", status: "ready", modelId: "fake", dimension: 8, textHash: `sha256:${"d".repeat(64)}`, filePaths: ["papers/2608.00001.pdf"], observationFingerprints: [`sha256:${"c".repeat(64)}`], chunkCount: 1, updatedAt: "2026-08-03T00:00:00.000Z" },
        "arxiv:2608.00002": { paperKey: "arxiv:2608.00002", status: "ready", modelId: "fake", dimension: 8, textHash: `sha256:${"d".repeat(64)}`, filePaths: ["papers/2608.00002.pdf"], observationFingerprints: [`sha256:${"e".repeat(64)}`], chunkCount: 1, updatedAt: "2026-08-03T00:00:00.000Z" },
      },
    }),
    loadPaper: async (paperKey: string) => documents.get(paperKey) ?? null,
    replaceManifest: async () => { throw new Error("not used"); },
    savePaper: async () => {},
    removePaper: async () => {},
    removeAll: async () => {},
  };
}
