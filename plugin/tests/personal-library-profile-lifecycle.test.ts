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
  const connection = authorizeLibraryConnection(rawConnection, settings.llm.baseUrl, new Date("2026-08-03T00:00:00Z"));
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
    schemaVersion: 2,
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
  it("builds an immutable authorized eligible daily discovery snapshot without private catalog data", () => {
    const { plugin, internals, catalog } = fixture();
    internals.libraryProfile = confirmedProfile(catalog);
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};

    const pipeline = internals.buildPipeline();
    const deps = pipeline.deps;
    expect(deps.personalizedDiscovery).toEqual({ directions: [{
      id: "direction-confirmed",
      name: "Confirmed reliable agents",
      description: "Researcher-reviewed reliable agent methods.",
      discoveryCues: ["agent evaluation"],
      representatives: [{
        paperKey: "arxiv:2608.00001",
        title: "Reliable agents",
        evidenceDepth: "metadata-and-abstract",
      }],
    }] });
    expect(Object.isFrozen(deps.personalizedDiscovery)).toBe(true);
    const serialized = JSON.stringify(deps.personalizedDiscovery);
    expect(serialized).not.toContain("/private/library");
    expect(serialized).not.toContain("papers/2608.00001.pdf");
    expect(serialized).not.toContain("Evidence");
    expect(serialized).not.toContain("sha256:");
    expect(serialized).not.toContain("secret");
    expect(serialized).not.toContain("authorization");

    internals.libraryProfile.directions[0].name = "Mutated later";
    internals.libraryCatalog.papers["arxiv:2608.00001"].title = "Mutated later";
    plugin.settings.arxiv.topics.push({ tag: "later", description: "later" });
    expect(deps.personalizedDiscovery.directions[0].name).toBe("Confirmed reliable agents");
    expect(deps.personalizedDiscovery.directions[0].representatives[0].title).toBe("Reliable agents");
    expect(deps.arxiv.topics).toEqual([]);
  });

  it.each([
    ["no connection", (internals: any) => { internals.libraryConnection = undefined; }],
    ["revoked", (internals: any) => { internals.libraryConnection = createLibraryConnection("/private/library", "1:2"); }],
    ["catalog load error", (internals: any) => { internals.libraryCatalogLoadError = { kind: "catalog", code: "broken", message: "broken" }; }],
    ["profile load error", (internals: any) => { internals.libraryProfileLoadError = { kind: "profile", code: "broken", message: "broken" }; }],
    ["stale representative", (internals: any) => { internals.libraryCatalog.papers["arxiv:2608.00001"].title = "stale"; }],
    ["empty profile", (internals: any) => { internals.libraryProfile.directions = []; }],
  ])("degrades %s to a successful manual-only pipeline build", (_name, invalidate) => {
    const { plugin, internals, catalog } = fixture();
    internals.libraryProfile = confirmedProfile(catalog);
    invalidate(internals);
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    expect(internals.buildPipeline().deps.personalizedDiscovery).toBeUndefined();
  });

  it("rebuilds fresh snapshots at the shared daily convergence seam", () => {
    const { internals, catalog } = fixture();
    internals.libraryProfile = confirmedProfile(catalog);
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    const first = internals.buildPipeline();
    internals.libraryProfile.directions[0].name = "Fresh confirmed name";
    const second = internals.buildPipeline();
    expect(first.deps.personalizedDiscovery.directions[0].name).toBe("Confirmed reliable agents");
    expect(second.deps.personalizedDiscovery.directions[0].name).toBe("Fresh confirmed name");
  });

  it("does not reopen authorization across a newer queued root or output transition", async () => {
    const { plugin, internals } = fixture();
    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    let releaseQueue!: () => void;
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });
    const authorize = plugin.authorizeLibraryProcessing();
    internals.markPersonalizedDailyDiscoveryUnavailable("newer root transition");
    const newerRevision = internals.personalizedDailyDiscoveryRevision;
    releaseQueue();
    await expect(authorize).rejects.toThrow("superseded");
    expect(internals.personalizedDailyDiscoveryRevision).toBe(newerRevision);
    expect(internals.personalizedDailyDiscoveryAvailable).toBe(false);

    internals.libraryConnection = createLibraryConnection("/private/library", "1:2");
    await plugin.authorizeLibraryProcessing();
    internals.personalizedDailyDiscoveryAvailable = false;
    internals.libraryMutationQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });
    const second = plugin.authorizeLibraryProcessing();
    internals.markPersonalizedDailyDiscoveryUnavailable("newer output transition");
    releaseQueue();
    await expect(second).rejects.toThrow("superseded");
    expect(internals.personalizedDailyDiscoveryAvailable).toBe(false);
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

  it("proposal-only mutations do not cancel captured personalized daily discovery", async () => {
    const { plugin, internals, catalog } = fixture();
    internals.libraryProfile = confirmedProfile(catalog);
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    const proposal = await plugin.generatePersonalLibraryDirections();
    const pipeline = internals.buildPipeline();
    const candidate = proposal.candidates[0]!;
    await plugin.updatePersonalLibraryProposalCandidate({
      candidateId: candidate.id,
      patch: { name: "Proposal-only edit" },
    });
    expect(pipeline.deps.personalizedDiscoverySignal.aborted).toBe(false);
  });

  it.each([
    ["disable", (plugin: ArxivDailyPlugin) => plugin.disablePersonalLibraryConfirmedDirection("direction-confirmed")],
    ["update", (plugin: ArxivDailyPlugin) => plugin.updatePersonalLibraryConfirmedDirection({
      directionId: "direction-confirmed", patch: { name: "Updated direction" },
    })],
    ["remove", (plugin: ArxivDailyPlugin) => plugin.removePersonalLibraryConfirmedDirection({
      directionId: "direction-confirmed", mode: "restrict",
    })],
  ])("aborts captured personalized discovery before committed %s installs", async (_name, mutate) => {
    const { plugin, internals, catalog, storage, scopeFingerprint, identificationFingerprint } = fixture();
    internals.libraryProfile = confirmedProfile(catalog);
    const store = new PersonalLibraryInterestProfileStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    );
    await store.replace(internals.libraryProfile, 0);
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    const pipeline = internals.buildPipeline();
    const mutation = mutate(plugin);
    expect(pipeline.deps.personalizedDiscoverySignal.aborted).toBe(true);
    await mutation;
  });

  it("aborts captured personalized discovery before confirmation installs a new profile", async () => {
    const { plugin, internals } = fixture();
    vi.spyOn(LlmClient.prototype, "call").mockResolvedValue(modelResult());
    const proposal = await plugin.generatePersonalLibraryDirections();
    const first = proposal.candidates[0]!;
    await plugin.confirmPersonalLibraryProposalCandidate({
      candidateId: first.id, directionId: "direction-first", status: "active",
      draft: {
        name: first.name, description: first.description, discoveryCues: first.discoveryCues,
        representativePaperKeys: first.representatives.map(({ paperKey }) => paperKey),
      },
    });
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    const pipeline = internals.buildPipeline();
    const second = plugin.getPersonalLibraryDirectionProposal()!.candidates[0]!;
    const confirmation = plugin.confirmPersonalLibraryProposalCandidate({
      candidateId: second.id, directionId: "direction-second", status: "active",
      draft: {
        name: second.name, description: second.description, discoveryCues: second.discoveryCues,
        representativePaperKeys: second.representatives.map(({ paperKey }) => paperKey),
      },
    });
    expect(pipeline.deps.personalizedDiscoverySignal.aborted).toBe(true);
    await confirmation;
  });

  it("profile reload invalidates captured personalized discovery before installing durable state", async () => {
    const { plugin, internals, catalog, storage, scopeFingerprint, identificationFingerprint } = fixture();
    internals.libraryProfile = confirmedProfile(catalog);
    await new PersonalLibraryInterestProfileStore(
      storage, plugin.settings.output, scopeFingerprint, identificationFingerprint,
    ).replace(internals.libraryProfile, 0);
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    const pipeline = internals.buildPipeline();
    const reload = plugin.reloadPersonalLibraryProfileDocuments();
    expect(pipeline.deps.personalizedDiscoverySignal.aborted).toBe(true);
    await reload;
  });

  it("cancels only personalized daily runs on revocation, not scans or review/detail work", async () => {
    const { plugin, internals, catalog } = fixture();
    internals.libraryProfile = confirmedProfile(catalog);
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    const pipeline = internals.buildPipeline();
    const scan = plugin.operations.begin("personal-library-scan", "scan");
    const review = plugin.operations.begin("personal-library-direction-generation", "review");
    const detail = plugin.operations.begin("detail-summary", "detail");

    await plugin.revokeLibraryProcessing();

    expect(pipeline.deps.personalizedDiscoverySignal.aborted).toBe(true);
    expect(scan.signal.aborted).toBe(false);
    expect(review.signal.aborted).toBe(true);
    expect(detail.signal.aborted).toBe(false);
    scan.finish(); review.finish(); detail.finish();
  });

  it("excludes proposals and unrelated catalog papers from daily discovery", () => {
    const { plugin, internals, catalog } = fixture();
    internals.libraryProfile = confirmedProfile(catalog);
    internals.libraryProposal = { candidates: [{ name: "RAW PROPOSAL" }] };
    internals.libraryCatalog.papers["arxiv:2608.99999"] = {
      ...structuredClone(catalog.papers["arxiv:2608.00001"]),
      paperKey: "arxiv:2608.99999", externalId: "2608.99999", title: "UNRELATED CATALOG PAPER",
      filePaths: ["private/unrelated.pdf"],
    };
    internals.libraryCatalog.files["private/unrelated.pdf"] = {
      path: "private/unrelated.pdf", status: "ready",
      observationFingerprint: `sha256:${"d".repeat(64)}`,
      paperKey: "arxiv:2608.99999", arxivId: "2608.99999",
      updatedAt: "2026-08-03T00:00:00.000Z",
    };
    internals.buildSharedDeps = vi.fn(() => ({ llm: {}, fetcher: {}, paperFetcher: {}, writer: {} }));
    internals.buildPaperIndex = vi.fn(() => ({}));
    internals.host.markupParser = {};
    const serialized = JSON.stringify(internals.buildPipeline().deps.personalizedDiscovery);
    expect(serialized).not.toContain("RAW PROPOSAL");
    expect(serialized).not.toContain("UNRELATED CATALOG PAPER");
    expect(serialized).not.toContain("private/unrelated.pdf");
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
