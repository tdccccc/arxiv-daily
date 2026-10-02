import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  Logger,
  OperationRegistry,
  PersonalLibraryDirectionProposalStore,
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseStore,
  type FullTextPaperDocument,
  type HttpClient,
  type PersonalLibraryCatalog,
  type StorageAdapter,
} from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import { authorizeLibraryConnection, createLibraryConnection } from "../src/library/connection";

/**
 * The guard that refuses to finish a generation whose evidence changed
 * underneath it. It fingerprints what the run is based on, so it has to
 * describe every library the run can start from — including one whose files
 * carry no arXiv identity, where the catalog is empty and the index holds
 * everything. Describing that as "no selection at all" made the guard throw
 * before generation began.
 */
const scopeFingerprint = createPersonalLibraryScopeFingerprint({
  rootIdentity: "1:2",
  eligibleExtensions: [".pdf"],
});
const identificationFingerprint = createPersonalLibraryIdentificationFingerprint([".pdf"]);

function emptyCatalog(): PersonalLibraryCatalog {
  return createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint);
}

function pluginWith(indexedPapers: Array<{ paperKey: string; title: string }>) {
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  Object.assign(plugin, { libraryIndexedPapers: indexedPapers });
  return plugin as unknown as {
    selectedCatalogFingerprint(catalog: PersonalLibraryCatalog): string;
  };
}

const paper = (index: number) => ({
  paperKey: `file:sha256:${String(index).padStart(64, "0")}`,
  title: `Local Paper ${index}`,
});

describe("generation evidence guard", () => {
  it("describes a library whose catalog is empty and whose index is not", () => {
    const plugin = pluginWith([paper(1), paper(2)]);
    expect(() => plugin.selectedCatalogFingerprint(emptyCatalog())).not.toThrow();
    expect(plugin.selectedCatalogFingerprint(emptyCatalog())).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it("is stable for the same evidence", () => {
    const first = pluginWith([paper(1), paper(2)]).selectedCatalogFingerprint(emptyCatalog());
    const second = pluginWith([paper(1), paper(2)]).selectedCatalogFingerprint(emptyCatalog());
    expect(first).toBe(second);
  });

  it("changes when an indexed paper joins, leaves, or is renamed", () => {
    const base = pluginWith([paper(1), paper(2)]).selectedCatalogFingerprint(emptyCatalog());
    // Without covering the index the guard would sleep through exactly the
    // evidence a non-arXiv library runs on.
    expect(pluginWith([paper(1), paper(2), paper(3)]).selectedCatalogFingerprint(emptyCatalog()))
      .not.toBe(base);
    expect(pluginWith([paper(1)]).selectedCatalogFingerprint(emptyCatalog())).not.toBe(base);
    expect(pluginWith([{ ...paper(1), title: "Retitled" }, paper(2)])
      .selectedCatalogFingerprint(emptyCatalog())).not.toBe(base);
  });

  it("distinguishes an empty library from one with indexed papers", () => {
    expect(pluginWith([]).selectedCatalogFingerprint(emptyCatalog()))
      .not.toBe(pluginWith([paper(1)]).selectedCatalogFingerprint(emptyCatalog()));
  });
});

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

function memoryStorage(): StorageAdapter {
  const files = new Map<string, string>();
  const directories = new Set<string>();
  return {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/"),
    readText: async (path) => {
      const text = files.get(path);
      if (text === undefined) throw new Error(`missing ${path}`);
      return text;
    },
    writeText: async (path, text) => { files.set(path, text); },
    writeTextAtomic: async (path, text) => { files.set(path, text); },
    exists: async (path) => files.has(path) || directories.has(path),
    mkdir: async (path) => { directories.add(path); },
    remove: async (path) => { files.delete(path); directories.delete(path); },
    rename: async (from, to) => {
      const text = files.get(from);
      if (text === undefined) throw new Error(`missing ${from}`);
      files.set(to, text);
      files.delete(from);
    },
  };
}

/** Four indexed PDFs form two evidence groups through the real clusterer. */
function indexedKnowledgeBase(): FullTextKnowledgeBaseStore {
  const documents = new Map<string, FullTextPaperDocument>();
  const manifest: FullTextKnowledgeBaseManifest = {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision: 1, scopeFingerprint, identificationFingerprint,
    modelId: "fixture-model", dimension: 2, updatedAt: "2026-09-07T00:00:00.000Z", papers: {},
  };
  for (const index of [1, 2, 3, 4]) {
    const indexed = paper(index);
    const textHash = `sha256:${String(index).padStart(64, "0")}`;
    const content = {
      paperKey: indexed.paperKey, modelId: "fixture-model", dimension: 2,
      textHash, contentHash: textHash, title: indexed.title, abstract: `Abstract for local paper ${index}`,
      filePaths: [`local-${index}.pdf`], observationFingerprints: [textHash],
      updatedAt: "2026-09-07T00:00:00.000Z",
    };
    documents.set(indexed.paperKey, {
      ...content, schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      chunks: [{ index: 0, page: 1, text: content.abstract }],
      vectors: new Float32Array(index <= 2 ? [1, 0] : [0, 1]),
    });
    manifest.papers[indexed.paperKey] = { ...content, status: "ready", chunkCount: 1 };
  }
  return {
    paths: {
      directory: "kb", papersDirectory: "kb/papers",
      manifest: { directory: "kb", documentPath: "kb/manifest.json", backupPath: "kb/manifest.backup" },
    },
    loadManifest: async () => structuredClone(manifest),
    loadPaper: async (paperKey) => structuredClone(documents.get(paperKey) ?? null),
    replaceManifest: async () => { throw new Error("unexpected knowledge base write"); },
    savePaper: async () => { throw new Error("unexpected knowledge base write"); },
    removePaper: async () => { throw new Error("unexpected knowledge base removal"); },
    removeAll: async () => { throw new Error("unexpected knowledge base removal"); },
  };
}

interface OrganizationRequest {
  existingTopics: Array<{ id: string; name: string; directions: Array<{ id: string; text: string }> }>;
  groups: Array<{ id: string; papers: Array<{ paperKey: string }> }>;
}

function generationPlugin(pauseResponse?: () => Promise<void>) {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.llm.baseUrl = "https://models.example/v1";
  settings.llm.model = "fixture-model";
  settings.embedding.mode = "local";
  settings.arxiv.topics = [{
    id: "existing-topic", name: "Research already followed", tag: "existing-topic",
    description: "First research direction", detail: false,
    directions: [
      { id: "first-direction", text: "First research direction", origin: "manual" },
      { id: "second-direction", text: "Second research direction", origin: "library" },
    ],
  }];
  const connection = authorizeLibraryConnection(
    createLibraryConnection("/private/library", "1:2"),
    { llmBaseUrl: settings.llm.baseUrl },
  );
  const requests: OrganizationRequest[] = [];
  const http: HttpClient = {
    request: async (request) => {
      if (typeof request.body !== "string") throw new Error("expected a JSON chat request");
      const body = JSON.parse(request.body) as { messages: Array<{ role: string; content: string }> };
      const user = body.messages.find(({ role }) => role === "user")?.content ?? "";
      const data = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(user);
      if (!data) throw new Error("organization evidence was missing from the real LLM request");
      const parsed = JSON.parse(data[1]!) as OrganizationRequest;
      requests.push(parsed);
      await pauseResponse?.();
      const content = JSON.stringify({
        topics: parsed.groups.map((group, index) => ({
          suggestedName: `Generated field ${index + 1}`,
          directions: [{
            text: `Generated research direction ${index + 1}`, discoveryCues: [`Evidence cue ${index + 1}`],
            groupIds: [group.id], representativePaperKeys: [group.papers[0]!.paperKey],
          }],
        })),
        coveredGroups: [],
      });
      return {
        status: 200, headers: { "content-type": "text/event-stream" },
        bodyText: `data: ${JSON.stringify({ choices: [{ delta: { content }, finish_reason: null }] })}\n\n`
          + `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`
          + "data: [DONE]\n\n",
      };
    },
  };
  const storage = memoryStorage();
  const store = new PersonalLibraryDirectionProposalStore(
    storage, settings.output, scopeFingerprint, identificationFingerprint,
  );
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  Object.assign(plugin, {
    settings, logger: new Logger("error"), host: { storage, http, markupParser: {} },
    progress: { setTask: () => undefined, setComplete: () => undefined, setError: () => undefined },
    operations: new OperationRegistry(), libraryConnection: connection, libraryCatalog: emptyCatalog(),
    libraryProposal: null, libraryConnectionRevision: 0, libraryOutputRevision: 0,
    libraryMutationQueue: Promise.resolve(), libraryIndexedPapers: [1, 2, 3, 4].map(paper),
    buildPersonalLibraryProfileStores: () => ({ proposal: store }),
    buildFullTextKnowledgeBaseStore: () => indexedKnowledgeBase(),
  });
  return { plugin, settings, requests, store };
}

describe("direction generation uses current research settings", () => {
  it("does not start a generation after the review caller has closed", async () => {
    const { plugin, requests, store } = generationPlugin();
    const caller = new AbortController();
    caller.abort("review closed");
    await expect(plugin.generatePersonalLibraryDirections(undefined, caller.signal)).rejects.toThrow();
    expect(requests).toHaveLength(0);
    expect(await store.load()).toBeNull();
    expect(plugin.operations.snapshot()).toEqual([]);
  });

  it("cancels only the caller's generation and does not save its late response", async () => {
    const response = deferred();
    const { plugin, requests, store } = generationPlugin(() => response.promise);
    const caller = new AbortController();
    const other = plugin.operations.begin("paper-note", "Unrelated operation", "another-paper");
    const generating = plugin.generatePersonalLibraryDirections(undefined, caller.signal).then(
      () => "completed", () => "cancelled",
    );
    try {
      await vi.waitFor(() => expect(requests).toHaveLength(1));
      caller.abort("review closed");
      response.resolve();
      expect(await generating).toBe("cancelled");
      expect(await store.load()).toBeNull();
      expect(other.signal.aborted).toBe(false);
    } finally {
      response.resolve();
      await generating;
      other.finish();
    }
    expect(plugin.operations.snapshot()).toEqual([]);
  });

  it("opens reviewed evidence through the indexed PDF path and refuses unknown papers", async () => {
    const { plugin } = generationPlugin();
    const paths: Array<{ paperKey: string; filePath: string }> = [];
    plugin.openPersonalLibraryFullTextEvidence = async (input) => { paths.push(input); return "file-fallback"; };
    await plugin.openPersonalLibraryReviewPaper(paper(1).paperKey);
    expect(paths).toEqual([{ paperKey: paper(1).paperKey, filePath: "local-1.pdf" }]);
    await expect(plugin.openPersonalLibraryReviewPaper("file:sha256:" + "f".repeat(64))).rejects.toThrow(/available|index/i);
    expect(paths).toHaveLength(1);
  });

  it.each([false, true])("previews local evidence without writes and rejects category changes: %s", async (changeCategories) => {
    const { plugin, store } = generationPlugin();
    const proposal = await plugin.generatePersonalLibraryDirections();
    const before = JSON.stringify(plugin.settings);
    const candidate = proposal.topics[0]!.directions[0]!;
    const http: HttpClient = { request: async (request) => {
      const body = JSON.parse(String(request.body));
      expect(body.messages[0].content).toContain("Previewed research methods");
      const ids = [...body.messages[1].content.matchAll(/^ID: (sample-\d+)$/gm)].map((match: RegExpMatchArray) => match[1]);
      expect(ids.length).toBeGreaterThan(0);
      if (changeCategories) plugin.settings.arxiv.categories = ["cs.LG"];
      const content = JSON.stringify({ papers: ids.map((id, index) => ({
        id, category: index === 0 ? "preview" : "skip", directions: index === 0 ? ["preview#1"] : [], relevanceScore: index === 0 ? 90 : 0,
      })) });
      return { status: 200, headers: { "content-type": "text/event-stream" }, bodyText:
        `data: ${JSON.stringify({ choices: [{ delta: { content }, finish_reason: null }] })}\n\n`
        + `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n` + "data: [DONE]\n\n" };
    } };
    Object.assign(plugin, { host: { http } });
    const preview = plugin.previewPersonalLibraryDirection({ candidateId: candidate.id, text: "Previewed research methods" });
    if (changeCategories) {
      await expect(preview).rejects.toMatchObject({ code: "conflict" });
      expect(await store.load()).toEqual(proposal);
      return;
    }
    const result = await preview;
    expect(result.papers.filter(({ matched }) => matched)).toHaveLength(1);
    expect(result.papers[0]!.paperKey).toMatch(/^file:sha256:/);
    expect(JSON.stringify(plugin.settings)).toBe(before);
    expect(await store.load()).toEqual(proposal);
  });

  it("persists edits to directions generated from non-arXiv PDFs through the actual review controller", async () => {
    const { plugin, store } = generationPlugin();
    const proposal = await plugin.generatePersonalLibraryDirections();
    const candidate = proposal.topics[0]!.directions[0]!;
    const representativePaperKeys = candidate.representatives.map(({ paperKey }) => paperKey);
    expect(representativePaperKeys[0]).toMatch(/^file:sha256:/);
    const result = await plugin.updatePersonalLibraryProposalCandidate({
      candidateId: candidate.id, patch: { text: "Reviewed local research direction" }, representativePaperKeys,
    });
    expect(result.proposal!.topics[0]!.directions[0]!.text).toBe("Reviewed local research direction");
    expect((await store.load())!.topics[0]!.directions[0]!.representatives).toEqual(candidate.representatives);
    expect((await store.load())!.topics[0]!.directions[0]!.text).toBe("Reviewed local research direction");
  });

  it("sends every existing topic and direction identity with its reviewed text to the actual LLM request", async () => {
    const { plugin, requests, store } = generationPlugin();
    const proposal = await plugin.generatePersonalLibraryDirections();
    expect(await store.load()).toEqual(proposal);
    expect(requests).toHaveLength(1);
    expect(requests[0]!.existingTopics).toEqual([{
      id: "existing-topic", name: "Research already followed",
      directions: [
        { id: "first-direction", text: "First research direction" },
        { id: "second-direction", text: "Second research direction" },
      ],
    }]);
  });

  it("does not persist a generation when a non-first direction changes while the LLM is running", async () => {
    const response = deferred();
    const { plugin, settings, requests, store } = generationPlugin(() => response.promise);
    const generating = plugin.generatePersonalLibraryDirections().then(
      (proposal) => ({ kind: "completed" as const, proposal }),
      (error: unknown) => ({ kind: "rejected" as const, error }),
    );
    try {
      await vi.waitFor(() => expect(requests).toHaveLength(1));
      settings.arxiv.topics[0]!.directions[1]!.text = "Research direction edited during generation";
      response.resolve();
      const result = await generating;

      expect(await store.load()).toBeNull();
      expect(result).toMatchObject({ kind: "rejected", error: expect.any(Error) });
      expect(settings.arxiv.topics[0]!.directions[1]!.text).toBe("Research direction edited during generation");
    } finally {
      response.resolve();
      await generating;
    }
  });
});
