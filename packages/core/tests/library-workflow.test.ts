import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import type { DocumentParser } from "../src/documents/parsed-document";
import type { EmbeddingModel } from "../src/library/fulltext/ports";
import { FullTextGenerationIndexStore } from "../src/library/fulltext/generation-index-store";
import { FullTextKnowledgeBaseFileStore } from "../src/library/fulltext/knowledge-base-store";
import { LibraryWorkflow, type LibraryTopicSettingsSnapshot, type LibraryWorkflowOptions } from "../src/library/library-workflow";
import { createLibraryConnection } from "../src/library/library-connection";
import { decodePersonalLibraryCatalog, PersonalLibraryCatalogStore } from "../src/library/personal-library-catalog";
import { decodePersonalLibraryDirectionProposal } from "../src/library/personal-library-interest-profile";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { PDF_IDENTIFICATION_HEAD_BYTES, PDF_IDENTIFICATION_TAIL_BYTES } from "../src/library/pdf-library-file-identifier";

function memoryStorage() {
  const text = new Map<string, string>();
  const binary = new Map<string, Uint8Array>();
  const dirs = new Set<string>();
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    readText: async (path) => { if (!text.has(path)) throw new Error(`missing ${path}`); return text.get(path)!; },
    writeText: async (path, value) => { text.set(path, value); },
    writeTextAtomic: vi.fn(async (path, value) => { text.set(path, value); }),
    createTextExclusive: async (path, value) => {
      if (await storage.exists(path)) return false;
      text.set(path, value); return true;
    },
    exists: async (path) => text.has(path) || binary.has(path) || dirs.has(path),
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => {
      for (const map of [text, binary, dirs]) {
        for (const key of map.keys()) if (key === path || key.startsWith(`${path}/`)) map.delete(key);
      }
    },
    rename: async (from, to) => { text.set(to, await storage.readText(from)); text.delete(from); },
    writeBinary: async (path, bytes) => { binary.set(path, new Uint8Array(bytes).slice()); },
    readBinary: async (path) => { if (!binary.has(path)) throw new Error(`missing ${path}`); return binary.get(path)!.slice().buffer; },
    list: async (directory) => {
      const entries = new Map<string, "file" | "folder">();
      for (const path of [...text.keys(), ...binary.keys(), ...dirs]) {
        if (!path.startsWith(`${directory}/`)) continue;
        const [head, ...rest] = path.slice(directory.length + 1).split("/");
        if (head) entries.set(`${directory}/${head}`, rest.length || dirs.has(`${directory}/${head}`) ? "folder" : "file");
      }
      return [...entries].map(([path, type]) => ({ path, type }));
    },
  };
  return { storage, text, binary };
}

function fixture(overrides: Partial<LibraryWorkflowOptions> = {}) {
  const memory = memoryStorage();
  const documents = new Map(Array.from({ length: 6 }, (_, i) => [
    `2610.${String(i + 1).padStart(5, "0")}.pdf`, new TextEncoder().encode(`${i < 3 ? "agents" : "vision"} paper ${i + 1}: meaningful experimental evidence. `.repeat(30)),
  ]));
  let id = 0;
  const parser: DocumentParser = {
    capabilities: ["page-text"], provenance: { id: "fixture", version: "1" },
    parse: vi.fn(async (bytes) => ({ mediaType: "application/pdf", blocks: [{ kind: "page", text: new TextDecoder().decode(bytes), locator: { page: 1, block: 0 } }] })),
  };
  const embedding: EmbeddingModel = {
    modelId: "fixture", dimension: 2, prefixPolicy: "none",
    embed: vi.fn(async (texts) => texts.map((text) => new Float32Array(text.includes("agents") ? [1, 0] : [0, 1]))),
  };
  let topicState: LibraryTopicSettingsSnapshot = { topics: [], acceptances: [] };
  const options: LibraryWorkflowOptions = {
    topicSettings: { read: async () => structuredClone(topicState), change: async compute => { topicState = structuredClone(await compute(structuredClone(topicState))); } },
    storage: memory.storage, output: DEFAULT_SETTINGS.output,
    connection: createLibraryConnection("/readonly/papers", "1:42"),
    source: {
      inventory: vi.fn(async () => ({ entries: [...documents].map(([path, bytes]) => ({ path, type: "file" as const, size: bytes.length, mtimeMs: 1 })), truncated: false })),
      readBinary: vi.fn(async (path, opts) => documents.get(path)!.slice(opts?.start ?? 0, opts?.end).buffer),
    },
    http: { request: vi.fn(async () => { throw new Error("unexpected HTTP"); }) },
    arxivFetcher: { fetchMetadataByIds: vi.fn(async (ids) => new Map(ids.map((key) => [key, {
      id: key, title: `Paper ${key}`, authors: "Researcher", authorNames: ["Researcher"], abstract: `Evidence for ${key}`,
      published: "2026-10-01T00:00:00Z", updated: "2026-10-01T00:00:00Z", primaryCategory: "cs.AI", categories: ["cs.AI"],
    }]))) },
    createParser: vi.fn(async () => ({ parser })), createEmbedding: vi.fn(async () => embedding),
    llm: { call: vi.fn(async (messages) => {
      const raw = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(messages.find((m) => m.role === "user")!.content)![1]!;
      const parsed = JSON.parse(raw) as { groups: {id: string; papers: {paperKey: string}[]}[] };
      return JSON.stringify({ topics: parsed.groups.map((group, index) => ({
        suggestedName: `Research topic ${index + 1}`,
        directions: [{ text: `Research direction ${index + 1}`, discoveryCues: ["methods", "evaluation"], groupIds: [group.id], representativePaperKeys: group.papers.slice(0, 3).map(p => p.paperKey) }],
      })) });
    }) },
    assertCurrent: vi.fn(), assertAuthorized: vi.fn(), createId: () => `workflow-id-${++id}`,
    now: () => new Date("2026-10-01T12:00:00.000Z"), ...overrides,
  };
  return { ...memory, documents, parser, embedding, options, workflow: new LibraryWorkflow(options) };
}

async function prepareDirections() {
  const f = fixture();
  await f.workflow.scan();
  await f.workflow.index();
  const proposal = await f.workflow.propose();
  const candidate = proposal.topics[0]!.directions[0]!;
  const confirmInput = { candidateId: candidate.id, status: "active" as const, draft: {
    text: candidate.text, discoveryCues: candidate.discoveryCues,
    representativePaperKeys: candidate.representatives.map((p) => p.paperKey),
  } };
  return { ...f, proposal, candidate, confirmInput };
}

describe("shared LibraryWorkflow", () => {
  it("scans, indexes and accepts proposals into normal topic settings with a durable receipt", async () => {
    const f = await prepareDirections();
    const before = [...f.documents].map(([path, bytes]) => [path, [...bytes]]);
    const initial = await f.workflow.review();
    expect(decodePersonalLibraryCatalog(initial.catalog)).not.toBeNull();
    expect(initial.catalog.lastScan?.papers).toBe(6);
    expect(decodePersonalLibraryDirectionProposal(f.proposal)).not.toBeNull();
    expect(f.proposal.topics).toHaveLength(2);
    expect(initial.topics).toHaveLength(0);
    const confirmed = await f.workflow.confirm(f.confirmInput);
    expect(confirmed.topics).toHaveLength(1);
    expect(confirmed.topics[0]!.directions[0]!.text).toBe(f.candidate.text);
    expect(confirmed.acceptances[0]!.processedCandidateIds).toContain(f.candidate.id);
    expect((await new LibraryWorkflow(f.options).review()).topics).toEqual(confirmed.topics);
    expect([...f.documents].map(([path, bytes]) => [path, [...bytes]])).toEqual(before);
    expect([...f.text.keys()].some(path => path.includes("interest-profile.json"))).toBe(false);
    const generationStore = new FullTextGenerationIndexStore(f.storage, f.options.output, initial.catalog.scopeFingerprint, initial.catalog.identificationFingerprint);
    const current = await generationStore.openCurrent();
    expect(current).not.toBeNull();
    await current?.close();
  });

  it("reuses unchanged PDFs and never calls direction generation during indexing", async () => {
    const f = fixture();
    await f.workflow.scan(); await f.workflow.index();
    const count = vi.mocked(f.parser.parse).mock.calls.length;
    expect(await f.workflow.index()).toMatchObject({ indexed: 0, reused: 6 });
    expect(f.parser.parse).toHaveBeenCalledTimes(count);
    expect(f.options.llm.call).not.toHaveBeenCalled();
    expect(vi.mocked(f.parser.parse).mock.calls[0]![1]).toMatchObject({ maxPages: 2 });
  });

  it("requires model authorization before proposals and remote embedding, while local indexing remains available", async () => {
    const denied = vi.fn(() => { throw new Error("authorization required"); });
    const f = fixture({ assertAuthorized: denied });
    await f.workflow.scan(); await f.workflow.index();
    await expect(f.workflow.propose()).rejects.toThrow("authorization required");
    expect(f.options.llm.call).not.toHaveBeenCalled();

    const remote = new LibraryWorkflow({ ...f.options, embeddingRequiresAuthorization: true });
    const calls = vi.mocked(f.options.createEmbedding).mock.calls.length;
    await expect(remote.index()).rejects.toThrow("authorization required");
    expect(f.options.createEmbedding).toHaveBeenCalledTimes(calls);
  });

  it("does not persist a scan after its host connection becomes stale", async () => {
    const f = fixture();
    vi.mocked(f.options.arxivFetcher.fetchMetadataByIds).mockImplementationOnce(async () => {
      vi.mocked(f.options.assertCurrent).mockImplementation(() => { throw new Error("connection changed"); });
      return new Map();
    });
    await expect(f.workflow.scan()).rejects.toThrow("connection changed");
    expect(f.storage.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("retains per-paper failures and supports the existing migration fallback store", async () => {
    const f = fixture();
    delete f.storage.writeBinary;
    vi.mocked(f.parser.parse).mockRejectedValueOnce(new Error("bad PDF"));
    await f.workflow.scan();
    expect(await f.workflow.index()).toMatchObject({ indexed: 5, failed: 1 });
    const { catalog } = await f.workflow.review();
    const kb = new FullTextKnowledgeBaseFileStore(f.storage, f.options.output, catalog.scopeFingerprint, catalog.identificationFingerprint);
    expect(Object.values((await kb.loadManifest()).papers).filter((p) => p.status === "failed")).toHaveLength(1);
  });

  it("identifies a renamed large PDF with bounded reads and the independent title witness", async () => {
    const f = fixture();
    f.documents.clear();
    const pdf = new Uint8Array(10 * 1024 * 1024);
    pdf.set(new TextEncoder().encode("%PDF-1.4\n1 0 obj\n<< /arXivID (https://arxiv.org/abs/2610.00001v1) /Title (Reliable Scientific Agents With Evidence) >>\nendobj\n"));
    f.documents.set("renamed.pdf", pdf);
    vi.mocked(f.options.http.request).mockResolvedValue({ status: 200, headers: {}, bodyText: "<feed><entry><id>https://arxiv.org/abs/2610.00002</id><title>Reliable Scientific Agents With Evidence</title></entry></feed>" });
    const catalog = await f.workflow.scan();
    expect(catalog.files["renamed.pdf"]).toMatchObject({ status: "ready", arxivId: "2610.00002" });
    expect(f.options.source.readBinary).toHaveBeenCalledTimes(2);
    expect(f.options.source.readBinary).toHaveBeenCalledWith("renamed.pdf", expect.objectContaining({ start: 0, end: PDF_IDENTIFICATION_HEAD_BYTES }));
    expect(f.options.source.readBinary).toHaveBeenCalledWith("renamed.pdf", expect.objectContaining({ start: pdf.length - PDF_IDENTIFICATION_TAIL_BYTES, end: pdf.length }));
    expect(f.parser.parse).not.toHaveBeenCalled();
  });

  it("preserves the committed title-and-abstract index when generation synchronization fails", async () => {
    const f = fixture();
    await f.workflow.scan();
    f.storage.writeBinary = async () => { throw new Error("generation disk failure"); };
    await expect(f.workflow.index()).rejects.toThrow("failed to spool or verify generation object");
    const review = await f.workflow.review();
    const kb = new FullTextKnowledgeBaseFileStore(f.storage, f.options.output, review.catalog.scopeFingerprint, review.catalog.identificationFingerprint);
    expect((await kb.loadManifest()).revision).toBe(1);
    expect(f.options.llm.call).not.toHaveBeenCalled();
  });

  it("rejects a stale proposal when its catalog changes during the model step", async () => {
    const f = fixture();
    await f.workflow.scan(); await f.workflow.index();
    const call = f.options.llm.call;
    let updated = false;
    f.options.llm = { call: async (messages, opts) => {
      if (!updated) {
        updated = true;
        const { catalog } = await f.workflow.review();
        catalog.papers["arxiv:2610.00001"]!.title = "New evidence";
        await new PersonalLibraryCatalogStore(f.storage, f.options.output).replace(catalog);
      }
      return call(messages, opts);
    } };
    await expect(f.workflow.propose()).rejects.toThrow("catalog changed");
    expect((await f.workflow.review()).proposal).toBeNull();
  });

  it("honors cancellation before reading the library or writing records", async () => {
    const f = fixture();
    const controller = new AbortController(); controller.abort();
    await expect(f.workflow.scan({ signal: controller.signal })).rejects.toThrow();
    await expect(f.workflow.index({ signal: controller.signal })).rejects.toThrow();
    expect(f.options.source.inventory).not.toHaveBeenCalled();
    expect(f.storage.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("rejects stale proposal acceptance and edits", async () => {
    const f = await prepareDirections();
    await expect(f.workflow.confirm({ ...f.confirmInput, expectedProposalRevision: f.proposal.revision + 1 })).rejects.toThrow("revision");
    const result = await f.workflow.updateCandidate({ candidateId: f.candidate.id, patch: { text: "Reviewed direction" }, expectedProposalRevision: f.proposal.revision });
    expect(result.proposal?.topics[0]!.directions[0]!.text).toBe("Reviewed direction");
    await expect(f.workflow.updateCandidate({ candidateId: f.candidate.id, patch: { text: "Stale" }, expectedProposalRevision: f.proposal.revision })).rejects.toThrow("revision");
    expect(result.topics).toHaveLength(0);
  });

  it("saves topics and receipts together and preserves edits on repeated acceptance", async () => {
    const f = await prepareDirections();
    const input = { topicIds: [f.proposal.topics[0]!.id], candidateIds: [f.candidate.id], expectedProposalRevision: f.proposal.revision };
    const accepted = await f.workflow.acceptTopics(input);
    await f.options.topicSettings!.change(async current => ({ ...current, topics: current.topics.map(topic => ({ ...topic, directions: [] })) }));
    const repeated = await f.workflow.acceptTopics(input);
    expect(repeated.topics[0]!.directions).toHaveLength(0);
    expect(repeated.acceptances).toEqual(accepted.acceptances);
  });

  it("does not publish acceptance when the settings transaction fails", async () => {
    const f = await prepareDirections();
    const read = f.options.topicSettings!.read;
    f.options.topicSettings = { read, change: async compute => { await compute(await read()); throw new Error("disk full"); } };
    await expect(f.workflow.acceptTopics({ topicIds: [f.proposal.topics[0]!.id] })).rejects.toThrow("disk full");
    const snapshot = await f.workflow.review();
    expect(snapshot.topics).toEqual([]);
    expect(snapshot.acceptances).toEqual([]);
    expect(snapshot.proposal).toEqual(f.proposal);
  });

  it("requires a host settings transaction and rejects disabled legacy profiles", async () => {
    const f = await prepareDirections();
    await expect(f.workflow.confirm({ ...f.confirmInput, status: "disabled" })).rejects.toThrow("retired");
    delete f.options.topicSettings;
    await expect(f.workflow.confirm(f.confirmInput)).rejects.toThrow("transactional topic settings");
    expect((await f.workflow.review()).proposal).toEqual(f.proposal);
  });
});
