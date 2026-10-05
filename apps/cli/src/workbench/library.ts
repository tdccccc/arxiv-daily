import { watch } from "node:fs";
import * as path from "node:path";
import { searchFullTextKnowledgeBase, LibrarySourceError, type FullTextKnowledgeBaseManifest, type PersonalLibraryCatalog, type EmbeddingModel, type PersonalLibraryPaperRecord, type PersonalLibraryScanSummary } from "@arxiv-daily/core";
import { openScopedLibrarySource } from "@arxiv-daily/node-runtime";
import { CliConfigError, type CliRuntimeConfig } from "../config";
import { createCliLibraryContext, type CliLibraryOptions } from "../library-cmd";
import { WorkbenchError } from "./documents";
import { createCliTopicSettings } from "../library-topic-settings";

export type WorkbenchLibraryPaper = Omit<PersonalLibraryPaperRecord, "source"> & { source: "arxiv" | "file"; pdfAvailable: boolean; score?: number };
export interface WorkbenchLibraryCatalog {
  papers: WorkbenchLibraryPaper[];
  total: number;
  offset: number;
  nextOffset: number | null;
  summary: PersonalLibraryScanSummary | null;
  connected: boolean;
}
export interface WorkbenchLibrarySearch {
  query: string;
  mode?: "lexical" | "hybrid" | "dense";
  limit?: number;
}
type Context = Awaited<ReturnType<typeof createCliLibraryContext>>;
const PDF_LIMIT = 25 * 1024 * 1024;

/** Structured host adapter: catalog, retrieval and bounded source reads share CLI/core scope. */
export class WorkbenchLibrary {
  constructor(private readonly config: CliRuntimeConfig, private readonly options: CliLibraryOptions = {}) {}

  async catalog(params: URLSearchParams): Promise<WorkbenchLibraryCatalog> {
    const offset = integer(params.get("offset") ?? 0, 0, Number.MAX_SAFE_INTEGER, "offset");
    const limit = integer(params.get("limit") ?? 20, 1, 100, "limit");
    const query = (params.get("q") ?? "").trim().toLocaleLowerCase();
    if (query.length > 4096) throw new WorkbenchError(400, "Library query is too long");
    if (!this.config.libraryConnection) {
      try { await createCliTopicSettings(this.config).assertCurrent(); }
      catch (error) { if (error instanceof CliConfigError) throw new WorkbenchError(409, error.message); throw error; }
      if (this.config.libraryConnectionError) throw new WorkbenchError(409, this.config.libraryConnectionError);
      return { papers: [], total: 0, offset, nextOffset: null, summary: null, connected: false };
    }
    return this.withContext(async (context, signal) => {
      const { catalog } = await context.workflow.review();
      const manifest = await context.knowledgeBase.loadManifest();
      const papers = Object.values(projectPapers(catalog, manifest)).filter(paper => !query || [paper.title, paper.abstract, paper.externalId, ...paper.authors, ...paper.categories].join(" ").toLocaleLowerCase().includes(query))
        .sort((a, b) => b.published.localeCompare(a.published) || a.paperKey.localeCompare(b.paperKey));
      const selected = await Promise.all(papers.slice(offset, offset + limit).map(paper => this.present(context, paper, signal)));
      return { papers: selected, total: papers.length, offset, nextOffset: offset + limit < papers.length ? offset + limit : null, summary: catalog.lastScan, connected: true };
    });
  }

  async search(raw: unknown, signal?: AbortSignal): Promise<{ papers: WorkbenchLibraryPaper[]; total: number }> {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new WorkbenchError(400, "Library search requires an object");
    const input = raw as Record<string, unknown>;
    if (typeof input.query !== "string" || !input.query.trim() || input.query.length > 4096) throw new WorkbenchError(400, "Library query must contain 1..4096 characters");
    const mode = (input.mode ?? "lexical") as WorkbenchLibrarySearch["mode"] & string;
    const query = input.query.trim();
    if (!["lexical", "hybrid", "dense"].includes(mode)) throw new WorkbenchError(400, "search mode must be lexical, hybrid or dense");
    const limit = integer(input.limit ?? 20, 1, 100, "limit");
    return this.withContext(async (context, currentSignal) => {
      if (mode !== "lexical" && this.config.settings.embedding.mode === "remote") await context.assertAuthorized("embedding");
      const { catalog } = await context.workflow.review();
      // BM25 needs the index identity for core compatibility checks, never a model runtime.
      const manifest = await context.knowledgeBase.loadManifest();
      const embedding: EmbeddingModel = mode === "lexical" ? {
        modelId: manifest.modelId, dimension: manifest.dimension, prefixPolicy: "none",
        embed: async () => { throw new Error("Lexical retrieval must not invoke embeddings"); },
      } : await context.createEmbedding(currentSignal);
      const records = projectPapers(catalog, manifest);
      const matches = await searchFullTextKnowledgeBase({ store: context.knowledgeBase, generationStore: context.generationStore,
        embedding, sourceManifest: manifest, queryText: query, mode, limit, signal: currentSignal,
        titles: new Map(Object.values(records).map(paper => [paper.paperKey, paper.title])), logger: context.logger });
      const papers: WorkbenchLibraryPaper[] = [];
      for (const match of matches) {
        const paper = records[match.paperKey];
        if (paper) papers.push({ ...await this.present(context, paper, currentSignal), score: match.rankingScore });
      }
      return { papers, total: papers.length };
    }, signal);
  }

  async pdf(key: string): Promise<ArrayBuffer> {
    return this.withContext(async (context, signal) => {
      const { catalog } = await context.workflow.review();
      const records = projectPapers(catalog, await context.knowledgeBase.loadManifest());
      const paper = Object.prototype.hasOwnProperty.call(records, key) ? records[key] : undefined;
      if (!paper) throw new WorkbenchError(404, "Library paper was not found");
      for (const file of paper.filePaths.filter(isPdf)) {
        try {
          const bytes = await context.source.readBinary(file, { maxBytes: PDF_LIMIT, signal });
          if (new TextDecoder().decode(bytes.slice(0, 5)) !== "%PDF-") continue;
          return bytes;
        } catch (error) { signal.throwIfAborted(); if (paper.filePaths.filter(isPdf).length === 1) throw error; }
      }
      throw new WorkbenchError(404, "Library PDF is unavailable");
    });
  }

  private async present(context: Context, paper: Omit<WorkbenchLibraryPaper, "pdfAvailable">, signal: AbortSignal): Promise<WorkbenchLibraryPaper> {
    let pdfAvailable = false;
    for (const file of paper.filePaths.filter(isPdf)) {
      try {
        const header = await context.source.readBinary(file, { start: 0, end: 5, maxBytes: PDF_LIMIT, signal });
        if (new TextDecoder().decode(header) === "%PDF-") { pdfAvailable = true; break; }
      } catch { signal.throwIfAborted(); }
    }
    return { ...paper, pdfAvailable };
  }

  private async withContext<T>(run: (context: Context, signal: AbortSignal) => Promise<T>, signal?: AbortSignal): Promise<T> {
    try { return await this.runContext(run, signal); }
    catch (error) {
      if (error instanceof WorkbenchError) throw error;
      if (error instanceof LibrarySourceError) throw new WorkbenchError(error.kind === "limit-exceeded" ? 413 : error.kind === "not-found" ? 404 : 409, error.message);
      if (error instanceof CliConfigError) throw new WorkbenchError(409, error.message);
      if (error instanceof Error && /identity changed|authorization|Personal library is busy/.test(error.message)) throw new WorkbenchError(409, error.message);
      throw error;
    }
  }

  private async runContext<T>(run: (context: Context, signal: AbortSignal) => Promise<T>, signal?: AbortSignal): Promise<T> {
    const controller = new AbortController();
    const combined = AbortSignal.any([controller.signal, ...[this.options.signal, signal].filter((value): value is AbortSignal => !!value)]);
    combined.throwIfAborted();
    const context = await createCliLibraryContext(this.config, { ...this.options, signal: combined });
    const lease = await context.storage.acquireLock("personal-library-workflow", { wait: true, signal: combined, timeoutMs: 30_000 });
    if (!lease) throw new Error("Personal library is busy");
    let watcher: ReturnType<typeof watch> | undefined;
    try {
      const assertCurrent = async () => {
        await context.assertCurrent();
        const source = await openScopedLibrarySource(this.config.libraryConnection!.selectedRoot);
        if (source.rootIdentity !== context.source.rootIdentity || source.canonicalRoot !== context.source.canonicalRoot) throw new Error("Library directory identity changed; connect it again");
      };
      await assertCurrent();
      watcher = watch(path.dirname(this.config.configPath), { persistent: false }, () => { void assertCurrent().catch(error => controller.abort(error)); });
      watcher.on("error", error => controller.abort(error));
      const result = await run(context, combined);
      await assertCurrent();
      return result;
    } finally { watcher?.close(); await lease.release(); }
  }
}
function isPdf(file: string): boolean { return /\.pdf$/i.test(file); }
function integer(value: unknown, min: number, max: number, name: string): number {
  const number = Number(value);
  if ((typeof value !== "number" && typeof value !== "string") || (typeof value === "string" && !/^\d+$/.test(value)) || !Number.isSafeInteger(number) || number < min || number > max) throw new WorkbenchError(400, `${name} must be ${min}..${max}`);
  return number;
}

function projectPapers(catalog: PersonalLibraryCatalog, manifest: FullTextKnowledgeBaseManifest): Record<string, Omit<WorkbenchLibraryPaper, "pdfAvailable">> {
  const papers: Record<string, Omit<WorkbenchLibraryPaper, "pdfAvailable">> = { ...catalog.papers };
  for (const [key, record] of Object.entries(manifest.papers)) {
    if (papers[key] || !key.startsWith("file:") || record.status !== "ready") continue;
    papers[key] = {
      paperKey: key, source: "file", externalId: "", title: record.title ?? record.filePaths[0] ?? key,
      abstract: record.abstract ?? "", authors: [], published: "", updated: record.updatedAt,
      primaryCategory: "", categories: [], evidenceDepth: "metadata-and-abstract", filePaths: [...record.filePaths],
    };
  }
  return papers;
}
