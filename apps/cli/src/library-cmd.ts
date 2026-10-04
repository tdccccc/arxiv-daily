import * as fs from "node:fs/promises";
import { watch } from "node:fs";
import * as path from "node:path";
import {
  ArxivFetcher, LibraryWorkflow, Logger, LlmClient,
  FullTextKnowledgeBaseFileStore, FullTextGenerationIndexStore,
  createPersonalLibraryScopeFingerprint,
  createPersonalLibraryIdentificationFingerprint, createRemoteEmbeddingModel,
  searchFullTextKnowledgeBase, redactText,
  validateLlmConfig, validateEmbeddingConfig,
  type HttpClient, type DocumentParser, type DocumentParserSelector, type EmbeddingModel,
  type LibraryReviewSnapshot, type PersonalLibraryDirectionTextPatch,
} from "@arxiv-daily/core";
import {
  NodeStorageAdapter, NodeHttpClient, LinkedomMarkupParser, openScopedLibrarySource,
  prepareNodeLibraryRuntime, createNodeLibraryDocumentParser, createNodeLibraryEmbeddingModel,
} from "@arxiv-daily/node-runtime";
import { CliConfigError, type CliRuntimeConfig } from "./config";
import { connectCliLibrary, authorizeCliLibrary, revokeCliLibrary, inspectCliLibraryConnection } from "./library-connection-cmd";
import { createCliTopicSettings } from "./library-topic-settings";
import type { CliIo } from "./main-types";

export interface CliLibraryOptions {
  http?: HttpClient;
  createParser?: (signal?: AbortSignal) => Promise<{ parser?: DocumentParser; parserSelector?: DocumentParserSelector }>;
  createEmbedding?: (signal?: AbortSignal) => Promise<EmbeddingModel>;
  signal?: AbortSignal;
}

/** The CLI is an application host; the shared workflow owns all library decisions and writes. */
export async function runCliLibrary(config: CliRuntimeConfig, args: string[], io: CliIo, options: CliLibraryOptions = {}): Promise<number> {
  const secrets = [config.settings.llm.apiKey, config.settings.embedding.apiKey, config.settings.email.apiKey ?? ""];
  const print = (value: unknown) => io.stdout.write(redactText(JSON.stringify(value), { secrets }) + "\n");
  const controller = new AbortController();
  const signal = options.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal;
  const cancel = () => controller.abort(new Error("Library task cancelled"));
  process.once("SIGINT", cancel);
  process.once("SIGTERM", cancel);
  let stopWatch: (() => void) | undefined;
  let lease: Awaited<ReturnType<NodeStorageAdapter["acquireLock"]>> = null;
  try {
    signal.throwIfAborted();
    const [command = "status", ...rest] = args;
    if (command === "update") throw new CliConfigError("Library profile updates are retired; generate proposals with library propose and accept them into research topic settings");
    if (command === "connect") {
      if (rest.length !== 1) throw new CliConfigError("library connect requires a directory path");
      print(inspectCliLibraryConnection(await connectCliLibrary(config, rest[0]!)));
      return 0;
    }
    if (command === "authorize") {
      const flags = parseFlags(rest, ["fingerprint"]);
      print(inspectCliLibraryConnection(await authorizeCliLibrary(config, required(flags.fingerprint, "fingerprint"))));
      return 0;
    }
    if (command === "revoke") {
      if (rest.length) throw new CliConfigError("library revoke takes no arguments");
      print(inspectCliLibraryConnection(await revokeCliLibrary(config)));
      return 0;
    }
    if (command === "status") {
      if (rest.length) throw new CliConfigError("library status takes no arguments");
      print(inspectCliLibraryConnection(config));
      return 0;
    }
    if (command === "prepare") {
      if (rest.length) throw new CliConfigError("library prepare takes no arguments");
      io.stderr.write("Preparing local PDF and model runtime dependencies…\n");
      const prepared = await prepareNodeLibraryRuntime({ cacheDir: config.cacheDir, localEmbedding: config.settings.embedding.mode === "local", signal });
      print({ prepared });
      return 0;
    }
    if (!["scan", "index", "propose", "directions", "confirm", "search", "review"].includes(command)) throw new CliConfigError(`Unknown library command: ${command}`);
    const context = await libraryContext(config, { ...options, signal });
    lease = await context.storage.acquireLock("personal-library-workflow", { wait: true, signal, timeoutMs: 30_000 });
    if (!lease) throw new Error("Personal library is busy");
    await context.assertCurrent();
    const watcher = watch(path.dirname(config.configPath), { persistent: false }, () => {
      void context.assertCurrent().catch(error => controller.abort(error));
    });
    watcher.on("error", error => controller.abort(error));
    stopWatch = () => watcher.close();
    if (["scan", "index", "propose", "directions"].includes(command) && rest.length) throw new CliConfigError(`library ${command} takes no arguments`);
    if (command === "scan") {
      const catalog = await context.workflow.scan({ signal });
      print({ revision: catalog.revision, lastScan: catalog.lastScan, paperCount: Object.keys(catalog.papers).length });
    } else if (command === "index") {
      print(await context.workflow.index({ signal, onProgress: detail => io.stderr.write(redactText(detail, { secrets }) + "\n") }));
    } else if (command === "propose") {
      print(await context.workflow.propose({ signal }));
    } else if (command === "directions") {
      print(reviewView(await context.workflow.review()));

    } else if (command === "confirm") {
      const flags = parseFlags(rest, ["candidate", "proposal-revision"]);
      const snapshot = await context.workflow.review();
      const candidate = snapshot.proposal?.topics.flatMap(topic => topic.directions).find(candidate => candidate.id === flags.candidate);
      if (!candidate) throw new Error("Direction candidate was not found; inspect library directions again");
      print(reviewView(await context.workflow.confirm({
        candidateId: candidate.id, status: "active", signal,
        expectedProposalRevision: revision(flags["proposal-revision"]),
        draft: { text: candidate.text, discoveryCues: candidate.discoveryCues, representativePaperKeys: candidate.representatives.map(paper => paper.paperKey) },
      })));
    } else if (command === "search") {
      const flags = parseFlags(rest, ["query", "mode", "limit"]);
      const queryText = required(flags.query, "query");
      const mode = flags.mode ?? "hybrid";
      if (!["lexical", "hybrid", "dense"].includes(mode)) throw new CliConfigError("search mode must be lexical, hybrid or dense");
      const limit = Number(flags.limit ?? 10);
      if (!Number.isInteger(limit) || limit < 1 || limit > 50) throw new CliConfigError("search limit must be 1..50");
      if (mode !== "lexical" && config.settings.embedding.mode === "remote") await context.assertAuthorized("embedding");
      const snapshot = await context.workflow.review();
      const embedding = await context.createEmbedding(signal);
      const matches = await searchFullTextKnowledgeBase({
        store: context.knowledgeBase, generationStore: context.generationStore, embedding,
        queryText, mode: mode as "lexical" | "hybrid" | "dense", limit, maxHitsPerPaper: 3, signal,
        titles: new Map(Object.values(snapshot.catalog.papers).map(paper => [paper.paperKey, paper.title])),
        logger: context.logger,
      });
      await context.assertCurrent();
      print({ matches });
    } else if (command === "review") {
      const flags = parseFlags(rest, ["input"]);
      const file = required(flags.input, "input");
      if ((await fs.stat(file)).size > 256 * 1024) throw new CliConfigError("review input exceeds 256 KiB");
      let request: Record<string, unknown>;
      try { request = JSON.parse(await fs.readFile(file, "utf8")); }
      catch { throw new CliConfigError("review input must contain valid JSON"); }
      print(reviewView(await applyReview(context.workflow, request, signal)));
    }
    return 0;
  } catch (error) {
    io.stderr.write(redactText(error instanceof Error ? error.message : String(error), { secrets }) + "\n");
    return error instanceof CliConfigError ? 2 : 1;
  } finally {
    stopWatch?.();
    await lease?.release();
    process.off("SIGINT", cancel);
    process.off("SIGTERM", cancel);
  }
}

async function libraryContext(config: CliRuntimeConfig, options: CliLibraryOptions = {}) {
  const connection = config.libraryConnection;
  if (!connection) throw new CliConfigError(config.libraryConnectionError ?? "Connect a personal library first: arxiv-daily library connect PATH");
  const source = await openScopedLibrarySource(connection.selectedRoot);
  if (source.canonicalRoot !== connection.selectedRoot || source.rootIdentity !== connection.rootIdentity) throw new Error("Library directory identity changed; connect it again");
  const storage = new NodeStorageAdapter(config.vaultRoot);
  const logger = new Logger(config.settings.advanced.logLevel);
  logger.setSensitiveValues([config.settings.llm.apiKey, config.settings.embedding.apiKey]);
  const topicSettings = createCliTopicSettings(config);
  const assertCurrent = async () => {
    options.signal?.throwIfAborted();
    await topicSettings.assertCurrent();
  };
  const assertAuthorized = async (_purpose: "directions" | "embedding") => {
    await assertCurrent();
    if (inspectCliLibraryConnection(config).status.kind !== "authorized") throw new Error("Library model processing requires current authorization; inspect library status and authorize its fingerprint");
    if (_purpose === "directions" && !validateLlmConfig(config.settings).ok) throw new Error("Complete the model configuration before generating research directions");
  };
  const transport = options.http ?? new NodeHttpClient();
  const http: HttpClient = { async request(request) { await assertCurrent(); return transport.request(request); } };
  const fetcher = new ArxivFetcher({ category: config.settings.arxiv.category, categories: config.settings.arxiv.categories, http, markupParser: new LinkedomMarkupParser(), logger, requestDelayMs: config.settings.advanced.requestDelayMs });
  let embedding: EmbeddingModel | undefined;
  const createEmbedding = async (signal?: AbortSignal) => {
    if (embedding) return embedding;
    if (options.createEmbedding) return embedding = await options.createEmbedding(signal);
    if (config.settings.embedding.mode === "remote") {
      if (!validateEmbeddingConfig(config.settings).ok) throw new Error("Complete remote embedding configuration before indexing");
      const { baseUrl, apiKey, model, dimension } = config.settings.embedding;
      return embedding = createRemoteEmbeddingModel({ baseUrl, apiKey, model, dimension, http });
    }
    return embedding = createNodeLibraryEmbeddingModel(config.cacheDir, { signal });
  };
  // Library indexing extracts bounded page text for titles and abstracts.
  // Structured-document sidecars belong to document reading, not this index.
  const createParser = options.createParser ?? (async () => ({
    parser: await createNodeLibraryDocumentParser(config.cacheDir),
  }));
  const scope = createPersonalLibraryScopeFingerprint(connection);
  const identity = createPersonalLibraryIdentificationFingerprint(connection.eligibleExtensions);
  return {
    storage, logger, assertCurrent, assertAuthorized, createEmbedding,
    knowledgeBase: new FullTextKnowledgeBaseFileStore(storage, config.settings.output, scope, identity),
    generationStore: new FullTextGenerationIndexStore(storage, config.settings.output, scope, identity),
    workflow: new LibraryWorkflow({
      storage, output: config.settings.output, connection, source, http, arxivFetcher: fetcher,
      createParser, createEmbedding, llm: new LlmClient(config.settings.llm, logger, http),
      topicSettings, assertCurrent, assertAuthorized, embeddingRequiresAuthorization: config.settings.embedding.mode === "remote", logger,
    }),
  };
}

function reviewView(snapshot: LibraryReviewSnapshot) {
  return { catalogSummary: snapshot.catalog.lastScan, topics: snapshot.topics, proposal: snapshot.proposal, acceptances: snapshot.acceptances };
}
function parseFlags(args: string[], accepted: string[]) {
  const flags: Record<string, string> = {};
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i]?.replace(/^--/, "");
    if (!args[i]?.startsWith("--") || !key || !accepted.includes(key) || !args[i + 1] || args[i + 1]!.startsWith("--") || flags[key] !== undefined) throw new CliConfigError("Invalid library command arguments");
    flags[key] = args[i + 1]!;
  }
  return flags;
}
function required(value: unknown, name: string): string {
  if (typeof value !== "string" || !value.trim()) throw new CliConfigError(`${name} is required`);
  return value.trim();
}
function revision(value: unknown): number {
  const parsed = Number(value);
  if ((typeof value !== "number" && typeof value !== "string") || value === "" || !Number.isSafeInteger(parsed) || parsed < 0) throw new CliConfigError("A current revision is required");
  return parsed;
}
async function applyReview(workflow: LibraryWorkflow, request: Record<string, unknown>, signal: AbortSignal) {
  if (!request || typeof request !== "object" || Array.isArray(request)) throw new CliConfigError("review input must be an object");
  if (request.operation === "update-candidate") return workflow.updateCandidate({ signal, expectedProposalRevision: revision(request.expectedProposalRevision), candidateId: required(request.candidateId, "candidateId"), patch: request.patch as PersonalLibraryDirectionTextPatch, ...(request.representativePaperKeys ? { representativePaperKeys: request.representativePaperKeys as string[] } : {}) });
  if (request.operation === "accept-topics") {
    if (!Array.isArray(request.topicIds) || !request.topicIds.every(value => typeof value === "string" && value.trim())) throw new CliConfigError("topicIds must be an array of topic IDs");
    if (request.candidateIds !== undefined && (!Array.isArray(request.candidateIds) || !request.candidateIds.every(value => typeof value === "string" && value.trim()))) throw new CliConfigError("candidateIds must be an array of direction IDs");
    return workflow.acceptTopics({ signal, topicIds: request.topicIds as string[], candidateIds: request.candidateIds as string[] | undefined, expectedProposalRevision: revision(request.expectedProposalRevision) });
  }
  if (["dismiss", "enable", "disable", "lock", "unlock", "update-direction", "apply"].includes(String(request.operation))) throw new CliConfigError("Library profiles are retired; accept proposed directions into normal research topic settings and edit them there");
  throw new CliConfigError("Unknown review operation");
}
