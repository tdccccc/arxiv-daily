import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  ArxivFetcher, ArxivSourceAdapter, DEFAULT_SETTINGS, HtmlCache, Logger, PaperContentFetcher,
  isArxivRetryDeferredError, legacyContentFromNormalized, modernArxivResources, summarizeDailyPaper, summarizePaperDetail,
  type ArxivRequestCoordinator, type ArxivSettings, type HttpClient, type LlmClient, type Topic,
} from "@arxiv-daily/core";
import { LinkedomMarkupParser, NodeHttpClient, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { isContentFetchFailure } from "../../packages/core/src/pipeline/content-fetch-failure";
import { assertWritableTarget } from "./evaluate";
import { requireDate, resolveVaultLayout } from "./summary-check";
import { loadSummaryCorpus } from "./summary-corpus";

const CACHE_DAYS = 3650;
const FAILURES_FILE = "failures.json";
const MAX_CONSECUTIVE_FAILURES = 10;
const SECONDS_PER_REQUEST = 3;

export type SourcePath = "html" | "latex" | "abs";

export interface Budgets { sectionCharLimit: number; paperCharLimit: number }

/** What the daily pipeline fetched for one paper, as the summarizers see it. */
export interface SourceRecord {
  schemaVersion: 1;
  id: string;
  fetchedAt: string;
  /** Which of the pipeline's three paths produced the text: rendered HTML, LaTeX source, or only the abs page. */
  sourcePath: SourcePath;
  /** Best effort: read from the fetched page when it names a version; null when it does not (always for LaTeX). */
  arxivVersion: string | null;
  budgets: Budgets;
  content: {
    /** The abstract text without its "## Abstract" heading. */
    abstract: string;
    abstractConclusion: string;
    fullSections: string | null;
    fullTextFailure: string | null;
  };
  /** SHA-256 of abstractConclusion, a NUL, and fullSections (empty when null). */
  contentHash: string;
}

export interface LegacyContent {
  abstractConclusion: string;
  fullSections: string | null;
  fullTextSource?: "arxiv-html" | "arxiv-source";
  fullTextFailure?: string;
}

export type ContentClass =
  | { kind: "ok"; sourcePath: SourcePath }
  | { kind: "failed"; reason: string; rateLimited?: true };

const RATE_LIMITED = /requested a later retry|HTTP 429/i;
// Failures of the transport (not a 404 or an unextractable page): the same fetch may work later, so it is not cached.
const TRANSIENT = /rendered HTML\/cache failed|source fetch failed|abs\/cache failed/;

/**
 * Whether a fetch produced something worth caching. Full text is always worth it. An abs-only result is too,
 * unless one of the paths failed on the network rather than answering "not there": caching that would freeze a
 * worse input than the generator may have had, so it is reported as a failure and retried next run.
 */
export function classifyContent(content: LegacyContent): ContentClass {
  const failure = content.fullTextFailure ?? "";
  if (content.fullSections) {
    return { kind: "ok", sourcePath: content.fullTextSource === "arxiv-source" ? "latex" : "html" };
  }
  if (isContentFetchFailure(content.abstractConclusion) || TRANSIENT.test(failure)) {
    const reason = failure || "no content could be fetched";
    return RATE_LIMITED.test(reason) ? { kind: "failed", reason, rateLimited: true } : { kind: "failed", reason };
  }
  return { kind: "ok", sourcePath: "abs" };
}

function abstractOf(abstractConclusion: string): string {
  const match = /^##[ \t]+Abstract[ \t]*\r?\n([\s\S]*?)(?=\n\n##[ \t]|$)/.exec(abstractConclusion.trimStart());
  return (match ? match[1]! : abstractConclusion).trim();
}

const contentHash = (abstractConclusion: string, fullSections: string | null): string =>
  createHash("sha256").update(`${abstractConclusion}\u0000${fullSections ?? ""}`).digest("hex");

export interface FetchSourcesOptions {
  sourcesDir: string;
  /** Without it only the plan is returned: fetching makes many requests to arXiv. */
  yes?: boolean;
  /** Fetch again even when a record exists. */
  refresh?: boolean;
  /** Fetch at most this many papers in this run. */
  limit?: number;
  budgets?: Budgets;
  requestDelayMs?: number;
}

export interface FetchSourcesDeps {
  http?: HttpClient;
  /** Defaults to the process-wide coordinator, which spaces requests and honours arXiv's cooldowns. */
  requestCoordinator?: ArxivRequestCoordinator;
  logger?: Logger;
  now?: () => Date;
  /** Replaces the daily pipeline's content fetch; tests use it to simulate arXiv answers. */
  fetchContent?: (id: string, budgets: Budgets) => Promise<LegacyContent>;
  onProgress?: (message: string) => void;
}

export interface SourceFailure { id: string; reason: string; at: string }

export interface FetchSourcesPlan {
  total: number;
  cached: number;
  toFetch: number;
  invalid: number;
  limit: number | null;
  budgets: Budgets;
}

export interface FetchSourcesResult {
  sourcesDir: string;
  dryRun: boolean;
  plan: FetchSourcesPlan;
  fetched: number;
  skipped: number;
  failed: number;
  /** Ids left unfetched because of --limit or because the run stopped early. */
  remaining: number;
  stopped: "rate-limit" | "network" | null;
  failures: SourceFailure[];
  byPath: Record<SourcePath, number>;
}

async function readRecord(file: string): Promise<SourceRecord | null> {
  try {
    const record = JSON.parse(await readFile(file, "utf8")) as SourceRecord;
    return record.schemaVersion === 1 && typeof record.id === "string" ? record : null;
  } catch {
    return null;
  }
}

/** The cached record of a paper, or null when there is none (or it is not a readable record of this format). */
export async function readSourceRecord(sourcesDir: string, id: string): Promise<SourceRecord | null> {
  const canonical = modernArxivResources(id)?.id;
  return canonical ? readRecord(path.join(sourcesDir, `${canonical}.json`)) : null;
}

async function readFailures(sourcesDir: string): Promise<SourceFailure[]> {
  try {
    const parsed = JSON.parse(await readFile(path.join(sourcesDir, FAILURES_FILE), "utf8")) as { failures?: SourceFailure[] };
    return Array.isArray(parsed.failures) ? parsed.failures : [];
  } catch {
    return [];
  }
}

/** Write through a temporary file, so a crash never leaves half a record that would later count as cached. */
async function writeAtomic(file: string, text: string): Promise<void> {
  const temporary = `${file}.tmp`;
  await writeFile(temporary, text);
  await rename(temporary, file);
}

/**
 * Cache the source text the generator would have seen, one JSON per paper. The fetch is the daily pipeline's own
 * (ArxivSourceAdapter over PaperContentFetcher: rendered HTML, then LaTeX source, then the abs page) with the same
 * section and paper budgets. Records already on disk are skipped, so an interrupted run resumes where it stopped.
 */
export async function fetchSources(ids: readonly string[], options: FetchSourcesOptions, deps: FetchSourcesDeps = {}): Promise<FetchSourcesResult> {
  const sourcesDir = path.resolve(options.sourcesDir);
  const budgets = options.budgets ?? {
    sectionCharLimit: DEFAULT_SETTINGS.advanced.sectionCharLimit, paperCharLimit: DEFAULT_SETTINGS.advanced.paperCharLimit,
  };
  const now = deps.now ?? (() => new Date());
  await assertWritableTarget(sourcesDir);

  const canonical = new Map<string, string | null>();
  for (const id of ids) canonical.set(id, modernArxivResources(id)?.id ?? null);
  const distinct = [...new Set(ids)].sort();
  const invalid = distinct.filter((id) => canonical.get(id) === null);
  const valid = [...new Set(distinct.filter((id) => canonical.get(id) !== null).map((id) => canonical.get(id)!))].sort();

  const cached: string[] = [];
  const pending: string[] = [];
  for (const id of valid) {
    const record = options.refresh ? null : await readRecord(path.join(sourcesDir, `${id}.json`));
    // A record made under other budgets is not what the generator would see now.
    if (record && record.budgets.sectionCharLimit === budgets.sectionCharLimit && record.budgets.paperCharLimit === budgets.paperCharLimit) cached.push(id);
    else pending.push(id);
  }
  const toFetch = options.limit === undefined ? pending : pending.slice(0, options.limit);
  const plan: FetchSourcesPlan = {
    total: distinct.length, cached: cached.length, toFetch: toFetch.length, invalid: invalid.length, limit: options.limit ?? null, budgets,
  };
  const result: FetchSourcesResult = {
    sourcesDir, dryRun: !options.yes, plan, fetched: 0, skipped: cached.length, failed: 0,
    remaining: pending.length - toFetch.length, stopped: null, failures: [], byPath: { html: 0, latex: 0, abs: 0 },
  };
  if (!options.yes) return result;

  await mkdir(sourcesDir, { recursive: true });
  // The page caches would otherwise answer a refresh without asking arXiv again.
  if (options.refresh) await rm(path.join(sourcesDir, ".cache"), { recursive: true, force: true });
  const attemptedAt = now().toISOString();
  const failed = new Map<string, SourceFailure>();
  for (const id of invalid) failed.set(id, { id, reason: `invalid arXiv id: ${id}`, at: attemptedAt });

  const logger = deps.logger ?? new Logger("error");
  const storage = new NodeStorageAdapter(sourcesDir);
  const parser = new LinkedomMarkupParser();
  const htmlCache = new HtmlCache({ rootDir: ".cache/html", expiryDays: CACHE_DAYS, storage });
  const fetcher = new ArxivFetcher({
    http: deps.http ?? new NodeHttpClient(), markupParser: parser, logger,
    requestDelayMs: options.requestDelayMs ?? DEFAULT_SETTINGS.advanced.requestDelayMs, requestCoordinator: deps.requestCoordinator,
  });
  const paperFetcher = new PaperContentFetcher(fetcher, htmlCache, logger, parser, { storage, cacheDir: ".cache/source", expiryDays: CACHE_DAYS });
  const adapter = new ArxivSourceAdapter({ fetcher, paperFetcher, markupParser: parser, logger, defaultCategories: [] });
  const fetchContent = deps.fetchContent ?? (async (id: string, limits: Budgets): Promise<LegacyContent> =>
    legacyContentFromNormalized(await adapter.fetchContent(id, { wantFullText: true, ...limits })));

  let consecutiveFailures = 0;
  let attempted = 0;
  try {
    for (const id of toFetch) {
      attempted += 1;
      deps.onProgress?.(`[${attempted}/${toFetch.length}] ${id}`);
      let outcome: ContentClass;
      let content: LegacyContent | null = null;
      try {
        content = await fetchContent(id, budgets);
        outcome = classifyContent(content);
      } catch (error) {
        const reason = (error as Error).message;
        outcome = isArxivRetryDeferredError(error) || RATE_LIMITED.test(reason)
          ? { kind: "failed", reason, rateLimited: true } : { kind: "failed", reason };
      }

      if (outcome.kind === "ok" && content) {
        const record: SourceRecord = {
          schemaVersion: 1, id, fetchedAt: now().toISOString(), sourcePath: outcome.sourcePath,
          arxivVersion: await detectVersion(htmlCache, id, outcome.sourcePath), budgets,
          content: {
            abstract: abstractOf(content.abstractConclusion), abstractConclusion: content.abstractConclusion,
            fullSections: content.fullSections, fullTextFailure: content.fullTextFailure ?? null,
          },
          contentHash: contentHash(content.abstractConclusion, content.fullSections),
        };
        await writeAtomic(path.join(sourcesDir, `${id}.json`), `${JSON.stringify(record, null, 2)}\n`);
        result.fetched += 1;
        result.byPath[outcome.sourcePath] += 1;
        consecutiveFailures = 0;
        continue;
      }

      const reason = outcome.kind === "failed" ? outcome.reason : "no content";
      failed.set(id, { id, reason, at: attemptedAt });
      consecutiveFailures += 1;
      deps.onProgress?.(`  failed: ${reason}`);
      if (outcome.kind === "failed" && outcome.rateLimited) {
        result.stopped = "rate-limit";
        break;
      }
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        result.stopped = "network";
        break;
      }
    }
  } finally {
    // Failures are kept across runs until the paper is cached; ids that are cached now drop out.
    const previous = (await readFailures(sourcesDir)).filter((failure) => !failed.has(failure.id));
    const cachedNow = new Set((await readdir(sourcesDir)).filter((name) => /^\d.*\.json$/.test(name)).map((name) => name.slice(0, -5)));
    const failures = [...previous.filter((failure) => !cachedNow.has(failure.id)), ...failed.values()]
      .sort((a, b) => a.id.localeCompare(b.id));
    await writeAtomic(path.join(sourcesDir, FAILURES_FILE), `${JSON.stringify({ schemaVersion: 1, updatedAt: attemptedAt, failures }, null, 2)}\n`);
    result.failures = failures;
  }
  result.failed = failed.size;
  result.remaining += toFetch.length - attempted;
  return result;
}

// The page names its version in the <base> or canonical link; the pipeline keeps no version, so this is best effort.
const VERSION_IN_PAGE = /\/(?:html|abs|pdf)\/\d{4}\.\d{4,5}(v\d+)/;

async function detectVersion(cache: HtmlCache, id: string, sourcePath: SourcePath): Promise<string | null> {
  if (sourcePath === "latex") return null;
  const page = sourcePath === "html" ? await cache.get(`html/${id}`, "html") : await cache.get(`abs/${id}`, "abs");
  return page ? VERSION_IN_PAGE.exec(page)?.[1] ?? null : null;
}

export interface SummarySourcesOptions extends FetchSourcesOptions {
  vaultRoot?: string;
  configPath?: string;
  dailyDir?: string;
  papersDir?: string;
  from?: string;
  to?: string;
  /** Also fetch the papers whose daily summary fell back to the abstract. */
  fallbacks?: boolean;
}

export function parseSummarySourcesArgs(argv: readonly string[]): Partial<SummarySourcesOptions> & { help?: boolean } {
  const parsed: Partial<SummarySourcesOptions> & { help?: boolean } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    if (flag === "--yes") { parsed.yes = true; continue; }
    if (flag === "--refresh") { parsed.refresh = true; continue; }
    if (flag === "--fallbacks") { parsed.fallbacks = true; continue; }
    const take = (): string => {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) throw new Error(`${flag} needs a value`);
      index += 1;
      return value;
    };
    switch (flag) {
      case "--vault": parsed.vaultRoot = take(); break;
      case "--config": parsed.configPath = take(); break;
      case "--daily-dir": parsed.dailyDir = take(); break;
      case "--papers-dir": parsed.papersDir = take(); break;
      case "--sources": parsed.sourcesDir = take(); break;
      case "--from": parsed.from = requireDate(flag, take()); break;
      case "--to": parsed.to = requireDate(flag, take()); break;
      case "--limit": {
        const value = take();
        if (!/^\d+$/.test(value) || Number(value) < 1) throw new Error("--limit must be a whole number of at least 1");
        parsed.limit = Number(value);
        break;
      }
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.sourcesDir) throw new Error("--sources <directory> is required");
  if (parsed.from && parsed.to && parsed.from > parsed.to) {
    throw new Error(`--from ${parsed.from} is after --to ${parsed.to}; the start date is after the end date`);
  }
  return parsed;
}

/** Fetch the source text of every paper in the vault's daily entries and detail notes (see {@link fetchSources}). */
export async function runSummarySources(options: SummarySourcesOptions, deps: FetchSourcesDeps = {}): Promise<FetchSourcesResult> {
  const { vaultRoot, dailyDir, papersDir, settings } = await resolveVaultLayout(options);
  const corpus = await loadSummaryCorpus({ vaultRoot, dailyDir, papersDir, from: options.from, to: options.to });
  const ids = [
    ...corpus.entries.map((entry) => entry.id),
    ...corpus.notes.map((note) => note.id),
    ...(options.fallbacks ? corpus.fallbacks.map((fallback) => fallback.id) : []),
  ];
  if (ids.length === 0) {
    throw new Error(`no daily entries or detail notes found under ${vaultRoot} (looked in ${dailyDir} and ${papersDir}) for ${options.from ?? "the first day"} to ${options.to ?? "the last day"}`);
  }
  return fetchSources(ids, {
    ...options,
    budgets: options.budgets ?? { sectionCharLimit: settings.advanced.sectionCharLimit, paperCharLimit: settings.advanced.paperCharLimit },
    requestDelayMs: options.requestDelayMs ?? settings.advanced.requestDelayMs,
  }, deps);
}

/** Plan line for a run that has not been confirmed: how much it would ask of arXiv. */
export function describePlan(plan: FetchSourcesPlan): string {
  const requests = plan.toFetch;
  const minutes = Math.ceil((requests * SECONDS_PER_REQUEST) / 60);
  return [
    `${plan.toFetch} paper${plan.toFetch === 1 ? "" : "s"} would be fetched from arXiv (${plan.cached} already cached${plan.invalid ? `, ${plan.invalid} invalid ids` : ""}${plan.limit === null ? "" : `, limited to ${plan.limit}`}).`,
    `Each paper takes one request when rendered HTML exists and up to three (HTML, LaTeX source, abs page) when it does not, at least ${SECONDS_PER_REQUEST} s apart: at least ${minutes} minute${minutes === 1 ? "" : "s"}.`,
    `Section budget ${plan.budgets.sectionCharLimit} characters, paper budget ${plan.budgets.paperCharLimit}.`,
    "Pass --yes to make the requests.",
  ].join("\n");
}

class CapturedMessages extends Error {
  constructor(readonly messages: Array<{ role: string; content: string }>, readonly options: { temperature?: number }) { super("captured"); }
}

const capturingLlm = { call: async (messages: Array<{ role: string; content: string }>, options: { temperature?: number } = {}) => { throw new CapturedMessages(messages, { temperature: options.temperature }); } } as unknown as LlmClient;

export interface RebuiltSummaryInput {
  messages: Array<{ role: string; content: string }>;
  options: { temperature?: number };
}

export interface InputIdentity { title: string; authors: string; summaryLanguage?: "zh" | "en" }

type DailyPaperArg = Parameters<typeof summarizeDailyPaper>[0];
type DetailPaperArg = Parameters<typeof summarizePaperDetail>[0];
type DetailDeps = Parameters<typeof summarizePaperDetail>[1];

/** The paper the pipeline hands the daily summarizer, made from a cached record. */
export function dailyPaperFromRecord(record: SourceRecord, identity: InputIdentity): DailyPaperArg {
  return {
    id: record.id, title: identity.title, authors: identity.authors,
    abstractConclusion: record.content.abstractConclusion, fullSections: record.content.fullSections,
  };
}

/** The paper and the settings the pipeline hands the detail summarizer (everything but the model client), from a cached record. */
export function detailCallFromRecord(
  record: SourceRecord,
  identity: InputIdentity & { topic: string; topics?: ReadonlyArray<Pick<Topic, "tag" | "name">> },
): { paper: DetailPaperArg; deps: Omit<DetailDeps, "llm"> } {
  return {
    paper: {
      id: record.id, title: identity.title, authors: identity.authors, category: identity.topic,
      abstractConclusion: record.content.abstractConclusion, fullSections: record.content.fullSections,
    } as unknown as DetailPaperArg,
    deps: {
      llmSettings: DEFAULT_SETTINGS.llm, logger: new Logger("error"), advanced: DEFAULT_SETTINGS.advanced,
      arxivSettings: { ...DEFAULT_SETTINGS.arxiv, topics: [...(identity.topics ?? [])] } as ArxivSettings,
      summaryLanguage: identity.summaryLanguage,
    },
  };
}

/**
 * The messages the daily summarizer would send for a cached paper, built by the summarizer itself (its prompt
 * rendering and `<paper_data>` builder run with a client that only records the request). Null when the cached text
 * is not enough to summarize from, which the pipeline treats as a fallback. The report keeps only the first author
 * ("A. Author et al."), so the authors line is as faithful as the caller's `authors`.
 */
export async function rebuildDailyInput(record: SourceRecord, identity: InputIdentity): Promise<RebuiltSummaryInput | null> {
  try {
    await summarizeDailyPaper(dailyPaperFromRecord(record, identity), { llm: capturingLlm, summaryLanguage: identity.summaryLanguage });
  } catch (error) {
    if (error instanceof CapturedMessages) return { messages: error.messages, options: error.options };
    if (/no source text/.test((error as Error).message)) return null;
    throw error;
  }
  return null;
}

/** The messages the detail summarizer would send for a cached paper; null when there is no full text to send. */
export async function rebuildDetailInput(
  record: SourceRecord,
  identity: InputIdentity & { topic: string; topics?: ReadonlyArray<Pick<Topic, "tag" | "name">> },
): Promise<RebuiltSummaryInput | null> {
  if (!record.content.fullSections) return null;
  try {
    const { paper, deps } = detailCallFromRecord(record, identity);
    await summarizePaperDetail(paper, { ...deps, llm: capturingLlm });
  } catch (error) {
    if (error instanceof CapturedMessages) return { messages: error.messages, options: error.options };
    throw error;
  }
  return null;
}
