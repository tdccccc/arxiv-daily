import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import {
  ArxivFetcher,
  AtomMetadataCache,
  DEFAULT_SETTINGS,
  FilterDecisionRecordStore,
  HtmlCache,
  Logger,
  extractAbsPageAbstract,
  isArxivHttpError,
  isArxivRetryDeferredError,
  modernArxivResources,
  type ArxivRequestCoordinator,
  type FilterDecisionCandidate,
  type FilterDecisionRecordDocument,
  type HttpClient,
  type OutputSettings,
} from "@arxiv-daily/core";
import { LinkedomMarkupParser, NodeHttpClient, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../../apps/cli/src/config";
import { encodeCsv } from "./csv";

export const SHEET_COLUMNS = [
  "date", "id", "title", "abstract", "had_abstract_at_filter",
  "model_category", "model_directions", "model_score", "model_outcome",
  "label", "label_direction", "note",
] as const;

export const DEFAULT_SKIPPED_SAMPLE = 50;
const ABSTRACT_BATCH = 100;
const ABSTRACT_CACHE_DIR = "abstract-cache";
const ABS_PAGE_CACHE_DIR = "abs-cache";
const ABSTRACT_CACHE_DAYS = 3650;
const ABS_PAGE_FAILURE_LIMIT = 3;
const SHEET_FILES = { csv: "labels.csv", guide: "labels.guide.md", meta: "labels.meta.json" } as const;

export interface ExportOptions {
  /** Vault root; defaults to the CLI config's `vault_root`. */
  vaultRoot?: string;
  /** CLI config supplying the vault root and output layout; the default config path when no vault is given. */
  configPath?: string;
  outDir: string;
  from?: string;
  to?: string;
  /** How many skipped candidates to add to the selected ones; deterministic for a seed. */
  skippedSample?: number;
  seed?: string;
  force?: boolean;
}

export interface ExportDeps {
  http?: HttpClient;
  logger?: Logger;
  requestCoordinator?: ArxivRequestCoordinator;
  now?: () => Date;
  /** Called as warnings arise, so a long export can report them live. */
  onWarning?: (message: string) => void;
  onProgress?: (message: string) => void;
}

export interface ExportResult {
  outDir: string;
  rows: number;
  selected: number;
  skippedSampled: number;
  withoutAbstract: number;
  warnings: string[];
  files: string[];
}

export interface SampledCandidate {
  date: string;
  candidate: FilterDecisionCandidate;
}

export interface LegendEntry {
  number: number;
  tag: string;
  id: string;
  text: string;
  dates: string[];
}

export function parseExportArgs(argv: readonly string[]): Partial<ExportOptions> & { help?: boolean } {
  const parsed: Partial<ExportOptions> & { help?: boolean } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    if (flag === "--force") { parsed.force = true; continue; }
    const take = (): string => {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) throw new Error(`${flag} needs a value`);
      index += 1;
      return value;
    };
    switch (flag) {
      case "--vault": parsed.vaultRoot = take(); break;
      case "--config": parsed.configPath = take(); break;
      case "--out": parsed.outDir = take(); break;
      case "--seed": parsed.seed = take(); break;
      case "--from": parsed.from = requireDate(flag, take()); break;
      case "--to": parsed.to = requireDate(flag, take()); break;
      case "--skipped": {
        const value = take();
        if (!/^\d+$/.test(value)) throw new Error("--skipped must be a non-negative integer");
        parsed.skippedSample = Number(value);
        break;
      }
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  if (parsed.from && parsed.to && parsed.from > parsed.to) {
    throw new Error(`--from ${parsed.from} is after --to ${parsed.to}; the start date is after the end date`);
  }
  return parsed;
}

function requireDate(flag: string, value: string): string {
  const date = /^\d{4}-\d{2}-\d{2}$/.test(value) ? new Date(`${value}T00:00:00Z`) : null;
  // `new Date("2026-13-01T00:00:00Z")` is invalid and would throw on toISOString with an unfriendly
  // RangeError, so check the parsed instant before reading it back. The round trip also rejects
  // dates the calendar normalizes, such as 2026-02-30.
  if (!date || Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== value) {
    throw new Error(`${flag} must be a date like 2026-10-14`);
  }
  return value;
}

/**
 * Every selected candidate (the model chose a topic) plus a seeded sample of
 * the skipped ones, ordered by date then id. The skipped pool is sorted before
 * it is shuffled, so the choice depends on the seed and the data, not on the
 * order records were read in.
 */
export function sampleCandidates(
  items: readonly SampledCandidate[],
  options: { skippedSample: number; seed: string },
): SampledCandidate[] {
  const order = (left: SampledCandidate, right: SampledCandidate): number =>
    compare(left.date, right.date) || compare(left.candidate.id, right.candidate.id);
  const sorted = [...items].sort(order);
  const selected = sorted.filter((item) => item.candidate.category !== "skip");
  const pool = sorted.filter((item) => item.candidate.category === "skip");
  const random = seededRandom(options.seed);
  const count = Math.min(pool.length, Math.max(0, options.skippedSample));
  for (let index = 0; index < count; index += 1) {
    const pick = index + Math.floor(random() * (pool.length - index));
    [pool[index], pool[pick]] = [pool[pick]!, pool[index]!];
  }
  return [...selected, ...pool.slice(0, count)].sort(order);
}

function compare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function seededRandom(seed: string): () => number {
  let hash = 2166136261;
  for (const char of seed) hash = Math.imul(hash ^ char.codePointAt(0)!, 16777619) >>> 0;
  let state = hash;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), state | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Number the directions across the exported days, first seen first. A direction
 * whose text was edited between days counts as a different direction, so a
 * label never silently refers to wording the model did not see.
 */
export function buildLegend(records: readonly FilterDecisionRecordDocument[]): {
  entries: LegendEntry[];
  numberOf: (direction: { tag: string; id: string; text: string }, date: string) => number;
} {
  const entries: LegendEntry[] = [];
  const byKey = new Map<string, LegendEntry>();
  const numberOf = (direction: { tag: string; id: string; text: string }, date: string): number => {
    const key = JSON.stringify([direction.tag, direction.id, direction.text]);
    let entry = byKey.get(key);
    if (!entry) {
      entry = { number: entries.length + 1, tag: direction.tag, id: direction.id, text: direction.text, dates: [] };
      entries.push(entry);
      byKey.set(key, entry);
    }
    if (!entry.dates.includes(date)) entry.dates.push(date);
    return entry.number;
  };
  for (const record of records) for (const direction of record.directions) numberOf(direction, record.reportDate);
  return { entries, numberOf };
}

export async function runExport(options: ExportOptions, deps: ExportDeps = {}): Promise<ExportResult> {
  // The fetcher's own retry chatter stays quiet; the caller hears about problems through onWarning.
  const logger = deps.logger ?? new Logger("error");
  const warnings: string[] = [];
  const warn = (message: string): void => { warnings.push(message); deps.onWarning?.(message); };
  const skippedSample = options.skippedSample ?? DEFAULT_SKIPPED_SAMPLE;
  const seed = options.seed ?? "quality-eval";
  const outDir = path.resolve(options.outDir);
  if (!options.force && existsSync(path.join(outDir, SHEET_FILES.csv))) {
    throw new Error(`${path.join(outDir, SHEET_FILES.csv)} already exists; it may hold labels. Choose another --out or pass --force to replace it`);
  }

  const config = options.configPath || !options.vaultRoot
    ? await loadCliConfig({ configPath: options.configPath })
    : undefined;
  const vaultRoot = path.resolve(options.vaultRoot ?? config!.vaultRoot);
  const output: OutputSettings = config?.settings.output ?? DEFAULT_SETTINGS.output;

  const store = new FilterDecisionRecordStore(new NodeStorageAdapter(vaultRoot), output, {
    onWarning: (message) => warn(message),
  });
  const records: FilterDecisionRecordDocument[] = [];
  for (const date of await store.listReportDates()) {
    if ((options.from && date < options.from) || (options.to && date > options.to)) continue;
    const record = await store.load(date);
    if (record) records.push(record);
  }
  if (records.length === 0) {
    throw new Error(`no filter decision records found under ${vaultRoot} for ${options.from ?? "the first day"} to ${options.to ?? "the last day"}`);
  }

  const items = records.flatMap((record) => record.candidates.map((candidate) => ({ date: record.reportDate, candidate })));
  const sampled = sampleCandidates(items, { skippedSample, seed });
  const legend = buildLegend(records);

  await mkdir(outDir, { recursive: true });
  deps.onProgress?.(`fetching abstracts for ${new Set(sampled.map((item) => item.candidate.id)).size} papers from arXiv`);
  const abstracts = await fetchAbstracts(sampled.map((item) => item.candidate.id), outDir, deps, logger, warn);

  const rows = sampled.map(({ date, candidate }) => [
    date,
    candidate.id,
    candidate.title,
    abstracts.get(candidate.id) ?? "",
    String(candidate.hadAbstract),
    candidate.category,
    candidate.directions.map((direction) => legend.numberOf(direction, date)).join(";"),
    String(candidate.relevanceScore),
    candidate.outcome,
    "", "", "",
  ]);
  const withoutAbstract = rows.filter((row) => row[3] === "").length;
  if (withoutAbstract > 0) {
    const ids = rows.filter((row) => row[3] === "").map((row) => row[1]!);
    warn(`no abstract could be fetched for ${withoutAbstract} papers (left empty): ${ids.slice(0, 20).join(", ")}${ids.length > 20 ? ", ..." : ""}`);
  }

  const selected = sampled.filter((item) => item.candidate.category !== "skip").length;
  const counts = {
    rows: rows.length,
    selected,
    skippedSampled: rows.length - selected,
    skippedAvailable: items.filter((item) => item.candidate.category === "skip").length,
    withoutAbstract,
    hadAbstractAtFilter: sampled.filter((item) => item.candidate.hadAbstract).length,
  };
  const meta = {
    schemaVersion: 1,
    createdAt: (deps.now?.() ?? new Date()).toISOString(),
    seed,
    skippedSample,
    range: { from: options.from ?? null, to: options.to ?? null },
    sourceDates: records.map((record) => record.reportDate),
    columns: SHEET_COLUMNS,
    counts,
    directions: legend.entries,
    records: records.map((record) => ({
      reportDate: record.reportDate,
      writtenAt: record.writtenAt,
      promptContractVersion: record.promptContractVersion,
      resultContractVersion: record.resultContractVersion,
      generation: record.generation,
      candidates: record.candidates.length,
    })),
  };

  const files = Object.values(SHEET_FILES).map((name) => path.join(outDir, name));
  await writeFile(path.join(outDir, SHEET_FILES.csv), encodeCsv([[...SHEET_COLUMNS], ...rows]));
  await writeFile(path.join(outDir, SHEET_FILES.guide), renderGuide(legend.entries, meta.sourceDates));
  await writeFile(path.join(outDir, SHEET_FILES.meta), `${JSON.stringify(meta, null, 2)}\n`);
  return { outDir, rows: rows.length, selected, skippedSampled: counts.skippedSampled, withoutAbstract, warnings, files };
}

/** Whether the query API said stop: a 429 (not retried since #95) or the fail-fast cooldown it leaves behind. */
function isThrottledQueryApi(error: unknown): boolean {
  return isArxivRetryDeferredError(error) || (isArxivHttpError(error) && error.status === 429);
}

function roughAbsPageDuration(papers: number): string {
  const seconds = papers * 3;
  if (seconds < 120) return `about ${seconds} seconds`;
  const minutes = Math.ceil(seconds / 60);
  return `about ${minutes} minute${minutes === 1 ? "" : "s"}`;
}

/**
 * Abstracts come through core's arXiv fetcher, which batches `id_list` queries
 * and reads/writes a metadata cache that here lives beside the sheet. Cached
 * abstracts are read up front, before the API is asked: a failed batch is
 * all-or-nothing, so this keeps its cached papers out of the failure. A 429
 * from the query API is not retried (see #95); within one process the host
 * then fails every later request at once, so this export stops asking and
 * reads all remaining papers from their arxiv.org/abs pages (about 3 s each)
 * after one summary warning instead of one warning per batch.
 */
async function fetchAbstracts(
  ids: readonly string[],
  outDir: string,
  deps: ExportDeps,
  logger: Logger,
  warn: (message: string) => void,
): Promise<Map<string, string>> {
  const metadataCache = new AtomMetadataCache({
    rootDir: ABSTRACT_CACHE_DIR,
    expiryDays: ABSTRACT_CACHE_DAYS,
    storage: new NodeStorageAdapter(outDir),
  });
  const fetcher = new ArxivFetcher({
    http: deps.http ?? new NodeHttpClient(),
    markupParser: new LinkedomMarkupParser(),
    logger,
    requestDelayMs: 3000,
    requestCoordinator: deps.requestCoordinator,
    metadataCache,
  });
  const absPages = new HtmlCache({
    rootDir: ABS_PAGE_CACHE_DIR,
    expiryDays: ABSTRACT_CACHE_DAYS,
    storage: new NodeStorageAdapter(outDir),
  });
  const parser = new LinkedomMarkupParser();
  const abstracts = new Map<string, string>();
  const unique = [...new Set(ids)];
  // Papers an earlier run already resolved from their abs page need no API request: the API
  // caches only what it returned, so without this a rerun would ask for them again.
  for (const id of unique) {
    const page = await absPages.get(`abs/${modernArxivResources(id)?.id ?? id}`, "abs");
    const abstract = page === null ? "" : extractAbsPageAbstract(page, parser);
    if (abstract) abstracts.set(id, abstract);
  }
  const pending = unique.filter((id) => !abstracts.has(id));
  // Read the metadata cache before batching, for the same reason as the abs cache above: a batch
  // that fails would otherwise discard the papers of that batch that were already cached.
  const uncached: string[] = [];
  for (const id of pending) {
    const canonical = modernArxivResources(id)?.id;
    const cached = canonical ? await metadataCache.get(canonical) : null;
    if (cached?.abstract) abstracts.set(id, cached.abstract);
    else uncached.push(id);
  }
  for (let start = 0; start < uncached.length; start += ABSTRACT_BATCH) {
    const batch = uncached.slice(start, start + ABSTRACT_BATCH);
    try {
      const found = await fetcher.fetchMetadataByIds(batch);
      for (const id of batch) {
        const canonical = modernArxivResources(id)?.id ?? id;
        const abstract = found.get(canonical)?.abstract;
        if (abstract) abstracts.set(id, abstract);
      }
    } catch (error) {
      const reason = (error as Error).message.replace(/:?\s*https?:\/\/\S+/g, "").trim();
      if (isThrottledQueryApi(error)) {
        const remaining = uncached.length - start;
        warn(`arXiv rate-limited the query API (${reason}); stopping query API requests for this export. The remaining ${remaining} papers will be read from arxiv.org/abs pages (${roughAbsPageDuration(remaining)})`);
        break;
      }
      warn(`abstract fetch failed for ${batch.length} papers (${reason}); the abstracts fetched so far are cached, and a later run reuses them`);
    }
  }

  const missing = pending.filter((id) => !abstracts.has(id));
  if (missing.length > 0) {
    deps.onProgress?.(`reading ${missing.length} abstracts from arxiv.org/abs pages`);
    await fillFromAbsPages(missing, abstracts, fetcher, absPages, parser, warn);
  }
  return abstracts;
}

async function fillFromAbsPages(
  ids: readonly string[],
  abstracts: Map<string, string>,
  fetcher: ArxivFetcher,
  cache: HtmlCache,
  parser: LinkedomMarkupParser,
  warn: (message: string) => void,
): Promise<void> {
  let consecutiveFailures = 0;
  for (const [index, id] of ids.entries()) {
    const canonical = modernArxivResources(id)?.id;
    if (!canonical) continue;
    try {
      let page = await cache.get(`abs/${canonical}`, "abs");
      if (page === null) {
        page = await fetcher.fetchPaperAbsPage(canonical);
        await cache.set(`abs/${canonical}`, "abs", page).catch(() => undefined);
      }
      consecutiveFailures = 0;
      const abstract = extractAbsPageAbstract(page, parser);
      if (abstract) abstracts.set(id, abstract);
    } catch (error) {
      consecutiveFailures += 1;
      if (consecutiveFailures >= ABS_PAGE_FAILURE_LIMIT) {
        const reason = (error as Error).message.replace(/:?\s*https?:\/\/\S+/g, "").trim();
        warn(`arxiv.org/abs page fetching stopped after ${consecutiveFailures} consecutive failures (${reason}); ${ids.length - index - 1} papers were not tried`);
        return;
      }
    }
  }
}

/** Escape one Markdown table cell: backslashes first, then pipes, and collapse whitespace (as report.ts does). */
function markdownCell(text: string): string {
  return text.replace(/\\/g, "\\\\").replace(/\|/g, "\\|").replace(/\s+/g, " ");
}

function renderGuide(legend: readonly LegendEntry[], dates: readonly string[]): string {
  const rows = legend.map((entry) => `| ${entry.number} | ${markdownCell(entry.tag)} | ${markdownCell(entry.text)} |`);
  return `# 筛选标注说明

这份表格用来检查日报的论文筛选准不准。每一行是一篇论文：先读标题和摘要，再告诉我们“这篇是否应该被选进日报”。覆盖的日报日期：${dates.join("、")}。

## 怎么填

只需要填 \`label\`、\`label_direction\`、\`note\` 三列，其余列请不要修改。

- \`label\`：\`y\` 表示这篇论文应该被选中（它确实属于你关注的某个方向），\`n\` 表示不应该。拿不准的行可以留空，评估时会当作未标注。
- \`label_direction\`：当 \`label\` 是 \`y\` 时，填它属于下面方向表里的哪个编号；属于多个方向时用英文分号隔开，例如 \`1;2\`。\`label\` 是 \`n\` 时留空。
- \`note\`：想备注的话写在这里，不影响评估。

## 建议

- 标注时先把 \`model_category\`、\`model_directions\`、\`model_score\`、\`model_outcome\` 这几列隐藏起来，不要看模型当时怎么判断的，这样标注才不会被模型带偏。
- \`had_abstract_at_filter\` 为 \`false\` 说明筛选当时模型只看到了标题（拿不到摘要）。这类行也请照常标注，评估时会单独统计。
- \`abstract\` 为空表示这次没能从 arXiv 取回摘要，请按标题判断，或打开论文链接 https://arxiv.org/abs/<id> 查看。
- 用 Excel 或 WPS 打开时，请先把 \`id\`、\`date\` 两列设为文本格式，保存时保留 CSV（UTF-8）格式。否则表格软件可能把 \`2610.10230\` 存成 \`2610.1023\`，或把日期改成别的格式；评估时这两列会按原样校验，格式不对会报错。

## 方向表

\`model_directions\` 和 \`label_direction\` 里的数字都对应下表：

| 编号 | 主题标签 | 方向 |
| --- | --- | --- |
${rows.join("\n")}
`;
}
