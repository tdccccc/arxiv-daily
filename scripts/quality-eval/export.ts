import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import {
  ArxivFetcher,
  AtomMetadataCache,
  DEFAULT_SETTINGS,
  FilterDecisionRecordStore,
  Logger,
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
const ABSTRACT_CACHE_DAYS = 3650;
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
  return parsed;
}

function requireDate(flag: string, value: string): string {
  const valid = /^\d{4}-\d{2}-\d{2}$/.test(value) && new Date(`${value}T00:00:00Z`).toISOString().startsWith(value);
  if (!valid) throw new Error(`${flag} must be a date like 2026-10-14`);
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
  deps.onProgress?.(`fetching abstracts for ${new Set(sampled.map((item) => item.candidate.id)).size} papers from arXiv (waits and retries while arXiv rate-limits)`);
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

/**
 * Abstracts come through core's arXiv fetcher: it batches `id_list` queries,
 * waits out HTTP 429 with backoff, and reads/writes a metadata cache, which
 * here lives beside the sheet so an interrupted export resumes without
 * refetching. A failed batch costs only that batch's abstracts.
 */
async function fetchAbstracts(
  ids: readonly string[],
  outDir: string,
  deps: ExportDeps,
  logger: Logger,
  warn: (message: string) => void,
): Promise<Map<string, string>> {
  const fetcher = new ArxivFetcher({
    http: deps.http ?? new NodeHttpClient(),
    markupParser: new LinkedomMarkupParser(),
    logger,
    requestDelayMs: 3000,
    requestCoordinator: deps.requestCoordinator,
    metadataCache: new AtomMetadataCache({
      rootDir: ABSTRACT_CACHE_DIR,
      expiryDays: ABSTRACT_CACHE_DAYS,
      storage: new NodeStorageAdapter(outDir),
    }),
  });
  const abstracts = new Map<string, string>();
  const unique = [...new Set(ids)];
  for (let start = 0; start < unique.length; start += ABSTRACT_BATCH) {
    const batch = unique.slice(start, start + ABSTRACT_BATCH);
    try {
      const found = await fetcher.fetchMetadataByIds(batch);
      for (const id of batch) {
        const canonical = modernArxivResources(id)?.id ?? id;
        const abstract = found.get(canonical)?.abstract;
        if (abstract) abstracts.set(id, abstract);
      }
    } catch (error) {
      const reason = (error as Error).message.replace(/:?\s*https?:\/\/\S+/g, "").trim();
      warn(`abstract fetch failed for ${batch.length} papers (${reason}); fetched ones stay cached, so a later run with --force picks up where this one stopped`);
    }
  }
  return abstracts;
}

function renderGuide(legend: readonly LegendEntry[], dates: readonly string[]): string {
  const rows = legend.map((entry) => `| ${entry.number} | ${entry.tag} | ${entry.text.replace(/\|/g, "\\|").replace(/\s+/g, " ")} |`);
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
- 用 Excel 或 WPS 保存时请保留 CSV（UTF-8）格式。

## 方向表

\`model_directions\` 和 \`label_direction\` 里的数字都对应下表：

| 编号 | 主题标签 | 方向 |
| --- | --- | --- |
${rows.join("\n")}
`;
}
