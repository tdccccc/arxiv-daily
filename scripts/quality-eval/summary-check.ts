import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { DEFAULT_SETTINGS, modernArxivResources, type Settings } from "@arxiv-daily/core";
import dailyPromptZh from "../../packages/core/src/prompts/daily-paper-summary.system.md";
import detailPromptZh from "../../packages/core/src/prompts/paper-detail.system.md";
import { loadCliConfig } from "../../apps/cli/src/config";
import { assertWritableTarget, prepareOutDir } from "./evaluate";
import {
  aggregateGrounding, checkGrounding, indexOfSource, stripFencedCode,
  type GroundingAggregate, type SourceIndex, type SourceTexts, type UngroundedNumber,
} from "./summary-grounding";
import {
  checkSummaries, MAIN_RESULT_LENGTH_RANGE, monthlyMainResult, SUMMARY_FIELDS,
  type MonthlyMainResult, type SummaryChecksResult, type SummaryFieldKey,
} from "./summary-checks";
import { checkNotes, type NoteChecksResult } from "./summary-note-checks";
import { loadSummaryCorpus, type SummaryCorpus } from "./summary-corpus";
import { renderSummaryReport } from "./summary-report";

const REPORT_FILES = { report: "report.md", findings: "findings.json" } as const;

export interface SummaryCheckOptions {
  /** Vault root; defaults to the CLI config's `vault_root`. */
  vaultRoot?: string;
  /** CLI config supplying the vault root and output layout; the default config path when no vault is given. */
  configPath?: string;
  /** Relative to the vault; defaults to the config's (or the built-in) daily directory. */
  dailyDir?: string;
  papersDir?: string;
  from?: string;
  to?: string;
  /** Cached sources from summary-sources.mjs; when given, the numbers are looked up in them. */
  sourcesDir?: string;
  outDir: string;
  /** Replace an existing report.md/findings.json in `outDir`. */
  force?: boolean;
}

export interface SummaryCheckDeps { now?: () => Date }

export interface PromptHash { file: string; sha256: string }

export interface SummaryCheckReport {
  schemaVersion: 1;
  generatedAt: string;
  scope: { vaultRoot: string; dailyDir: string; papersDir: string; range: { from: string | null; to: string | null } };
  /** The prompts as bundled when the check ran; reports and notes do not record which prompt wrote them. */
  prompts: { dailySummary: PromptHash; paperDetail: PromptHash };
  limits: { mainResultRange: { min: number; max: number } };
  corpus: Pick<SummaryCorpus, "counts" | "legacyReportDates" | "fallbacks">;
  daily: SummaryChecksResult;
  notes: NoteChecksResult;
  monthlyMainResult: MonthlyMainResult[];
  grounding: GroundingReport;
}

/** How one entry's or note's numbers fared against its cached source. */
export interface GroundingItem {
  id: string;
  date: string | null;
  title: string;
  sourcePath: string;
  arxivVersion: string | null;
  fetchedAt: string;
  total: number;
  grounded: number;
  byPlace: Record<string, { grounded: number; total: number }>;
  ungrounded: UngroundedNumber[];
}

export interface SourceCoverage { withSource: number; withoutSource: number }

export type GroundingReport =
  | { run: false }
  | {
    run: true;
    sourcesDir: string;
    sources: {
      daily: SourceCoverage;
      notes: SourceCoverage;
      /** The cached records the check used (one per paper, however many entries cite it). */
      records: { count: number; bySourcePath: Record<string, number>; byArxivVersion: Record<string, number>; fetchedAt: { first: string; last: string } | null };
    };
    daily: { results: GroundingItem[]; aggregate: GroundingAggregate };
    notes: { results: GroundingItem[]; aggregate: GroundingAggregate };
  };

export interface SummaryCheckResult { outDir: string; files: string[]; report: SummaryCheckReport }

export function parseSummaryCheckArgs(argv: readonly string[]): Partial<SummaryCheckOptions> & { help?: boolean } {
  const parsed: Partial<SummaryCheckOptions> & { help?: boolean } = {};
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
      case "--daily-dir": parsed.dailyDir = take(); break;
      case "--papers-dir": parsed.papersDir = take(); break;
      case "--out": parsed.outDir = take(); break;
      case "--sources": parsed.sourcesDir = take(); break;
      case "--from": parsed.from = requireDate(flag, take()); break;
      case "--to": parsed.to = requireDate(flag, take()); break;
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  if (parsed.from && parsed.to && parsed.from > parsed.to) {
    throw new Error(`--from ${parsed.from} is after --to ${parsed.to}; the start date is after the end date`);
  }
  return parsed;
}

export function requireDate(flag: string, value: string): string {
  const date = /^\d{4}-\d{2}-\d{2}$/.test(value) ? new Date(`${value}T00:00:00Z`) : null;
  if (!date || Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== value) {
    throw new Error(`${flag} must be a date like 2026-10-14`);
  }
  return value;
}

export interface VaultLayoutOptions {
  vaultRoot?: string;
  configPath?: string;
  dailyDir?: string;
  papersDir?: string;
}

/**
 * Where the daily reports and detail notes are. `--vault` wins over the config's vault root, explicit
 * directories win over the config's output layout, and the built-in defaults fill the rest. `settings`
 * are the config's when one was read, the defaults otherwise.
 */
export async function resolveVaultLayout(options: VaultLayoutOptions): Promise<{ vaultRoot: string; dailyDir: string; papersDir: string; settings: Settings }> {
  const config = options.configPath || !options.vaultRoot
    ? await loadCliConfig({ configPath: options.configPath })
    : undefined;
  const settings = config?.settings ?? DEFAULT_SETTINGS;
  return {
    vaultRoot: path.resolve(options.vaultRoot ?? config!.vaultRoot),
    dailyDir: options.dailyDir ?? settings.output.dailyDir,
    papersDir: options.papersDir ?? settings.output.papersDir,
    settings,
  };
}

const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");

/** Read the vault's daily entries and detail notes, run the offline checks and write report.md and findings.json. No model, no network. */
export async function runSummaryCheck(options: SummaryCheckOptions, deps: SummaryCheckDeps = {}): Promise<SummaryCheckResult> {
  const outDir = path.resolve(options.outDir);
  const files = [path.join(outDir, REPORT_FILES.report), path.join(outDir, REPORT_FILES.findings)];
  if (!options.force) {
    const existing = files.find((file) => existsSync(file));
    if (existing) throw new Error(`${existing} already exists; choose another --out or pass --force to replace it`);
  }
  await assertWritableTarget(outDir);
  if (options.sourcesDir) await requireDirectory(options.sourcesDir, "--sources");

  const { vaultRoot, dailyDir, papersDir } = await resolveVaultLayout(options);

  const corpus = await loadSummaryCorpus({ vaultRoot, dailyDir, papersDir, from: options.from, to: options.to });
  if (corpus.counts.reports === 0 && corpus.counts.notes === 0 && corpus.counts.legacyNotes === 0) {
    throw new Error(`no daily reports or detail notes found under ${vaultRoot} (looked in ${dailyDir} and ${papersDir}) for ${options.from ?? "the first day"} to ${options.to ?? "the last day"}`);
  }

  const report: SummaryCheckReport = {
    schemaVersion: 1,
    generatedAt: (deps.now?.() ?? new Date()).toISOString(),
    scope: { vaultRoot, dailyDir, papersDir, range: { from: options.from ?? null, to: options.to ?? null } },
    prompts: {
      dailySummary: { file: "packages/core/src/prompts/daily-paper-summary.system.md", sha256: sha256(dailyPromptZh) },
      paperDetail: { file: "packages/core/src/prompts/paper-detail.system.md", sha256: sha256(detailPromptZh) },
    },
    limits: { mainResultRange: { ...MAIN_RESULT_LENGTH_RANGE } },
    corpus: { counts: corpus.counts, legacyReportDates: corpus.legacyReportDates, fallbacks: corpus.fallbacks },
    daily: checkSummaries(corpus),
    notes: checkNotes(corpus.notes),
    monthlyMainResult: monthlyMainResult(corpus.entries),
    grounding: options.sourcesDir ? await checkAgainstSources(corpus, path.resolve(options.sourcesDir)) : { run: false },
  };

  await prepareOutDir(outDir);
  await writeFile(files[0]!, renderSummaryReport(report));
  await writeFile(files[1]!, `${JSON.stringify(report, null, 2)}\n`);
  return { outDir, files, report };
}

async function requireDirectory(directory: string, flag: string): Promise<void> {
  const info = await stat(directory).catch(() => null);
  if (!info) throw new Error(`${flag} ${directory} does not exist`);
  if (!info.isDirectory()) throw new Error(`${flag} ${directory} is not a directory`);
}

interface CachedSource {
  record: { id: string; fetchedAt: string; sourcePath: string; arxivVersion: string | null; content: SourceTexts & { abstract: string } };
  index: SourceIndex;
}

/** Look the numbers of every entry and note that has a cached source up in that source; the rest are counted, not checked. */
async function checkAgainstSources(corpus: SummaryCorpus, sourcesDir: string): Promise<GroundingReport> {
  const cache = new Map<string, CachedSource | null>();
  const load = async (id: string): Promise<CachedSource | null> => {
    const canonical = modernArxivResources(id)?.id;
    if (!canonical) return null;
    if (!cache.has(canonical)) {
      let found: CachedSource | null = null;
      try {
        const record = JSON.parse(await readFile(path.join(sourcesDir, `${canonical}.json`), "utf8")) as CachedSource["record"] & { schemaVersion?: number };
        if (record.schemaVersion === 1 && record.content) found = { record, index: indexOfSource(record.content) };
      } catch { /* no usable record: the paper takes no part */ }
      cache.set(canonical, found);
    }
    return cache.get(canonical)!;
  };
  const item = (id: string, date: string | null, title: string, source: CachedSource, written: Array<{ where: string; text: string }>): GroundingItem => {
    const result = checkGrounding(written, source.index);
    return {
      id, date, title, sourcePath: source.record.sourcePath, arxivVersion: source.record.arxivVersion, fetchedAt: source.record.fetchedAt,
      total: result.total, grounded: result.grounded, byPlace: result.byWhere, ungrounded: result.ungrounded,
    };
  };

  const daily: GroundingItem[] = [];
  const notes: GroundingItem[] = [];
  const coverage = { daily: { withSource: 0, withoutSource: 0 }, notes: { withSource: 0, withoutSource: 0 } };
  for (const entry of corpus.entries) {
    const source = await load(entry.id);
    if (!source) { coverage.daily.withoutSource += 1; continue; }
    coverage.daily.withSource += 1;
    daily.push(item(entry.id, entry.date, entry.title, source, SUMMARY_FIELDS.map((field: SummaryFieldKey) => ({ where: field, text: entry.fields[field] }))));
  }
  for (const note of corpus.notes) {
    const source = await load(note.id);
    if (!source) { coverage.notes.withoutSource += 1; continue; }
    coverage.notes.withSource += 1;
    notes.push(item(note.id, note.date, note.title, source, note.sections.map((section) => ({ where: section.heading, text: stripFencedCode(section.body) }))));
  }

  const used = [...cache.values()].filter((value): value is CachedSource => value !== null)
    .filter((value) => [...daily, ...notes].some((entry) => entry.id === value.record.id));
  const count = (values: string[]) => values.reduce<Record<string, number>>((all, value) => ({ ...all, [value]: (all[value] ?? 0) + 1 }), {});
  const times = used.map((value) => value.record.fetchedAt).sort();
  const asResult = (items: GroundingItem[]) => items.map((entry) => ({ total: entry.total, grounded: entry.grounded, byWhere: entry.byPlace, ungrounded: entry.ungrounded }));
  return {
    run: true, sourcesDir,
    sources: {
      ...coverage,
      records: {
        count: used.length, bySourcePath: count(used.map((value) => value.record.sourcePath)),
        byArxivVersion: count(used.map((value) => value.record.arxivVersion ?? "unknown")),
        fetchedAt: times.length ? { first: times[0]!, last: times.at(-1)! } : null,
      },
    },
    daily: { results: daily, aggregate: aggregateGrounding(asResult(daily)) },
    notes: { results: notes, aggregate: aggregateGrounding(asResult(notes)) },
  };
}
