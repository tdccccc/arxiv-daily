import { access, constants, mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import type { LlmCallMetrics } from "@arxiv-daily/core";
import {
  disagreementsOf, isLabelled, metricSet, topicMetrics,
  type Disagreement, type LabelledRow, type MetricSet, type TopicMetrics,
} from "./metrics";
import { renderRecordedReport, renderRerunReport } from "./report";
import { runRerun, type RerunReport } from "./rerun";
import { parseSheet, type SheetRow } from "./sheet";
import {
  RULES, readMeta, skippedAvailable, summarizeSheet, summarizeSource,
  type MetaDirection, type SheetMeta, type SheetSummary, type SourceSummary,
} from "./source";

export const MAX_RUNS = 20;

export interface EvaluateOptions {
  sheet: string;
  /** Defaults to labels.meta.json beside the sheet. */
  meta?: string;
  outDir: string;
  /** Score the decisions recorded in the sheet instead of calling the model. */
  recorded?: boolean;
  /** Rerun the real filter this many times (default 1). */
  runs?: number;
  /** Confirm that the model calls may be made; without it the plan is printed and nothing is called. */
  yes?: boolean;
  /** CLI config supplying the LLM settings (and topics with `topicsFromConfig`). */
  configPath?: string;
  /** Use the config's topics instead of the direction snapshot stored with the sheet. */
  topicsFromConfig?: boolean;
  /** Replace an existing report.md/report.json in `outDir`. */
  force?: boolean;
}

export interface EvaluateDeps {
  now?: () => Date;
  /** A model client; tests pass a fake, otherwise one is built from the config. */
  llm?: { call(messages: Array<{ role: "system" | "user" | "assistant"; content: string }>, options?: { temperature?: number; onMetrics?: (metrics: LlmCallMetrics) => void }): Promise<string> };
  /** Progress lines, including the plan printed before any model call. */
  log?: (line: string) => void;
}

export interface EvaluateResult {
  outDir: string;
  files: string[];
  report: RecordedReport | RerunReport | null;
  /** True when `--yes` was missing: the plan was printed and no model was called. */
  dryRun?: boolean;
  allRunsFailed?: boolean;
}

export interface RecordedReport {
  schemaVersion: 1;
  kind: "recorded";
  generatedAt: string;
  rules: { positive: string; direction: string; weighting: string };
  sheet: SheetSummary;
  source: SourceSummary;
  directions: MetaDirection[];
  sets: { all: MetricSet; withAbstract: MetricSet };
  topics: TopicMetrics[];
  topicsWithoutDirection: number;
  disagreements: {
    falseNegatives: Disagreement[];
    falsePositives: Disagreement[];
    directionMismatches: Disagreement[];
  };
}

export function parseEvaluateArgs(argv: readonly string[]): Partial<EvaluateOptions> & { help?: boolean } {
  const parsed: Partial<EvaluateOptions> & { help?: boolean } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    if (flag === "--recorded") { parsed.recorded = true; continue; }
    if (flag === "--yes") { parsed.yes = true; continue; }
    if (flag === "--force") { parsed.force = true; continue; }
    if (flag === "--topics-from-config") { parsed.topicsFromConfig = true; continue; }
    const take = (): string => {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) throw new Error(`${flag} needs a value`);
      index += 1;
      return value;
    };
    switch (flag) {
      case "--sheet": parsed.sheet = take(); break;
      case "--meta": parsed.meta = take(); break;
      case "--out": parsed.outDir = take(); break;
      case "--config": parsed.configPath = take(); break;
      case "--runs": {
        const value = take();
        if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > MAX_RUNS) {
          throw new Error(`--runs must be a whole number from 1 to ${MAX_RUNS}`);
        }
        parsed.runs = Number(value);
        break;
      }
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.sheet) throw new Error("--sheet <labels.csv> is required");
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  if (parsed.recorded) {
    const conflicting = (["runs", "yes", "configPath", "topicsFromConfig"] as const).filter((key) => parsed[key] !== undefined);
    if (conflicting.length > 0) throw new Error("--recorded scores saved decisions and cannot be combined with --runs, --yes, --config or --topics-from-config");
  }
  return parsed;
}

export async function runEvaluate(options: EvaluateOptions, deps: EvaluateDeps = {}): Promise<EvaluateResult> {
  const sheetPath = path.resolve(options.sheet);
  const metaPath = path.resolve(options.meta ?? path.join(path.dirname(sheetPath), "labels.meta.json"));
  const meta = await readMeta(metaPath, options.meta === undefined);
  const rows = parseSheet(await readFile(sheetPath, "utf8"), new Set(meta.directions.map((direction) => direction.number)));
  const context = { sheetPath, metaPath, now: deps.now?.() ?? new Date() };
  const outDir = path.resolve(options.outDir);
  const files = [path.join(outDir, "report.md"), path.join(outDir, "report.json")];

  // Prepare the output before any model call: a bad directory or an existing report would
  // otherwise surface only after the paid calls finished, losing them.
  if (!options.force) {
    const existing = files.find((file) => existsSync(file));
    if (existing) {
      throw new Error(`${existing} already exists; choose another --out or pass --force to replace it`);
    }
  }
  const dryRun = !options.recorded && !options.yes;
  if (dryRun) {
    // A dry run must not create anything (its plan is printed and nothing is written), but it
    // can still fail early when the nearest existing parent could not hold the report.
    await assertWritableTarget(outDir);
  } else {
    await prepareOutDir(outDir);
  }

  let report: RecordedReport | RerunReport;
  let allRunsFailed = false;
  if (options.recorded) {
    report = buildRecordedReport(rows, meta, context);
  } else {
    const rerun = await runRerun(rows, meta, options, deps, context);
    if (rerun.dryRun) return { outDir, files: [], report: null, dryRun: true };
    report = rerun.report!;
    allRunsFailed = rerun.allRunsFailed;
  }

  await writeFile(files[0]!, report.kind === "recorded" ? renderRecordedReport(report) : renderRerunReport(report));
  await writeFile(files[1]!, `${JSON.stringify(report, null, 2)}\n`);
  return { outDir, files, report, allRunsFailed };
}

/** The nearest path that exists, so a missing `outDir` can be checked without creating it. */
async function nearestExistingPath(target: string): Promise<string> {
  let current = target;
  while (true) {
    try {
      await stat(current);
      return current;
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "ENOENT" && code !== "ENOTDIR") throw error;
      const parent = path.dirname(current);
      if (parent === current) return current;
      current = parent;
    }
  }
}

/** Fail with a clear message when `target` cannot become a writable directory (creating nothing). */
async function assertWritableTarget(target: string): Promise<void> {
  const existing = await nearestExistingPath(target);
  const info = await stat(existing);
  if (!info.isDirectory()) {
    throw new Error(`cannot write reports to ${target}: ${existing} is not a directory`);
  }
  try {
    await access(existing, constants.W_OK);
  } catch {
    throw new Error(`cannot write reports to ${target}: ${existing} is not writable`);
  }
}

async function prepareOutDir(outDir: string): Promise<void> {
  await assertWritableTarget(outDir);
  await mkdir(outDir, { recursive: true });
  // Write and remove a probe so a directory that exists but is not writable fails here,
  // before the model calls, rather than when the report is finally written.
  const probe = path.join(outDir, `.write-test-${process.pid}-${Date.now()}`);
  try {
    await writeFile(probe, "");
  } catch (error) {
    throw new Error(`cannot write to output directory ${outDir}: ${(error as Error).message}`);
  } finally {
    await rm(probe, { force: true }).catch(() => undefined);
  }
}

export function buildRecordedReport(
  rows: readonly SheetRow[],
  meta: SheetMeta,
  context: { sheetPath: string; metaPath: string; now: Date },
): RecordedReport {
  const labelled: LabelledRow[] = rows.filter(isLabelled);
  const available = skippedAvailable(meta);
  const tagOf = new Map(meta.directions.map((direction) => [direction.number, direction.tag]));
  const tags = [...new Set(meta.directions.map((direction) => direction.tag))];
  const { topics, withoutDirection } = topicMetrics(labelled, tags, (number) => tagOf.get(number));

  return {
    schemaVersion: 1,
    kind: "recorded",
    generatedAt: context.now.toISOString(),
    rules: { ...RULES },
    sheet: summarizeSheet(rows, context),
    source: summarizeSource(meta),
    directions: meta.directions,
    sets: {
      all: metricSet(labelled, available),
      // How many skipped candidates had an abstract is not recorded, so this line stays unweighted.
      withAbstract: metricSet(labelled.filter((row) => row.hadAbstractAtFilter), null),
    },
    topics,
    topicsWithoutDirection: withoutDirection,
    disagreements: disagreementsOf(labelled),
  };
}
