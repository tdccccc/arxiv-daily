import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { isLabelled, metricSet, topicMetrics, type LabelledRow, type MetricSet, type TopicMetrics } from "./metrics";
import { renderRecordedReport } from "./report";
import { parseSheet, type SheetRow } from "./sheet";

export interface EvaluateOptions {
  sheet: string;
  /** Defaults to labels.meta.json beside the sheet. */
  meta?: string;
  outDir: string;
  /** Score the decisions recorded in the sheet instead of calling the model. */
  recorded?: boolean;
}

export interface EvaluateDeps {
  now?: () => Date;
}

export interface EvaluateResult {
  outDir: string;
  files: string[];
  report: RecordedReport;
}

interface MetaDirection { number: number; tag: string; id: string; text: string }
interface MetaRecord {
  reportDate: string;
  promptContractVersion: number;
  resultContractVersion: number;
  generation: { provider: string; model: string };
}
interface SheetMeta {
  seed: string;
  sourceDates: string[];
  counts: { skippedAvailable?: number; skippedSampled?: number };
  directions: MetaDirection[];
  records: MetaRecord[];
}

export interface Disagreement {
  row: number;
  date: string;
  id: string;
  title: string;
  score: number;
  modelCategory: string;
  modelDirections: number[];
  labelDirections: number[];
  note: string;
  hadAbstractAtFilter: boolean;
}

export interface RecordedReport {
  schemaVersion: 1;
  kind: "recorded";
  generatedAt: string;
  rules: { positive: string; direction: string; weighting: string };
  sheet: {
    path: string;
    metaPath: string;
    rows: number;
    labelled: number;
    unlabelled: number;
    labelledYes: number;
    labelledNo: number;
    withoutAbstractAtFilter: number;
  };
  source: {
    seed: string;
    dates: string[];
    skipped: { available: number | null; sampled: number | null };
    records: Array<{
      reportDate: string;
      promptContractVersion: number;
      resultContractVersion: number;
      provider: string;
      model: string;
    }>;
  };
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

export const RULES = {
  positive: "A paper counts as selected by the filter when model_category is not skip; the daily cap and ignore marks are not filter decisions.",
  direction: "A true positive is right about the direction when any label_direction number is among model_directions; true positives with no label_direction are counted apart.",
  weighting: "Selected papers were all exported; skipped papers were sampled, so each labelled skipped row stands for skipped-available divided by the labelled skipped rows.",
} as const;

export function parseEvaluateArgs(argv: readonly string[]): Partial<EvaluateOptions> & { help?: boolean } {
  const parsed: Partial<EvaluateOptions> & { help?: boolean } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    if (flag === "--recorded") { parsed.recorded = true; continue; }
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
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.sheet) throw new Error("--sheet <labels.csv> is required");
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  return parsed;
}

export async function runEvaluate(options: EvaluateOptions, deps: EvaluateDeps = {}): Promise<EvaluateResult> {
  if (!options.recorded) {
    throw new Error("only --recorded scoring is available so far; rerunning the model against the sheet is not implemented yet");
  }
  const sheetPath = path.resolve(options.sheet);
  const metaPath = path.resolve(options.meta ?? path.join(path.dirname(sheetPath), "labels.meta.json"));
  const meta = await readMeta(metaPath, options.meta === undefined);
  const rows = parseSheet(await readFile(sheetPath, "utf8"), new Set(meta.directions.map((direction) => direction.number)));
  const report = buildRecordedReport(rows, meta, { sheetPath, metaPath, now: deps.now?.() ?? new Date() });

  const outDir = path.resolve(options.outDir);
  await mkdir(outDir, { recursive: true });
  const files = [path.join(outDir, "report.md"), path.join(outDir, "report.json")];
  await writeFile(files[0]!, renderRecordedReport(report));
  await writeFile(files[1]!, `${JSON.stringify(report, null, 2)}\n`);
  return { outDir, files, report };
}

export function buildRecordedReport(
  rows: readonly SheetRow[],
  meta: SheetMeta,
  context: { sheetPath: string; metaPath: string; now: Date },
): RecordedReport {
  const labelled: LabelledRow[] = rows.filter(isLabelled);
  const available = typeof meta.counts.skippedAvailable === "number" ? meta.counts.skippedAvailable : null;
  const tagOf = new Map(meta.directions.map((direction) => [direction.number, direction.tag]));
  const tags = [...new Set(meta.directions.map((direction) => direction.tag))];
  const { topics, withoutDirection } = topicMetrics(labelled, tags, (number) => tagOf.get(number));

  const disagreement = (row: LabelledRow): Disagreement => ({
    row: row.row, date: row.date, id: row.id, title: row.title, score: row.modelScore, modelCategory: row.modelCategory,
    modelDirections: row.modelDirections, labelDirections: row.labelDirections, note: row.note, hadAbstractAtFilter: row.hadAbstractAtFilter,
  });
  const byDateAndId = (left: Disagreement, right: Disagreement): number =>
    left.date < right.date ? -1 : left.date > right.date ? 1 : left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
  const selected = (row: LabelledRow): boolean => row.modelCategory !== "skip";
  const truePositives = labelled.filter((row) => row.label === "y" && selected(row));

  return {
    schemaVersion: 1,
    kind: "recorded",
    generatedAt: context.now.toISOString(),
    rules: { ...RULES },
    sheet: {
      path: context.sheetPath,
      metaPath: context.metaPath,
      rows: rows.length,
      labelled: labelled.length,
      unlabelled: rows.length - labelled.length,
      labelledYes: labelled.filter((row) => row.label === "y").length,
      labelledNo: labelled.filter((row) => row.label === "n").length,
      withoutAbstractAtFilter: rows.filter((row) => !row.hadAbstractAtFilter).length,
    },
    source: {
      seed: meta.seed,
      dates: meta.sourceDates,
      skipped: {
        available,
        sampled: typeof meta.counts.skippedSampled === "number" ? meta.counts.skippedSampled : null,
      },
      records: meta.records.map((record) => ({
        reportDate: record.reportDate,
        promptContractVersion: record.promptContractVersion,
        resultContractVersion: record.resultContractVersion,
        provider: record.generation.provider,
        model: record.generation.model,
      })),
    },
    directions: meta.directions,
    sets: {
      all: metricSet(labelled, available),
      // How many skipped candidates had an abstract is not recorded, so this line stays unweighted.
      withAbstract: metricSet(labelled.filter((row) => row.hadAbstractAtFilter), null),
    },
    topics,
    topicsWithoutDirection: withoutDirection,
    disagreements: {
      falseNegatives: labelled.filter((row) => row.label === "y" && !selected(row)).map(disagreement).sort(byDateAndId),
      falsePositives: labelled.filter((row) => row.label === "n" && selected(row)).map(disagreement).sort(byDateAndId),
      directionMismatches: truePositives
        .filter((row) => row.labelDirections.length > 0 && !row.labelDirections.some((number) => row.modelDirections.includes(number)))
        .map(disagreement).sort(byDateAndId),
    },
  };
}

async function readMeta(metaPath: string, implicit: boolean): Promise<SheetMeta> {
  let raw: string;
  try {
    raw = await readFile(metaPath, "utf8");
  } catch (error) {
    throw new Error(`cannot read ${metaPath}${implicit ? " (expected beside the sheet; pass --meta <labels.meta.json>)" : ""}: ${(error as Error).message}`);
  }
  let meta: Partial<SheetMeta>;
  try {
    meta = JSON.parse(raw) as Partial<SheetMeta>;
  } catch {
    throw new Error(`${metaPath} is not valid JSON`);
  }
  if (!Array.isArray(meta.directions) || !Array.isArray(meta.records) || !Array.isArray(meta.sourceDates) || typeof meta.counts !== "object" || meta.counts === null) {
    throw new Error(`${metaPath} is not a labels.meta.json written by export`);
  }
  return { seed: String(meta.seed ?? ""), ...meta } as SheetMeta;
}
