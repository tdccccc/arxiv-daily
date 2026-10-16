import { readFile } from "node:fs/promises";
import { isLabelled } from "./metrics";
import type { SheetRow } from "./sheet";

export interface MetaDirection { number: number; tag: string; id: string; text: string; dates?: string[] }
interface MetaRecord {
  reportDate: string;
  promptContractVersion: number;
  resultContractVersion: number;
  generation: { provider: string; model: string };
}
export interface SheetMeta {
  seed: string;
  sourceDates: string[];
  counts: { skippedAvailable?: number; skippedSampled?: number };
  directions: MetaDirection[];
  records: MetaRecord[];
}

export interface SheetSummary {
  path: string;
  metaPath: string;
  rows: number;
  labelled: number;
  unlabelled: number;
  labelledYes: number;
  labelledNo: number;
  withoutAbstractAtFilter: number;
}

export interface SourceSummary {
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
}

export const RULES = {
  positive: "A paper counts as selected by the filter when model_category is not skip; the daily cap and ignore marks are not filter decisions.",
  direction: "A true positive is right about the direction when any label_direction number is among model_directions; true positives with no label_direction are counted apart.",
  weighting: "Selected papers were all exported; skipped papers were sampled, so each labelled skipped row stands for skipped-available divided by the labelled skipped rows.",
} as const;

export function summarizeSheet(rows: readonly SheetRow[], context: { sheetPath: string; metaPath: string }): SheetSummary {
  const labelled = rows.filter(isLabelled);
  return {
    path: context.sheetPath,
    metaPath: context.metaPath,
    rows: rows.length,
    labelled: labelled.length,
    unlabelled: rows.length - labelled.length,
    labelledYes: labelled.filter((row) => row.label === "y").length,
    labelledNo: labelled.filter((row) => row.label === "n").length,
    withoutAbstractAtFilter: rows.filter((row) => !row.hadAbstractAtFilter).length,
  };
}

export function summarizeSource(meta: SheetMeta): SourceSummary {
  return {
    seed: meta.seed,
    dates: meta.sourceDates,
    skipped: {
      available: skippedAvailable(meta),
      sampled: typeof meta.counts.skippedSampled === "number" ? meta.counts.skippedSampled : null,
    },
    records: meta.records.map((record) => ({
      reportDate: record.reportDate,
      promptContractVersion: record.promptContractVersion,
      resultContractVersion: record.resultContractVersion,
      provider: record.generation.provider,
      model: record.generation.model,
    })),
  };
}

/** How many skipped candidates the exported skipped rows were sampled from, when the meta says. */
export function skippedAvailable(meta: SheetMeta): number | null {
  return typeof meta.counts.skippedAvailable === "number" ? meta.counts.skippedAvailable : null;
}

export async function readMeta(metaPath: string, implicit: boolean): Promise<SheetMeta> {
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
