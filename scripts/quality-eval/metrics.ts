import type { SheetRow } from "./sheet";

export const SCORE_THRESHOLDS = [0, 50, 60, 70, 80, 90] as const;

/** A ratio and the count behind it; `value` is null when that count is zero. */
export interface Ratio {
  value: number | null;
  /** The denominator in rows. */
  n: number;
  /** The numerator, in rows. */
  hits: number;
}

export interface Confusion {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface ScoreSummary {
  precision: Ratio;
  recall: Ratio;
  f1: Ratio;
  counts: Confusion;
}

export interface WeightedSummary {
  /** How many skipped candidates each labelled skipped row stands for. */
  skippedWeight: number;
  precision: Ratio;
  recall: Ratio;
  f1: Ratio;
}

export interface DirectionAccuracy extends Ratio {
  correct: number;
  /** True positives whose label names no direction, left out of the accuracy. */
  withoutLabelledDirection: number;
}

export interface MetricSet extends ScoreSummary {
  rows: number;
  /** Null when the number of skipped candidates the sample stands for is unknown. */
  weighted: WeightedSummary | null;
  direction: DirectionAccuracy;
  thresholds: Array<ScoreSummary & { threshold: number; weighted: WeightedSummary | null }>;
}

export interface TopicMetrics {
  tag: string;
  precision: Ratio;
  recall: Ratio;
  /**
   * The same two metrics with skipped-stratum rows weighted like the overall recall;
   * null when the skipped population or the labelled skipped rows are unknown.
   */
  weighted: { precision: Ratio; recall: Ratio } | null;
}

export type LabelledRow = SheetRow & { label: "y" | "n" };

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

export const isLabelled = (row: SheetRow): row is LabelledRow => row.label !== null;
const isSelected = (row: SheetRow): boolean => row.modelCategory !== "skip";
/** Whether the row came from the down-sampled stratum of papers the recorded filter skipped. */
const fromSkippedStratum = (row: SheetRow): boolean => (row.sampledAs ?? (isSelected(row) ? "selected" : "skipped")) === "skipped";

function ratio(hits: number, n: number): Ratio {
  return { value: n > 0 ? hits / n : null, n, hits };
}

/**
 * F1 is undefined only when there is nothing to score at all (TP, FP and FN are all zero).
 * With no true positives but some errors the F1 is 0, even though precision or recall alone
 * may be undefined; showing n/a there would hide a rerun that selected nothing.
 */
function f1Of(tp: number, fp: number, fn: number): Ratio {
  const denominator = 2 * tp + fp + fn;
  return { value: denominator > 0 ? (2 * tp) / denominator : null, n: denominator, hits: 2 * tp };
}

/** Weighted confusion sums for a rule that decides which rows count as selected. */
function confuse(rows: readonly LabelledRow[], selected: (row: SheetRow) => boolean, weightOf: (row: SheetRow) => number): Confusion {
  const counts: Confusion = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (const row of rows) {
    const weight = weightOf(row);
    const positive = row.label === "y";
    if (selected(row)) counts[positive ? "tp" : "fp"] += weight;
    else counts[positive ? "fn" : "tn"] += weight;
  }
  return counts;
}

function summarize(counts: Confusion): ScoreSummary {
  const precision = ratio(counts.tp, counts.tp + counts.fp);
  const recall = ratio(counts.tp, counts.tp + counts.fn);
  return { precision, recall, f1: f1Of(counts.tp, counts.fp, counts.fn), counts };
}

/**
 * The skipped sample stands for `skippedAvailable` candidates, so each labelled skipped
 * row counts for that many divided by the labelled skipped rows; selected rows were all
 * exported and count once. Null when the population is unknown or no skipped row is labelled.
 */
export function skippedWeight(rows: readonly LabelledRow[], skippedAvailable: number | null): number | null {
  const labelledSkipped = rows.filter(fromSkippedStratum).length;
  return skippedAvailable !== null && skippedAvailable > 0 && labelledSkipped > 0 ? skippedAvailable / labelledSkipped : null;
}

function weighted(rows: readonly LabelledRow[], selected: (row: SheetRow) => boolean, weight: number | null): WeightedSummary | null {
  if (weight === null) return null;
  const { precision, recall, f1 } = summarize(confuse(rows, selected, (row) => (fromSkippedStratum(row) ? weight : 1)));
  return { skippedWeight: weight, precision, recall, f1 };
}

/**
 * A true positive is right about the direction when any labelled direction is one of the
 * model's directions. Rows labelled y with no direction cannot be judged and are counted apart.
 */
export function directionAccuracy(rows: readonly LabelledRow[]): DirectionAccuracy {
  const truePositives = rows.filter((row) => row.label === "y" && isSelected(row));
  const judged = truePositives.filter((row) => row.labelDirections.length > 0);
  const correct = judged.filter((row) => row.labelDirections.some((number) => row.modelDirections.includes(number))).length;
  return { ...ratio(correct, judged.length), correct, withoutLabelledDirection: truePositives.length - judged.length };
}

/**
 * `skippedAvailable` is the population the exported skipped rows were sampled from;
 * pass null where that population is not known for these rows.
 */
export function metricSet(rows: readonly LabelledRow[], skippedAvailable: number | null): MetricSet {
  const weight = skippedWeight(rows, skippedAvailable);
  return {
    rows: rows.length,
    ...summarize(confuse(rows, isSelected, () => 1)),
    weighted: weighted(rows, isSelected, weight),
    direction: directionAccuracy(rows),
    thresholds: SCORE_THRESHOLDS.map((threshold) => {
      const selected = (row: SheetRow): boolean => isSelected(row) && row.modelScore >= threshold;
      return { threshold, ...summarize(confuse(rows, selected, () => 1)), weighted: weighted(rows, selected, weight) };
    }),
  };
}

/**
 * Per topic, over the labelled rows: precision is how many of the papers the model filed
 * under the topic were labelled y; recall is how many papers labelled y with a direction
 * of that topic the model filed under it. Papers labelled y without a direction cannot be
 * assigned a topic and are counted in the second return value.
 *
 * The in-sample ratios count every row once. The weighted ones use the same rule as the
 * overall recall: a row from the sampled skipped stratum stands for `skippedWeight`
 * candidates, every other row for one. `skippedAvailable` is the population the exported
 * skipped rows were sampled from; pass null where that population is not known.
 */
export function topicMetrics(
  rows: readonly LabelledRow[],
  tags: readonly string[],
  tagOfNumber: (number: number) => string | undefined,
  skippedAvailable: number | null = null,
): { topics: TopicMetrics[]; withoutDirection: number } {
  const weight = skippedWeight(rows, skippedAvailable);
  const weightedRatio = (hits: readonly LabelledRow[], total: readonly LabelledRow[]): Ratio => {
    let hitWeight = 0;
    let totalWeight = 0;
    for (const row of total) {
      const rowWeight = weight !== null && fromSkippedStratum(row) ? weight : 1;
      totalWeight += rowWeight;
      if (hits.includes(row)) hitWeight += rowWeight;
    }
    return { value: totalWeight > 0 ? hitWeight / totalWeight : null, n: totalWeight, hits: hitWeight };
  };
  const topics = tags.map((tag) => {
    const filed = rows.filter((row) => row.modelCategory === tag);
    const wanted = rows.filter((row) => row.label === "y" && row.labelDirections.some((number) => tagOfNumber(number) === tag));
    const filedYes = filed.filter((row) => row.label === "y");
    const wantedFiled = wanted.filter((row) => row.modelCategory === tag);
    return {
      tag,
      precision: ratio(filedYes.length, filed.length),
      recall: ratio(wantedFiled.length, wanted.length),
      weighted: weight === null ? null : { precision: weightedRatio(filedYes, filed), recall: weightedRatio(wantedFiled, wanted) },
    };
  });
  return { topics, withoutDirection: rows.filter((row) => row.label === "y" && row.labelDirections.length === 0).length };
}

/** Papers the model got wrong, and true positives filed under another direction, ordered by date then id. */
export function disagreementsOf(rows: readonly LabelledRow[]): {
  falseNegatives: Disagreement[];
  falsePositives: Disagreement[];
  directionMismatches: Disagreement[];
} {
  const describe = (row: LabelledRow): Disagreement => ({
    row: row.row, date: row.date, id: row.id, title: row.title, score: row.modelScore, modelCategory: row.modelCategory,
    modelDirections: row.modelDirections, labelDirections: row.labelDirections, note: row.note, hadAbstractAtFilter: row.hadAbstractAtFilter,
  });
  const order = (left: Disagreement, right: Disagreement): number =>
    left.date < right.date ? -1 : left.date > right.date ? 1 : left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
  const pick = (keep: (row: LabelledRow) => boolean): Disagreement[] => rows.filter(keep).map(describe).sort(order);
  return {
    falseNegatives: pick((row) => row.label === "y" && !isSelected(row)),
    falsePositives: pick((row) => row.label === "n" && isSelected(row)),
    directionMismatches: pick((row) =>
      row.label === "y" && isSelected(row) && row.labelDirections.length > 0
      && !row.labelDirections.some((number) => row.modelDirections.includes(number))),
  };
}
