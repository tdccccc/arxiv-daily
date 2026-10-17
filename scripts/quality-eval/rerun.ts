import {
  DAILY_FILTER_PROMPT_CONTRACT_VERSION,
  DAILY_FILTER_RESULT_CONTRACT_VERSION,
  LlmClient,
  Logger,
  buildCheckpointGenerationIdentity,
  buildPaperFilterRequest,
  decodePaperFilterResponse,
  isPaperFilterResponseValidationError,
  normalizeTopic,
  redactText,
  type ArxivSettings,
  type CheckpointGenerationIdentity,
  type FilterRecord,
  type LlmCallMetrics,
  type PaperFilterRequest,
  type Topic,
} from "@arxiv-daily/core";
import { NodeHttpClient } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../../apps/cli/src/config";
import type { EvaluateDeps, EvaluateOptions } from "./evaluate";
import {
  disagreementsOf, isLabelled, metricSet, topicMetrics,
  type Disagreement, type DirectionAccuracy, type LabelledRow, type MetricSet, type Ratio, type TopicMetrics, type WeightedSummary,
} from "./metrics";
import type { SheetRow } from "./sheet";
import {
  RULES, skippedAvailable, summarizeSheet, summarizeSource,
  type MetaDirection, type SheetMeta, type SheetSummary, type SourceSummary,
} from "./source";

/** A metric set whose direction accuracy may be withheld. */
export type RunMetricSet = Omit<MetricSet, "direction"> & { direction: DirectionAccuracy | null };

export interface RunDecision {
  id: string;
  category: string;
  directions: number[];
  score: number;
  /** Whether this run's request carried an abstract from the sheet. */
  requestHadAbstract: boolean;
}

export interface RerunRun {
  run: number;
  status: "ok" | "failed";
  /** Who failed: the model call, or the response's validity. */
  failure?: "model" | "validation";
  error?: string;
  usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number; elapsedMs?: number };
  /** Papers sent that the model left out of its answer; they count as skipped. */
  omitted?: number;
  sets?: { all: RunMetricSet; withAbstract: RunMetricSet };
  topics?: TopicMetrics[] | null;
  topicsWithoutDirection?: number | null;
  disagreements?: {
    falseNegatives: Disagreement[];
    falsePositives: Disagreement[];
    directionMismatches: Disagreement[] | null;
  };
  decisions?: RunDecision[];
}

export interface Comparison {
  name: string;
  kind: "recorded" | "run";
  rows: number;
  precision: Ratio;
  recall: Ratio;
  f1: Ratio;
  /** Weighted by the sample's original strata; null when the population is unknown. */
  weightedPrecision: Ratio | null;
  weightedRecall: Ratio | null;
  weightedF1: Ratio | null;
}

export interface Agreement {
  /** How many successful runs were compared. */
  runs: number;
  papers: number;
  identical: Ratio;
  flipFlops: Array<{ id: string; title: string; label: "y" | "n"; selected: boolean[] }>;
}

export interface RerunReport {
  schemaVersion: 1;
  kind: "rerun";
  generatedAt: string;
  rules: { positive: string; direction: string; weighting: string };
  sheet: SheetSummary;
  source: SourceSummary;
  current: {
    promptContractVersion: number;
    resultContractVersion: number;
    provider: string;
    model: string;
    mode: CheckpointGenerationIdentity["mode"];
    categories: string[];
    topicSource: "sheet-snapshot" | "config";
    papers: number;
    requestCharacters: number;
    runs: number;
  };
  directions: MetaDirection[];
  warnings: string[];
  /** Why direction accuracy is not reported, or null when it is. */
  directionWithheld: string | null;
  recorded: { all: MetricSet; withAbstract: MetricSet };
  runs: RerunRun[];
  /** Recorded decisions and reruns over the labelled rows only, weighted by stratum. */
  comparison: Comparison[];
  /**
   * The same comparison over the labelled rows that had an abstract at filter time,
   * so both sides saw the same input; this subset is not weighted.
   */
  comparisonWithAbstract: Comparison[];
  agreement: Agreement | null;
}

export interface RerunOutcome {
  dryRun: boolean;
  report: RerunReport | null;
  allRunsFailed: boolean;
}

const keyOf = (tag: string, text: string): string => JSON.stringify([tag, text]);

/**
 * Rebuild the topics the recorded run offered the model. The filter prompt uses only a topic's
 * tag and its directions' text in order, so the sheet's numbered legend carries all of it. A
 * direction that was not on every exported day means the days were filtered against different
 * lists, and there is no single list to rerun with.
 */
export function topicsFromSnapshot(meta: SheetMeta): Topic[] {
  const stale = meta.directions.filter((direction) => direction.dates && meta.sourceDates.some((date) => !direction.dates!.includes(date)));
  if (stale.length > 0) {
    throw new Error(
      `the sheet covers days with different direction lists (${stale.map((direction) => `"${direction.text}"`).join(", ")} was not on every day), so there is no single list to rerun with; ` +
      "export one day at a time, or pass --topics-from-config to use the CLI config's topics",
    );
  }
  const tags = [...new Set(meta.directions.map((direction) => direction.tag))];
  return tags.map((tag) => normalizeTopic({
    id: tag, name: tag, tag, detail: false,
    directions: meta.directions.filter((direction) => direction.tag === tag).map((direction) => ({ id: direction.id, text: direction.text, origin: "manual" })),
  }));
}

function redactor(settings: { apiKey: string; baseUrl: string }): (value: unknown) => string {
  const secrets = [settings.apiKey];
  try {
    const endpoint = new URL(settings.baseUrl);
    secrets.push(settings.baseUrl, endpoint.host, endpoint.hostname, endpoint.username, endpoint.password, ...endpoint.searchParams.values());
  } catch { /* an invalid endpoint cannot be called anyway */ }
  const ordered = [...new Set(secrets.filter(Boolean))].sort((left, right) => right.length - left.length);
  return (value) => {
    let text = redactText(value, { secrets: ordered });
    // Short secrets are ignored by the shared redactor, so replace them explicitly, then drop any URL.
    for (const secret of ordered) text = text.split(secret).join("[REDACTED]");
    return text.replace(/https?:\/\/[^\s"')]+/g, "[url]");
  };
}

export async function runRerun(
  rows: readonly SheetRow[],
  meta: SheetMeta,
  options: EvaluateOptions,
  deps: EvaluateDeps,
  context: { sheetPath: string; metaPath: string; now: Date },
): Promise<RerunOutcome> {
  const log = deps.log ?? (() => undefined);
  const labelled: LabelledRow[] = rows.filter(isLabelled);
  if (labelled.length === 0) throw new Error("the sheet has no labelled rows; fill label with y or n first");
  const seen = new Set<string>();
  for (const row of labelled) {
    if (seen.has(row.id)) throw new Error(`paper ${row.id} appears more than once among the labelled rows (row ${row.row}); one request cannot carry the same paper twice, so export a single day or remove the duplicate`);
    seen.add(row.id);
  }

  const config = await loadCliConfig({ configPath: options.configPath });
  const topics = options.topicsFromConfig ? configTopics(config.settings.arxiv.topics) : topicsFromSnapshot(meta);
  const arxivSettings: ArxivSettings = {
    category: config.settings.arxiv.category,
    categories: config.settings.arxiv.categories,
    timezone: config.settings.arxiv.timezone,
    topics,
  };
  const papers = labelled.map((row) => ({ id: row.id, title: row.title, authors: "", abstract: row.abstract }));
  const request = buildPaperFilterRequest(papers, arxivSettings);
  const requestCharacters = request.messages.reduce((sum, message) => sum + message.content.length, 0);
  const runs = options.runs ?? 1;
  const generation = buildCheckpointGenerationIdentity(config.settings.llm, request.options.temperature);

  const legendKey = new Map(meta.directions.map((direction) => [keyOf(direction.tag, direction.text), direction.number] as const));
  const snapshotKeys = meta.directions.map((direction) => keyOf(direction.tag, direction.text));
  const currentKeys = request.identity.directions.map((direction) => keyOf(direction.tag, direction.text));
  const sameDirections = snapshotKeys.length === currentKeys.length && snapshotKeys.every((key, index) => key === currentKeys[index]);
  const directionWithheld = sameDirections ? null
    : `方向列表与标注时不同，方向准确率和分主题结果已隐藏（选中与否的指标不受影响）。标注时：${meta.directions.map((d) => `${d.tag}：${d.text}`).join("；")}。现在：${request.identity.directions.map((d) => `${d.tag}：${d.text}`).join("；")}。`;
  const warnings = directionWithheld === null ? [] : [directionWithheld];
  if (meta.records.some((record) => record.promptContractVersion !== DAILY_FILTER_PROMPT_CONTRACT_VERSION)) {
    warnings.push(`记录的判定用的提示词版本（${[...new Set(meta.records.map((record) => record.promptContractVersion))].join("、")}）与当前代码（${DAILY_FILTER_PROMPT_CONTRACT_VERSION}）不同，重跑与记录的差异里包含提示词本身的变化。`);
  }
  // The rerun sends the sheet's abstract (fetching it at export time for papers the filter
  // saw title-only), so a recorded-vs-rerun difference on those rows mixes in an input change.
  const titleOnly = labelled.filter((row) => !row.hadAbstractAtFilter).length;
  const titleOnlyWithAbstract = labelled.filter((row) => !row.hadAbstractAtFilter && row.abstract.trim() !== "").length;
  if (titleOnly > 0) {
    warnings.push(titleOnlyWithAbstract > 0
      ? `${titleOnly} 篇已标注论文在筛选时只看到标题（没有摘要），其中 ${titleOnlyWithAbstract} 篇在重跑时发送了标注表里的摘要；与记录的判定比较时，这部分差异来自输入变化。“记录的判定与重跑对比”一节另给出只看筛选时有摘要的行的对比。`
      : `${titleOnly} 篇已标注论文在筛选时只看到标题（没有摘要），标注表里也没有取到摘要，重跑请求与当时一样不带摘要。`);
  }

  const plan = `This will make ${runs} model call${runs === 1 ? "" : "s"} (one request per run), each with ${papers.length} papers ` +
    `(about ${requestCharacters} characters of prompt), using ${config.settings.llm.model} via ${config.settings.llm.provider}. ` +
    `Topics come from ${options.topicsFromConfig ? "the CLI config" : "the sheet's direction snapshot"}.`;
  if (!options.yes) {
    log(plan);
    log("No model was called. Add --yes to make these calls.");
    return { dryRun: true, report: null, allRunsFailed: false };
  }
  log(plan);

  const redact = redactor(config.settings.llm);
  const llm = deps.llm ?? new LlmClient(config.settings.llm, new Logger("error"), new NodeHttpClient());
  const results: RerunRun[] = [];
  for (let index = 1; index <= runs; index += 1) {
    log(`run ${index}/${runs}: calling ${config.settings.llm.model}`);
    results.push(await oneRun(index, llm, request, labelled, { legendKey, available: skippedAvailable(meta), directionWithheld: directionWithheld !== null, redact }));
  }

  const available = skippedAvailable(meta);
  const recordedAll = metricSet(labelled, available);
  const recordedWithAbstract = metricSet(labelled.filter((row) => row.hadAbstractAtFilter), null);
  const report: RerunReport = {
    schemaVersion: 1,
    kind: "rerun",
    generatedAt: context.now.toISOString(),
    rules: { ...RULES },
    sheet: summarizeSheet(rows, context),
    source: summarizeSource(meta),
    current: {
      promptContractVersion: DAILY_FILTER_PROMPT_CONTRACT_VERSION,
      resultContractVersion: DAILY_FILTER_RESULT_CONTRACT_VERSION,
      provider: generation.provider,
      model: generation.model,
      mode: generation.mode,
      categories: config.settings.arxiv.categories,
      topicSource: options.topicsFromConfig ? "config" : "sheet-snapshot",
      papers: papers.length,
      requestCharacters,
      runs,
    },
    directions: meta.directions,
    warnings,
    directionWithheld,
    recorded: { all: recordedAll, withAbstract: recordedWithAbstract },
    runs: results,
    comparison: [comparisonOf("recorded", "recorded", recordedAll), ...results.filter((run) => run.sets).map((run) => comparisonOf(`run ${run.run}`, "run", run.sets!.all))],
    comparisonWithAbstract: [
      comparisonOf("recorded", "recorded", recordedWithAbstract),
      ...results.filter((run) => run.sets).map((run) => comparisonOf(`run ${run.run}`, "run", run.sets!.withAbstract)),
    ],
    agreement: agreementOf(results, labelled),
  };
  return { dryRun: false, report, allRunsFailed: results.every((run) => run.status === "failed") };
}

function configTopics(all: readonly Topic[]): Topic[] {
  const usable = all.filter((topic) => topic.directions.length > 0);
  if (usable.length === 0) throw new Error("the CLI config has no topics with directions to rerun with; omit --topics-from-config to use the sheet's direction snapshot");
  return usable;
}

async function oneRun(
  run: number,
  llm: NonNullable<EvaluateDeps["llm"]>,
  request: PaperFilterRequest,
  labelled: readonly LabelledRow[],
  context: { legendKey: Map<string, number>; available: number | null; directionWithheld: boolean; redact: (value: unknown) => string },
): Promise<RerunRun> {
  const usage: NonNullable<RerunRun["usage"]> = {};
  let records: FilterRecord[];
  try {
    const raw = await llm.call(request.messages, {
      ...request.options,
      onMetrics: (metrics: LlmCallMetrics) => {
        usage.inputTokens = (usage.inputTokens ?? 0) + (metrics.inputTokens ?? 0);
        usage.outputTokens = (usage.outputTokens ?? 0) + (metrics.outputTokens ?? 0);
        usage.totalTokens = (usage.totalTokens ?? 0) + (metrics.totalTokens ?? 0);
        usage.elapsedMs = (usage.elapsedMs ?? 0) + metrics.elapsedMs;
      },
    });
    records = decodePaperFilterResponse(raw, request);
  } catch (error) {
    return {
      run, status: "failed", usage,
      failure: isPaperFilterResponseValidationError(error) ? "validation" : "model",
      error: context.redact((error as Error).message),
    };
  }

  const decided = new Map(records.map((record) => [record.id, record] as const));
  const refNumber = new Map(request.identity.directions.map((direction) => [direction.ref, context.legendKey.get(keyOf(direction.tag, direction.text))] as const));
  type RerunRow = LabelledRow & { requestHadAbstract: boolean };
  const rerunRows: RerunRow[] = labelled.map((row) => {
    const record = decided.get(row.id);
    return {
      ...row,
      modelCategory: record?.category ?? "skip",
      modelDirections: (record?.directions ?? []).flatMap((ref) => refNumber.get(ref) ?? []),
      modelScore: record?.relevanceScore ?? 0,
      modelOutcome: "",
      // Keep the sheet's value: it says whether the recorded filter saw an abstract.
      // The request may still carry the abstract export fetched later, recorded apart below.
      requestHadAbstract: row.abstract.trim() !== "",
      // The sample was drawn from the recorded decisions; a new decision does not move a row between strata.
      sampledAs: row.modelCategory === "skip" ? "skipped" : "selected",
    };
  });
  const withheld = <T extends MetricSet>(set: T): RunMetricSet => ({ ...set, direction: context.directionWithheld ? null : set.direction });
  const tags = [...new Set(request.identity.directions.map((direction) => direction.tag))];
  const legendTag = new Map<number, string>();
  for (const direction of request.identity.directions) {
    const number = context.legendKey.get(keyOf(direction.tag, direction.text));
    if (number !== undefined) legendTag.set(number, direction.tag);
  }
  const topics = topicMetrics(rerunRows, tags, (number) => legendTag.get(number), context.available);
  const disagreements = disagreementsOf(rerunRows);
  return {
    run, status: "ok", usage,
    omitted: request.identity.knownIds.filter((id) => !decided.has(id)).length,
    sets: {
      all: withheld(metricSet(rerunRows, context.available)),
      withAbstract: withheld(metricSet(rerunRows.filter((row) => row.hadAbstractAtFilter), null)),
    },
    topics: context.directionWithheld ? null : topics.topics,
    topicsWithoutDirection: context.directionWithheld ? null : topics.withoutDirection,
    disagreements: { ...disagreements, directionMismatches: context.directionWithheld ? null : disagreements.directionMismatches },
    decisions: rerunRows.map((row) => ({ id: row.id, category: row.modelCategory, directions: row.modelDirections, score: row.modelScore, requestHadAbstract: row.requestHadAbstract })),
  };
}

function comparisonOf(name: string, kind: Comparison["kind"], set: { rows: number; precision: Ratio; recall: Ratio; f1: Ratio; weighted: WeightedSummary | null }): Comparison {
  return {
    name, kind, rows: set.rows, precision: set.precision, recall: set.recall, f1: set.f1,
    weightedPrecision: set.weighted?.precision ?? null,
    weightedRecall: set.weighted?.recall ?? null,
    weightedF1: set.weighted?.f1 ?? null,
  };
}

/** Whether each labelled paper got the same selected/not decision in every successful run. */
function agreementOf(runs: readonly RerunRun[], labelled: readonly LabelledRow[]): Agreement | null {
  const ok = runs.filter((run) => run.status === "ok");
  if (ok.length < 2) return null;
  const selectedBy = ok.map((run) => new Map(run.decisions!.map((decision) => [decision.id, decision.category !== "skip"] as const)));
  const flipFlops: Agreement["flipFlops"] = [];
  const ordered = [...labelled].sort((left, right) => (left.date < right.date ? -1 : left.date > right.date ? 1 : left.id < right.id ? -1 : left.id > right.id ? 1 : 0));
  let identical = 0;
  for (const row of ordered) {
    const selected = selectedBy.map((byId) => byId.get(row.id) ?? false);
    if (selected.every((value) => value === selected[0])) identical += 1;
    else flipFlops.push({ id: row.id, title: row.title, label: row.label, selected });
  }
  return { runs: ok.length, papers: ordered.length, identical: { value: ordered.length > 0 ? identical / ordered.length : null, n: ordered.length, hits: identical }, flipFlops };
}
