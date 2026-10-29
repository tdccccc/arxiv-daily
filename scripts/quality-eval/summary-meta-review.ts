import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { LlmClient, Logger, type LlmCallMetrics, type LlmSettings } from "@arxiv-daily/core";
import { NodeHttpClient } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../../apps/cli/src/config";
import { prepareOutDir } from "./evaluate";
import { redactor } from "./rerun";
import { cell } from "./report";
import { requireDirectory } from "./summary-check";
import { rate } from "./summary-checks";
import { JUDGE_DIMENSIONS, type JudgeDimension, type JudgeIssue } from "./summary-judge-contract";
import { prepareJudgeInput, type JudgeItem } from "./summary-judge-input";
import { prepareRegeneratedJudgeInput } from "./summary-judge-regenerated";
import { loadJudgments, type JudgmentRecord } from "./summary-judge-report";
import { CHARS_PER_TOKEN } from "./summary-judge-run";
import { buildMetaCorrection, buildMetaSystemPrompt, checkMetaOutput, type MetaReview, type MetaSeverityOpinion, type MetaVerdict } from "./summary-meta-review-contract";
import { locationLabel } from "./summary-review";
import { capped, percent, table } from "./summary-report";

const MAX_CONSECUTIVE_ERRORS = 3;
const REQUESTS_PER_ITEM = 2;
const SHOWN_SKIPPED = 10;
const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");
const sameModel = (left: string, right: string): boolean => left.trim().toLowerCase() === right.trim().toLowerCase();
const DIMENSION_LABELS: Record<JudgeDimension, string> = { accuracy: "准确", completeness: "完整", clarity: "清楚" };
const SEVERITY_LABELS = { major: "大问题", minor: "小问题" } as const;
const VERDICT_LABELS: Record<MetaVerdict, string> = { agree: "同意", disagree: "不同意", unsure: "拿不准" };
const OPINION_LABELS: Record<MetaSeverityOpinion, string> = { ok: "合适", "too-high": "偏重", "too-low": "偏轻" };

export interface MetaReviewOptions {
  /** Output directory of summary-judge.mjs. */
  judgedDir: string;
  sourcesDir: string;
  /** CLI config.toml: its `[llm]` endpoint and key are used, with the model and effort below. */
  configPath: string;
  reviewerModel: string;
  reviewerEffort: string;
  /** Needed only to rebuild the input of judgments of historical (vault) summaries. */
  vaultRoot?: string;
  dailyDir?: string;
  papersDir?: string;
  outDir: string;
  yes?: boolean;
  rereview?: boolean;
}

export interface MetaModelLike {
  call(messages: Array<{ role: "system" | "user" | "assistant"; content: string }>, options?: { onMetrics?: (metrics: LlmCallMetrics) => void }): Promise<string>;
}

export interface MetaReviewDeps {
  llm?: MetaModelLike;
  now?: () => Date;
  log?: (line: string) => void;
  /** The CLI config's `[llm]` settings; tests pass a fixture. */
  loadReviewerBase?: (configPath: string) => Promise<LlmSettings>;
}

/** The reviewer's settings: the CLI config's endpoint and key, the model and effort asked for, thinking on. */
export function reviewerSettings(base: LlmSettings, choice: { model: string; effort: string }): LlmSettings {
  return { ...base, model: choice.model.trim(), reasoningEffort: choice.effort.trim(), thinkingMode: true };
}

export interface MetaSkip { kind: "daily" | "note"; id: string; reason: "no-vault" | "not-rebuilt" | "summary-changed" | "source-changed"; detail: string }

export interface MetaPlan {
  items: number;
  issues: number;
  alreadyReviewed: number;
  toReview: number;
  skipped: MetaSkip[];
  maxRequests: number;
  totalChars: number;
  estimatedTokens: number;
}

export interface MetaCounts { items: number; reviewed: number; invalid: number; errors: number; issues: number }

export interface MetaReviewResult {
  dryRun: boolean;
  outDir: string;
  plan: MetaPlan;
  counts: (MetaCounts & { reused: number; remaining: number }) | null;
  stopped: "errors" | null;
}

export interface MetaArgs extends Partial<MetaReviewOptions> { help?: boolean; reportOnly?: boolean }

export function parseSummaryMetaReviewArgs(argv: readonly string[]): MetaArgs {
  const parsed: MetaArgs = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    if (flag === "--yes") { parsed.yes = true; continue; }
    if (flag === "--rereview") { parsed.rereview = true; continue; }
    if (flag === "--report-only") { parsed.reportOnly = true; continue; }
    const take = (): string => {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) throw new Error(`${flag} needs a value`);
      index += 1;
      return value;
    };
    switch (flag) {
      case "--judged": parsed.judgedDir = take(); break;
      case "--sources": parsed.sourcesDir = take(); break;
      case "--config": parsed.configPath = take(); break;
      case "--reviewer-model": parsed.reviewerModel = take(); break;
      case "--reviewer-effort": parsed.reviewerEffort = take(); break;
      case "--vault": parsed.vaultRoot = take(); break;
      case "--daily-dir": parsed.dailyDir = take(); break;
      case "--papers-dir": parsed.papersDir = take(); break;
      case "--out": parsed.outDir = take(); break;
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  if (!parsed.reportOnly) {
    for (const [value, flag] of [[parsed.judgedDir, "--judged <judge output directory>"], [parsed.sourcesDir, "--sources <directory>"], [parsed.configPath, "--config <CLI config.toml>"], [parsed.reviewerModel, "--reviewer-model <name>"], [parsed.reviewerEffort, "--reviewer-effort <low|medium|high|xhigh>"]] as const) {
      if (!value?.trim()) throw new Error(`${flag} is required`);
    }
  }
  return parsed;
}

async function defaultLoadReviewerBase(configPath: string): Promise<LlmSettings> {
  return (await loadCliConfig({ configPath })).settings.llm;
}

async function writeAtomic(file: string, text: string): Promise<void> {
  const temporary = `${file}.tmp`;
  await writeFile(temporary, text);
  await rename(temporary, file);
}

const keyOf = (kind: string, id: string): string => `${kind}-${id}`;
const fileFor = (outDir: string, kind: string, id: string): string => path.join(outDir, "meta", `${kind}-${id}.json`);

interface Target {
  record: JudgmentRecord & { judgment: NonNullable<JudgmentRecord["judgment"]> };
  item: JudgeItem;
  issues: JudgeIssue[];
  judgmentHash: string;
  systemPrompt: string;
  promptHash: string;
  messages: Array<{ role: "system" | "user"; content: string }>;
}

function issuesBlock(kind: "daily" | "note", issues: readonly JudgeIssue[]): string {
  return issues.map((issue, index) => [
    `[${index}] 维度：${DIMENSION_LABELS[issue.dimension]}（${issue.dimension}）；严重程度：${SEVERITY_LABELS[issue.severity]}（${issue.severity}）；位置：${issue.location}（${locationLabel(kind, issue.location)}）`,
    `摘要原句：${issue.summaryQuote}`,
    `原文依据：${issue.sourceQuote === null ? "（评审认为原文未提及）" : issue.sourceQuote}`,
    `评审说明：${issue.explanation}`,
  ].join("\n")).join("\n\n");
}

/** Everything a reviewer request needs for the judged items that have issues, and the ones that cannot be rebuilt. */
async function collectTargets(options: MetaReviewOptions, records: readonly JudgmentRecord[]): Promise<{ targets: Target[]; skipped: MetaSkip[] }> {
  const withIssues = records.filter((record): record is Target["record"] => record.status === "judged" && record.judgment !== null && record.judgment.issues.length > 0);
  const built = new Map<string, JudgeItem>();
  const notBuilt = new Map<string, string>();
  const regeneratedDirs = new Set(withIssues.flatMap((record) => (record.origin?.kind === "regenerated" ? [record.origin.summariesDir] : [])));
  for (const dir of regeneratedDirs) {
    const prepared = await prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: options.sourcesDir });
    for (const item of prepared.items) built.set(`${dir}|${keyOf(item.kind, item.id)}`, item);
    for (const skip of prepared.skipped) notBuilt.set(`${dir}|${keyOf(skip.kind, skip.id)}`, `${skip.reason}: ${skip.detail}`);
  }
  const historical = withIssues.filter((record) => record.origin?.kind !== "regenerated");
  if (historical.length > 0 && options.vaultRoot) {
    const prepared = await prepareJudgeInput({ vaultRoot: options.vaultRoot, dailyDir: options.dailyDir, papersDir: options.papersDir, sourcesDir: options.sourcesDir, ids: [...new Set(historical.map((record) => record.item.id))] });
    for (const item of prepared.items) built.set(`|${keyOf(item.kind, item.id)}`, item);
    for (const skip of prepared.skipped) notBuilt.set(`|${keyOf(skip.kind === "id" ? "daily" : skip.kind, skip.id)}`, `${skip.reason}: ${skip.detail}`);
  }

  const targets: Target[] = [];
  const skipped: MetaSkip[] = [];
  for (const record of withIssues) {
    const kind = record.item.kind;
    const skip = (reason: MetaSkip["reason"], detail: string) => skipped.push({ kind, id: record.item.id, reason, detail });
    const dir = record.origin?.kind === "regenerated" ? record.origin.summariesDir : "";
    if (!dir && !options.vaultRoot) { skip("no-vault", "the judged summary is from the vault; pass --vault to rebuild what the judge saw"); continue; }
    const item = built.get(`${dir}|${keyOf(kind, record.item.id)}`);
    if (!item) { skip("not-rebuilt", notBuilt.get(`${dir}|${keyOf(kind, record.item.id)}`) ?? "the judge's input could not be rebuilt"); continue; }
    if (record.summaryHash !== undefined && record.summaryHash !== item.summaryHash) { skip("summary-changed", "the summary is not the text the judge saw"); continue; }
    if (item.source.contentHash !== record.item.source.contentHash) { skip("source-changed", "the cached source text is not the text the judge saw"); continue; }
    const issues = record.judgment.issues;
    const systemPrompt = buildMetaSystemPrompt({ kind });
    targets.push({
      record, item, issues, systemPrompt, promptHash: sha256(systemPrompt),
      judgmentHash: sha256(JSON.stringify({ summaryHash: item.summaryHash, source: item.source.contentHash, issues })),
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: `${item.paperData}\n\n<summary_under_review>\n${item.judgedText}\n</summary_under_review>\n\n<judge_issues>\n${issuesBlock(kind, issues)}\n</judge_issues>` },
      ],
    });
  }
  targets.sort((a, b) => a.record.item.kind.localeCompare(b.record.item.kind) || a.record.item.id.localeCompare(b.record.item.id));
  return { targets, skipped };
}

interface MetaRecord {
  schemaVersion: 1;
  status: "reviewed" | "invalid" | "error";
  item: { kind: "daily" | "note"; id: string; title: string; date: string | null; topic: string | null; version?: string; source: JudgeItem["source"]; origin: JudgeItem["origin"] };
  judge: { provider: string; model: string; thinkingMode?: boolean; reasoningEffort?: string };
  issues: JudgeIssue[];
  reviewer: { provider: string; model: string; thinkingMode: boolean; reasoningEffort: string };
  reviewerPromptHash: string;
  judgmentHash: string;
  attempts: Array<{ raw: string; reasons: string[] | null; usage: unknown; elapsedMs: number }>;
  reviews: MetaReview[] | null;
  error?: string;
  startedAt: string;
  finishedAt: string;
  elapsedMs: number;
}

async function isCurrent(file: string, target: Target, reviewer: LlmSettings): Promise<boolean> {
  try {
    const record = JSON.parse(await readFile(file, "utf8")) as MetaRecord;
    return (record.status === "reviewed" || record.status === "invalid")
      && record.reviewer.model === reviewer.model && record.reviewer.provider === reviewer.provider && record.reviewer.reasoningEffort === reviewer.reasoningEffort
      && record.reviewerPromptHash === target.promptHash && record.judgmentHash === target.judgmentHash;
  } catch {
    return false;
  }
}

function usageOf(metrics: LlmCallMetrics | undefined): unknown {
  return metrics ? { inputTokens: metrics.inputTokens, outputTokens: metrics.outputTokens, totalTokens: metrics.totalTokens, usageComplete: metrics.usageComplete } : null;
}

async function reviewTarget(target: Target, llm: MetaModelLike, now: () => Date) {
  const attempts: MetaRecord["attempts"] = [];
  let messages = target.messages as Array<{ role: "system" | "user" | "assistant"; content: string }>;
  for (let attempt = 1; attempt <= REQUESTS_PER_ITEM; attempt += 1) {
    const started = now().getTime();
    let metrics: LlmCallMetrics | undefined;
    let raw: string;
    try {
      raw = await llm.call(messages, { onMetrics: (observed) => { metrics = observed; } });
    } catch (error) {
      return { attempts, reviews: null as MetaReview[] | null, error };
    }
    const checked = checkMetaOutput(raw, { count: target.issues.length, paperData: target.item.paperData });
    attempts.push({ raw, reasons: checked.ok ? null : checked.reasons, usage: usageOf(metrics), elapsedMs: now().getTime() - started });
    if (checked.ok) return { attempts, reviews: checked.reviews as MetaReview[] | null };
    messages = [...messages, { role: "assistant", content: raw }, { role: "user", content: buildMetaCorrection(checked.reasons) }];
  }
  return { attempts, reviews: null as MetaReview[] | null };
}

/**
 * Have a second model re-check every issue the judge raised, one request per judged item with issues: it sees the
 * judge's own input and the numbered issues, and answers agree / disagree / unsure with a severity opinion and a reason.
 * Without `yes` it only prints the plan.
 */
export async function runSummaryMetaReview(options: MetaReviewOptions, deps: MetaReviewDeps = {}): Promise<MetaReviewResult> {
  const log = deps.log ?? (() => undefined);
  const now = deps.now ?? (() => new Date());
  const outDir = path.resolve(options.outDir);
  await requireDirectory(options.sourcesDir, "--sources");

  const base = await (deps.loadReviewerBase ?? defaultLoadReviewerBase)(options.configPath);
  const reviewer = reviewerSettings(base, { model: options.reviewerModel, effort: options.reviewerEffort });
  const { run, records } = await loadJudgments(path.resolve(options.judgedDir));
  const judgeModels = [...new Set(records.map((record) => record.judge.model))];
  const clash = judgeModels.find((model) => sameModel(model, reviewer.model));
  if (clash) throw new Error(`The reviewer must be a different model from the judge, but both are "${reviewer.model}". Use another --reviewer-model.`);
  if (run.generatorModel && sameModel(run.generatorModel, reviewer.model)) log(`Note: the reviewer is the same model as the generator (${reviewer.model}); that is allowed.`);

  const { targets, skipped } = await collectTargets(options, records);
  const pending: Target[] = [];
  let reused = 0;
  for (const target of targets) {
    if (!options.rereview && await isCurrent(fileFor(outDir, target.record.item.kind, target.record.item.id), target, reviewer)) reused += 1;
    else pending.push(target);
  }
  const totalChars = pending.reduce((sum, target) => sum + target.messages.reduce((size, message) => size + message.content.length, 0), 0);
  const plan: MetaPlan = {
    items: targets.length, issues: targets.reduce((sum, target) => sum + target.issues.length, 0), alreadyReviewed: reused, toReview: pending.length, skipped,
    maxRequests: pending.length * REQUESTS_PER_ITEM, totalChars, estimatedTokens: Math.ceil(totalChars / CHARS_PER_TOKEN),
  };
  log(`Reviewer model: ${reviewer.model} (${reviewer.provider}), reasoning effort ${reviewer.reasoningEffort}; judge model: ${judgeModels.join(", ")}.`);
  log(`${plan.items} items with judge issues (${plan.issues} issues). Already reviewed with this reviewer, effort, prompt and judgment: ${reused}. To review: ${pending.length}.`);
  if (skipped.length > 0) log(`Skipped ${skipped.length}: ${skipped.slice(0, SHOWN_SKIPPED).map((entry) => `${entry.id} [${entry.reason}]`).join(", ")}${skipped.length > SHOWN_SKIPPED ? ", ..." : ""}`);
  log(`This ${options.yes ? "will" : "would"} make at most ${plan.maxRequests} requests (${pending.length} items x ${REQUESTS_PER_ITEM}: one retry each when the reply fails validation), sending ${totalChars} characters in the first requests, about ${plan.estimatedTokens} input tokens (estimated at ${CHARS_PER_TOKEN} characters per token for mixed English and Chinese text).`);
  if (!options.yes) {
    log("No model was called and nothing was written. Add --yes to review.");
    return { dryRun: true, outDir, plan, counts: null, stopped: null };
  }

  await prepareOutDir(outDir);
  await mkdir(path.join(outDir, "meta"), { recursive: true });
  const redact = redactor(reviewer);
  const llm: MetaModelLike = deps.llm ?? (() => {
    const logger = new Logger("error");
    logger.setSensitiveValues([reviewer.apiKey]);
    return new LlmClient(reviewer, logger, new NodeHttpClient());
  })();
  const identity = { provider: reviewer.provider, model: reviewer.model, thinkingMode: reviewer.thinkingMode, reasoningEffort: reviewer.reasoningEffort };
  const runInfo = {
    schemaVersion: 1, startedAt: now().toISOString(), finishedAt: null as string | null,
    // The CLI config path and the secrets in it are not recorded.
    judged: { dir: path.resolve(options.judgedDir), judgeModels, generatorModel: run.generatorModel ?? null }, sourcesDir: path.resolve(options.sourcesDir),
    reviewer: identity, plan, skipped, counts: null as unknown, stopped: null as "errors" | null,
  };
  await writeAtomic(path.join(outDir, "run.json"), `${JSON.stringify(runInfo, null, 2)}\n`);

  let consecutiveErrors = 0;
  let stopped: "errors" | null = null;
  let attempted = 0;
  for (const target of pending) {
    if (stopped) break;
    attempted += 1;
    const started = now();
    const outcome = await reviewTarget(target, llm, now);
    const item = target.item;
    let status: MetaRecord["status"];
    let error: string | undefined;
    if ("error" in outcome && outcome.error !== undefined) {
      status = "error";
      error = redact((outcome.error as Error)?.message ?? outcome.error) || "unknown error";
    } else status = outcome.reviews ? "reviewed" : "invalid";
    const finished = now();
    const record: MetaRecord = {
      schemaVersion: 1, status,
      item: { kind: item.kind, id: item.id, title: item.title, date: item.date, topic: item.topic, ...(item.version ? { version: item.version } : {}), source: item.source, origin: item.origin },
      judge: target.record.judge, issues: target.issues, reviewer: identity, reviewerPromptHash: target.promptHash, judgmentHash: target.judgmentHash,
      attempts: outcome.attempts, reviews: outcome.reviews, ...(error !== undefined ? { error } : {}),
      startedAt: started.toISOString(), finishedAt: finished.toISOString(), elapsedMs: finished.getTime() - started.getTime(),
    };
    await writeAtomic(fileFor(outDir, item.kind, item.id), `${JSON.stringify(record, null, 2)}\n`);
    if (status === "error") {
      consecutiveErrors += 1;
      log(`[${attempted}/${pending.length}] ${item.kind} ${item.id}: error: ${error}`);
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) stopped = "errors";
    } else {
      consecutiveErrors = 0;
      log(`[${attempted}/${pending.length}] ${item.kind} ${item.id}: ${status} (${outcome.attempts.length} attempt${outcome.attempts.length === 1 ? "" : "s"})`);
    }
  }

  runInfo.finishedAt = now().toISOString();
  runInfo.stopped = stopped;
  const report = await writeMetaReport(outDir, { now, run: runInfo });
  runInfo.counts = report.data.counts;
  await writeAtomic(path.join(outDir, "run.json"), `${JSON.stringify(runInfo, null, 2)}\n`);
  const remaining = pending.length - attempted;
  log(`Report: ${report.files.join(" and ")}`);
  log(`Done: reviewed ${report.data.counts.reviewed}, invalid ${report.data.counts.invalid}, errors ${report.data.counts.errors}, already reviewed ${reused}, skipped ${skipped.length}, not attempted ${remaining}.`);
  if (stopped) log(`Stopped after ${MAX_CONSECUTIVE_ERRORS} errors in a row; run the same command again to continue.`);
  return { dryRun: false, outDir, plan, counts: { ...report.data.counts, reused, remaining }, stopped };
}

// ---- report ----------------------------------------------------------------------------------------

interface Tally { agree: number; disagree: number; unsure: number; rate: number | null }
const emptyTally = (): Tally => ({ agree: 0, disagree: 0, unsure: 0, rate: null });
const finishTally = (tally: Tally): Tally => ({ ...tally, rate: tally.agree + tally.disagree === 0 ? null : tally.agree / (tally.agree + tally.disagree) });
type Opinion = Record<MetaSeverityOpinion, number>;
const emptyOpinion = (): Opinion => ({ ok: 0, "too-high": 0, "too-low": 0 });

export interface MetaAggregate {
  overall: Tally;
  byDimension: Record<JudgeDimension, Tally>;
  bySeverity: Record<"major" | "minor", Tally>;
  byKind: Record<"daily" | "note", Tally>;
  severityOpinion: { overall: Opinion; byJudgeSeverity: Record<"major" | "minor", Opinion> };
}

export interface MetaReportData {
  schemaVersion: 1;
  generatedAt: string;
  run: unknown;
  counts: MetaCounts;
  aggregate: MetaAggregate;
  disputes: Array<{ kind: "daily" | "note"; id: string; title: string; location: string; locationLabel: string; dimension: JudgeDimension; severity: "major" | "minor"; verdict: MetaVerdict; summaryQuote: string; judgeExplanation: string; reviewerReason: string; evidenceQuote: string | null; severityOpinion: MetaSeverityOpinion }>;
  records: MetaRecord[];
}

async function loadMetaRecords(outDir: string): Promise<{ run: unknown; records: MetaRecord[] }> {
  const dir = path.join(outDir, "meta");
  const names = existsSync(dir) ? (await readdir(dir)).filter((name) => name.endsWith(".json")).sort() : [];
  const records: MetaRecord[] = [];
  for (const name of names) {
    try {
      const record = JSON.parse(await readFile(path.join(dir, name), "utf8")) as MetaRecord;
      if (record.schemaVersion === 1 && record.item) records.push(record);
    } catch { /* not a record */ }
  }
  if (records.length === 0) throw new Error(`no reviews found in ${dir}; run summary-meta-review.mjs with --yes first`);
  let run: unknown = null;
  try { run = JSON.parse(await readFile(path.join(outDir, "run.json"), "utf8")); } catch { /* the report can still be built */ }
  return { run, records };
}

export function buildMetaReport(run: unknown, records: readonly MetaRecord[], now: Date): MetaReportData {
  const reviewed = records.filter((record) => record.status === "reviewed" && record.reviews);
  const overall = emptyTally();
  const byDimension = Object.fromEntries(JUDGE_DIMENSIONS.map((dimension) => [dimension, emptyTally()])) as Record<JudgeDimension, Tally>;
  const bySeverity = { major: emptyTally(), minor: emptyTally() };
  const byKind = { daily: emptyTally(), note: emptyTally() };
  const opinion = { overall: emptyOpinion(), byJudgeSeverity: { major: emptyOpinion(), minor: emptyOpinion() } };
  const disputes: MetaReportData["disputes"] = [];
  let issues = 0;
  for (const record of reviewed) {
    for (const review of record.reviews!) {
      const issue = record.issues[review.index]!;
      issues += 1;
      for (const tally of [overall, byDimension[issue.dimension], bySeverity[issue.severity], byKind[record.item.kind]]) tally[review.verdict] += 1;
      opinion.overall[review.severity] += 1;
      opinion.byJudgeSeverity[issue.severity][review.severity] += 1;
      if (review.verdict !== "agree") {
        disputes.push({
          kind: record.item.kind, id: record.item.id, title: record.item.title, location: issue.location, locationLabel: locationLabel(record.item.kind, issue.location),
          dimension: issue.dimension, severity: issue.severity, verdict: review.verdict, summaryQuote: issue.summaryQuote, judgeExplanation: issue.explanation,
          reviewerReason: review.reason, evidenceQuote: review.evidenceQuote, severityOpinion: review.severity,
        });
      }
    }
  }
  return {
    schemaVersion: 1, generatedAt: now.toISOString(), run,
    counts: { items: records.length, reviewed: reviewed.length, invalid: records.filter((record) => record.status === "invalid").length, errors: records.filter((record) => record.status === "error").length, issues },
    aggregate: {
      overall: finishTally(overall),
      byDimension: Object.fromEntries(JUDGE_DIMENSIONS.map((dimension) => [dimension, finishTally(byDimension[dimension])])) as Record<JudgeDimension, Tally>,
      bySeverity: { major: finishTally(bySeverity.major), minor: finishTally(bySeverity.minor) },
      byKind: { daily: finishTally(byKind.daily), note: finishTally(byKind.note) },
      severityOpinion: opinion,
    },
    disputes, records: [...records],
  };
}

const rateText = (tally: Tally): string => (tally.rate === null ? "n/a" : percent(rate(tally.agree, tally.agree + tally.disagree)));
const tallyRow = (label: string, tally: Tally): (string | number)[] => [label, tally.agree, tally.disagree, tally.unsure, rateText(tally)];
const clip = (text: string, limit: number): string => (text.length > limit ? `${text.slice(0, limit)}…` : text);

export function renderMetaReport(data: MetaReportData): string {
  const run = (data.run ?? {}) as { reviewer?: { model: string; provider: string; thinkingMode: boolean; reasoningEffort: string }; judged?: { judgeModels?: string[]; generatorModel?: string | null }; startedAt?: string; finishedAt?: string | null; skipped?: MetaSkip[] };
  const a = data.aggregate;
  const header = ["分组", "同意", "不同意", "拿不准", "复核一致率"];
  const opinionRow = (label: string, o: Opinion): (string | number)[] => [label, o.ok, o["too-high"], o["too-low"]];
  const lines = [
    "# 评审问题复核报告",
    "",
    "本报告由第二个模型逐条复核评审模型提出的问题：每个问题是否真的存在（同意 / 不同意 / 拿不准），以及评审定的严重程度合不合适。复核模型看到的原文和被评摘要与评审模型完全相同，评审引用是否属实由程序另外核对。",
    "",
    "## 复核对象",
    "",
    `- 复核模型：${run.reviewer ? `${run.reviewer.model}（${run.reviewer.provider}）；思考模式 ${run.reviewer.thinkingMode ? "开" : "关"}，推理强度 ${run.reviewer.reasoningEffort}` : "-"}`,
    `- 被复核的评审模型：${run.judged?.judgeModels?.join("、") ?? "-"}；生成摘要的模型：${run.judged?.generatorModel ?? "-"}`,
    `- 复核时间：${run.startedAt ?? "-"}${run.finishedAt ? ` 至 ${run.finishedAt}` : ""}；报告生成时间 ${data.generatedAt}`,
    `- 结果：有评审问题的条目 ${data.counts.items} 条，复核成功 ${data.counts.reviewed} 条，无效 ${data.counts.invalid} 条，出错 ${data.counts.errors} 条；复核了 ${data.counts.issues} 个问题。`,
    ...(run.skipped?.length ? [`- 没有复核的条目 ${run.skipped.length} 条：${run.skipped.slice(0, SHOWN_SKIPPED).map((entry) => `${entry.id}（${entry.reason}）`).join("、")}`] : []),
    "",
    "## 复核一致率",
    "",
    "复核一致率 = 同意 /（同意 + 不同意）。“拿不准”不计入一致率，单独列出。",
    "",
    ...table(header, [tallyRow("全部问题", a.overall)]),
    "### 按维度",
    "",
    ...table(header, JUDGE_DIMENSIONS.map((dimension) => tallyRow(DIMENSION_LABELS[dimension], a.byDimension[dimension]))),
    "### 按严重程度（评审给的）",
    "",
    ...table(header, [tallyRow(SEVERITY_LABELS.major, a.bySeverity.major), tallyRow(SEVERITY_LABELS.minor, a.bySeverity.minor)]),
    "### 按类型",
    "",
    ...table(header, [tallyRow("日报条目", a.byKind.daily), tallyRow("详细笔记", a.byKind.note)]),
    "## 严重程度意见",
    "",
    "复核模型认为评审给的严重程度：合适、偏重（定重了）、偏轻（定轻了）。认为问题不成立时严重程度意见按“合适”记。",
    "",
    ...table(["评审给的", OPINION_LABELS.ok, OPINION_LABELS["too-high"], OPINION_LABELS["too-low"]], [
      opinionRow("全部", a.severityOpinion.overall), opinionRow(SEVERITY_LABELS.major, a.severityOpinion.byJudgeSeverity.major), opinionRow(SEVERITY_LABELS.minor, a.severityOpinion.byJudgeSeverity.minor),
    ]),
    "## 不同意和拿不准的问题",
    "",
    "这些是需要研究者裁定的分歧。",
    "",
    ...capped(["编号", "标题", "位置", "维度", "严重程度", "复核", "摘要原句", "评审说明", "复核理由"], data.disputes
      .sort((x, y) => Number(y.verdict === "disagree") - Number(x.verdict === "disagree") || x.id.localeCompare(y.id))
      .map((entry) => ({ cells: [entry.id, cell(clip(entry.title, 40)), cell(entry.locationLabel), DIMENSION_LABELS[entry.dimension], SEVERITY_LABELS[entry.severity], VERDICT_LABELS[entry.verdict], cell(clip(entry.summaryQuote, 200)), cell(entry.judgeExplanation), cell(entry.reviewerReason)] })), "meta-review.json"),
  ];
  return `${lines.join("\n")}\n`;
}

export async function writeMetaReport(outDir: string, options: { now?: () => Date; run?: unknown } = {}): Promise<{ files: string[]; data: MetaReportData }> {
  const loaded = await loadMetaRecords(outDir);
  const data = buildMetaReport(options.run ?? loaded.run, loaded.records, (options.now ?? (() => new Date()))());
  const files = [path.join(outDir, "meta-review.md"), path.join(outDir, "meta-review.json")];
  await writeAtomic(files[0]!, renderMetaReport(data));
  await writeAtomic(files[1]!, `${JSON.stringify(data, null, 2)}\n`);
  return { files, data };
}

export async function runMetaReportOnly(options: { outDir: string }, deps: { now?: () => Date } = {}): Promise<{ files: string[] }> {
  const { files } = await writeMetaReport(path.resolve(options.outDir), { now: deps.now });
  return { files };
}

