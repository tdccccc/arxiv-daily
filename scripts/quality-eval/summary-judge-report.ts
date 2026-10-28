import { existsSync } from "node:fs";
import { readFile, readdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { cell } from "./report";
import { extractNumbers } from "./summary-grounding";
import { JUDGE_DIMENSIONS, type JudgeDimension, type JudgeIssue, type Judgment } from "./summary-judge-contract";
import { capped, percent, table } from "./summary-report";
import { rate } from "./summary-checks";

const DIMENSION_LABELS: Record<JudgeDimension, string> = { accuracy: "准确", completeness: "完整", clarity: "清楚" };
const DIMENSION_ISSUE_TITLES: Record<JudgeDimension, string> = { accuracy: "准确性问题", completeness: "完整性问题", clarity: "清楚性问题" };
const KIND_LABELS = { daily: "日报条目", note: "详细笔记" } as const;
const TITLE_CHARS = 40;
const QUOTE_CHARS = 200;
const TOP_REASONS = 10;
const NOT_IN_SOURCE = "原文未提及";

/** One file under `<out>/judgments/`, as summary-judge-run writes it. */
export interface JudgmentRecord {
  schemaVersion: 1;
  status: "judged" | "invalid" | "error";
  invalid: boolean;
  item: {
    kind: "daily" | "note"; id: string; title: string; date: string | null; topic: string | null; version?: "current" | "older";
    source: { sourcePath: string; arxivVersion: string | null; fetchedAt: string; contentHash: string };
  };
  judge: { provider: string; model: string; thinkingMode?: boolean; reasoningEffort?: string };
  judgePromptHash: string;
  attempts: Array<{ raw: string; reasons: string[] | null; usage: unknown; elapsedMs: number }>;
  judgment: Judgment | null;
  error?: string;
}

export interface JudgeRunInfo {
  schemaVersion: 1;
  startedAt: string;
  finishedAt: string | null;
  seed: string;
  requested: { daily: number; notes: number };
  allocation: Record<string, number>;
  judge: { provider: string; model: string; thinkingMode?: boolean; reasoningEffort?: string };
  generatorModel: string;
  sample: { daily: string[]; notes: string[] };
  skipped: Array<{ kind: string; id: string; reason: string; detail: string }>;
  counts: unknown;
  stopped: string | null;
}

type Tally = Record<"ok" | "minor" | "major", number>;
type DimensionTallies = Record<JudgeDimension, Tally>;
export interface VerdictDistribution extends DimensionTallies { judged: number }
type SeverityTally = { major: number; minor: number };

export interface CrossCheckEntry { kind: "daily" | "note"; id: string; date: string | null; location: string; token: string; context: string; explanation?: string }

export interface CrossCheck {
  counts: { both: number; onlyOffline: number; onlyJudge: number; judgeWithoutNumber: number };
  both: CrossCheckEntry[];
  onlyOffline: CrossCheckEntry[];
  onlyJudge: CrossCheckEntry[];
  /** Judged items the offline data has nothing on, left out of the comparison. */
  notCompared: number;
}

export interface JudgeAggregate {
  verdicts: { daily: VerdictDistribution; note: VerdictDistribution; dailyByMonth: Record<string, VerdictDistribution> };
  issues: { daily: Record<string, Record<JudgeDimension, SeverityTally>>; note: Record<string, Record<JudgeDimension, SeverityTally>> };
  quality: {
    counts: { validFirstTry: number; validAfterRetry: number; invalid: number; errors: number; total: number };
    reasons: Array<{ reason: string; count: number }>;
  };
}

export interface JudgeReportData {
  schemaVersion: 1;
  generatedAt: string;
  run: JudgeRunInfo;
  records: JudgmentRecord[];
  aggregate: JudgeAggregate;
  crossCheck: CrossCheck | null;
}

export async function loadJudgments(outDir: string): Promise<{ run: JudgeRunInfo; records: JudgmentRecord[] }> {
  const dir = path.join(outDir, "judgments");
  const names = existsSync(dir) ? (await readdir(dir)).filter((name) => name.endsWith(".json")).sort() : [];
  const records: JudgmentRecord[] = [];
  for (const name of names) {
    try {
      const record = JSON.parse(await readFile(path.join(dir, name), "utf8")) as JudgmentRecord;
      if (record.schemaVersion === 1 && record.item) records.push(record);
    } catch { /* a half-written or foreign file is not a judgment */ }
  }
  if (records.length === 0) throw new Error(`no judgments found in ${dir}; run summary-judge.mjs with --yes first`);
  let run: JudgeRunInfo;
  try { run = JSON.parse(await readFile(path.join(outDir, "run.json"), "utf8")) as JudgeRunInfo; }
  catch { throw new Error(`${path.join(outDir, "run.json")} is missing or unreadable; the judgments cannot be described without it`); }
  records.sort((a, b) => a.item.kind.localeCompare(b.item.kind) || (a.item.date ?? "").localeCompare(b.item.date ?? "") || a.item.id.localeCompare(b.item.id));
  return { run, records };
}

const emptyTally = (): Tally => ({ ok: 0, minor: 0, major: 0 });
const emptyDistribution = (): VerdictDistribution => ({ judged: 0, accuracy: emptyTally(), completeness: emptyTally(), clarity: emptyTally() });

function addVerdicts(target: VerdictDistribution, judgment: Judgment): void {
  target.judged += 1;
  for (const dimension of JUDGE_DIMENSIONS) target[dimension][judgment.verdicts[dimension]] += 1;
}

/** The part of a validation reason that does not depend on the particular reply: path without index, up to the first quoted value. */
export function reasonCategory(reason: string): string {
  const cut = reason.search(/["“：，]/u);
  return (cut < 0 ? reason : reason.slice(0, cut)).replace(/\[\d+\]/g, "[]").trim().slice(0, 80);
}

const judged = (records: readonly JudgmentRecord[]): Array<JudgmentRecord & { judgment: Judgment }> =>
  records.filter((record): record is JudgmentRecord & { judgment: Judgment } => record.status === "judged" && record.judgment !== null);

function aggregate(run: JudgeRunInfo, records: readonly JudgmentRecord[]): JudgeAggregate {
  const verdicts = { daily: emptyDistribution(), note: emptyDistribution(), dailyByMonth: {} as Record<string, VerdictDistribution> };
  for (const record of records) {
    if (record.item.kind === "daily" && record.item.date) verdicts.dailyByMonth[record.item.date.slice(0, 7)] ??= emptyDistribution();
  }
  for (const month of Object.keys(run.allocation ?? {})) verdicts.dailyByMonth[month] ??= emptyDistribution();
  const issues: JudgeAggregate["issues"] = { daily: {}, note: {} };
  for (const record of judged(records)) {
    const kind = record.item.kind;
    addVerdicts(verdicts[kind], record.judgment);
    if (kind === "daily" && record.item.date) addVerdicts(verdicts.dailyByMonth[record.item.date.slice(0, 7)]!, record.judgment);
    for (const issue of record.judgment.issues) {
      const place = (issues[kind][issue.location] ??= { accuracy: { major: 0, minor: 0 }, completeness: { major: 0, minor: 0 }, clarity: { major: 0, minor: 0 } });
      place[issue.dimension][issue.severity] += 1;
    }
  }
  verdicts.dailyByMonth = Object.fromEntries(Object.entries(verdicts.dailyByMonth).sort(([a], [b]) => a.localeCompare(b)));

  const counts = { validFirstTry: 0, validAfterRetry: 0, invalid: 0, errors: 0, total: records.length };
  const reasons = new Map<string, number>();
  for (const record of records) {
    if (record.status === "judged") (record.attempts.length <= 1 ? counts.validFirstTry += 1 : counts.validAfterRetry += 1);
    else if (record.status === "invalid") counts.invalid += 1;
    else counts.errors += 1;
    for (const attempt of record.attempts) for (const reason of attempt.reasons ?? []) reasons.set(reasonCategory(reason), (reasons.get(reasonCategory(reason)) ?? 0) + 1);
  }
  return {
    verdicts, issues,
    quality: { counts, reasons: [...reasons].map(([reason, count]) => ({ reason, count })).sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason)) },
  };
}

interface OfflineItem { id: string; date?: string | null; ungrounded: Array<{ where: string; token: string; context: string }> }

async function readOffline(file: string): Promise<{ daily: Map<string, OfflineItem>; notes: Map<string, OfflineItem> }> {
  let parsed: { grounding?: { run?: boolean; daily?: { results?: OfflineItem[] }; notes?: { results?: OfflineItem[] } } };
  try { parsed = JSON.parse(await readFile(file, "utf8")); }
  catch { throw new Error(`--offline ${file} is not readable JSON`); }
  if (!parsed.grounding?.run) throw new Error(`--offline ${file} has no grounding data; run summary-check.mjs with --sources to produce it`);
  const index = (items: OfflineItem[] | undefined) => new Map((items ?? []).map((item) => [item.id, item] as const));
  return { daily: index(parsed.grounding.daily?.results), notes: index(parsed.grounding.notes?.results) };
}

const unsigned = (token: string): string => token.replace(/^-/, "");
const numbersOf = (text: string): string[] => extractNumbers(text, { keepSmallIntegers: true }).map((number) => unsigned(number.token));

async function crossCheck(records: readonly JudgmentRecord[], offlinePath: string): Promise<CrossCheck> {
  const offline = await readOffline(offlinePath);
  const result: CrossCheck = { counts: { both: 0, onlyOffline: 0, onlyJudge: 0, judgeWithoutNumber: 0 }, both: [], onlyOffline: [], onlyJudge: [], notCompared: 0 };
  for (const record of judged(records)) {
    const item = (record.item.kind === "daily" ? offline.daily : offline.notes).get(record.item.id);
    if (!item) { result.notCompared += 1; continue; }
    const accuracy = record.judgment.issues.filter((issue) => issue.dimension === "accuracy");
    const matched = new Set<JudgeIssue>();
    for (const unfound of item.ungrounded) {
      const hit = accuracy.find((issue) => issue.location === unfound.where && numbersOf(issue.summaryQuote).includes(unsigned(unfound.token)));
      const entry: CrossCheckEntry = { kind: record.item.kind, id: record.item.id, date: record.item.date, location: unfound.where, token: unfound.token, context: unfound.context };
      if (hit) { matched.add(hit); result.both.push({ ...entry, explanation: hit.explanation }); }
      else result.onlyOffline.push(entry);
    }
    for (const issue of accuracy) {
      if (matched.has(issue)) continue;
      const numbers = numbersOf(issue.summaryQuote);
      if (numbers.length === 0) { result.counts.judgeWithoutNumber += 1; continue; }
      result.onlyJudge.push({ kind: record.item.kind, id: record.item.id, date: record.item.date, location: issue.location, token: numbers.join("、"), context: issue.summaryQuote, explanation: issue.explanation });
    }
  }
  result.counts.both = result.both.length;
  result.counts.onlyOffline = result.onlyOffline.length;
  result.counts.onlyJudge = result.onlyJudge.length;
  return result;
}

export function buildJudgeReport(run: JudgeRunInfo, records: JudgmentRecord[], now: Date, cross: CrossCheck | null): JudgeReportData {
  return { schemaVersion: 1, generatedAt: now.toISOString(), run, records, aggregate: aggregate(run, records), crossCheck: cross };
}

// ---- rendering -------------------------------------------------------------------------------------

const clip = (text: string, limit: number): string => (text.length > limit ? `${text.slice(0, limit)}…` : text);
const tally = (counts: Tally, judgedCount: number): string[] => (["ok", "minor", "major"] as const).map((level) => `${counts[level]}（${(judgedCount === 0 ? 0 : (counts[level] / judgedCount) * 100).toFixed(1)}%）`);

function distributionTable(distribution: VerdictDistribution): string[] {
  return table(["维度", "ok", "minor", "major"], JUDGE_DIMENSIONS.map((dimension) => [DIMENSION_LABELS[dimension], ...tally(distribution[dimension], distribution.judged)]));
}

function issueRows(records: readonly JudgmentRecord[], kind: "daily" | "note", dimension: JudgeDimension) {
  const rows = judged(records).filter((record) => record.item.kind === kind).flatMap((record) =>
    record.judgment.issues.filter((issue) => issue.dimension === dimension).map((issue) => ({ record, issue })));
  rows.sort((a, b) => Number(b.issue.severity === "major") - Number(a.issue.severity === "major")
    || (a.record.item.date ?? "").localeCompare(b.record.item.date ?? "") || a.record.item.id.localeCompare(b.record.item.id));
  return rows.map(({ record, issue }) => ({
    cells: [
      record.item.date ?? "-", record.item.id, cell(clip(record.item.title, TITLE_CHARS)), cell(issue.location), issue.severity,
      cell(clip(issue.summaryQuote, QUOTE_CHARS)), issue.sourceQuote === null ? NOT_IN_SOURCE : cell(clip(issue.sourceQuote, QUOTE_CHARS)), cell(issue.explanation),
    ],
  }));
}

function crossTable(entries: readonly CrossCheckEntry[], withExplanation: boolean) {
  const header = withExplanation ? ["日期", "编号", "位置", "数字", "上下文", "评审的说明"] : ["日期", "编号", "位置", "数字", "上下文"];
  return capped(header, entries.map((entry) => ({
    cells: [entry.date ?? "-", entry.id, cell(entry.location), cell(entry.token), cell(clip(entry.context, QUOTE_CHARS)), ...(withExplanation ? [cell(entry.explanation ?? "")] : [])],
  })), "judgments.json");
}

export function renderJudgeReport(data: JudgeReportData): string {
  const { run, records, aggregate: a } = data;
  const rec = judged(records);
  const sources = records.map((record) => record.item.source);
  const count = (values: readonly string[]) => Object.entries(values.reduce<Record<string, number>>((all, value) => ({ ...all, [value]: (all[value] ?? 0) + 1 }), {})).map(([key, value]) => `${key} ${value}`).join("、") || "-";
  const fetched = sources.map((source) => source.fetchedAt).sort();
  const older = records.filter((record) => record.item.version === "older").length;
  const q = a.quality.counts;

  const lines: string[] = [
    "# 摘要大模型评审报告",
    "",
    "本报告由评审模型对照原文，按准确、完整、清楚三项检查已生成的日报条目和详细笔记。评审的引用已由程序核对过：摘要原句必须出现在被评审的位置，原文依据必须出现在评审看到的原文里，对不上的判定记为无效。格式、长度和公式写法不在评审范围内，由离线检查负责。",
    "",
    "## 评审对象",
    "",
    `- 评审模型：${run.judge.model}（${run.judge.provider}）；思考模式 ${run.judge.thinkingMode === undefined ? "-" : run.judge.thinkingMode ? "开" : "关"}，推理强度 ${run.judge.reasoningEffort ?? "-"}。`,
    `- 生成这些摘要的模型：${run.generatorModel}。评审模型与生成模型不同。`,
    `- 评审时间：${run.startedAt}${run.finishedAt ? ` 至 ${run.finishedAt}` : ""}；报告生成时间 ${data.generatedAt}。`,
    `- 抽样：种子 ${run.seed}，要求日报条目 ${run.requested.daily} 条、详细笔记 ${run.requested.notes} 篇；实际抽到日报条目 ${run.sample.daily.length} 条、笔记 ${run.sample.notes.length} 篇${older > 0 ? `（其中 ${older} 篇是更早提示词写的笔记）` : ""}。`,
    `- 结果：评审成功 ${q.validFirstTry + q.validAfterRetry} 条，无效 ${q.invalid} 条，调用出错 ${q.errors} 条。`,
    `- 评审看到的原文是评估时重新取回的（取回时间 ${fetched.length ? `${fetched[0]} 至 ${fetched.at(-1)}` : "-"}；取回路径：${count(sources.map((source) => source.sourcePath))}；arXiv 版本：${count(sources.map((source) => source.arxivVersion ?? "unknown"))}），论文可能已更新，与生成摘要时模型读到的不一定相同。`,
    "- 评审参考的生成提示词是当前版本的提示词；更早生成的内容也按当前提示词的要求评审。",
    "",
    "### 日报条目按月份的抽样数",
    "",
    ...table(["月份", "抽样条数"], Object.entries(run.allocation ?? {}).map(([month, n]) => [month, n])),
    "### 没有评审的条目",
    "",
    ...(run.skipped.length === 0 ? ["没有。", ""] : table(["编号", "类型", "原因", "说明"], run.skipped.map((entry) => [entry.id, entry.kind, entry.reason, cell(entry.detail)]))),
    "## 判定分布",
    "",
    "只统计评审成功的条目；比例的分母是该类评审成功的条数。",
    "",
    `### ${KIND_LABELS.daily}（${a.verdicts.daily.judged} 条）`,
    "",
    ...distributionTable(a.verdicts.daily),
    `### ${KIND_LABELS.note}（${a.verdicts.note.judged} 篇）`,
    "",
    ...distributionTable(a.verdicts.note),
    "### 日报条目按月份",
    "",
    ...table(["月份", "评审条数", "准确（ok/minor/major）", "完整（ok/minor/major）", "清楚（ok/minor/major）"],
      Object.entries(a.verdicts.dailyByMonth).map(([month, d]) => [month, d.judged, ...JUDGE_DIMENSIONS.map((dimension) => `${d[dimension].ok}/${d[dimension].minor}/${d[dimension].major}`)])),
    "## 问题分布",
    "",
    "每格是“大问题/小问题”的个数。",
    "",
  ];
  for (const kind of ["daily", "note"] as const) {
    const entries = Object.entries(a.issues[kind]);
    lines.push(`### ${KIND_LABELS[kind]}`, "", ...(entries.length === 0 ? ["没有问题。", ""] : table([kind === "daily" ? "字段" : "章节", ...JUDGE_DIMENSIONS.map((dimension) => DIMENSION_LABELS[dimension])],
      entries.map(([location, d]) => [cell(location), ...JUDGE_DIMENSIONS.map((dimension) => `${d[dimension].major}/${d[dimension].minor}`)]))));
  }

  lines.push("## 问题清单", "", "大问题在前。每类最多列出 30 条，完整数据在 judgments.json。“原文未提及”表示评审认为摘要的说法在原文里没有依据。", "");
  for (const kind of ["daily", "note"] as const) {
    for (const dimension of JUDGE_DIMENSIONS) {
      lines.push(`### ${KIND_LABELS[kind]}：${DIMENSION_ISSUE_TITLES[dimension]}`, "",
        ...capped(["日期", "编号", "标题", "位置", "严重程度", "摘要原句", "原文依据", "说明"], issueRows(records, kind, dimension), "judgments.json"));
    }
  }

  lines.push(
    "## 评审质量",
    "",
    "这一节说明评审模型有没有遵守输出合同，不是摘要的质量。",
    "",
    `- 第一次就合格：${q.validFirstTry} 条`,
    `- 重试后合格：${q.validAfterRetry} 条`,
    `- 重试后仍无效：${q.invalid} 条`,
    `- 调用出错：${q.errors} 条`,
    `- 合计：${q.total} 条；合格率 ${percent(rate(q.validFirstTry + q.validAfterRetry, q.total))}`,
    "",
    "### 校验失败最常见的原因",
    "",
    ...(a.quality.reasons.length === 0 ? ["没有校验失败。", ""] : table(["原因", "次数"], a.quality.reasons.slice(0, TOP_REASONS).map((entry) => [cell(entry.reason), entry.count]))),
  );
  if (rec.length === 0) lines.push("没有评审成功的条目，上面的分布为空。", "");

  if (data.crossCheck) {
    const c = data.crossCheck;
    lines.push(
      "## 与离线检查对照",
      "",
      "把离线检查（summary-check 的数字溯源）在原文里找不到的数字，与评审在同一位置提出的准确性问题对照：评审的问题引用了这个数字，就算两边都指出。",
      "",
      `- 两边都指出：${c.counts.both} 个数字`,
      `- 只有离线检查指出：${c.counts.onlyOffline} 个数字（评审没有在该位置对它提问题）`,
      `- 只有评审指出：${c.counts.onlyJudge} 个（评审的准确性问题引用了数字，但离线检查认为这些数字能在原文里找到）`,
      `- 评审的准确性问题引用里没有数字，无法对照：${c.counts.judgeWithoutNumber} 个`,
      `- 离线数据里没有的评审条目：${c.notCompared} 条，未参与对照`,
      "",
      "### 两边都指出", "", ...crossTable(c.both, true),
      "### 只有离线检查指出", "", ...crossTable(c.onlyOffline, false),
      "### 只有评审指出", "", ...crossTable(c.onlyJudge, true),
    );
  }
  return `${lines.join("\n")}\n`;
}

export interface WriteReportOptions { now?: () => Date; offlinePath?: string }

async function writeAtomic(file: string, text: string): Promise<void> {
  const temporary = `${file}.tmp`;
  await writeFile(temporary, text);
  await rename(temporary, file);
}

/** Write `report.md` and `judgments.json` from the judgments already in `outDir`. Calls no model. */
export async function writeJudgeReport(outDir: string, options: WriteReportOptions = {}): Promise<{ files: string[]; data: JudgeReportData }> {
  const { run, records } = await loadJudgments(outDir);
  const cross = options.offlinePath ? await crossCheck(records, options.offlinePath) : null;
  const data = buildJudgeReport(run, records, (options.now ?? (() => new Date()))(), cross);
  const files = [path.join(outDir, "report.md"), path.join(outDir, "judgments.json")];
  await writeAtomic(files[0]!, renderJudgeReport(data));
  await writeAtomic(files[1]!, `${JSON.stringify(data, null, 2)}\n`);
  return { files, data };
}

/** `--report-only`: rebuild the report from what is on disk. */
export async function runJudgeReportOnly(options: { outDir: string; offlinePath?: string }, deps: { now?: () => Date } = {}): Promise<{ files: string[] }> {
  const { files } = await writeJudgeReport(path.resolve(options.outDir), { now: deps.now, offlinePath: options.offlinePath });
  return { files };
}
