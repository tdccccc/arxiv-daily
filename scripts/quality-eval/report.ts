import type { RecordedReport } from "./evaluate";
import { SCORE_THRESHOLDS, type Disagreement, type MetricSet, type Ratio, type TopicMetrics, type WeightedSummary } from "./metrics";
import type { Comparison, RerunReport, RunMetricSet } from "./rerun";
import type { MetaDirection, SourceSummary } from "./source";

const NA = "n/a";

const percent = (ratio: Ratio): string => (ratio.value === null ? NA : `${(ratio.value * 100).toFixed(1)}%`);
/** A percentage with the rows behind it, e.g. `66.7%（4/6）`. */
const withCount = (ratio: Ratio): string => (ratio.value === null ? `${NA}（n=${ratio.n}）` : `${percent(ratio)}（${ratio.hits}/${ratio.n}）`);
const estimate = (ratio: Ratio): string => (ratio.value === null ? NA : percent(ratio));

/** One-line safe text for a Markdown table cell or list item: no newlines, pipes or emphasis. */
function cell(text: string): string {
  return text.replace(/\s+/g, " ").replace(/\\/g, "\\\\").replace(/\|/g, "\\|").replace(/([*_`])/g, "\\$1").trim() || " ";
}

function numbers(values: readonly number[]): string {
  return values.length > 0 ? values.join(";") : "-";
}

/** Weighted precision / recall / F1 in one cell, since the weights make the counts fractional. */
function weightedSummary(weighted: WeightedSummary | null): string {
  return weighted === null ? "仅未加权" : `${estimate(weighted.precision)} / ${estimate(weighted.recall)} / ${estimate(weighted.f1)}`;
}

function weightedTriple(precision: Ratio | null, recall: Ratio | null, f1: Ratio | null): string {
  return precision === null || recall === null || f1 === null ? "仅未加权" : `${estimate(precision)} / ${estimate(recall)} / ${estimate(f1)}`;
}

const WEIGHTED_HEADER = "加权估计（精确率 / 召回率 / F1）";

function disagreementTable(rows: readonly Disagreement[], empty: string): string[] {
  if (rows.length === 0) return [empty, ""];
  return [
    "| 表格行 | 日期 | 编号 | 标题 | 模型分数 | 模型方向 | 标注方向 | 筛选时有摘要 | 备注 |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ...rows.map((row) => `| ${row.row} | ${row.date} | ${row.id} | ${cell(row.title)} | ${row.score} | ${numbers(row.modelDirections)} | ${numbers(row.labelDirections)} | ${row.hadAbstractAtFilter ? "是" : "否"} | ${cell(row.note)} |`),
    "",
  ];
}

/** What the metric sections show; a withheld part (direction accuracy, topics) is null. */
interface MetricView {
  sets: { all: RunMetricSet; withAbstract: RunMetricSet };
  topics: TopicMetrics[] | null;
  topicsWithoutDirection: number | null;
  disagreements: { falseNegatives: Disagreement[]; falsePositives: Disagreement[]; directionMismatches: Disagreement[] | null };
  skipped: SourceSummary["skipped"];
  /** What the second metric line leaves out. */
  withAbstractLabel: string;
}

/** The metric sections shared by every report, at heading `level` (subsections one deeper). */
function metricSections(view: MetricView, level: number): string[] {
  const h = "#".repeat(level);
  const { all, withAbstract } = view.sets;
  const lines: string[] = [
    `${h} 主要结果`,
    "",
    "“被选中”指模型给了某个主题（不是 skip）；每日篇数上限和忽略标记不算筛选判定。",
    "样本内指标按表里的行直接计数；加权估计把被抽样标注的“原本被判不相关”行按抽样比例还原。",
    "",
    `| 范围 | 已标注行数 | 精确率（样本内） | 召回率（样本内） | F1（样本内） | ${WEIGHTED_HEADER} |`,
    "| --- | --- | --- | --- | --- | --- |",
    `| 全部已标注行 | ${all.rows} | ${withCount(all.precision)} | ${withCount(all.recall)} | ${estimate(all.f1)} | ${weightedSummary(all.weighted)} |`,
    `| ${view.withAbstractLabel} | ${withAbstract.rows} | ${withCount(withAbstract.precision)} | ${withCount(withAbstract.recall)} | ${estimate(withAbstract.f1)} | ${weightedSummary(withAbstract.weighted)} |`,
    "",
  ];
  if (all.weighted === null) {
    lines.push("全部行的加权估计不可用：元数据缺少被判不相关的论文总数，或没有标注任何被判不相关的行；以上为仅未加权结果。", "");
  } else {
    lines.push(
      `加权方式：每个已标注的“原本被判不相关”行代表 ${all.weighted.skippedWeight.toFixed(2)} 篇论文；被选中的论文在导出时全部在表中，权重为 1，所以记录的判定里加权精确率与样本内精确率相同。`,
      "重跑时如果新模型选中了抽样的跳过层行，样本内精确率会偏高，比较改动前后应看加权列。",
      `“${view.withAbstractLabel}”没有加权，因为这类论文在被判不相关的总数里占多少没有记录。`,
      "",
    );
  }

  lines.push(`${h} 方向判对了多少`, "");
  if (all.direction === null || withAbstract.direction === null) {
    lines.push("方向列表与标注时不同，方向准确率不可用。", "");
  } else {
    lines.push(
      "只看模型选中、标注也是 y 的论文：标注的方向只要有一个出现在模型给的方向里，就算判对。没有填标注方向的论文不计入。",
      "",
      `- 全部行：${withCount(all.direction)}；另有 ${all.direction.withoutLabelledDirection} 篇没有填标注方向。`,
      `- ${view.withAbstractLabel}：${withCount(withAbstract.direction)}；另有 ${withAbstract.direction.withoutLabelledDirection} 篇没有填标注方向。`,
      "",
    );
  }

  lines.push(
    `${h} 不同分数线下的结果`,
    "",
    `只有分数不低于分数线、且不是 skip 的论文才算被选中（全部已标注行，括号里是行数）。分数是模型给的 0–100 相关度，分数线取 ${SCORE_THRESHOLDS.join("、")}。`,
    "",
    "| 分数线 | 精确率（样本内） | 召回率（样本内） | " + WEIGHTED_HEADER + " |",
    "| --- | --- | --- | --- |",
    ...all.thresholds.map((entry) => `| ≥ ${entry.threshold} | ${withCount(entry.precision)} | ${withCount(entry.recall)} | ${weightedSummary(entry.weighted)} |`),
    "",
    `${h} 分主题`,
    "",
  );
  if (view.topics === null) {
    lines.push("方向列表与标注时不同，分主题结果不可用。", "");
  } else {
    lines.push(
      "精确率：模型归入该主题的论文里，标注为 y 的比例。召回率：标注方向属于该主题的 y 论文里，模型归入该主题的比例。样本内按行计数；加权估计把被抽样标注的“原本被判不相关”行按抽样比例还原。",
      "",
      "| 主题 | 精确率（样本内） | 召回率（样本内） | 加权估计（精确率 / 召回率） |",
      "| --- | --- | --- | --- |",
      ...view.topics.map((topic) => `| ${cell(topic.tag)} | ${withCount(topic.precision)} | ${withCount(topic.recall)} | ${topic.weighted === null ? "仅未加权" : `${estimate(topic.weighted.precision)} / ${estimate(topic.weighted.recall)}`} |`),
      "",
      `另有 ${view.topicsWithoutDirection ?? 0} 篇标注为 y 但没有填标注方向，无法归入主题召回率。`,
      "",
    );
  }

  lines.push(
    `${h} 分歧列表`,
    "",
    `${h}# 漏选（标注应选，模型判为不相关）`,
    "",
    ...disagreementTable(view.disagreements.falseNegatives, "没有。"),
    `${h}# 误选（标注不应选，模型选中了）`,
    "",
    ...disagreementTable(view.disagreements.falsePositives, "没有。"),
    `${h}# 方向不一致（选对了，但方向没对上）`,
    "",
    ...(view.disagreements.directionMismatches === null ? ["方向列表与标注时不同，不判断方向是否一致。", ""] : disagreementTable(view.disagreements.directionMismatches, "没有。")),
  );
  return lines;
}

function sampleLines(report: { sheet: RecordedReport["sheet"]; source: SourceSummary }): string[] {
  const { sheet, source } = report;
  return [
    `- 表格共 ${sheet.rows} 行，已标注 ${sheet.labelled} 行（应选 y：${sheet.labelledYes}，不应选 n：${sheet.labelledNo}），未标注 ${sheet.unlabelled} 行（不计入任何指标）。`,
    `- 筛选当时只看到标题、没有摘要的行：${sheet.withoutAbstractAtFilter} 行。`,
    source.skipped.available === null
      ? "- 元数据没有记录被判不相关的论文总数，召回率只能给出未加权结果。"
      : `- 被模型判为不相关的论文共 ${source.skipped.available} 篇，表中抽样了 ${source.skipped.sampled ?? "?"} 篇。抽样会让表里的召回率偏高，所以另给出按抽样比例加权的估计。`,
  ];
}

function describeSource(source: SourceSummary): string[] {
  const models = [...new Set(source.records.map((record) => `${record.model}（${record.provider}）`))];
  const prompts = [...new Set(source.records.map((record) => `提示词 v${record.promptContractVersion} / 结果 v${record.resultContractVersion}`))];
  return [
    `- 日报日期：${source.dates.join("、") || "-"}`,
    `- 当时的模型：${models.join("、") || "-"}`,
    `- 当时的提示词版本：${prompts.join("、") || "-"}`,
    `- 抽样种子：${source.seed || "-"}`,
  ];
}

const directionLine = (directions: readonly MetaDirection[]): string =>
  `- 方向（编号对应标注表）：${directions.map((direction) => `${direction.number}. ${cell(direction.tag)}：${cell(direction.text)}`).join("；") || "-"}`;

export function renderRecordedReport(report: RecordedReport): string {
  const { sheet, source } = report;
  const lines: string[] = [
    "# 筛选质量评估报告（记录的判定）",
    "",
    "本报告只对日报当时记录下来的筛选判定打分，未调用模型。",
    "",
    "## 评估对象",
    "",
    `- 生成时间：${report.generatedAt}`,
    `- 标注表：${sheet.path}`,
    `- 元数据：${sheet.metaPath}`,
    ...describeSource(source),
    directionLine(report.directions),
    "",
    "## 样本",
    "",
    ...sampleLines(report),
    "",
    ...metricSections({
      sets: report.sets, topics: report.topics, topicsWithoutDirection: report.topicsWithoutDirection,
      disagreements: report.disagreements, skipped: source.skipped, withAbstractLabel: "排除筛选时无摘要的行",
    }, 2),
  ];
  return `${lines.join("\n")}\n`;
}

const selectedMark = (selected: boolean): string => (selected ? "选中" : "未选");

function comparisonTable(entries: readonly Comparison[]): string[] {
  return [
    `| 来源 | 已标注行数 | 精确率（样本内） | 召回率（样本内） | F1（样本内） | ${WEIGHTED_HEADER} |`,
    "| --- | --- | --- | --- | --- | --- |",
    ...entries.map((entry) => `| ${entry.kind === "recorded" ? "记录的判定" : `第 ${entry.name.replace("run ", "")} 次重跑`} | ${entry.rows} | ${withCount(entry.precision)} | ${withCount(entry.recall)} | ${estimate(entry.f1)} | ${weightedTriple(entry.weightedPrecision, entry.weightedRecall, entry.weightedF1)} |`),
    "",
  ];
}

export function renderRerunReport(report: RerunReport): string {
  const { sheet, source, current } = report;
  const okRuns = report.runs.filter((run) => run.status === "ok");
  const lines: string[] = [
    "# 筛选质量评估报告（真实模型重跑）",
    "",
    "本报告用当前的提示词和模型，对标注过的论文重新跑了筛选，并与日报当时记录的判定放在一起比较。",
    "",
  ];
  for (const warning of report.warnings) lines.push(`> **注意**：${cell(warning)}`, "");
  lines.push(
    "## 评估对象",
    "",
    `- 生成时间：${report.generatedAt}`,
    `- 标注表：${sheet.path}`,
    `- 元数据：${sheet.metaPath}`,
    `- 重跑使用的模型：${current.model}（${current.provider}），模式：${JSON.stringify(current.mode)}`,
    `- 重跑使用的提示词版本：提示词 v${current.promptContractVersion} / 结果 v${current.resultContractVersion}`,
    `- 方向来源：${current.topicSource === "config" ? "CLI 配置里的主题" : "标注表记录的方向快照"}`,
    `- 请求里的 arXiv 分类：${current.categories.join("、") || "-"}`,
    `- 每次运行一个请求，共 ${current.papers} 篇已标注论文（约 ${current.requestCharacters} 个字符的提示词），运行 ${current.runs} 次`,
    ...describeSource(source),
    directionLine(report.directions),
    "",
    "## 样本",
    "",
    ...sampleLines(report),
    "- 模型没有在回答里提到的论文按 skip 处理。",
    "",
    "## 记录的判定与重跑对比",
    "",
    "同一批已标注的行；加权估计按抽样时的分层计算，与新判定无关。比较提示词或模型的改动时，请看加权列。",
    "",
    ...comparisonTable(report.comparison),
    "### 只看筛选时有摘要的行",
    "",
    "这里只比较筛选时已有摘要的论文：记录的判定和重跑看到的是同一批论文、同样的输入，可以直接对照。这些行没有加权估计。",
    "",
    ...comparisonTable(report.comparisonWithAbstract),
    "## 运行之间是否稳定",
    "",
  );
  if (report.agreement === null) {
    lines.push(okRuns.length === 0 ? "没有成功的运行，无法比较。" : "只有一次成功的运行，无法比较运行之间的一致性。", "");
  } else {
    lines.push(
      `${report.agreement.runs} 次成功的运行里，有 ${withCount(report.agreement.identical)} 的已标注论文每次都得到同样的选中与否。`,
      "",
      "### 运行之间摇摆的论文",
      "",
    );
    if (report.agreement.flipFlops.length === 0) lines.push("没有。", "");
    else {
      lines.push(
        "| 编号 | 标题 | 标注 | 各次运行 |",
        "| --- | --- | --- | --- |",
        ...report.agreement.flipFlops.map((row) => `| ${row.id} | ${cell(row.title)} | ${row.label} | ${row.selected.map(selectedMark).join(" / ")} |`),
        "",
      );
    }
  }
  if (okRuns.length === 0) lines.push(`全部 ${report.runs.length} 次运行都失败了，没有可打分的结果。`, "");

  for (const run of report.runs) {
    lines.push(`## 第 ${run.run} 次重跑`, "");
    if (run.status === "failed" || !run.sets || !run.disagreements) {
      lines.push(`这次运行失败（${run.failure === "validation" ? "模型的回答不符合筛选的格式要求" : "模型调用失败"}）：${run.error ?? "未知原因"}`, "");
      continue;
    }
    lines.push(
      `- 模型漏掉（没有在回答里提到）的论文：${run.omitted ?? 0} 篇，已按 skip 处理。`,
      ...(run.usage?.inputTokens !== undefined || run.usage?.outputTokens !== undefined ? [`- 用量：输入 ${run.usage.inputTokens ?? "-"} 个 token，输出 ${run.usage.outputTokens ?? "-"} 个 token。`] : []),
      "",
      ...metricSections({
        sets: run.sets, topics: run.topics ?? null, topicsWithoutDirection: run.topicsWithoutDirection ?? null,
        disagreements: run.disagreements, skipped: source.skipped, withAbstractLabel: "排除筛选时无摘要的行",
      }, 3),
    );
  }
  return `${lines.join("\n")}\n`;
}
