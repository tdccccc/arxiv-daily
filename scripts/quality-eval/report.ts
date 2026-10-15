import type { Disagreement, RecordedReport } from "./evaluate";
import { SCORE_THRESHOLDS, type MetricSet, type Ratio, type WeightedSummary } from "./metrics";

const NA = "n/a";

const percent = (ratio: Ratio): string => (ratio.value === null ? NA : `${(ratio.value * 100).toFixed(1)}%`);
/** A percentage with the rows behind it, e.g. `66.7%（4/6）`. */
const withCount = (ratio: Ratio): string => (ratio.value === null ? `${NA}（n=${ratio.n}）` : `${percent(ratio)}（${ratio.hits}/${ratio.n}）`);
const estimate = (ratio: Ratio): string => (ratio.value === null ? NA : percent(ratio));

function cell(text: string): string {
  return text.replace(/\s+/g, " ").replace(/\|/g, "\\|").trim() || " ";
}

function numbers(values: readonly number[]): string {
  return values.length > 0 ? values.join(";") : "-";
}

function mainRows(label: string, set: MetricSet | null): string[] {
  if (set === null) return [];
  return [
    `| ${label} | ${set.rows} | ${withCount(set.precision)} | ${withCount(set.recall)} | ${estimate(set.f1)} | ${weightedRecall(set.weighted)} |`,
  ];
}

function weightedRecall(weighted: WeightedSummary | null): string {
  return weighted === null ? "仅未加权" : `${estimate(weighted.recall)}（F1 ${estimate(weighted.f1)}）`;
}

function disagreementTable(rows: readonly Disagreement[], empty: string): string[] {
  if (rows.length === 0) return [empty, ""];
  return [
    "| 表格行 | 日期 | 编号 | 标题 | 模型分数 | 模型方向 | 标注方向 | 筛选时有摘要 | 备注 |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ...rows.map((row) => `| ${row.row} | ${row.date} | ${row.id} | ${cell(row.title)} | ${row.score} | ${numbers(row.modelDirections)} | ${numbers(row.labelDirections)} | ${row.hadAbstractAtFilter ? "是" : "否"} | ${cell(row.note)} |`),
    "",
  ];
}

export function renderRecordedReport(report: RecordedReport): string {
  const { sheet, source, sets } = report;
  const models = [...new Set(source.records.map((record) => `${record.model}（${record.provider}）`))];
  const prompts = [...new Set(source.records.map((record) => `提示词 v${record.promptContractVersion} / 结果 v${record.resultContractVersion}`))];
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
    `- 日报日期：${source.dates.join("、") || "-"}`,
    `- 当时的模型：${models.join("、") || "-"}`,
    `- 当时的提示词版本：${prompts.join("、") || "-"}`,
    `- 抽样种子：${source.seed || "-"}`,
    `- 方向（编号对应标注表）：${report.directions.map((direction) => `${direction.number}. ${direction.tag}：${direction.text}`).join("；") || "-"}`,
    "",
    "## 样本",
    "",
    `- 表格共 ${sheet.rows} 行，已标注 ${sheet.labelled} 行（应选 y：${sheet.labelledYes}，不应选 n：${sheet.labelledNo}），未标注 ${sheet.unlabelled} 行（不计入任何指标）。`,
    `- 筛选当时只看到标题、没有摘要的行：${sheet.withoutAbstractAtFilter} 行。`,
    source.skipped.available === null
      ? "- 元数据没有记录被判不相关的论文总数，召回率只能给出未加权结果。"
      : `- 被模型判为不相关的论文共 ${source.skipped.available} 篇，表中抽样了 ${source.skipped.sampled ?? "?"} 篇。抽样会让表里的召回率偏高，所以另给出按抽样比例加权的估计。`,
    "",
    "## 主要结果",
    "",
    "“被选中”指模型给了某个主题（不是 skip）；每日篇数上限和忽略标记不算筛选判定。",
    "",
    "| 范围 | 已标注行数 | 精确率（选中的有多少该选） | 召回率（该选的选中了多少，样本内） | F1 | 召回率估计（按抽样加权） |",
    "| --- | --- | --- | --- | --- | --- |",
    ...mainRows("全部已标注行", sets.all),
    ...mainRows("排除筛选时无摘要的行", sets.withAbstract),
    "",
  ];
  if (sets.all.weighted === null) {
    lines.push("全部行的加权估计不可用：元数据缺少被判不相关的论文总数，或没有标注任何被判不相关的行；以上为仅未加权结果。", "");
  } else {
    lines.push(
      `加权方式：每个已标注的“被判不相关”行代表 ${sets.all.weighted.skippedWeight.toFixed(2)} 篇论文；被选中的论文在导出时全部在表中，权重为 1。`,
      "“排除筛选时无摘要的行”没有加权，因为这类论文在被判不相关的总数里占多少没有记录。",
      "",
    );
  }
  const direction = sets.all.direction;
  const directionWithAbstract = sets.withAbstract.direction;
  lines.push(
    "## 方向判对了多少",
    "",
    "只看模型选中、标注也是 y 的论文：标注的方向只要有一个出现在模型给的方向里，就算判对。没有填标注方向的论文不计入。",
    "",
    `- 全部行：${withCount(direction)}；另有 ${direction.withoutLabelledDirection} 篇没有填标注方向。`,
    `- 排除筛选时无摘要的行：${withCount(directionWithAbstract)}；另有 ${directionWithAbstract.withoutLabelledDirection} 篇没有填标注方向。`,
    "",
    "## 不同分数线下的结果",
    "",
    `只有分数不低于分数线、且不是 skip 的论文才算被选中（全部已标注行，括号里是行数）。分数是模型给的 0–100 相关度，分数线取 ${SCORE_THRESHOLDS.join('、')}。`,
    "",
    "| 分数线 | 精确率 | 召回率（样本内） | 召回率估计（加权） |",
    "| --- | --- | --- | --- |",
    ...sets.all.thresholds.map((entry) => `| ≥ ${entry.threshold} | ${withCount(entry.precision)} | ${withCount(entry.recall)} | ${entry.weighted === null ? "仅未加权" : estimate(entry.weighted.recall)} |`),
    "",
    "## 分主题",
    "",
    "精确率：模型归入该主题的论文里，标注为 y 的比例。召回率：标注方向属于该主题的 y 论文里，模型归入该主题的比例。",
    "",
    "| 主题 | 精确率 | 召回率 |",
    "| --- | --- | --- |",
    ...report.topics.map((topic) => `| ${cell(topic.tag)} | ${withCount(topic.precision)} | ${withCount(topic.recall)} |`),
    "",
    `另有 ${report.topicsWithoutDirection} 篇标注为 y 但没有填标注方向，无法归入主题召回率。`,
    "",
    "## 分歧列表",
    "",
    "### 漏选（标注应选，模型判为不相关）",
    "",
    ...disagreementTable(report.disagreements.falseNegatives, "没有。"),
    "### 误选（标注不应选，模型选中了）",
    "",
    ...disagreementTable(report.disagreements.falsePositives, "没有。"),
    "### 方向不一致（选对了，但方向没对上）",
    "",
    ...disagreementTable(report.disagreements.directionMismatches, "没有。"),
  );
  return `${lines.join("\n")}\n`;
}
