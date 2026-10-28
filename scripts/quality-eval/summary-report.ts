import { cell } from "./report";
import {
  SUMMARY_FIELDS, type EntryChecks, type LengthDistribution, type Rate, type SummaryFieldKey, type SummaryFinding,
} from "./summary-checks";
import type { NoteChecks, NoteFinding } from "./summary-note-checks";
import type { GroundingItem, SummaryCheckReport } from "./summary-check";
import type { GroundingAggregate } from "./summary-grounding";

/** How many rows of one kind the report lists; findings.json has them all. */
export const LIST_CAP = 30;
const HASH_LENGTH = 12;

const FIELD_LABELS: Record<SummaryFieldKey, string> = {
  coreProblem: "研究问题", keyMethod: "方法设计", mainResult: "核心结果", whyRelevant: "研究价值", limitations: "适用边界",
};

const CODE_MEANINGS: Record<string, string> = {
  "bare-tex": "TeX 命令写在公式外面",
  "paren-delimiter": "使用了提示词禁止的 \\(..\\) 或 \\[..\\]",
  "split-fragment": "一个公式被拆成相邻的 $..$ 片段",
  "display-math": "日报字段里出现 $$ 独立公式",
  "unclosed-delimiter": "$ 没有闭合",
  "nested-delimiter": "公式里又出现了 $",
  "unmatched-delimiter": "孤立的 \\) 或 \\]",
  "unbalanced-parentheses": "公式里括号不成对",
  "unbalanced-braces": "公式里花括号不成对",
  "unbalanced-optional-argument": "公式里方括号不成对",
  "unclosed-display-math": "$$ 独立公式没有结束",
  "raw-html-in-math": "公式里有 HTML 形状的尖括号",
  "empty-math": "空公式",
  "extra-heading": "多出提示词没有的章节",
  "duplicate-heading": "章节重复",
  "out-of-order": "章节顺序与提示词不同",
};

export const percent = (rate: Rate): string => `${(rate.rate * 100).toFixed(1)}%（${rate.count}/${rate.of}）`;
const short = (hash: string): string => hash.slice(0, HASH_LENGTH);

export function table(header: readonly string[], rows: readonly (readonly (string | number)[])[]): string[] {
  return [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
    "",
  ];
}

function distributionRow(label: string, spread: LengthDistribution): (string | number)[] {
  return [label, spread.min, spread.median, spread.p90, spread.max];
}

function codeTable(codes: Record<string, number>): string[] {
  const entries = Object.entries(codes).sort(([, a], [, b]) => b - a);
  if (entries.length === 0) return ["没有。", ""];
  return table(["代码", "含义", "次数"], entries.map(([code, count]) => [cell(code), cell(CODE_MEANINGS[code] ?? "-"), count]));
}

export interface ListRow { cells: string[] }

/** A capped table of rows, or a single line saying there are none. */
export function capped(header: readonly string[], rows: readonly ListRow[], fullListIn = "findings.json"): string[] {
  if (rows.length === 0) return ["没有。", ""];
  const shown = rows.slice(0, LIST_CAP);
  return [
    ...(rows.length > shown.length ? [`共 ${rows.length} 条，仅列出前 ${LIST_CAP} 条，完整清单见 ${fullListIn}。`, ""] : [`共 ${rows.length} 条。`, ""]),
    ...table(header, shown.map((row) => row.cells)),
  ];
}

const DAILY_HEADER = ["日期", "编号", "字段", "代码", "片段"];
const NOTE_HEADER = ["日期", "编号", "章节", "代码", "片段"];

function dailyRows(results: readonly EntryChecks[], keep: (finding: SummaryFinding) => boolean): ListRow[] {
  return results.flatMap((result) => result.findings.filter(keep).map((finding) => ({
    cells: [result.date, result.id, FIELD_LABELS[finding.field], cell(finding.code), cell(finding.snippet)],
  })));
}

function noteRows(results: readonly NoteChecks[], keep: (finding: NoteFinding) => boolean): ListRow[] {
  return results.flatMap((result) => result.findings.filter(keep).map((finding) => ({
    cells: [result.date ?? "-", result.id, cell(finding.heading ?? "（整篇）"), cell(finding.code), cell(finding.snippet)],
  })));
}

function groundingRows(items: readonly GroundingItem[], placeLabel: (place: string) => string): ListRow[] {
  return items.flatMap((item) => item.ungrounded.map((number) => ({
    cells: [item.date ?? "-", item.id, cell(placeLabel(number.where)), cell(number.token), cell(number.context)],
  })));
}

function groundingTable(aggregate: GroundingAggregate, placeHeader: string, placeLabel: (place: string) => string): string[] {
  return [
    `有数字的${placeHeader === "字段" ? "条目" : "笔记"} ${aggregate.withNumbers} 个；其中数字全部找到的占 ${percent({ ...aggregate.withUngrounded, count: aggregate.withUngrounded.of - aggregate.withUngrounded.count, rate: 1 - aggregate.withUngrounded.rate })}，有至少一个没找到的占 ${percent(aggregate.withUngrounded)}。每${placeHeader === "字段" ? "条" : "篇"}找到的比例：最小 ${(aggregate.perItemShare.min * 100).toFixed(1)}%，中位数 ${(aggregate.perItemShare.median * 100).toFixed(1)}%，最大 ${(aggregate.perItemShare.max * 100).toFixed(1)}%。`,
    "",
    ...table([placeHeader, "找到的数字比例"], Object.entries(aggregate.byPlace).map(([place, value]) => [cell(placeLabel(place)), percent(value)])),
  ];
}

function groundingSection(report: SummaryCheckReport): string[] {
  const { grounding } = report;
  if (!grounding.run) {
    return [
      "## 数字能否在原文中找到",
      "",
      "本次没有指定 --sources，未做数字溯源检查。要做这项检查，先用 summary-sources.mjs 把论文原文取回缓存，再加上 `--sources <目录>` 重跑。",
      "",
    ];
  }
  const { sources, daily, notes } = grounding;
  const spread = (values: Record<string, number>): string => Object.entries(values).map(([key, value]) => `${key} ${value}`).join("、") || "-";
  return [
    "## 数字能否在原文中找到",
    "",
    "从日报字段和笔记正文里抽出数字（包括公式里的数值，统一负号、±、千分位、科学计数法和百分号），到缓存的原文（摘要、摘要与结论、各章节摘录）里查找。写法不同但数值相同、按摘要自己的精度取整后相同（0.25 对 0.248）、百分数与小数（25% 对 0.25）、“万”“亿”“million”换算后相同，都算找到。单独出现的 0–10 的整数、图表和章节编号、公式编号、引用编号、名称里的数字（DR10、W1、M_{500}）不检查。",
    "",
    `- 原文是评估时重新取回的，可能与当时生成摘要时模型读到的不同（论文可能已更新版本）。用到的原文 ${sources.records.count} 篇；取回时间 ${sources.records.fetchedAt ? `${sources.records.fetchedAt.first} 至 ${sources.records.fetchedAt.last}` : "-"}；取回路径：${spread(sources.records.bySourcePath)}；arXiv 版本：${spread(sources.records.byArxivVersion)}（读不到的记为 unknown）。`,
    `- 无原文的条目不参与这一项：日报条目 ${sources.daily.withoutSource} 条（有原文 ${sources.daily.withSource} 条），详细笔记 ${sources.notes.withoutSource} 篇（有原文 ${sources.notes.withSource} 篇）。`,
    "- “没找到”只说明原文里查不到这个数值，可能是模型编造，也可能是我们的规范化没覆盖（比如换算了单位），或原文版本不同，需要人工核对。",
    "",
    "### 日报条目",
    "",
    `找到的数字比例：${percent(daily.aggregate.numbers)}。`,
    "",
    ...groundingTable(daily.aggregate, "字段", (place) => FIELD_LABELS[place as SummaryFieldKey] ?? place),
    "### 详细笔记",
    "",
    `找到的数字比例：${percent(notes.aggregate.numbers)}。`,
    "",
    ...groundingTable(notes.aggregate, "章节", (place) => place),
  ];
}

export function renderSummaryReport(report: SummaryCheckReport): string {
  const { scope, corpus, daily, notes, prompts } = report;
  const { counts } = corpus;
  const a = daily.aggregate;
  const n = notes.aggregate;
  const { min, max } = report.limits.mainResultRange;
  const range = scope.range.from || scope.range.to ? `${scope.range.from ?? "最早"} 至 ${scope.range.to ?? "最新"}（含两端）` : "全部日期";
  const isDefect = (finding: { knownFalsePositive?: true }): boolean => !finding.knownFalsePositive;
  const isFalsePositive = (finding: { knownFalsePositive?: true }): boolean => finding.knownFalsePositive === true;

  const lines: string[] = [
    "# 摘要质量离线检查报告",
    "",
    "本报告只检查已经写进 Vault 的日报条目和详细笔记，没有调用模型，也没有联网。它检查格式、公式、空话和章节结构，字段长度只作为参考数据给出。它不判断内容是否准确、是否完整，这由之后的模型评审来做。",
    "",
    "## 评估范围",
    "",
    `- 生成时间：${report.generatedAt}`,
    `- Vault：${scope.vaultRoot}（日报目录 ${scope.dailyDir}，笔记目录 ${scope.papersDir}）`,
    `- 日期范围：${range}。笔记按它链接的日报日期或 published 日期归入范围，没有日期的笔记始终包含在内。`,
    `- 日报：共 ${counts.reports} 份，其中 ${counts.evaluableReports} 份有五字段条目或回退条目，纳入评估；${counts.legacyReports} 份是旧格式（没有五字段），只计数、不评估。`,
    `- 日报条目：五字段条目 ${counts.entries} 条（其中字段不全 ${counts.incompleteEntries} 条），回退条目 ${counts.fallbacks} 条。`,
    `- 详细笔记：当前格式 ${counts.notes} 篇纳入评估；旧格式 ${counts.legacyNotes} 篇只计数、不评估。`,
    `- 提示词文件的 SHA-256（前 ${HASH_LENGTH} 位，评估时仓库里打包的版本）：daily-paper-summary.system.md ${short(prompts.dailySummary.sha256)}；paper-detail.system.md ${short(prompts.paperDetail.sha256)}。`,
    "- 日报和笔记都没有记录生成时用的模型和提示词版本，所以这两个哈希只说明“评估时的提示词是什么”，不能证明这些内容就是用它们生成的。",
    "",
    "## 日报条目",
    "",
    "日报里的字段是渲染后的文本：空白压成一行，`[[` 被转义。公式检查先还原这种转义，再用核心代码的公式规则检查。",
    "",
    "### 回退",
    "",
    `回退比例 = 回退条目 /（五字段条目 + 回退条目）：${percent(a.fallback.overall)}。`,
    "",
    ...table(["月份", "回退比例"], Object.entries(a.fallback.byMonth).map(([month, value]) => [month, percent(value)])),
    "### 公式",
    "",
    `有公式问题的条目：${percent(a.entriesWith.math)}。`,
    "",
    ...codeTable(a.codes.math),
    `另有 ${a.coreFalsePositives.findings} 处（${a.coreFalsePositives.entries} 条）是核心规则误报，渲染正常，不计入上面的数字，见“问题清单”。`,
    "",
    "### 空话与“原文未说明”",
    "",
    `- 含提示词禁止的空话：${percent(a.entriesWith["generic-phrase"])}。`,
    `- 含“原文未说明”的条目：${percent(a.entriesWith["not-stated"])}。`,
    `- “作者声称”：共出现 ${a.claims.occurrences} 次，涉及 ${percent(a.claims.entriesWith)}。`,
    "",
    ...table(
      ["字段", "只写“原文未说明”", "含“原文未说明”", "“作者声称”次数"],
      SUMMARY_FIELDS.map((field) => [FIELD_LABELS[field], a.notStated[field].exact, a.notStated[field].partial, a.claims.byField[field]]),
    ),
    "### 字段长度",
    "",
    "长度只是参考数据，不算问题：下面的分布和按月表用来观察长度的变化，超出范围的条目不会进入问题清单（逐条数据仍在 findings.json）。长度 = 去掉 `$...$` 公式和所有空白后的字符数（标点、字母、数字都算），只统计该字段有内容的条目。",
    "",
    ...table(["字段", "最短", "中位数", "第 90 百分位", "最长"], SUMMARY_FIELDS.map((field) => distributionRow(FIELD_LABELS[field], a.lengths[field]))),
    `提示词对核心结果的要求是“通常 ${min}–${max} 个汉字”，超出这个范围的条目占 ${percent(a.mainResultOutsideRange)}。下表按日报月份看核心结果的长度。`,
    "",
    ...table(
      ["月份", "条目数", "中位数", "第 90 百分位", `落在 ${min}–${max} 内`],
      report.monthlyMainResult.map((row) => [row.month, row.entries, row.median, row.p90, percent(row.inRange)]),
    ),
    "## 详细笔记",
    "",
    `当前格式笔记 ${n.notes} 篇。笔记里的公式按行检查，允许 \`$$\` 独立公式和代码块。`,
    "",
    "### 章节分组",
    "",
    `笔记含提示词现在要求的全部章节，算“当前版本”（${n.versions.current} 篇）；缺章节的是更早的提示词写的“旧版本”（${n.versions.older} 篇），单独汇总，不算违规。`,
    "",
    ...table(
      ["篇数", "版本", "章节顺序", "示例"],
      n.groups.map((group) => [group.count, group.version === "current" ? "当前" : "旧版本", cell(group.sequence.join(" > ")), group.exampleIds.join("、")]),
    ),
    `当前版本笔记中章节结构有问题的：${percent(n.structure.notes)}。`,
    "",
    ...codeTable(n.structure.codes),
    "### “原文未说明”等占位语",
    "",
    ...(Object.keys(n.placeholders).length === 0
      ? ["没有。", ""]
      : table(["章节", "占位语", "篇数", "次数"], Object.entries(n.placeholders).flatMap(([heading, phrases]) =>
        Object.entries(phrases).map(([phrase, value]) => [cell(heading), cell(phrase), value.notes, value.occurrences])))),
    "### 公式与空话",
    "",
    `- 有公式问题的笔记：${percent(n.notesWith.math)}；另有 ${n.coreFalsePositives.findings} 处（${n.coreFalsePositives.notes} 篇）核心规则误报，不计入。`,
    `- 含提示词禁止的空话：${percent(n.notesWith["generic-phrase"])}。`,
    "",
    ...codeTable(n.codes.math),
    "### 正文中的英文占比",
    "",
    "英文占比 = 英文单词数 /（英文单词数 + 汉字数），只数公式、代码、链接地址之外的正文。只作参考，没有门槛。",
    "",
    ...table(["最小", "中位数", "第 90 百分位", "最大"], [[n.englishShare.min, n.englishShare.median, n.englishShare.p90, n.englishShare.max].map((value) => `${(value * 100).toFixed(1)}%`)]),
    ...groundingSection(report),
    "## 问题清单",
    "",
    `每类最多列出 ${LIST_CAP} 条，完整清单在 findings.json。`,
    "",
    "### 日报条目：公式问题",
    "",
    ...capped(DAILY_HEADER, dailyRows(daily.results, (finding) => finding.kind === "math" && isDefect(finding))),
    "### 日报条目：空话",
    "",
    ...capped(DAILY_HEADER, dailyRows(daily.results, (finding) => finding.kind === "generic-phrase")),
    "### 日报条目：含“原文未说明”（仅供参考）",
    "",
    ...capped(DAILY_HEADER, dailyRows(daily.results, (finding) => finding.kind === "not-stated")),
    "### 详细笔记：章节结构",
    "",
    ...capped(NOTE_HEADER, noteRows(notes.results, (finding) => finding.kind === "structure")),
    "### 详细笔记：公式问题",
    "",
    ...capped(NOTE_HEADER, noteRows(notes.results, (finding) => finding.kind === "math" && isDefect(finding))),
    "### 详细笔记：空话",
    "",
    ...capped(NOTE_HEADER, noteRows(notes.results, (finding) => finding.kind === "generic-phrase")),
    ...(report.grounding.run
      ? [
        "### 疑似无出处数字（需人工核对，可能是规范化遗漏或原文版本不同）",
        "",
        "日报条目：",
        "",
        ...capped(["日期", "编号", "字段", "数字", "片段"], groundingRows(report.grounding.daily.results, (place) => FIELD_LABELS[place as SummaryFieldKey] ?? place)),
        "详细笔记：",
        "",
        ...capped(["日期", "编号", "章节", "数字", "片段"], groundingRows(report.grounding.notes.results, (place) => place)),
      ]
      : []),
    "### core 规则误报（渲染正常）",
    "",
    "核心代码的公式规则把 `$` 后面紧跟数字当作货币符号，所以 `$6:1$`、`$8:1:1$` 这样以数字开头的比值会连带报“未闭合”“嵌套”；半开区间 `(0, 0.15]`、`[0,1)` 则被报括号不成对。这些在 Obsidian 里渲染正常，不当作缺陷，也不计入上面的数字，只在这里列出。",
    "",
    "日报条目：",
    "",
    ...capped(DAILY_HEADER, dailyRows(daily.results, (finding) => finding.kind === "math" && isFalsePositive(finding))),
    "详细笔记：",
    "",
    ...capped(NOTE_HEADER, noteRows(notes.results, (finding) => finding.kind === "math" && isFalsePositive(finding))),
  ];
  return `${lines.join("\n")}\n`;
}
