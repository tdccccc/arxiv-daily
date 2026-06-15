import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import { throwIfCancelled } from "../services/cancellation";
import type {
  AdvancedSettings,
  ArxivSettings,
  LinkStyle,
} from "../settings/types";
import { formatArxivCategories } from "../settings/categories";
import type { PaperIndexEntry, PaperStatus } from "../services/paper-index";
import type { FilteredPaper } from "./paper-filter";
import { renderPrompt } from "../prompts/render";
import dailySystemTemplate from "../prompts/daily-summary.system.md";

export interface DailyPaperWithContent extends FilteredPaper {
  abstractConclusion: string;
  fullSections: string | null;
  inboxStatus?: PaperStatus;
  seenBefore?: boolean;
  paperPath?: string | null;
  detailLink?: string;
  indexEntry?: PaperIndexEntry;
}

export interface SummarizerDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
  advanced: AdvancedSettings;
  llmTemperature: number;
  linkStyle?: LinkStyle;
  signal?: AbortSignal;
}

function extractSectionTitles(markdown: string | null | undefined): string[] {
  if (!markdown) return [];
  const titles: string[] = [];
  const re = /^##\s+(.+)$/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(markdown)) !== null) {
    const title = m[1].trim();
    if (title && !titles.includes(title)) titles.push(title);
  }
  return titles;
}

function paperSourceSections(p: DailyPaperWithContent): string {
  const titles = [
    ...extractSectionTitles(p.abstractConclusion),
    ...extractSectionTitles(p.fullSections),
  ].filter((title, i, arr) => arr.indexOf(title) === i);
  if (titles.length) return titles.join(", ");
  if (p.abstractConclusion.startsWith("[获取失败]")) return "获取失败";
  return "正文摘录";
}

function buildDailyContent(p: DailyPaperWithContent): string {
  const parts = [p.abstractConclusion.trim()].filter(Boolean);
  if (p.fullSections?.trim()) {
    parts.push(`Full-text excerpts:\n${p.fullSections.trim()}`);
  }
  return parts.join("\n\n");
}

function buildPaperBlock(p: DailyPaperWithContent): string {
  const detailMark =
    p.isDetail || p.paperPath ? ` → ${p.detailLink ?? `[[${p.id}]]`}` : "";
  const inboxLine =
    `Inbox: ${p.seenBefore ? "seen_before" : "new"}, ` +
    `status: ${p.inboxStatus ?? "inbox"}, ` +
    `note: ${detailMark ? "local_note" : "arxiv_only"}\n`;
  return (
    `=== Paper: ${p.id} [category: ${p.category}]${detailMark} ===\n` +
    `Title: ${p.title}\n` +
    `Authors: ${p.authors}\n` +
    `Source sections: ${paperSourceSections(p)}\n` +
    inboxLine +
    `${buildDailyContent(p)}\n\n`
  );
}

function splitBatches(
  papers: DailyPaperWithContent[],
  charLimit: number,
): DailyPaperWithContent[][] {
  const batches: DailyPaperWithContent[][] = [];
  let cur: DailyPaperWithContent[] = [];
  let size = 0;
  for (const p of papers) {
    const bs = buildPaperBlock(p).length;
    if (cur.length && size + bs > charLimit) {
      batches.push(cur);
      cur = [];
      size = 0;
    }
    cur.push(p);
    size += bs;
  }
  if (cur.length) batches.push(cur);
  return batches;
}

async function callDailyLlm(
  papers: DailyPaperWithContent[],
  dateStr: string,
  nTotal: number,
  nDetail: number,
  isPartial: boolean,
  deps: SummarizerDeps,
): Promise<string> {
  const { llm, arxivSettings, llmTemperature } = deps;
  const categoryList = arxivSettings.topics
    .map((t) => `- ${t.tag} → ${t.name}`)
    .join("\n");
  const papersInfo = papers.map(buildPaperBlock).join("");
  const detailLinkTemplate =
    deps.linkStyle === "relative"
      ? `[YYMM.NNNNN](../papers/YYMM.NNNNN.md)`
      : `[[YYMM.NNNNN]]`;
  const partialNote = isPartial
    ? `\n注意：这是分批处理的一部分（本批 ${papers.length} 篇），请只为本批论文生成总结，不要输出标题头和统计行。\n`
    : "";
  const headerFmt = isPartial
    ? ""
    : `# arXiv ${formatArxivCategories(arxivSettings)} 每日追踪 ${dateStr}\n` +
      `共 ${nTotal} 篇相关论文，其中 ${nDetail} 篇详细收录。\n\n`;

  const systemPrompt = renderPrompt(dailySystemTemplate, {
    categoryList,
    partialNote,
    headerFmt,
    detailLinkTemplate,
  });

  return llm.call(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: `以下是今日筛选出的论文：\n\n${papersInfo}` },
    ],
    { temperature: llmTemperature, signal: deps.signal },
  );
}

export async function summarizeDaily(
  papers: DailyPaperWithContent[],
  dateStr: string,
  deps: SummarizerDeps,
): Promise<string> {
  throwIfCancelled(deps.signal);
  const nTotal = papers.length;
  const nDetail = papers.filter((p) => p.isDetail).length;
  const totalChars = papers.reduce((s, p) => s + buildPaperBlock(p).length, 0);
  deps.logger.info(
    `summarizeDaily: ${totalChars} chars (limit ${deps.advanced.dailyCharLimit})`,
  );

  if (totalChars <= deps.advanced.dailyCharLimit) {
    const summary = await callDailyLlm(
      papers,
      dateStr,
      nTotal,
      nDetail,
      false,
      deps,
    );
    return normalizeDailySummary(summary, papers, deps.arxivSettings);
  }

  const batches = splitBatches(papers, deps.advanced.dailyCharLimit);
  deps.logger.info(
    `summarizeDaily: batching into ${batches.length} (${batches.map((b) => b.length).join(",")})`,
  );
  const header =
    `# arXiv ${formatArxivCategories(deps.arxivSettings)} 每日追踪 ${dateStr}\n` +
    `共 ${nTotal} 篇相关论文，其中 ${nDetail} 篇详细收录。\n`;
  const parts: string[] = [header];
  for (let i = 0; i < batches.length; i++) {
    throwIfCancelled(deps.signal);
    deps.logger.info(`summarizeDaily: batch ${i + 1}/${batches.length}`);
    parts.push(
      await callDailyLlm(batches[i], dateStr, nTotal, nDetail, true, deps),
    );
  }
  throwIfCancelled(deps.signal);
  return normalizeDailySummary(parts.join("\n\n"), papers, deps.arxivSettings);
}

function normalizeDailySummary(
  markdown: string,
  papers: DailyPaperWithContent[],
  arxivSettings: ArxivSettings,
): string {
  return mergeDuplicateCategorySections(
    canonicalizeDetailHeadingLinks(markdown, papers),
    arxivSettings.topics.map((topic) => topic.name),
  );
}

function canonicalizeDetailHeadingLinks(
  markdown: string,
  papers: DailyPaperWithContent[],
): string {
  const detailLinks = new Map(
    papers
      .filter((paper) => paper.isDetail || paper.paperPath)
      .map((paper) => [paper.id, paper.detailLink ?? `[[${paper.id}]]`]),
  );
  return markdown
    .split("\n")
    .map((line) => canonicalizeDetailHeadingLink(line, detailLinks))
    .join("\n");
}

function canonicalizeDetailHeadingLink(
  line: string,
  detailLinks: Map<string, string>,
): string {
  if (!line.startsWith("### ")) return line;
  const arrow = line.lastIndexOf(" → ");
  if (arrow === -1) return line;
  const suffix = line.slice(arrow + 3).trim();
  const id = /\b\d{4}\.\d{4,5}\b/.exec(suffix)?.[0];
  const looksLikeLink =
    suffix.includes("[[") || /\[[^\]]+\]\([^)]+\)/.test(suffix);
  if (!id) return looksLikeLink ? line.slice(0, arrow).trimEnd() : line;
  const detailLink = detailLinks.get(id);
  return detailLink
    ? `${line.slice(0, arrow).trimEnd()} → ${detailLink}`
    : line.slice(0, arrow).trimEnd();
}

function mergeDuplicateCategorySections(
  markdown: string,
  categoryNames: string[],
): string {
  const names = unique(categoryNames);
  const nameSet = new Set(names);
  if (nameSet.size === 0) return markdown;

  const lines = markdown.split("\n");
  const headingIndexes = lines
    .map((line, index) => (line.startsWith("## ") ? index : -1))
    .filter((index) => index !== -1);
  const counts = new Map<string, number>();
  for (const index of headingIndexes) {
    const name = lines[index].slice(3).trim();
    if (nameSet.has(name)) counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  if (![...counts.values()].some((count) => count > 1)) return markdown;

  const firstHeading = headingIndexes[0];
  const prelude =
    firstHeading === undefined ? lines : lines.slice(0, firstHeading);
  const blocksByName = new Map<string, string[][]>();
  const unknownBlocks: string[][] = [];

  for (let i = 0; i < headingIndexes.length; i++) {
    const start = headingIndexes[i];
    const end = headingIndexes[i + 1] ?? lines.length;
    const name = lines[start].slice(3).trim();
    const content = lines.slice(start + 1, end);
    if (nameSet.has(name)) {
      const blocks = blocksByName.get(name) ?? [];
      blocks.push(content);
      blocksByName.set(name, blocks);
    } else {
      unknownBlocks.push(lines.slice(start, end));
    }
  }

  const out = trimOuterBlank(prelude);
  for (const name of names) {
    const blocks = blocksByName.get(name);
    if (!blocks) continue;
    appendBlank(out);
    out.push(`## ${name}`);
    out.push(...mergeCategoryContentBlocks(blocks));
  }
  for (const block of unknownBlocks) {
    appendBlank(out);
    out.push(...trimOuterBlank(block));
  }
  return out.join("\n");
}

function mergeCategoryContentBlocks(blocks: string[][]): string[] {
  const cleaned = blocks.map(trimOuterBlank);
  const substantial = cleaned.filter((block) => !isNoUpdateBlock(block));
  const selected = substantial.length
    ? substantial
    : [cleaned.find((block) => block.length > 0) ?? ["今日无相关论文更新。"]];
  const out: string[] = [];
  for (const block of selected) {
    appendBlank(out);
    out.push(...block);
  }
  return out;
}

function isNoUpdateBlock(lines: string[]): boolean {
  const body = trimOuterBlank(lines);
  return (
    body.length === 0 ||
    body.every((line) => line.trim() === "今日无相关论文更新。")
  );
}

function trimOuterBlank(lines: string[]): string[] {
  let start = 0;
  let end = lines.length;
  while (start < end && lines[start].trim() === "") start++;
  while (end > start && lines[end - 1].trim() === "") end--;
  return lines.slice(start, end);
}

function appendBlank(lines: string[]): void {
  if (lines.length > 0 && lines[lines.length - 1] !== "") lines.push("");
}

function unique(values: string[]): string[] {
  return values.filter((value, index) => values.indexOf(value) === index);
}

export async function summarizePaperDetail(
  paper: DailyPaperWithContent,
  deps: SummarizerDeps,
): Promise<string> {
  throwIfCancelled(deps.signal);
  if (!paper.fullSections) {
    throw new Error(
      `summarizePaperDetail: paper ${paper.id} has no full sections`,
    );
  }

  const systemPrompt = `你是一个专业的研究助手。请根据提供的论文各章节内容，生成一篇详细的中文论文总结。

你的任务不是复述摘要，而是还原论文的贡献链条：研究问题 -> 方法设计 -> 关键证据 -> 主要结论 -> 适用边界。

请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，不要输出 YAML frontmatter，直接从 # 标题开始）：

# ${paper.title}

- **arXiv**: [${paper.id}](https://arxiv.org/abs/${paper.id})

## 研究问题
论文要解决的具体问题是什么？为什么这个问题值得研究？

## 方法设计
作者采用了什么核心方法、模型、数据、实验、观测、模拟或理论框架？

## 关键证据
作者用什么证据支持结论？优先保留数值、样本规模、误差、显著性、参数范围、基线对比或实验设置。

## 主要结论
论文最核心的发现或贡献是什么？区分作者已经证明的结果和作者提出的解释。

## 适用边界
结论在哪些条件下成立？有哪些限制、不确定性或未覆盖的问题？

## 一句话价值判断
用一句话说明这篇论文最值得关注的点，避免空泛评价。

注意：
- 使用中文撰写
- 保留关键英文术语（如专有名词、物理量）
- 数学公式、物理量和符号必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$
- 只基于输入内容回答，不要引入外部知识，不要补全输入中没有说明的数据、实验、指标或结论
- 如果某项信息在输入中没有说明，请写"原文未说明"
- 先在内部判断论文属于方法、观测、理论、模拟、数据发布、综述等哪类，但不要输出类型；根据论文类型组织重点
- 优先提取数值、误差、显著性、提升幅度、样本规模、参数范围、与前人/基线的对比
- 区分作者已经用数据/实验/理论推导支持的结果和作者提出的解释；证据细节不足时写"作者声称"
- 不要写"具有重要意义""提高了理解"这类空泛句子；每个价值判断必须说明具体改变了什么判断、约束了什么问题、或适用于什么场景`;

  const userContent =
    `论文 ID: ${paper.id}\n` +
    `标题: ${paper.title}\n` +
    `作者: ${paper.authors}\n\n` +
    `以下是论文各章节内容：\n\n${paper.fullSections}`;

  return deps.llm.call(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userContent },
    ],
    { temperature: deps.llmTemperature, signal: deps.signal },
  );
}
