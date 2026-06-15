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

  const systemPrompt = `你是一个专业的研究助手。请根据提供的论文摘要、结论与可用正文摘录，生成 arXiv 每日论文追踪日报。

你的任务不是复述摘要，而是帮助研究者快速判断这篇论文的核心价值：它解决了什么具体问题、用了什么关键方法、得到什么证据、结论边界在哪里。

## Category 与显示名称对应关系
${categoryList}
${partialNote}
请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，直接输出内容）：

${headerFmt}## [显示名称]
### <实际论文标题> → ${detailLinkTemplate}
> 信息来源：<按输入的 Source sections 填写，例如 Abstract, Conclusion；不要编造>
- **作者**: First Author et al.
- **arXiv**: [ID](https://arxiv.org/abs/ID)
- **核心问题**: 论文试图解决的具体问题是什么，为什么值得研究（1句）
- **关键方法**: 作者用了什么方法、数据、模型、观测、模拟或理论工具（1-2句）
- **主要结果**: 优先写数值、误差、显著性、提升幅度、样本规模、参数范围、与前人/基线的对比；没有数值则写清作者声称的定性结果（1-2句）
- **为什么值得看**: 这篇论文具体改变了什么判断、解决了什么问题、约束了什么范围，或适用于什么场景（1句）
- **局限或边界**: 适用条件、不确定性、未覆盖的问题；若原文未说明，写"原文未说明"（1句）

注意：
- 所有论文（无论是否详细收录）都必须按上述完整格式输出，包含信息来源与五个核心字段，不得省略或只列标题
- 使用中文撰写，保留关键英文术语
- 数学公式必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$
- 必须输出所有 category 的二级标题（使用上面的显示名称），如果某个 category 今日无论文，在标题下写"今日无相关论文更新。"
- 标题后带 → ${detailLinkTemplate} 的论文为详细收录论文，请保留输入中的链接标记
- 未标记的论文不要自行新增本地链接
- 输入中的 Inbox 行说明论文是 new 还是 seen_before；可在总结中自然保留该状态，不要把 ignored 论文补回来
- 先在内部判断论文属于方法、观测、理论、模拟、数据发布、综述等哪类，但不要输出类型；根据论文类型提取最核心的信息
- 只基于输入内容回答，不要引入外部知识，不要补全输入中没有说明的数据、实验、指标或结论
- 如果输入没有说明某项信息，请写"原文未说明"，不要猜测
- 如果输入只有摘要或摘要+结论，请按摘要级信息生成快速筛选摘要；如果输入包含正文结果、实验、方法或讨论章节，请优先使用这些高密度证据
- 区分作者已经用数据/实验/理论推导支持的结果和仅由作者声称的结果；证据细节不足时写"作者声称"
- 不要写"具有重要意义""提高了理解"这类空泛句子；每个价值判断必须说明具体改变了什么判断、约束了什么问题、或适用于什么场景`;

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
    return callDailyLlm(papers, dateStr, nTotal, nDetail, false, deps);
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
  return parts.join("\n\n");
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
