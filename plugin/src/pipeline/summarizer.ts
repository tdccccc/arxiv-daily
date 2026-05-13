import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import type { ArxivSettings, AdvancedSettings } from "../settings/types";
import type { FilteredPaper } from "./paper-filter";

export interface DailyPaperWithContent extends FilteredPaper {
  abstractConclusion: string;
  fullSections: string | null;
}

export interface SummarizerDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
  advanced: AdvancedSettings;
  llmTemperature: number;
}

function buildPaperBlock(p: DailyPaperWithContent): string {
  const detailMark = p.isDetail ? ` → [[${p.id}]]` : "";
  return (
    `=== Paper: ${p.id} [category: ${p.category}]${detailMark} ===\n` +
    `Title: ${p.title}\n` +
    `Authors: ${p.authors}\n` +
    `${p.abstractConclusion}\n\n`
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
  const partialNote = isPartial
    ? `\n注意：这是分批处理的一部分（本批 ${papers.length} 篇），请只为本批论文生成总结，不要输出标题头和统计行。\n`
    : "";
  const headerFmt = isPartial
    ? ""
    : `# arXiv ${arxivSettings.category} 每日追踪 ${dateStr}\n` +
      `共 ${nTotal} 篇相关论文，其中 ${nDetail} 篇详细收录。\n\n`;

  const systemPrompt = `你是一个专业的研究助手。请根据提供的论文摘要与结论，生成 arXiv 每日论文追踪日报。

## Category 与显示名称对应关系
${categoryList}
${partialNote}
请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，直接输出内容）：

${headerFmt}## [显示名称]
### <实际论文标题> → [[YYMM.NNNNN]]
- **作者**: First Author et al.
- **arXiv**: [ID](https://arxiv.org/abs/ID)
- **一句话总结**: 用一句话概括本文做了什么
- **数据**: 使用了什么数据集/样本/巡天（2-4句）
- **方法**: 采用了什么方法或模型，关键技术细节是什么（2-4句）
- **主要结果**: 核心发现是什么，给出关键定量数值（精度、误差、提升幅度等），与已有工作的对比（2-4句）
- **意义**: 对领域的贡献或启示，局限性，未来展望（1-2句）

注意：
- 所有论文（无论是否详细收录）都必须按上述完整格式输出，包含五个字段，不得省略或只列标题
- 使用中文撰写，保留关键英文术语
- 数学公式必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$
- 必须输出所有 category 的二级标题（使用上面的显示名称），如果某个 category 今日无论文，在标题下写"今日无相关论文更新。"
- 标题后带 → [[YYMM.NNNNN]] 的论文为详细收录论文，请保留此标记
- 未标记的论文不要加 [[]] 链接
- 重点提取定量结果，避免泛泛而谈`;

  return llm.call(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: `以下是今日筛选出的论文：\n\n${papersInfo}` },
    ],
    { temperature: llmTemperature },
  );
}

export async function summarizeDaily(
  papers: DailyPaperWithContent[],
  dateStr: string,
  deps: SummarizerDeps,
): Promise<string> {
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
    `# arXiv ${deps.arxivSettings.category} 每日追踪 ${dateStr}\n` +
    `共 ${nTotal} 篇相关论文，其中 ${nDetail} 篇详细收录。\n`;
  const parts: string[] = [header];
  for (let i = 0; i < batches.length; i++) {
    deps.logger.info(`summarizeDaily: batch ${i + 1}/${batches.length}`);
    parts.push(
      await callDailyLlm(batches[i], dateStr, nTotal, nDetail, true, deps),
    );
  }
  return parts.join("\n\n");
}

export async function summarizePaperDetail(
  paper: DailyPaperWithContent,
  deps: SummarizerDeps,
): Promise<string> {
  if (!paper.fullSections) {
    throw new Error(
      `summarizePaperDetail: paper ${paper.id} has no full sections`,
    );
  }

  const systemPrompt = `你是一个专业的研究助手。请根据提供的论文各章节内容，生成一篇详细的中文论文总结。

请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，不要输出 YAML frontmatter，直接从 # 标题开始）：

# ${paper.title}

- **arXiv**: [${paper.id}](https://arxiv.org/abs/${paper.id})

## 背景与动机
（研究背景、前人工作、本文动机）

## 数据
（使用了什么数据集、样本大小、数据处理方法）

## 方法
（核心方法/模型/算法的详细描述）

## 结果
（主要发现、定量结果、与前人工作的比较）

## 讨论
（结果的意义、局限性、与其他工作的对比）

## 结论
（核心结论、未来展望）

注意：
- 使用中文撰写
- 保留关键英文术语（如专有名词、物理量）
- 数学公式、物理量和符号必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$
- 尽可能包含定量结果（数值、误差）
- 如果某个章节的信息不足，可以简要说明`;

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
    { temperature: deps.llmTemperature },
  );
}
