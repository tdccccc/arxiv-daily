import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import type { ArxivSettings } from "../settings/types";
import type { PaperMeta } from "./arxiv-parser";

export interface FilteredPaper extends PaperMeta {
  category: string;
  isDetail: boolean;
}

export interface PaperFilterDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
}

export async function filterPapers(
  papers: PaperMeta[],
  deps: PaperFilterDeps,
): Promise<FilteredPaper[]> {
  const { llm, logger, arxivSettings } = deps;
  if (papers.length === 0) return [];

  const categories = Object.keys(arxivSettings.categoryDisplayMap ?? {});
  const categoryOptions = categories.length
    ? categories.join("|")
    : "photo-z|galaxy-cluster|ml|other";

  const papersText = papers
    .map(
      (p) =>
        `---\nID: ${p.id}\nTitle: ${p.title}\nAbstract: ${p.abstract}\n`,
    )
    .join("");

  const systemPrompt = `你是一位研究者的助手。请根据研究兴趣，从下方论文列表中筛选出相关论文。

## 研究兴趣
${arxivSettings.researchInterests}

## 详细收录标准
以下类型的论文应标记 detail: true（会生成详细报告）：
${arxivSettings.detailCriteria}

## 输出格式
请只输出一个 JSON 对象，不要输出任何其他内容：
{"papers": [
  {"id": "YYMM.NNNNN", "category": "${categoryOptions}", "detail": true/false},
  ...
]}

规则：
- 只收录与研究兴趣相关的论文，不相关的直接忽略
- category 从 ${categoryOptions} 中选择最匹配的一个
- detail 判定要从严：只有核心主题直接匹配详细收录标准时才设为 true
- 宁可漏选 detail 也不要错选——不确定时设为 false，日报已包含所有相关论文的总结
- 如果没有任何相关论文，返回 {"papers": []}`;

  const userContent = `以下是今日 arXiv ${arxivSettings.category} 的所有新论文：\n\n${papersText}`;

  let raw: string;
  try {
    raw = await llm.call(
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
      { temperature: 0 },
    );
  } catch (e) {
    logger.error("paper-filter: LLM call failed", e);
    return [];
  }

  let parsed: {
    papers?: Array<{ id?: string; category?: string; detail?: boolean }>;
  };
  try {
    parsed = JSON.parse(raw);
  } catch {
    const m = /\{[\s\S]*\}/.exec(raw);
    if (!m) {
      logger.error("paper-filter: no JSON in LLM response", raw.slice(0, 200));
      return [];
    }
    try {
      parsed = JSON.parse(m[0]);
    } catch (e) {
      logger.error("paper-filter: JSON parse failed", e);
      return [];
    }
  }

  const idMap = new Map(papers.map((p) => [p.id, p] as const));
  const out: FilteredPaper[] = [];
  for (const item of parsed.papers ?? []) {
    const id = item.id ?? "";
    const meta = idMap.get(id);
    if (!meta) {
      logger.warn(`paper-filter: unknown id ${id}, skipping`);
      continue;
    }
    const category = item.category ?? "other";
    let isDetail = Boolean(item.detail);
    if (isDetail && !(arxivSettings.detailCategories ?? []).includes(category)) {
      isDetail = false;
      logger.info(`paper-filter: demote detail for ${id} (category=${category})`);
    }
    out.push({ ...meta, category, isDetail });
  }
  logger.info(`paper-filter: kept ${out.length}/${papers.length} papers`);
  return out;
}
