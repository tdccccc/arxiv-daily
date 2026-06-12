import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { ArxivSettings, Topic } from "../settings/types";
import { formatArxivCategories } from "../settings/categories";
import type { PaperMeta } from "./arxiv-parser";

export interface FilteredPaper extends PaperMeta {
  category: string;
  isDetail: boolean;
}

export interface PaperFilterDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
  signal?: AbortSignal;
}

export async function filterPapers(
  papers: PaperMeta[],
  deps: PaperFilterDeps,
): Promise<FilteredPaper[]> {
  const { llm, logger, arxivSettings } = deps;
  throwIfCancelled(deps.signal);
  if (papers.length === 0) return [];

  const topics: Topic[] = arxivSettings.topics ?? [];
  if (topics.length === 0) {
    logger.warn("paper-filter: no topics configured, skipping LLM call");
    return [];
  }

  const topicLines = topics
    .map((t) => `- ${t.tag}${t.detail ? " [DETAIL]" : ""}: ${t.description}`)
    .join("\n");
  const tagOptions = topics.map((t) => t.tag).join("|") + "|skip";
  const validTags = new Set(topics.map((t) => t.tag));
  const topicByTag = new Map(topics.map((t) => [t.tag, t] as const));

  const papersText = papers
    .map((p) => `---\nID: ${p.id}\nTitle: ${p.title}\nAbstract: ${p.abstract}\n`)
    .join("");

  const systemPrompt = `你是一位研究者的助手。请根据下方主题列表，为每篇论文选择最匹配的主题。

## 主题列表
${topicLines}

## 输出格式
请只输出一个 JSON 对象，不要输出任何其他内容：
{"papers": [
  {"id": "YYMM.NNNNN", "category": "${tagOptions}", "detail": true/false},
  ...
]}

规则：
- category 选择最匹配的主题 tag；若与所有主题都不相关，返回 "skip"
- detail 仅在带 [DETAIL] 标记的主题上有意义；当且仅当该论文是该主题的核心贡献时设为 true，其余设为 false
- detail 判定从严：宁可漏选也不要错选——不确定时设为 false
- 如果没有任何相关论文，返回 {"papers": []}`;

  const userContent = `以下是今日 arXiv ${formatArxivCategories(arxivSettings)} 的所有新论文：\n\n${papersText}`;

  let raw: string;
  try {
    raw = await llm.call(
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
      { temperature: 0, signal: deps.signal },
    );
  } catch (e) {
    if (isCancellationError(e)) throw e;
    logger.error("paper-filter: LLM call failed", e);
    return [];
  }
  throwIfCancelled(deps.signal);

  let parsed: { papers?: Array<{ id?: string; category?: string; detail?: boolean }> };
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
    const category = item.category ?? "";
    if (category === "skip") continue;
    if (!validTags.has(category)) {
      logger.info(`paper-filter: unknown tag '${category}' for ${id}, dropping`);
      continue;
    }
    const topic = topicByTag.get(category)!;
    let isDetail = Boolean(item.detail);
    if (isDetail && !topic.detail) {
      isDetail = false;
      logger.info(`paper-filter: demote detail for ${id} (topic ${category} has detail=false)`);
    }
    out.push({ ...meta, category, isDetail });
  }
  logger.info(`paper-filter: kept ${out.length}/${papers.length} papers`);
  return out;
}
