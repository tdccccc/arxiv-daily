import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { ArxivSettings, Topic } from "../settings/types";
import { formatArxivCategories } from "../settings/categories";
import type { PaperMeta } from "./arxiv-parser";
import { renderPrompt } from "../prompts/render";
import filterSystemTemplate from "../prompts/paper-filter.system.md";
import injectionGuard from "../prompts/injection-guard.md";

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

  const systemPrompt = renderPrompt(filterSystemTemplate, {
    topicLines,
    tagOptions,
    injectionGuard,
  });

  const userContent = `以下是今日 arXiv ${formatArxivCategories(arxivSettings)} 的所有新论文：\n\n<paper_data>\n${papersText}</paper_data>`;

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
