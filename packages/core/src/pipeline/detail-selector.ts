import type { LlmClient } from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import injectionGuard from "../prompts/injection-guard.md";
import detailSelectorSystemTemplate from "../prompts/detail-selector.system.md";
import { renderPrompt } from "../prompts/render";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { Logger } from "../services/logger";
import type { Topic } from "../settings/types";
import type { DailyPaperWithContent } from "./summarizer";
import { escapePaperDataFence } from "./prompt-safety";

export const DETAIL_SELECTOR_FULL_TEXT_CHAR_LIMIT = 12_000;
export const DETAIL_SELECTOR_REASON_CHAR_LIMIT = 500;

export interface DetailSelectionPolicy {
  normalThreshold: number;
  exceptionalThreshold: number;
  softLimit: number;
}

export interface DetailSelectorDeps {
  llm: LlmClient;
  logger: Logger;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
}

export interface DetailSelectionEvaluation {
  id: string;
  score: number;
  reason: string;
}

export interface DetailSelectionResult {
  /** Evaluations for every eligible candidate, in candidate input order. */
  evaluations: DetailSelectionEvaluation[];
  /** Deterministically selected evaluations, ordered by score descending then ID. */
  selected: DetailSelectionEvaluation[];
}

interface DetailCandidate {
  paper: DailyPaperWithContent;
  topic: Topic;
}

const EMPTY_RESULT: DetailSelectionResult = {
  evaluations: [],
  selected: [],
};

export function buildDetailSelectorSystemPrompt(): string {
  return renderPrompt(detailSelectorSystemTemplate, { injectionGuard });
}

export async function selectDetailPapers(
  papers: readonly DailyPaperWithContent[],
  topics: readonly Topic[],
  policy: DetailSelectionPolicy,
  deps: DetailSelectorDeps,
): Promise<DetailSelectionResult> {
  throwIfCancelled(deps.signal);

  if (!isValidPolicy(policy)) {
    deps.logger.warn("detail-selector: invalid selection policy; selecting no papers");
    return emptyResult();
  }

  const topicByTag = new Map(topics.map((topic) => [topic.tag, topic] as const));
  const candidates = papers
    .map((paper): DetailCandidate | null => {
      const topic = topicByTag.get(paper.category);
      if (
        !topic?.detail ||
        !paper.fullSections?.trim() ||
        Boolean(paper.paperPath?.trim())
      ) {
        return null;
      }
      return { paper, topic };
    })
    .filter((candidate): candidate is DetailCandidate => candidate !== null);

  if (candidates.length === 0) {
    deps.logger.info("detail-selector: no eligible candidates; skipping LLM call");
    return emptyResult();
  }

  const candidateIds = new Set<string>();
  for (const { paper } of candidates) {
    if (candidateIds.has(paper.id)) {
      deps.logger.warn(
        `detail-selector: duplicate candidate id ${paper.id}; selecting no papers`,
      );
      return emptyResult();
    }
    candidateIds.add(paper.id);
  }

  const userContent = buildCandidateContent(candidates);
  deps.logger.info(
    `detail-selector: sending ${candidates.length} candidates to LLM for scoring`,
  );

  let raw: string;
  try {
    raw = await deps.llm.call(
      [
        { role: "system", content: buildDetailSelectorSystemPrompt() },
        { role: "user", content: userContent },
      ],
      { temperature: 0, signal: deps.signal, onMetrics: deps.onMetrics },
    );
  } catch (error) {
    if (isCancellationError(error)) throw error;
    deps.logger.warn("detail-selector: LLM call failed; selecting no papers", error);
    return emptyResult();
  }
  throwIfCancelled(deps.signal);

  const evaluations = parseEvaluations(raw, candidateIds);
  if (!evaluations.ok) {
    deps.logger.warn(
      `detail-selector: invalid LLM response (${evaluations.reason}); selecting no papers`,
    );
    return emptyResult();
  }

  const byId = new Map(evaluations.value.map((evaluation) => [evaluation.id, evaluation]));
  const inputOrdered = candidates.map(({ paper }) => byId.get(paper.id)!);
  const eligible = inputOrdered
    .filter((evaluation) => evaluation.score >= policy.normalThreshold)
    .sort(compareEvaluations);
  const selected = eligible.filter(
    (evaluation, index) =>
      index < policy.softLimit || evaluation.score >= policy.exceptionalThreshold,
  );

  deps.logger.info(
    `detail-selector: selected ${selected.length}/${candidates.length} candidates`,
  );
  return { evaluations: inputOrdered, selected };
}

function buildCandidateContent(candidates: readonly DetailCandidate[]): string {
  const blocks = candidates.map(({ paper, topic }) => {
    const fullText = paper.fullSections!.trim().slice(0, DETAIL_SELECTOR_FULL_TEXT_CHAR_LIMIT);
    return [
      "---",
      `ID: ${escapePaperDataFence(paper.id)}`,
      `Title: ${escapePaperDataFence(paper.title)}`,
      `Abstract: ${escapePaperDataFence(paper.abstract)}`,
      `Topic: ${escapePaperDataFence(topic.tag)}`,
      `Topic description: ${escapePaperDataFence(topic.description)}`,
      `Key full-text excerpt:\n${escapePaperDataFence(fullText)}`,
    ].join("\n");
  });
  return `Score every candidate below.\n\n<paper_data>\n${blocks.join("\n")}\n</paper_data>`;
}

function parseEvaluations(
  raw: string,
  candidateIds: ReadonlySet<string>,
): { ok: true; value: DetailSelectionEvaluation[] } | { ok: false; reason: string } {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return { ok: false, reason: "response is not strict JSON" };
  }

  if (!isPlainObject(value) || !hasExactKeys(value, ["papers"]) || !Array.isArray(value.papers)) {
    return { ok: false, reason: "root must be exactly {papers:[...]}" };
  }
  if (value.papers.length !== candidateIds.size) {
    return { ok: false, reason: "candidate record count mismatch" };
  }

  const seen = new Set<string>();
  const evaluations: DetailSelectionEvaluation[] = [];
  for (const record of value.papers) {
    if (!isPlainObject(record) || !hasExactKeys(record, ["id", "score", "reason"])) {
      return { ok: false, reason: "paper record has an invalid shape" };
    }
    if (typeof record.id !== "string" || !candidateIds.has(record.id)) {
      return { ok: false, reason: "paper record has an unknown id" };
    }
    if (seen.has(record.id)) {
      return { ok: false, reason: "paper record has a duplicate id" };
    }
    if (
      typeof record.score !== "number" ||
      !Number.isFinite(record.score) ||
      !Number.isInteger(record.score) ||
      record.score < 0 ||
      record.score > 100
    ) {
      return { ok: false, reason: `paper ${record.id} has an invalid score` };
    }
    if (
      typeof record.reason !== "string" ||
      record.reason.trim().length === 0 ||
      record.reason.length > DETAIL_SELECTOR_REASON_CHAR_LIMIT
    ) {
      return { ok: false, reason: `paper ${record.id} has an invalid reason` };
    }
    seen.add(record.id);
    evaluations.push({
      id: record.id,
      score: record.score,
      reason: record.reason.trim(),
    });
  }

  for (const id of candidateIds) {
    if (!seen.has(id)) return { ok: false, reason: `missing paper record for ${id}` };
  }
  return { ok: true, value: evaluations };
}

function isValidPolicy(policy: DetailSelectionPolicy): boolean {
  return (
    isThreshold(policy.normalThreshold) &&
    isThreshold(policy.exceptionalThreshold) &&
    policy.exceptionalThreshold >= policy.normalThreshold &&
    Number.isSafeInteger(policy.softLimit) &&
    policy.softLimit >= 0 &&
    policy.softLimit <= 20
  );
}

function isThreshold(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 100;
}

function compareEvaluations(
  left: DetailSelectionEvaluation,
  right: DetailSelectionEvaluation,
): number {
  return right.score - left.score || left.id.localeCompare(right.id);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index])
  );
}

function emptyResult(): DetailSelectionResult {
  return { ...EMPTY_RESULT, evaluations: [], selected: [] };
}
