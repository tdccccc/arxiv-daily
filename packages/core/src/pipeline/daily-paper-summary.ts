import type { LlmClient } from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import dailyPaperSummaryTemplateEn from "../prompts/daily-paper-summary.en.system.md";
import dailyPaperSummaryTemplate from "../prompts/daily-paper-summary.system.md";
import injectionGuard from "../prompts/injection-guard.md";
import { renderPrompt } from "../prompts/render";
import { throwIfCancelled } from "../services/cancellation";
import { normalizeSummaryLanguage } from "../settings/summary-language";
import type { SummaryLanguage } from "../settings/types";
import type { StructuredPaperSummary } from "./daily-summary-assembler";
import { escapePaperDataFence } from "./prompt-safety";

export interface DailyPaperSummaryInput {
  id: string;
  title: string;
  authors: string;
  abstractConclusion: string;
  fullSections: string | null;
}

export interface DailyPaperSummaryDeps {
  llm: LlmClient;
  summaryLanguage?: SummaryLanguage;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
}

const SUMMARY_KEYS = [
  "id",
  "coreProblem",
  "keyMethod",
  "mainResult",
  "whyRelevant",
  "limitations",
] as const satisfies ReadonlyArray<keyof StructuredPaperSummary>;

const SEMANTIC_KEYS = SUMMARY_KEYS.filter(
  (key): key is Exclude<(typeof SUMMARY_KEYS)[number], "id"> => key !== "id",
);

export async function summarizeDailyPaper(
  paper: DailyPaperSummaryInput,
  deps: DailyPaperSummaryDeps,
): Promise<StructuredPaperSummary> {
  throwIfCancelled(deps.signal);
  const language = normalizeSummaryLanguage(deps.summaryLanguage);
  const systemPrompt = renderPrompt(
    language === "en" ? dailyPaperSummaryTemplateEn : dailyPaperSummaryTemplate,
    { injectionGuard },
  );
  const raw = await deps.llm.call(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: buildPaperData(paper, language) },
    ],
    { temperature: 0, signal: deps.signal, onMetrics: deps.onMetrics },
  );
  throwIfCancelled(deps.signal);
  return parseDailyPaperSummary(raw, paper.id);
}

export function derivePaperSourceSections(
  paper: Pick<DailyPaperSummaryInput, "abstractConclusion" | "fullSections">,
): string {
  const titles = [
    ...extractSectionTitles(paper.abstractConclusion),
    ...extractSectionTitles(paper.fullSections),
  ].filter((title, index, values) => values.indexOf(title) === index);
  if (titles.length > 0) return titles.join(", ");
  if (paper.abstractConclusion.startsWith("[获取失败]")) return "获取失败";
  return "正文摘录";
}

function buildPaperData(
  paper: DailyPaperSummaryInput,
  language: SummaryLanguage,
): string {
  const content = [paper.abstractConclusion.trim()];
  if (paper.fullSections?.trim()) {
    content.push(
      language === "en"
        ? `Full-text excerpts:\n${paper.fullSections.trim()}`
        : `正文摘录：\n${paper.fullSections.trim()}`,
    );
  }
  const sourceSections = derivePaperSourceSections(paper);
  return (
    `<paper_data>\n` +
    `ID: ${escapePaperDataFence(paper.id)}\n` +
    `Title: ${escapePaperDataFence(paper.title)}\n` +
    `Authors: ${escapePaperDataFence(paper.authors)}\n` +
    `Source sections: ${escapePaperDataFence(sourceSections)}\n\n` +
    `${escapePaperDataFence(content.filter(Boolean).join("\n\n"))}\n` +
    `</paper_data>`
  );
}

function parseDailyPaperSummary(
  raw: string,
  expectedId: string,
): StructuredPaperSummary {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error(
      `summarizeDailyPaper: response for ${expectedId} is not strict JSON`,
    );
  }
  if (!isPlainObject(value)) {
    throw new Error(
      `summarizeDailyPaper: response for ${expectedId} must be a plain object`,
    );
  }
  if (!hasExactKeys(value, SUMMARY_KEYS)) {
    throw new Error(
      `summarizeDailyPaper: response for ${expectedId} must contain exactly: ${SUMMARY_KEYS.join(", ")}`,
    );
  }
  if (typeof value.id !== "string") {
    throw new Error(`summarizeDailyPaper: id for ${expectedId} must be a string`);
  }
  if (value.id !== expectedId) {
    throw new Error(
      `summarizeDailyPaper: response ID ${value.id} does not match ${expectedId}`,
    );
  }

  const summary = { id: expectedId } as StructuredPaperSummary;
  for (const key of SEMANTIC_KEYS) {
    const field = value[key];
    if (typeof field !== "string") {
      throw new Error(
        `summarizeDailyPaper: ${key} for ${expectedId} must be a string`,
      );
    }
    const trimmed = field.trim();
    if (!trimmed) {
      throw new Error(
        `summarizeDailyPaper: ${key} for ${expectedId} must be non-empty`,
      );
    }
    summary[key] = trimmed;
  }
  return summary;
}

function extractSectionTitles(markdown: string | null | undefined): string[] {
  if (!markdown) return [];
  const titles: string[] = [];
  const heading = /^##\s+(.+)$/gm;
  let match: RegExpExecArray | null;
  while ((match = heading.exec(markdown)) !== null) {
    const title = match[1]?.trim() ?? "";
    if (title && !titles.includes(title)) titles.push(title);
  }
  return titles;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index])
  );
}
