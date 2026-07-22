import type { LlmClient } from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import rescueSystemTemplateEn from "../prompts/daily-summary-rescue.en.system.md";
import rescueSystemTemplate from "../prompts/daily-summary-rescue.system.md";
import { throwIfCancelled } from "../services/cancellation";
import type { Logger } from "../services/logger";
import { formatArxivCategories } from "../settings/categories";
import {
  dailyCountLine,
  dailyHeader,
  noCategoryPapersText,
  normalizeSummaryLanguage,
} from "../settings/summary-language";
import type { SummaryLanguage } from "../settings/types";
import type {
  DailySummaryAssemblyInput,
  DailySummaryAssemblyPaper,
  StructuredPaperSummary,
} from "./daily-summary-assembler";
import {
  extractFallbackPaperIds,
  extractPaperSummaries,
} from "./daily-summary-parser";
import {
  fallbackCountLine,
  normalizeMarkdownLine,
  renderFallbackBlock,
  renderPaperHeader,
  renderStructuredFields,
  safeDetailLink,
  trustedArxivUrl,
} from "./daily-summary-rendering";

export const DAILY_SUMMARY_RESCUE_MAX_ATTEMPTS = 3;
export const DAILY_SUMMARY_RESCUE_VALIDATION_ERROR_CODE =
  "ARXIV_DAILY_SUMMARY_RESCUE_VALIDATION" as const;
export const DAILY_SUMMARY_RESCUE_EXHAUSTED_ERROR_CODE =
  "ARXIV_DAILY_SUMMARY_RESCUE_EXHAUSTED" as const;

export class DailySummaryRescueValidationError extends Error {
  readonly name = "DailySummaryRescueValidationError";
  readonly code = DAILY_SUMMARY_RESCUE_VALIDATION_ERROR_CODE;
  readonly failure: string;

  constructor(failure: string) {
    super(`Daily summary rescue postflight failed: ${failure}`);
    this.failure = failure;
  }
}

export class DailySummaryRescueExhaustedError extends Error {
  readonly name = "DailySummaryRescueExhaustedError";
  readonly code = DAILY_SUMMARY_RESCUE_EXHAUSTED_ERROR_CODE;
  readonly attempts = DAILY_SUMMARY_RESCUE_MAX_ATTEMPTS;
  readonly cause: DailySummaryRescueValidationError;

  constructor(cause: DailySummaryRescueValidationError) {
    super(
      `Daily summary rescue failed postflight after ${DAILY_SUMMARY_RESCUE_MAX_ATTEMPTS} attempts`,
    );
    this.cause = cause;
  }
}

export function isDailySummaryRescueValidationError(
  error: unknown,
): error is DailySummaryRescueValidationError {
  if (error instanceof DailySummaryRescueValidationError) return true;
  return isErrorLike(error) &&
    error.name === "DailySummaryRescueValidationError" &&
    error.code === DAILY_SUMMARY_RESCUE_VALIDATION_ERROR_CODE &&
    typeof error.failure === "string";
}

export function isDailySummaryRescueExhaustedError(
  error: unknown,
): error is DailySummaryRescueExhaustedError {
  if (error instanceof DailySummaryRescueExhaustedError) return true;
  if (!isErrorLike(error) ||
      error.name !== "DailySummaryRescueExhaustedError" ||
      error.code !== DAILY_SUMMARY_RESCUE_EXHAUSTED_ERROR_CODE ||
      error.attempts !== DAILY_SUMMARY_RESCUE_MAX_ATTEMPTS) return false;
  return isDailySummaryRescueValidationError(error.cause);
}

export interface DailySummaryRescueDeps {
  llm: LlmClient;
  logger: Logger;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
}

type RescueContract = {
  version: 1;
  language: SummaryLanguage;
  date: string;
  categories: string;
  topics: Array<{ tag: string; name: string }>;
  counts: { total: number; detail: number; fallback: number };
  slots: Array<{
    paper: DailySummaryAssemblyPaper & { hasDetail: boolean; arxivLink: string };
    result:
      | { kind: "structured"; summary: StructuredPaperSummary }
      | {
          kind: "fallback";
          reasonCode: "validation-exhausted" | "transport-exhausted";
          attempts: number;
          originalAbstract: string;
        };
  }>;
};

export async function rescueDailySummary(
  input: DailySummaryAssemblyInput,
  deps: DailySummaryRescueDeps,
): Promise<string> {
  const contract = buildDailySummaryRescueContract(input);
  const requiredMarkdown = renderDailySummaryRescueMarkdown(contract);
  let previousError: DailySummaryRescueValidationError | undefined;

  for (let attempt = 1; attempt <= DAILY_SUMMARY_RESCUE_MAX_ATTEMPTS; attempt += 1) {
    throwIfCancelled(deps.signal);
    const correction = previousError
      ? `\n\nPrevious postflight failure (attempt ${attempt - 1}): ${previousError.failure}\nReturn requiredMarkdown exactly.`
      : "";
    deps.logger.warn(
      `summarizeDaily: rescue call attempt=${attempt} slots=${contract.slots.length}`,
    );
    const raw = await deps.llm.call(
      [
        {
          role: "system",
          content: contract.language === "en"
            ? rescueSystemTemplateEn
            : rescueSystemTemplate,
        },
        {
          role: "user",
          content: `<rescue_contract>\n${JSON.stringify(contract)}\n</rescue_contract>${correction}`,
        },
      ],
      { temperature: 0, signal: deps.signal, onMetrics: deps.onMetrics },
    );
    throwIfCancelled(deps.signal);

    try {
      validateDailySummaryRescueMarkdown(raw, requiredMarkdown, contract);
      deps.logger.info(
        `summarizeDaily: rescue postflight passed attempt=${attempt} slots=${contract.slots.length}`,
      );
      return raw;
    } catch (error) {
      if (!isDailySummaryRescueValidationError(error)) throw error;
      previousError = error;
      deps.logger.warn(
        `summarizeDaily: rescue postflight failed attempt=${attempt} failure=${error.failure}`,
      );
      if (attempt === DAILY_SUMMARY_RESCUE_MAX_ATTEMPTS) {
        throw new DailySummaryRescueExhaustedError(error);
      }
    }
  }
  throw new DailySummaryRescueExhaustedError(previousError!);
}

export function buildDailySummaryRescueContract(
  input: DailySummaryAssemblyInput,
): RescueContract {
  const language = normalizeSummaryLanguage(input.summaryLanguage);
  const display = (value: string): string =>
    protectRescueContractDelimiter(normalizeMarkdownLine(value));
  const slots = input.slots.map(({ paper, result }) => {
    const hasDetail = paper.isDetail || Boolean(paper.paperPath);
    const detailLink = safeDetailLink(
      paper.id,
      paper.paperPath,
      paper.detailLink,
      hasDetail,
    );
    return {
      paper: {
        ...paper,
        title: display(paper.title),
        authors: display(paper.authors),
        category: protectRescueContractDelimiter(paper.category),
        sourceSections: display(paper.sourceSections),
        paperPath: paper.paperPath
          ? protectRescueContractDelimiter(paper.paperPath)
          : paper.paperPath,
        detailLink: detailLink
          ? protectRescueContractDelimiter(detailLink)
          : undefined,
        hasDetail,
        arxivLink: trustedArxivUrl(paper.id),
      },
      result: result.kind === "structured"
        ? {
            kind: "structured" as const,
            summary: {
              id: result.summary.id,
              coreProblem: display(result.summary.coreProblem),
              keyMethod: display(result.summary.keyMethod),
              mainResult: display(result.summary.mainResult),
              whyRelevant: display(result.summary.whyRelevant),
              limitations: display(result.summary.limitations),
            },
          }
        : {
            kind: "fallback" as const,
            reasonCode: result.reasonCode,
            attempts: result.attempts,
            originalAbstract: display(result.originalAbstract),
          },
    };
  });
  return {
    version: 1,
    language,
    date: display(input.dateStr),
    categories: display(formatArxivCategories(input.arxivSettings)),
    topics: input.arxivSettings.topics.map(({ tag, name }) => ({
      tag: protectRescueContractDelimiter(tag),
      name: display(name),
    })),
    counts: {
      total: slots.length,
      detail: slots.filter(({ paper }) => paper.hasDetail).length,
      fallback: slots.filter(({ result }) => result.kind === "fallback").length,
    },
    slots,
  };
}

export function validateDailySummaryRescueMarkdown(
  markdown: string,
  requiredMarkdown: string,
  contract?: RescueContract,
): void {
  if (markdown === requiredMarkdown) {
    if (contract) validateRescueParserProjection(markdown, contract);
    return;
  }
  const actualLines = markdown.split("\n");
  const expectedLines = requiredMarkdown.split("\n");
  if (actualLines.length !== expectedLines.length) {
    throw new DailySummaryRescueValidationError(
      `line count mismatch: expected ${expectedLines.length}, received ${actualLines.length}`,
    );
  }
  const line = expectedLines.findIndex((value, index) => value !== actualLines[index]);
  throw new DailySummaryRescueValidationError(
    `line ${line + 1} differs from the required trusted contract`,
  );
}

function validateRescueParserProjection(
  markdown: string,
  contract: RescueContract,
): void {
  const expectedFallbackIds = contract.slots
    .filter(({ result }) => result.kind === "fallback")
    .map(({ paper }) => paper.id);
  if (JSON.stringify(extractFallbackPaperIds(markdown)) !== JSON.stringify(expectedFallbackIds)) {
    throw new DailySummaryRescueValidationError(
      "fallback parser projection does not match trusted fallback slots",
    );
  }
  const parsed = extractPaperSummaries(markdown);
  const expectedStructured = contract.slots.filter(
    ({ result }) => result.kind === "structured",
  );
  if (Object.keys(parsed).length !== expectedStructured.length) {
    throw new DailySummaryRescueValidationError(
      "structured parser projection contains missing or unexpected papers",
    );
  }
  for (const slot of expectedStructured) {
    if (slot.result.kind !== "structured") continue;
    const actual = parsed[slot.paper.id];
    const expected = {
      sourceSections: normalizeMarkdownLine(slot.paper.sourceSections),
      coreProblem: normalizeMarkdownLine(slot.result.summary.coreProblem),
      keyMethod: normalizeMarkdownLine(slot.result.summary.keyMethod),
      mainResult: normalizeMarkdownLine(slot.result.summary.mainResult),
      whyRelevant: normalizeMarkdownLine(slot.result.summary.whyRelevant),
      limitations: normalizeMarkdownLine(slot.result.summary.limitations),
    };
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new DailySummaryRescueValidationError(
        `structured parser projection differs for ${slot.paper.id}`,
      );
    }
  }
}

export function renderDailySummaryRescueMarkdown(contract: RescueContract): string {
  const out = [
    "<!-- arxiv-daily-rescue-report:start -->",
    dailyHeader(
      contract.language,
      normalizeMarkdownLine(contract.categories),
      normalizeMarkdownLine(contract.date),
    ),
    dailyCountLine(contract.language, contract.counts.total, contract.counts.detail),
  ];
  if (contract.counts.fallback > 0) {
    out.push(fallbackCountLine(contract.language, contract.counts.fallback));
  }
  for (let topicIndex = 0; topicIndex < contract.topics.length; topicIndex += 1) {
    const topic = contract.topics[topicIndex]!;
    out.push(
      "",
      `<!-- arxiv-daily-rescue-topic:${topicIndex} -->`,
      `## ${normalizeMarkdownLine(topic.name)}`,
    );
    const topicSlots = contract.slots.filter(
      ({ paper }) => paper.category === topic.tag,
    );
    if (topicSlots.length === 0) {
      out.push(noCategoryPapersText(contract.language));
      continue;
    }
    for (const slot of topicSlots) out.push("", renderRescueSlot(slot, contract.language));
  }
  out.push("", "<!-- arxiv-daily-rescue-report:end -->");
  return out.join("\n");
}

function renderRescueSlot(
  slot: RescueContract["slots"][number],
  language: SummaryLanguage,
): string {
  const marker = `<!-- arxiv-daily-rescue-paper:${slot.paper.id}:${slot.result.kind} -->`;
  if (slot.result.kind === "fallback") {
    return renderFallbackBlock(
      slot.paper,
      slot.result.originalAbstract,
      language,
      [marker],
    );
  }
  return [
    ...renderPaperHeader(slot.paper, language, [marker]),
    ...renderStructuredFields(slot.result.summary, language),
  ].join("\n");
}

function protectRescueContractDelimiter(value: string): string {
  return value.replace(/<\/rescue_contract>/gi, "&lt;/rescue_contract&gt;");
}

function isErrorLike(value: unknown): value is Error & Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.name === "string" &&
    typeof candidate.message === "string" &&
    typeof candidate.stack === "string";
}
