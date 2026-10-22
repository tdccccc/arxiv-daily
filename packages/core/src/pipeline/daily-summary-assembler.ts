import { formatArxivCategories } from "../settings/categories";
import {
  dailyCountLine,
  dailyHeader,
  normalizeSummaryLanguage,
  omittedPapersText,
  titleOnlyCandidatesText,
} from "../settings/summary-language";
import type { ArxivSettings, SummaryLanguage } from "../settings/types";
import { normalizePaperDiscoveryProvenance } from "./discovery-provenance-marker";
import { normalizeTopicDirectionHits, type TopicDirectionHit } from "./topic-direction-hits";
import type { PaperDiscoveryProvenance } from "./discovery-provenance-marker";
import {
  DAILY_SUMMARY_EMERGENCY_MARKER,
  emergencyWarning,
  fallbackCountLine,
  renderFallbackBlock,
  renderPaperHeader,
  renderStructuredFields,
  renderEmptyTopics,
  normalizeMarkdownLine,
} from "./daily-summary-rendering";

export interface StructuredPaperSummary {
  id: string;
  coreProblem: string;
  keyMethod: string;
  mainResult: string;
  whyRelevant: string;
  limitations: string;
}

export interface DailySummaryAssemblyPaper {
  id: string;
  title: string;
  authors: string;
  category: string;
  sourceSections: string;
  isDetail: boolean;
  paperPath?: string | null;
  detailLink?: string;
  discoveryProvenance?: PaperDiscoveryProvenance;
  /**
   * Directions of this paper's own topic that selected it, in the topic's
   * order. Kept beside `discoveryProvenance` rather than inside it: a topic
   * direction carries no representative evidence (ADR 0012 §4).
   */
  topicDirections?: TopicDirectionHit[];
}

export type DailyPaperFallbackReasonCode =
  | "validation-exhausted"
  | "transport-exhausted"
  /** The model provider's content-safety filter refused this paper's text. */
  | "content-blocked"
  /** No paper text could be fetched and the listing had no abstract, so there was nothing to summarize. */
  | "content-unavailable";

export type DailyPaperResult =
  | { kind: "structured"; summary: StructuredPaperSummary }
  | {
      kind: "fallback";
      reasonCode: DailyPaperFallbackReasonCode;
      attempts: number;
      originalAbstract: string;
    };

export interface DailyPaperSlot {
  paper: DailySummaryAssemblyPaper;
  result: DailyPaperResult;
}

export interface DailySummaryAssemblyInput {
  slots: DailyPaperSlot[];
  dateStr: string;
  arxivSettings: ArxivSettings;
  summaryLanguage?: SummaryLanguage;
  /** Matched, non-ignored papers omitted only because of the daily limit. */
  omittedByTopic?: Readonly<Record<string, number>>;
  /** Candidates offered to the filter with no abstract; shown in the header when count > 0. */
  titleOnlyCandidates?: DailyTitleOnlyCandidates;
}

export interface DailyTitleOnlyCandidates {
  count: number;
  total: number;
}

/** Validate the trusted title-only snapshot shared by normal and rescue rendering. */
export function validateDailyTitleOnlyCandidates(
  input: Pick<DailySummaryAssemblyInput, "titleOnlyCandidates">,
): DailyTitleOnlyCandidates {
  const value: unknown = input.titleOnlyCandidates;
  if (value === undefined) return { count: 0, total: 0 };
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("preflightDailySummaryAssembly: invalid titleOnlyCandidates object");
  }
  const { count, total } = value as Record<string, unknown>;
  if (typeof count !== "number" || typeof total !== "number"
    || !Number.isSafeInteger(count) || !Number.isSafeInteger(total)
    || count < 0 || count > total) {
    throw new Error("preflightDailySummaryAssembly: invalid titleOnlyCandidates counts");
  }
  return { count, total };
}

/** Share the trusted omission snapshot between normal and rescue rendering. */
export function validateDailySummaryOmissions(
  input: Pick<DailySummaryAssemblyInput, "arxivSettings" | "omittedByTopic">,
): { total: number; byTopic: ReadonlyMap<string, number> } {
  const byTopic = new Map<string, number>();
  const counts = input.omittedByTopic;
  if (counts === undefined) return { total: 0, byTopic };
  const countsPrototype: unknown = typeof counts === "object" && counts !== null ? Object.getPrototypeOf(counts) : undefined;
  if (typeof counts !== "object" || counts === null || Array.isArray(counts)
    || (countsPrototype !== Object.prototype && countsPrototype !== null)) {
    throw new Error("preflightDailySummaryAssembly: invalid omittedByTopic object");
  }
  const tags = new Set(input.arxivSettings.topics.map(({ tag }) => tag));
  let total = 0;
  for (const [tag, count] of Object.entries(counts)) {
    if (!tags.has(tag) || !Number.isSafeInteger(count) || count < 0) {
      throw new Error(`preflightDailySummaryAssembly: invalid omittedByTopic count for ${tag}`);
    }
    total += count;
    if (!Number.isSafeInteger(total)) {
      throw new Error("preflightDailySummaryAssembly: invalid omittedByTopic total");
    }
    byTopic.set(tag, count);
  }
  return { total, byTopic };
}

const FALLBACK_REASON_CODES = new Set<DailyPaperFallbackReasonCode>([
  "validation-exhausted",
  "transport-exhausted",
  "content-blocked",
  "content-unavailable",
]);
const STRUCTURED_FIELDS = [
  "id",
  "coreProblem",
  "keyMethod",
  "mainResult",
  "whyRelevant",
  "limitations",
] as const satisfies ReadonlyArray<keyof StructuredPaperSummary>;
const MODERN_ARXIV_ID_RE = /^\d{4}\.\d{4,5}$/;

export function preflightDailySummaryPapers(
  papers: DailySummaryAssemblyPaper[],
  arxivSettings: ArxivSettings,
): void {
  const topicsByTag = new Map<string, true>();
  const topicNames = new Set<string>();
  for (const topic of arxivSettings.topics) {
    requireTrustedMetadata("topic tag", topic.tag);
    requireTrustedMetadata("topic name", topic.name);
    if (topicsByTag.has(topic.tag)) {
      throw new Error(`preflightDailySummaryAssembly: duplicate topic tag: ${topic.tag}`);
    }
    if (topicNames.has(topic.name)) {
      throw new Error(`preflightDailySummaryAssembly: duplicate topic name: ${topic.name}`);
    }
    topicsByTag.set(topic.tag, true);
    topicNames.add(topic.name);
  }

  const paperIds = new Set<string>();
  for (const paper of papers) {
    requireTrustedMetadata("paper ID", paper.id);
    if (!MODERN_ARXIV_ID_RE.test(paper.id)) {
      throw new Error(`preflightDailySummaryAssembly: invalid paper ID: ${paper.id}`);
    }
    requireTrustedMetadata(`paper ${paper.id} title`, paper.title);
    requireTrustedMetadata(`paper ${paper.id} authors`, paper.authors);
    requireTrustedMetadata(`paper ${paper.id} category`, paper.category);
    requireTrustedMetadata(`paper ${paper.id} source sections`, paper.sourceSections);
    if (paper.discoveryProvenance
      && !normalizePaperDiscoveryProvenance(paper.discoveryProvenance)) {
      throw new Error(`preflightDailySummaryAssembly: paper ${paper.id} has invalid discovery provenance`);
    }
    if (paper.topicDirections
      && !normalizeTopicDirectionHits(paper.topicDirections)) {
      throw new Error(`preflightDailySummaryAssembly: paper ${paper.id} has invalid topic directions`);
    }
    if (paperIds.has(paper.id)) {
      throw new Error(`preflightDailySummaryAssembly: duplicate input paper ID: ${paper.id}`);
    }
    paperIds.add(paper.id);
    if (!topicsByTag.has(paper.category)) {
      throw new Error(
        `preflightDailySummaryAssembly: paper ${paper.id} has unknown category tag: ${paper.category}`,
      );
    }
  }
}

export function preflightDailySummaryAssembly(input: DailySummaryAssemblyInput): void {
  preflightDailySummaryPapers(input.slots.map(({ paper }) => paper), input.arxivSettings);
  validateDailySummaryOmissions(input);
  validateDailyTitleOnlyCandidates(input);
  for (const { paper, result } of input.slots) {
    if (result.kind === "structured") {
      for (const field of STRUCTURED_FIELDS) {
        requireTrustedMetadata(`paper ${paper.id} summary ${field}`, result.summary[field]);
      }
      if (result.summary.id !== paper.id) {
        throw new Error(
          `preflightDailySummaryAssembly: summary ID ${result.summary.id} does not match paper ID: ${paper.id}`,
        );
      }
      continue;
    }
    if (!FALLBACK_REASON_CODES.has(result.reasonCode)) {
      throw new Error(`preflightDailySummaryAssembly: paper ${paper.id} has invalid fallback reasonCode`);
    }
    if (!Number.isInteger(result.attempts) || result.attempts < 1 || result.attempts > 3) {
      throw new Error(`preflightDailySummaryAssembly: paper ${paper.id} has invalid fallback attempts`);
    }
    if (typeof result.originalAbstract !== "string") {
      throw new Error(
        `preflightDailySummaryAssembly: missing trusted metadata: paper ${paper.id} originalAbstract`,
      );
    }
  }
}

export const DAILY_SUMMARY_ASSEMBLY_RUNTIME_ERROR_CODE =
  "ARXIV_DAILY_SUMMARY_ASSEMBLY_RUNTIME" as const;

export class DailySummaryAssemblyRuntimeError extends Error {
  readonly name = "DailySummaryAssemblyRuntimeError";
  readonly code = DAILY_SUMMARY_ASSEMBLY_RUNTIME_ERROR_CODE;
  readonly cause: unknown;

  constructor(cause: unknown) {
    super("Daily summary rendering failed after valid preflight");
    this.cause = cause;
  }
}

export function isDailySummaryAssemblyRuntimeError(
  error: unknown,
): error is DailySummaryAssemblyRuntimeError {
  if (error instanceof DailySummaryAssemblyRuntimeError) return true;
  return isErrorLike(error) &&
    error.name === "DailySummaryAssemblyRuntimeError" &&
    error.code === DAILY_SUMMARY_ASSEMBLY_RUNTIME_ERROR_CODE &&
    "cause" in error;
}

export interface DailySummaryAssemblerDeps {
  renderer?: (input: DailySummaryAssemblyInput) => string;
  /** @deprecated Use renderer. */
  render?: (input: DailySummaryAssemblyInput) => string;
}

export function assembleDailySummary(
  input: DailySummaryAssemblyInput,
  deps: DailySummaryAssemblerDeps = {},
): string {
  preflightDailySummaryAssembly(input);
  return renderAfterPreflight(input, deps.renderer ?? deps.render ?? renderDailySummary);
}

export function renderAfterPreflight(
  input: DailySummaryAssemblyInput,
  renderer: (input: DailySummaryAssemblyInput) => string,
): string {
  try {
    return renderer(input);
  } catch (error) {
    if (isDailySummaryAssemblyRuntimeError(error)) throw error;
    throw new DailySummaryAssemblyRuntimeError(error);
  }
}

function renderDailySummary(input: DailySummaryAssemblyInput): string {
  return renderDailySummarySlots(input, false);
}

/** Render a complete degraded report without another provider call. Input must be preflighted. */
export function assembleEmergencyDailySummary(input: DailySummaryAssemblyInput): string {
  preflightDailySummaryAssembly(input);
  return renderDailySummarySlots(input, true);
}

function renderDailySummarySlots(input: DailySummaryAssemblyInput, emergency: boolean): string {
  const { slots, dateStr, arxivSettings } = input;
  const language = normalizeSummaryLanguage(input.summaryLanguage);
  const slotsByTopic = groupSlots(slots, arxivSettings);
  const omissions = validateDailySummaryOmissions(input);
  const titleOnly = validateDailyTitleOnlyCandidates(input);
  const detailCount = slots.filter(({ paper }) => paper.isDetail || Boolean(paper.paperPath)).length;
  const fallbackCount = slots.filter(({ result }) => result.kind === "fallback").length;
  const out = emergency
    ? [DAILY_SUMMARY_EMERGENCY_MARKER, emergencyWarning(language)]
    : [];
  out.push(
    dailyHeader(
      language,
      normalizeMarkdownLine(formatArxivCategories(arxivSettings)),
      normalizeMarkdownLine(dateStr),
    ),
    dailyCountLine(language, slots.length, detailCount),
  );
  if (fallbackCount > 0) out.push(fallbackCountLine(language, fallbackCount));
  if (titleOnly.count > 0) out.push(titleOnlyCandidatesText(language, titleOnly.count, titleOnly.total));
  if (omissions.total > 0) out.push(omittedPapersText(language, omissions.total, true));

  const emptyTopics: Array<{ name: string; omittedCount: number }> = [];
  for (const topic of arxivSettings.topics) {
    const topicSlots = slotsByTopic.get(topic.tag) ?? [];
    const omittedCount = omissions.byTopic.get(topic.tag) ?? 0;
    if (topicSlots.length === 0) {
      emptyTopics.push({ name: topic.name, omittedCount });
      continue;
    }
    out.push("", `## ${normalizeMarkdownLine(topic.name)}`);
    if (omittedCount > 0) out.push(omittedPapersText(language, omittedCount, true));
    for (const slot of topicSlots) out.push("", renderSlot(slot, language, dateStr));
  }
  const emptyTopicLines = renderEmptyTopics(emptyTopics, language);
  if (emptyTopicLines.length > 0) out.push("", ...emptyTopicLines);
  return out.join("\n");
}

function groupSlots(
  slots: DailyPaperSlot[],
  arxivSettings: ArxivSettings,
): Map<string, DailyPaperSlot[]> {
  const slotsByTopic = new Map(
    arxivSettings.topics.map((topic) => [topic.tag, [] as DailyPaperSlot[]] as const),
  );
  for (const slot of slots) slotsByTopic.get(slot.paper.category)!.push(slot);
  return slotsByTopic;
}

function renderSlot(slot: DailyPaperSlot, language: SummaryLanguage, reportDate: string): string {
  if (slot.result.kind === "fallback") {
    return renderFallbackBlock(
      slot.paper,
      slot.result.originalAbstract,
      language,
      [],
      reportDate,
      slot.result.reasonCode,
    );
  }
  return [
    ...renderPaperHeader(slot.paper, language, [], reportDate),
    "",
    ...renderStructuredFields(slot.result.summary, language),
  ].join("\n");
}

function requireTrustedMetadata(label: string, value: unknown): void {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`preflightDailySummaryAssembly: missing trusted metadata: ${label}`);
  }
}

function isErrorLike(value: unknown): value is Error & Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.name === "string" &&
    typeof candidate.message === "string" &&
    typeof candidate.stack === "string";
}
