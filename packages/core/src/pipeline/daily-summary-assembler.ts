import { formatArxivCategories } from "../settings/categories";
import {
  dailyCountLine,
  dailyHeader,
  noCategoryPapersText,
  normalizeSummaryLanguage,
} from "../settings/summary-language";
import type { ArxivSettings, SummaryLanguage } from "../settings/types";
import { normalizePaperDiscoveryProvenance } from "./discovery-provenance-marker";
import { PERSONALIZED_LIBRARY_ONLY_CATEGORY } from "./personalized-paper-filter";
import type { PaperDiscoveryProvenance } from "./personalized-paper-filter";
import { normalizePersonalNoveltyWithBasis } from "./personalized-novelty";
import type { PersonalNoveltyWithBasis } from "./personalized-novelty";
import {
  DAILY_SUMMARY_EMERGENCY_MARKER,
  emergencyWarning,
  fallbackCountLine,
  renderFallbackBlock,
  renderPaperHeader,
  renderStructuredFields,
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
   * Validated personal novelty with trusted basis display titles, carried only
   * from library-derived papers with a novelty outcome. Novelty is not part of
   * daily summary generation prompts or the summary checkpoint identity, but —
   * like discovery provenance — the deterministic rescue/emergency re-render
   * contracts may carry it; persisted markers stay minimal.
   */
  personalNovelty?: PersonalNoveltyWithBasis;
}

export type DailyPaperFallbackReasonCode =
  | "validation-exhausted"
  | "transport-exhausted";

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
}

const FALLBACK_REASON_CODES = new Set<DailyPaperFallbackReasonCode>([
  "validation-exhausted",
  "transport-exhausted",
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
    if (paper.personalNovelty
      && !normalizePersonalNoveltyWithBasis(paper.personalNovelty)) {
      throw new Error(`preflightDailySummaryAssembly: paper ${paper.id} has invalid personal novelty`);
    }
    if (paperIds.has(paper.id)) {
      throw new Error(`preflightDailySummaryAssembly: duplicate input paper ID: ${paper.id}`);
    }
    paperIds.add(paper.id);
    if (!topicsByTag.has(paper.category)
      && paper.category !== PERSONALIZED_LIBRARY_ONLY_CATEGORY) {
      throw new Error(
        `preflightDailySummaryAssembly: paper ${paper.id} has unknown category tag: ${paper.category}`,
      );
    }
  }
}

export function preflightDailySummaryAssembly(input: DailySummaryAssemblyInput): void {
  preflightDailySummaryPapers(input.slots.map(({ paper }) => paper), input.arxivSettings);
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

  for (const topic of arxivSettings.topics) {
    out.push("", `## ${normalizeMarkdownLine(topic.name)}`);
    const topicSlots = slotsByTopic.get(topic.tag) ?? [];
    if (topicSlots.length === 0) {
      out.push(noCategoryPapersText(language));
      continue;
    }
    for (const slot of topicSlots) out.push("", renderSlot(slot, language, dateStr));
  }
  const librarySlots = slotsByTopic.get(PERSONALIZED_LIBRARY_ONLY_CATEGORY) ?? [];
  if (librarySlots.length > 0) {
    out.push("", `## ${language === "en" ? "Library-guided discoveries" : "个人文献库引导发现"}`);
    for (const slot of librarySlots) out.push("", renderSlot(slot, language, dateStr));
  }
  return out.join("\n");
}

function groupSlots(
  slots: DailyPaperSlot[],
  arxivSettings: ArxivSettings,
): Map<string, DailyPaperSlot[]> {
  const slotsByTopic = new Map([
    ...arxivSettings.topics.map((topic) => [topic.tag, [] as DailyPaperSlot[]] as const),
    [PERSONALIZED_LIBRARY_ONLY_CATEGORY, [] as DailyPaperSlot[]] as const,
  ]);
  for (const slot of slots) slotsByTopic.get(slot.paper.category)!.push(slot);
  return slotsByTopic;
}

function renderSlot(slot: DailyPaperSlot, language: SummaryLanguage, reportDate: string): string {
  if (slot.result.kind === "fallback") {
    return renderFallbackBlock(slot.paper, slot.result.originalAbstract, language, [], reportDate);
  }
  return [
    ...renderPaperHeader(slot.paper, language, [], reportDate),
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
