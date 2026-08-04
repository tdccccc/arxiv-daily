import type { PaperSummary } from "../services/paper-index";
import { stripGenerationMetrics } from "../metrics/generation";
import {
  DAILY_SUMMARY_ABSTRACT_ABSENT_MARKER_PREFIX,
  DAILY_SUMMARY_EMERGENCY_MARKER,
} from "./daily-summary-rendering";
import { dailySelectionMarkerRegExp } from "../services/daily-selection-marker";
export {
  parseDailyReportDiscoveryProvenance,
  parseDiscoveryProvenanceMarker,
} from "./discovery-provenance-marker";

export { DAILY_SUMMARY_EMERGENCY_MARKER };

const FALLBACK_MARKER_RE =
  /^<!-- arxiv-daily-fallback:(\d{4}\.\d{4,5}) -->$/m;
const ARXIV_BULLET_RE = /^[\t ]*[-*][\t ]+\*\*arXiv\*\*[:：]/im;

const FIELD_LABELS: Array<[keyof PaperSummary, string[]]> = [
  ["coreProblem", ["研究问题", "核心问题", "Research problem"]],
  ["keyMethod", ["方法设计", "关键方法", "Method design"]],
  ["mainResult", ["核心结果", "主要结果", "Core results"]],
  ["whyRelevant", ["研究价值", "为什么值得看", "Research value"]],
  ["limitations", ["适用边界", "局限或边界", "Scope and limits"]],
];

export function extractPaperSummaries(
  markdown: string,
): Record<string, PaperSummary> {
  const summaries: Record<string, PaperSummary> = {};
  const blocks = paperBlocks(markdown);

  for (const block of blocks) {
    if (extractFallbackId(block)) continue;
    const id = extractArxivId(block);
    if (!id) continue;

    const summary: PaperSummary = {};
    const sourceSections = extractSourceSections(block);
    if (sourceSections) summary.sourceSections = sourceSections;

    for (const [key, labels] of FIELD_LABELS) {
      const value = extractAnyBulletField(block, labels);
      if (value) summary[key] = value;
    }

    if (Object.keys(summary).length > 0) {
      summaries[id] = summary;
    }
  }

  return summaries;
}

export function hasEmergencyDailySummaryMarker(markdown: string): boolean {
  return standaloneLineRegExp(DAILY_SUMMARY_EMERGENCY_MARKER).test(
    stripGenerationMetrics(markdown),
  );
}

export function extractFallbackPaperIds(markdown: string): string[] {
  const ids: string[] = [];
  for (const block of paperBlocks(markdown)) {
    const id = extractFallbackId(block);
    if (id && !ids.includes(id)) ids.push(id);
  }
  return ids;
}

/** Extract localized original abstracts only from marker-confirmed fallback blocks. */
export function extractFallbackAbstracts(markdown: string): Record<string, string> {
  const abstracts: Record<string, string> = {};
  for (const block of paperBlocks(markdown)) {
    const id = extractFallbackId(block);
    if (!id || id in abstracts) continue;
    if (hasAbsentAbstractMarker(block, id)) continue;
    const value = extractAnyBulletField(block, ["原始摘要", "Original abstract"]);
    if (value) abstracts[id] = value;
  }
  return abstracts;
}

function paperBlocks(markdown: string): string[] {
  return stripGenerationMetrics(markdown).split(/^###\s+/m).slice(1);
}

function extractFallbackId(block: string): string | null {
  const marker = FALLBACK_MARKER_RE.exec(block);
  const arxivBulletIndex = block.search(ARXIV_BULLET_RE);
  if (!marker || arxivBulletIndex < 0 || marker.index > arxivBulletIndex) {
    return null;
  }
  const paperId = extractArxivId(block);
  return marker[1] === paperId ? marker[1]! : null;
}

function hasAbsentAbstractMarker(block: string, id: string): boolean {
  const escapedId = escapeRegExp(id);
  return new RegExp(
    `^<!-- ${DAILY_SUMMARY_ABSTRACT_ABSENT_MARKER_PREFIX}:${escapedId} -->$`,
    "m",
  ).test(block);
}

function extractArxivId(block: string): string | null {
  const firstHeadingLine = block.split(/\r?\n/, 1)[0] ?? "";
  return (
    dailySelectionMarkerRegExp("m").exec(block)?.[2] ??
    /^[\t ]*[-*][\t ]+\*\*arXiv\*\*[:：][^\r\n]*?arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})(?:v\d+)?/im.exec(block)?.[1] ??
    extractLegacyHeadingId(firstHeadingLine)
  );
}

function extractLegacyHeadingId(headingLine: string): string | null {
  return /(?:^|[\t ])\[(\d{4}\.\d{4,5})\](?=[\t ]*(?:$|→|[-—–|:：]))/.exec(
    headingLine,
  )?.[1] ?? null;
}

function extractSourceSections(block: string): string {
  return compact(
    /^>\s*信息来源[:：]\s*(.+)$/m.exec(block)?.[1] ??
      /^>\s*Source sections[:：]\s*(.+)$/im.exec(block)?.[1] ??
      /^Source sections[:：]\s*(.+)$/im.exec(block)?.[1] ??
      "",
  );
}

function extractBulletField(block: string, label: string): string {
  const escaped = escapeRegExp(label);
  const re = new RegExp(
    String.raw`^[\t ]*[-*][\t ]+\*\*${escaped}\*\*[:：][\t ]*([\s\S]*?)(?=^[\t ]*[-*][\t ]+\*\*|\n###\s+|\n##\s+|$)`,
    "m",
  );
  return compact(re.exec(block)?.[1] ?? "");
}

function extractAnyBulletField(block: string, labels: string[]): string {
  for (const label of labels) {
    const value = extractBulletField(block, label);
    if (value) return value;
  }
  return "";
}

function compact(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function standaloneLineRegExp(value: string): RegExp {
  return new RegExp(`^${escapeRegExp(value)}$`, "m");
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
