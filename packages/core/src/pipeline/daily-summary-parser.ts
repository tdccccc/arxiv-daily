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

    const fields = unquoteSummaryCallouts(block);
    for (const [key, labels] of FIELD_LABELS) {
      const value = extractAnyBulletField(fields, labels);
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

/** One paper of a committed daily report, as written (Markdown-normalized text). */
export interface DailyReportPaper {
  /** The `##` heading the paper sits under; null before the first one. */
  section: string | null;
  id: string;
  title: string;
  authors: string;
  /** The five summary fields of a structured paper; absent for a fallback block. */
  summary?: PaperSummary;
  /** A paper whose summary was replaced by its original abstract. */
  fallback: boolean;
  fallbackAbstract?: string;
}

// The writer appends " → [[id]]" or " → [id](path)" to the heading of a paper that has a detail note.
const DETAIL_LINK_TAIL_RE =
  /^[\t ]+(?:\[\[[^\]\r\n]+\]\]|\[[^\]\r\n]+\]\([^)\r\n]*\))[\t ]*$/;

function stripDetailLink(heading: string): string {
  // Only the last arrow can start the link, so no unanchored scan over spaces is needed.
  const arrow = heading.lastIndexOf("→");
  return arrow >= 0 && DETAIL_LINK_TAIL_RE.test(heading.slice(arrow + 1))
    ? heading.slice(0, arrow)
    : heading;
}

/**
 * Read the papers of a committed daily report in report order, each with the
 * `##` section it sits under. Blocks without a recognizable paper ID are left
 * out, so the result is only what the identity and field parsers can vouch for.
 */
export function parseDailyReportPapers(markdown: string): DailyReportPaper[] {
  const papers: DailyReportPaper[] = [];
  let section: string | null = null;
  let heading: string | undefined;
  let body: string[] = [];
  const flush = () => {
    if (heading === undefined) return;
    const block = [`### ${heading}`, ...body].join("\n");
    heading = undefined;
    const fallbackId = extractFallbackId(block);
    const id = fallbackId ?? extractArxivId(block);
    if (!id) return;
    const paper: DailyReportPaper = {
      section,
      id,
      title: compact(stripDetailLink(block.split("\n", 1)[0]!.slice(4))),
      authors: extractAnyBulletField(block, ["作者", "Authors"]),
      fallback: fallbackId !== null,
    };
    if (fallbackId) {
      paper.fallbackAbstract = extractFallbackAbstracts(block)[id] ?? "";
    } else {
      const summary = extractPaperSummaries(block)[id];
      if (summary) paper.summary = summary;
    }
    papers.push(paper);
  };
  for (const line of stripGenerationMetrics(markdown).split(/\r?\n/)) {
    // Match only the bounded heading prefix. Consuming whitespace followed by
    // `.*` can repeatedly rescan long tab runs when a Unicode separator follows.
    const headingPrefix = /^(#{1,3})[\t ]/.exec(line);
    if (headingPrefix?.[1] === "###") {
      flush();
      heading = line.slice(headingPrefix[0].length).trimStart();
      body = [];
      continue;
    }
    if (headingPrefix) {
      flush();
      if (headingPrefix[1] === "##") section = compact(line.slice(headingPrefix[0].length));
      continue;
    }
    if (heading !== undefined) body.push(line);
  }
  flush();
  return papers;
}

function paperBlocks(markdown: string): string[] {
  return stripGenerationMetrics(markdown).split(/^###\s+/m).slice(1)
    .map((block) => block.split(/^#{1,2}[\t ]+/m, 1)[0]!);
}

/** Unquote only summary callouts, leaving source annotations and identity untouched. */
function unquoteSummaryCallouts(block: string): string {
  let inSummary = false;
  return block.split(/\r?\n/).map((line) => {
    if (/^> \[!abstract\][+-]?(?:\s|$)/.test(line)) {
      inSummary = true;
      return "";
    }
    if (inSummary && (line === ">" || line.startsWith("> "))) return line.slice(2);
    inSummary = false;
    return line;
  }).join("\n");
}

function extractFallbackId(block: string): string | null {
  const marker = FALLBACK_MARKER_RE.exec(block);
  const arxivBulletIndex = block.search(ARXIV_BULLET_RE);
  if (!marker || arxivBulletIndex < 0 || marker.index > arxivBulletIndex) {
    return null;
  }
  const paperId = extractArxivId(block);
  return marker[1] === paperId ? marker[1] : null;
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
  // compact() below collapses and trims whatever (.+) captures, so a
  // redundant \s* immediately before it only duplicated what (.+) already
  // matches, letting CodeQL flag the ambiguous split on long whitespace runs
  // (js/polynomial-redos). (.+) alone is unambiguous and linear.
  return compact(
    /^>\s*信息来源[:：](.+)$/m.exec(block)?.[1] ??
      /^>\s*Source sections[:：](.+)$/im.exec(block)?.[1] ??
      /^Source sections[:：](.+)$/im.exec(block)?.[1] ??
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
