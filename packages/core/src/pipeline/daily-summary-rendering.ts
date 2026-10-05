import type { PaperSummary } from "../services/paper-index";
import type { SummaryLanguage } from "../settings/types";
import { noCategoryPapersText } from "../settings/summary-language";
import type {
  DailySummaryAssemblyPaper,
  StructuredPaperSummary,
} from "./daily-summary-assembler";
import { neutralizeRawHtml } from "./raw-html";
import {
  escapeDiscoveryProvenancePlainText,
  renderDiscoveryProvenanceMarker,
} from "./discovery-provenance-marker";
import { renderPersonalNoveltyMarker } from "./personal-novelty-marker";
import { renderTopicDirectionMarker } from "./topic-direction-marker";
import type {
  PersonalNoveltyDifferenceType,
} from "./personalized-novelty";

export const DAILY_SUMMARY_EMERGENCY_MARKER =
  "<!-- arxiv-daily-emergency-report:v1 -->";
export const DAILY_SUMMARY_FALLBACK_MARKER_PREFIX = "arxiv-daily-fallback";
export const DAILY_SUMMARY_ABSTRACT_ABSENT_MARKER_PREFIX =
  "arxiv-daily-fallback-abstract-absent";

export type SummaryField = Exclude<keyof PaperSummary, "sourceSections">;

// A template literal (rather than a regex literal) keeps this control-character
// check from tripping eslint's no-control-regex scanner-parity rule; the
// matched set (C0 controls + DEL) is unchanged.
const CONTROL_CHARACTER_RE = new RegExp(`[\u0000-\u001f\u007f]`);

export const DAILY_SUMMARY_FIELD_LABELS: Record<
  SummaryLanguage,
  Array<[SummaryField, string]>
> = {
  zh: [
    ["coreProblem", "研究问题"],
    ["keyMethod", "方法设计"],
    ["mainResult", "核心结果"],
    ["whyRelevant", "研究价值"],
    ["limitations", "适用边界"],
  ],
  en: [
    ["coreProblem", "Research problem"],
    ["keyMethod", "Method design"],
    ["mainResult", "Core results"],
    ["whyRelevant", "Research value"],
    ["limitations", "Scope and limits"],
  ],
};

/** Localized difference-type labels for the visible novelty line. */
export const PERSONAL_NOVELTY_DIFFERENCE_TYPE_LABELS: Record<
  SummaryLanguage,
  Record<PersonalNoveltyDifferenceType, string>
> = {
  zh: {
    "new-task": "新任务",
    "new-method": "新方法",
    "new-dataset": "新数据集",
    "new-experiment": "新实验",
    "efficiency-result": "效率结果",
    "counter-evidence": "反例证据",
  },
  en: {
    "new-task": "new task",
    "new-method": "new method",
    "new-dataset": "new dataset",
    "new-experiment": "new experiment",
    "efficiency-result": "efficiency result",
    "counter-evidence": "counter-evidence",
  },
};

/**
 * Compact interpolated prose to one physical Markdown line.
 *
 * CommonMark inline Markdown remains readable: links, emphasis, code spans,
 * URI/email autolinks, MathJax, comparisons, ampersands, and ordinary angle
 * brackets are not globally encoded. Actual CommonMark raw-HTML constructs are
 * made inert by encoding only their opening `<`.
 */
export function normalizeMarkdownLine(value: string): string {
  const compacted = value.replace(/\s+/gu, " ").trim();
  return mapOutsideCodeSpans(compacted, neutralizeRawHtml);
}

function mapOutsideCodeSpans(
  value: string,
  transform: (segment: string) => string,
): string {
  let out = "";
  let plainStart = 0;
  for (let index = 0; index < value.length;) {
    if (value[index] !== "`") {
      index += 1;
      continue;
    }
    const openerEnd = readBacktickRunEnd(value, index);
    const delimiterLength = openerEnd - index;
    const close = findMatchingBacktickRun(value, openerEnd, delimiterLength);
    if (close < 0) {
      // An unmatched run is literal text. Continue at the next run so it can
      // independently open a span; CommonMark gives backslashes no special
      // delimiter-disabling meaning for backtick strings.
      index = openerEnd;
      continue;
    }
    const closeEnd = close + delimiterLength;
    out += transform(value.slice(plainStart, index));
    out += value.slice(index, closeEnd);
    index = closeEnd;
    plainStart = index;
  }
  return out + transform(value.slice(plainStart));
}

function findMatchingBacktickRun(
  value: string,
  start: number,
  delimiterLength: number,
): number {
  for (let index = start; index < value.length;) {
    const runStart = value.indexOf("`", index);
    if (runStart < 0) return -1;
    const runEnd = readBacktickRunEnd(value, runStart);
    if (runEnd - runStart === delimiterLength) return runStart;
    index = runEnd;
  }
  return -1;
}

function readBacktickRunEnd(value: string, start: number): number {
  let end = start + 1;
  while (value[end] === "`") end += 1;
  return end;
}

export function trustedArxivUrl(id: string): string {
  return `https://arxiv.org/abs/${id}`;
}

export function safeDetailLink(
  paperId: string,
  paperPath: string | null | undefined,
  generatedLink: string | null | undefined,
  hasDetail: boolean,
): string | null {
  if (!hasDetail) return null;
  const value = generatedLink?.trim();
  if (!value) return null;

  // MarkdownWriter's wikilink contract is deliberately alias-free and ID-addressed.
  if (value === `[[${paperId}]]`) return value;

  // Its relative-link contract has an exact ID label and a local destination that
  // resolves to the trusted paperPath.
  const match = /^\[([^\]\r\n]+)\]\(([^)\r\n]+)\)$/.exec(value);
  if (!match || match[1] !== paperId) return null;
  const destination = match[2]!;
  let decoded: string;
  try {
    decoded = decodeURI(destination);
  } catch {
    return null;
  }
  if (!isExpectedLocalPaperTarget(decoded, paperId, paperPath)) return null;
  return `[${paperId}](${encodeRelativeLinkTarget(decoded)})`;
}

export function renderPaperHeader(
  paper: DailySummaryAssemblyPaper,
  language: SummaryLanguage,
  leadingMarkers: string[] = [],
  reportDate?: string,
): string[] {
  const detailLink = safeDetailLink(
    paper.id,
    paper.paperPath,
    paper.detailLink,
    paper.isDetail || Boolean(paper.paperPath),
  );
  const sourceLabel = language === "en" ? "Source sections:" : "信息来源：";
  const contextLabel = paper.topicDirections
    ? language === "en" ? "Matched directions and sources" : "命中方向与信息来源"
    : language === "en" ? "Source sections" : "信息来源";
  const authorLabel = language === "en" ? "Authors" : "作者";
  return [
    ...leadingMarkers,
    `### ${normalizeMarkdownLine(paper.title)}${detailLink ? ` → ${detailLink}` : ""}`,
    ...(paper.discoveryProvenance
      ? [
          renderDiscoveryProvenanceMarker(
            paper.discoveryProvenance,
            paper.id,
            requireReportDate(reportDate),
          ),
          renderVisibleDiscoveryProvenance(paper.discoveryProvenance, language),
        ]
      : []),
    ...(paper.personalNovelty
      ? [
          renderPersonalNoveltyMarker(
            paper.personalNovelty,
            paper.id,
            requireReportDate(reportDate),
          ),
          renderVisiblePersonalNovelty(paper.personalNovelty, language),
        ]
      : []),
    // Keep the metadata callout after the other two marker families, whose
    // canonical slots are counted from the paper heading.
    ...(paper.discoveryProvenance || paper.personalNovelty ? [""] : []),
    `> [!info]- ${contextLabel}`,
    ...(paper.topicDirections
      ? [
          `> ${renderTopicDirectionMarker(
            paper.topicDirections,
            paper.id,
            requireReportDate(reportDate),
          )}`,
          ...renderVisibleTopicDirections(paper.topicDirections, language),
          ">",
        ]
      : []),
    `> ${sourceLabel} ${normalizeMarkdownLine(paper.sourceSections)}`,
    "",
    `- **${authorLabel}**: ${normalizeMarkdownLine(paper.authors)}`,
    `- **arXiv**: [${paper.id}](${trustedArxivUrl(paper.id)})`,
  ];
}

/**
 * Each direction gets its own list item. The report's section heading already
 * names the topic. Direction text is researcher-authored literal text.
 */
function renderVisibleTopicDirections(
  hits: NonNullable<DailySummaryAssemblyPaper["topicDirections"]>,
  language: SummaryLanguage,
): string[] {
  const label = language === "en" ? "Matched directions" : "命中方向";
  return [
    `> **${label}**`,
    ...hits.map((hit) => `> - ${escapeDiscoveryProvenancePlainText(hit.text)}`),
  ];
}

function renderVisibleDiscoveryProvenance(
  provenance: NonNullable<DailySummaryAssemblyPaper["discoveryProvenance"]>,
  language: SummaryLanguage,
): string {
  const manual = provenance.manualTopicTags.map(escapeDiscoveryProvenancePlainText);
  const directions = provenance.directions.map((direction) => {
    const representatives = direction.representatives.map((representative) =>
      `${escapeDiscoveryProvenancePlainText(representative.title)} (${escapeDiscoveryProvenancePlainText(representative.paperKey)})`
    ).join(language === "en" ? "; " : "；");
    return `${escapeDiscoveryProvenancePlainText(direction.name)} [${representatives}]`;
  });
  const sources = [
    ...(manual.length > 0
      ? [language === "en" ? `manual topics: ${manual.join(", ")}` : `手动主题：${manual.join("、")}`]
      : []),
    ...(directions.length > 0
      ? [language === "en"
          ? `library directions: ${directions.join("; ")}`
          : `个人文献库方向：${directions.join("；")}`]
      : []),
  ];
  const depth = directions.length > 0
    ? language === "en" ? "; evidence depth: metadata and abstract" : "；证据深度：元数据与摘要"
    : "";
  return language === "en"
    ? `> Discovery source: ${sources.join("; ")}${depth}`
    : `> 发现来源：${sources.join("；")}${depth}`;
}

/**
 * Deterministic localized novelty line: localized difference-type label,
 * named representative prior papers (paperKey + trusted title carried on the
 * assembly paper), the explicit metadata-and-abstract evidence depth, and the
 * bounded explanation. Every literal is escaped to plain text like the
 * discovery-provenance projection.
 */
function renderVisiblePersonalNovelty(
  novelty: NonNullable<DailySummaryAssemblyPaper["personalNovelty"]>,
  language: SummaryLanguage,
): string {
  const typeLabel = PERSONAL_NOVELTY_DIFFERENCE_TYPE_LABELS[language][novelty.differenceType];
  const basis = novelty.comparisonBasis.map((paperKey) => {
    const title = novelty.comparisonBasisTitles[paperKey];
    const name = title
      ? escapeDiscoveryProvenancePlainText(title)
      : escapeDiscoveryProvenancePlainText(paperKey);
    return `${name} (${escapeDiscoveryProvenancePlainText(paperKey)})`;
  }).join(language === "en" ? "; " : "；");
  const depth = language === "en" ? "metadata and abstract" : "元数据与摘要";
  const explanation = escapeDiscoveryProvenancePlainText(novelty.explanation);
  return language === "en"
    ? `> Personal novelty: ${typeLabel} vs. prior papers: ${basis}; evidence depth: ${depth}; ${explanation}`
    : `> 个人新颖性：${typeLabel}，对比先验文献：${basis}；证据深度：${depth}；${explanation}`;
}

function requireReportDate(reportDate?: string): string {
  if (!reportDate) throw new TypeError("report date is required for paper markers");
  return reportDate;
}

export function renderStructuredFields(
  summary: StructuredPaperSummary,
  language: SummaryLanguage,
): string[] {
  const fields = DAILY_SUMMARY_FIELD_LABELS[language];
  const resultLabel = fields.find(([key]) => key === "mainResult")![1];
  const detailsLabel = language === "en" ? "Background, methods and limits" : "研究背景、方法与边界";
  return [
    `- **${resultLabel}**: ${normalizeMarkdownLine(summary.mainResult)}`,
    "",
    `> [!abstract]- ${detailsLabel}`,
    ...fields.filter(([key]) => key !== "mainResult").flatMap(
      ([key, label], index) => [
        ...(index > 0 ? [">"] : []),
        `> - **${label}**: ${normalizeMarkdownLine(summary[key])}`,
      ],
    ),
  ];
}

export function renderEmptyTopics(
  topics: ReadonlyArray<{ name: string; omittedCount: number }>,
  language: SummaryLanguage,
): string[] {
  if (topics.length === 0) return [];
  return [
    language === "en" ? "## Other followed topics" : "## 其他关注主题",
    ...topics.map(({ name, omittedCount }) =>
      `- **${escapeDiscoveryProvenancePlainText(normalizeMarkdownLine(name))}** — ${noCategoryPapersText(language, omittedCount)}`),
  ];
}

export function renderFallbackBlock(
  paper: DailySummaryAssemblyPaper,
  originalAbstract: string,
  language: SummaryLanguage,
  leadingMarkers: string[] = [],
  reportDate?: string,
): string {
  const lines = renderPaperHeader(paper, language, leadingMarkers, reportDate);
  const arxivUrl = trustedArxivUrl(paper.id);
  const warning = language === "en"
    ? `> **Summary unavailable.** Read the [original paper on arXiv](${arxivUrl}) directly.`
    : `> **自动摘要不可用。** 请直接阅读 [arXiv 原文](${arxivUrl})。`;
  const abstractLabel = language === "en" ? "Original abstract" : "原始摘要";
  const unavailable = language === "en" ? "Unavailable." : "不可用。";
  const abstract = normalizeMarkdownLine(originalAbstract);
  // The header ends with the author and arXiv bullets. Place the warning
  // outside the collapsed context, before the identity bullet used by parsers.
  lines.splice(
    lines.length - 2,
    0,
    warning,
    `<!-- ${DAILY_SUMMARY_FALLBACK_MARKER_PREFIX}:${paper.id} -->`,
    ...(!abstract
      ? [`<!-- ${DAILY_SUMMARY_ABSTRACT_ABSENT_MARKER_PREFIX}:${paper.id} -->`]
      : []),
    "",
  );
  lines.push("", `- **${abstractLabel}**: ${abstract || unavailable}`);
  return lines.join("\n");
}

export function fallbackCountLine(
  language: SummaryLanguage,
  fallbackCount: number,
): string {
  return language === "en"
    ? `${fallbackCount} ${fallbackCount === 1 ? "paper uses" : "papers use"} fallback content.`
    : `其中 ${fallbackCount} 篇使用回退内容。`;
}

export function emergencyWarning(language: SummaryLanguage): string {
  return language === "en"
    ? "> **Degraded emergency report.** Rescue generation failed; this report was assembled deterministically from validated local data."
    : "> **降级应急报告。** 救援生成失败；本报告由已验证的本地数据确定性组装。";
}

function isExpectedLocalPaperTarget(
  destination: string,
  id: string,
  paperPath: string | null | undefined,
): boolean {
  if (
    !destination ||
    destination.startsWith("/") ||
    destination.includes("\\") ||
    /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(destination) ||
    /[?#]/.test(destination)
  ) {
    return false;
  }
  const parts = destination.split("/");
  if (parts.some((part) => !part || part === ".")) return false;
  const firstLocalPart = parts.findIndex((part) => part !== "..");
  if (
    firstLocalPart < 0 ||
    parts.slice(firstLocalPart).some((part) => part === "..")
  ) return false;
  const normalized = normalizeLocalPath(parts);
  if (!normalized || normalized.at(-1) !== `${id}.md` || !paperPath) return false;
  const trusted = normalizeLocalPath(paperPath.replace(/\\/g, "/").split("/"));
  if (!trusted) return false;
  const localTail = normalized.filter((part) => part !== "..");
  return localTail.length > 0 &&
    trusted.join("/").endsWith(localTail.join("/"));
}

function normalizeLocalPath(parts: string[]): string[] | null {
  const normalized: string[] = [];
  for (const part of parts) {
    if (part === ".") continue;
    if (part === "..") {
      if (normalized.length > 0 && normalized.at(-1) !== "..") normalized.pop();
      else normalized.push(part);
      continue;
    }
    if (!part || CONTROL_CHARACTER_RE.test(part)) return null;
    normalized.push(part);
  }
  return normalized;
}

function encodeRelativeLinkTarget(path: string): string {
  return encodeURI(path).replace(/\(/g, "%28").replace(/\)/g, "%29");
}
