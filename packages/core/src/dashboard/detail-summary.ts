import { stripGenerationMetrics } from "../metrics/generation";

const MIN_DETAIL_SUMMARY_BODY_CHARS = 400;
const DETAIL_SUMMARY_HEADINGS = [
  "研究问题",
  "方法设计",
  "关键证据",
  "主要结论",
  "贡献与创新点",
  "适用边界",
  "学术价值判断",
];

export function looksLikeDetailSummary(markdown: string): boolean {
  const body = stripGenerationMetrics(stripYamlFrontmatter(markdown)).trim();
  if (body.length < MIN_DETAIL_SUMMARY_BODY_CHARS) return false;
  if (!/^#\s+\S.+$/m.test(body)) return false;

  const headings = extractH2Headings(body);
  const matchedHeadings = DETAIL_SUMMARY_HEADINGS.filter((heading) =>
    headings.includes(heading),
  ).length;
  if (matchedHeadings >= 3) return true;

  return headings.length >= 4 && !isLightweightNote(headings);
}

function stripYamlFrontmatter(markdown: string): string {
  return markdown.replace(/^---\r?\n[\s\S]*?\r?\n---\s*(?:\r?\n|$)/, "");
}

function isLightweightNote(headings: string[]): boolean {
  return headings.length <= 1 && headings.some((heading) => heading === "Notes");
}

const WHITESPACE = /\s/u;
const LINE_TERMINATOR = /[\r\n\u2028\u2029]/u;

/**
 * Collect "## Heading" titles with position scans instead of a backtracking
 * regex, so a body with long runs of whitespace after "##" never participates
 * in overlapping \s+ / \s* searches (CodeQL js/polynomial-redos).
 */
function extractH2Headings(body: string): string[] {
  const titles: string[] = [];
  const headingStarts = /^##/gm;
  let heading: RegExpExecArray | null;
  while ((heading = headingStarts.exec(body)) !== null) {
    let cursor = heading.index + 2;
    if (!WHITESPACE.test(body[cursor] ?? "")) continue;
    while (cursor < body.length && WHITESPACE.test(body[cursor]!) && !LINE_TERMINATOR.test(body[cursor]!)) {
      cursor += 1;
    }
    let end = cursor;
    while (end < body.length && !LINE_TERMINATOR.test(body[end]!)) end += 1;
    const title = body.slice(cursor, end).trimEnd();
    if (title) titles.push(title);
  }
  return titles;
}
