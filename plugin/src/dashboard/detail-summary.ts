const MIN_DETAIL_SUMMARY_BODY_CHARS = 400;
const DETAIL_SUMMARY_HEADINGS = [
  "研究问题",
  "方法设计",
  "关键证据",
  "主要结论",
  "贡献与创新点",
  "适用边界",
  "阅读价值",
];

export function looksLikeDetailSummary(markdown: string): boolean {
  const body = stripYamlFrontmatter(markdown).trim();
  if (body.length < MIN_DETAIL_SUMMARY_BODY_CHARS) return false;
  if (!/^#\s+\S.+$/m.test(body)) return false;

  const matchedHeadings = DETAIL_SUMMARY_HEADINGS.filter((heading) =>
    new RegExp(`^##\\s+${escapeRegExp(heading)}\\s*$`, "m").test(body),
  ).length;
  if (matchedHeadings >= 3) return true;

  const sectionCount = (body.match(/^##\s+\S+/gm) ?? []).length;
  return sectionCount >= 4 && !isLightweightNote(body);
}

function stripYamlFrontmatter(markdown: string): string {
  return markdown.replace(/^---\r?\n[\s\S]*?\r?\n---\s*(?:\r?\n|$)/, "");
}

function isLightweightNote(body: string): boolean {
  const sections = body.match(/^##\s+(.+)$/gm) ?? [];
  return sections.length <= 1 && sections.some((section) => /^##\s+Notes\s*$/.test(section));
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
