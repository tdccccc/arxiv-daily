import type { PaperSummary } from "../services/paper-index";

const FIELD_LABELS: Array<[keyof PaperSummary, string]> = [
  ["coreProblem", "核心问题"],
  ["keyMethod", "关键方法"],
  ["mainResult", "主要结果"],
  ["whyRelevant", "为什么值得看"],
  ["limitations", "局限或边界"],
];

export function extractPaperSummaries(
  markdown: string,
): Record<string, PaperSummary> {
  const summaries: Record<string, PaperSummary> = {};
  const blocks = markdown.split(/^###\s+/m).slice(1);

  for (const block of blocks) {
    const id = extractArxivId(block);
    if (!id) continue;

    const summary: PaperSummary = {};
    const sourceSections = extractSourceSections(block);
    if (sourceSections) summary.sourceSections = sourceSections;

    for (const [key, label] of FIELD_LABELS) {
      const value = extractBulletField(block, label);
      if (value) summary[key] = value;
    }

    if (Object.keys(summary).length > 0) {
      summaries[id] = summary;
    }
  }

  return summaries;
}

function extractArxivId(block: string): string | null {
  return (
    /arxiv-daily:(\d{4}\.\d{4,5}):(?:watch|highlight)/.exec(block)?.[1] ??
    /arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})(?:v\d+)?/i.exec(block)?.[1] ??
    /\[(\d{4}\.\d{4,5})\]/.exec(block)?.[1] ??
    null
  );
}

function extractSourceSections(block: string): string {
  return compact(
    /^>\s*信息来源[:：]\s*(.+)$/m.exec(block)?.[1] ??
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

function compact(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
