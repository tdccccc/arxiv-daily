import type { PaperSummary } from "../services/paper-index";

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
  const blocks = markdown.split(/^###\s+/m).slice(1);

  for (const block of blocks) {
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

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
