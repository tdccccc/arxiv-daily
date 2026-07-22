import type { PaperSummary } from "../services/paper-index";
import { formatArxivCategories } from "../settings/categories";
import {
  dailyCountLine,
  dailyHeader,
  noCategoryPapersText,
  normalizeSummaryLanguage,
} from "../settings/summary-language";
import type { ArxivSettings, SummaryLanguage } from "../settings/types";

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
}

export interface DailySummaryAssemblyInput {
  papers: DailySummaryAssemblyPaper[];
  summaries: StructuredPaperSummary[];
  dateStr: string;
  arxivSettings: ArxivSettings;
  summaryLanguage?: SummaryLanguage;
}

type SummaryField = Exclude<keyof PaperSummary, "sourceSections">;

const FIELD_LABELS: Record<
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

export function assembleDailySummary(input: DailySummaryAssemblyInput): string {
  const { papers, summaries, dateStr, arxivSettings } = input;
  const language = normalizeSummaryLanguage(input.summaryLanguage);
  const papersByTopic = validateAndGroupPapers(papers, arxivSettings);
  const summariesById = validateSummaries(papers, summaries);
  const detailCount = papers.filter(
    (paper) => paper.isDetail || Boolean(paper.paperPath),
  ).length;
  const out = [
    dailyHeader(language, formatArxivCategories(arxivSettings), dateStr),
    dailyCountLine(language, papers.length, detailCount),
  ];

  for (const topic of arxivSettings.topics) {
    out.push("", `## ${topic.name}`);
    const topicPapers = papersByTopic.get(topic.tag) ?? [];
    if (topicPapers.length === 0) {
      out.push(noCategoryPapersText(language));
      continue;
    }
    for (const paper of topicPapers) {
      out.push("", renderPaper(paper, summariesById.get(paper.id)!, language));
    }
  }

  return out.join("\n");
}

function validateAndGroupPapers(
  papers: DailySummaryAssemblyPaper[],
  arxivSettings: ArxivSettings,
): Map<string, DailySummaryAssemblyPaper[]> {
  const papersByTopic = new Map<string, DailySummaryAssemblyPaper[]>();
  const topicNames = new Set<string>();
  for (const topic of arxivSettings.topics) {
    if (papersByTopic.has(topic.tag)) {
      throw new Error(`assembleDailySummary: duplicate topic tag: ${topic.tag}`);
    }
    if (topicNames.has(topic.name)) {
      throw new Error(`assembleDailySummary: duplicate topic name: ${topic.name}`);
    }
    papersByTopic.set(topic.tag, []);
    topicNames.add(topic.name);
  }

  const paperIds = new Set<string>();
  for (const paper of papers) {
    if (paperIds.has(paper.id)) {
      throw new Error(`assembleDailySummary: duplicate input paper ID: ${paper.id}`);
    }
    paperIds.add(paper.id);
    const topicPapers = papersByTopic.get(paper.category);
    if (!topicPapers) {
      throw new Error(
        `assembleDailySummary: paper ${paper.id} has unknown category tag: ${paper.category}`,
      );
    }
    topicPapers.push(paper);
  }
  return papersByTopic;
}

function validateSummaries(
  papers: DailySummaryAssemblyPaper[],
  summaries: StructuredPaperSummary[],
): Map<string, StructuredPaperSummary> {
  const paperIds = new Set(papers.map((paper) => paper.id));
  const summariesById = new Map<string, StructuredPaperSummary>();
  for (const summary of summaries) {
    if (!paperIds.has(summary.id)) {
      throw new Error(`assembleDailySummary: unknown summary ID: ${summary.id}`);
    }
    if (summariesById.has(summary.id)) {
      throw new Error(`assembleDailySummary: duplicate summary ID: ${summary.id}`);
    }
    summariesById.set(summary.id, summary);
  }

  const missing = papers
    .map((paper) => paper.id)
    .filter((id) => !summariesById.has(id));
  if (missing.length > 0) {
    throw new Error(
      `assembleDailySummary: missing summary IDs: ${missing.join(", ")}`,
    );
  }
  return summariesById;
}

function renderPaper(
  paper: DailySummaryAssemblyPaper,
  summary: StructuredPaperSummary,
  language: SummaryLanguage,
): string {
  const detailLink =
    (paper.isDetail || paper.paperPath) && paper.detailLink?.trim()
      ? ` → ${paper.detailLink}`
      : "";
  const sourceLabel = language === "en" ? "Source sections:" : "信息来源：";
  const authorLabel = language === "en" ? "Authors" : "作者";
  const lines = [
    `### ${paper.title}${detailLink}`,
    `> ${sourceLabel} ${paper.sourceSections}`,
    `- **${authorLabel}**: ${paper.authors}`,
    `- **arXiv**: [${paper.id}](https://arxiv.org/abs/${paper.id})`,
  ];
  for (const [key, label] of FIELD_LABELS[language]) {
    lines.push(`- **${label}**: ${summary[key]}`);
  }
  return lines.join("\n");
}
