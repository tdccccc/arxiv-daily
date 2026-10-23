import {
  parseDailyReportPapers,
  type DailyReportPaper,
} from "../pipeline/daily-summary-parser";
import { normalizeMarkdownLine } from "../pipeline/daily-summary-rendering";
import { normalizeSummaryLanguage } from "../settings/summary-language";
import type {
  ArxivSettings,
  OutputSettings,
  SummaryLanguage,
  Topic,
} from "../settings/types";
import { arxivLinks, assembleDailyDigest } from "./digest";
import type { DailyDigest, DigestPaper, DigestStructuredFields } from "./types";

export interface DigestFromReportInput {
  /** The committed daily report, exactly as read from the vault. */
  markdown: string;
  date: string;
  arxiv: ArxivSettings;
  output: Pick<OutputSettings, "dailyDir" | "summaryLanguage">;
  dailyPath?: string;
}

/**
 * Rebuild the email digest of a daily report that is already on disk, so a run
 * that only repairs the report's index (because the run that wrote it failed
 * afterwards) can still hand its mail to delivery. Papers keep the report's
 * order and topic sections. Returns undefined when the file holds no paper the
 * report parsers can vouch for; nothing is guessed.
 */
export function digestFromDailyReport(
  input: DigestFromReportInput,
): DailyDigest | undefined {
  const reported = parseDailyReportPapers(input.markdown);
  if (reported.length === 0) return undefined;

  // Section headings are the Markdown-normalized topic names.
  const topicsByHeading = new Map<string, Topic>(
    input.arxiv.topics.map((topic) => [normalizeMarkdownLine(topic.name), topic]),
  );
  const language =
    reportLanguage(input.markdown) ??
    normalizeSummaryLanguage(input.output.summaryLanguage);
  const papers = reported.map((paper) =>
    digestPaperFromReport(paper, topicsByHeading, language),
  );
  return assembleDailyDigest(
    {
      date: input.date,
      arxiv: input.arxiv,
      output: { ...input.output, summaryLanguage: language },
      dailyPath: input.dailyPath,
    },
    papers,
  );
}

function digestPaperFromReport(
  paper: DailyReportPaper,
  topicsByHeading: ReadonlyMap<string, Topic>,
  language: SummaryLanguage,
): DigestPaper {
  const topicTag = paper.section
    ? (topicsByHeading.get(paper.section)?.tag ?? decodeNeutralizedHtml(paper.section))
    : language === "en" ? "Papers" : "论文";
  const base = {
    id: paper.id,
    title: decodeNeutralizedHtml(paper.title),
    authors: decodeNeutralizedHtml(paper.authors),
    topicTag,
    sourceSections: paper.summary?.sourceSections || undefined,
    ...arxivLinks(paper.id),
  };

  const summary = paper.summary;
  const fields: DigestStructuredFields = {
    coreProblem: decodeNeutralizedHtml(summary?.coreProblem ?? ""),
    keyMethod: decodeNeutralizedHtml(summary?.keyMethod ?? ""),
    mainResult: decodeNeutralizedHtml(summary?.mainResult ?? ""),
    whyRelevant: decodeNeutralizedHtml(summary?.whyRelevant ?? ""),
    limitations: decodeNeutralizedHtml(summary?.limitations ?? ""),
  };
  if (!paper.fallback && Object.values(fields).some(Boolean)) {
    return { ...base, kind: "structured", fields };
  }
  return {
    ...base,
    kind: "fallback",
    abstract: decodeNeutralizedHtml(paper.fallbackAbstract ?? ""),
  };
}

/** The header the report was written with; the settings may have changed since. */
function reportLanguage(markdown: string): SummaryLanguage | undefined {
  if (/^# arXiv .* Daily Digest \d{4}-\d{2}-\d{2}[\t ]*$/m.test(markdown)) return "en";
  if (/^# arXiv .* 每日追踪 \d{4}-\d{2}-\d{2}[\t ]*$/m.test(markdown)) return "zh";
  return undefined;
}

/** The report encodes the "<" of raw HTML so Markdown renders it inert; mail escapes text itself. */
function decodeNeutralizedHtml(text: string): string {
  return text.replace(/&lt;(?=[A-Za-z/!?])/g, "<");
}
