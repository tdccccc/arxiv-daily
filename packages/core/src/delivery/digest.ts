import type { DailyPaperSlot } from "../pipeline/daily-summary-assembler";
import { formatArxivCategories } from "../settings/categories";
import { normalizeSummaryLanguage } from "../settings/summary-language";
import type { ArxivSettings, OutputSettings, SummaryLanguage } from "../settings/types";
import { modernArxivResources } from "../utils/arxiv";
import type {
  DailyDigest,
  DigestPaper,
  DigestTopic,
} from "./types";

export interface BuildDailyDigestInput {
  date: string;
  arxiv: ArxivSettings;
  output: Pick<OutputSettings, "dailyDir" | "summaryLanguage">;
  /** Structured assembly slots when papers were summarized. */
  slots?: DailyPaperSlot[];
  /** Override vault daily path (defaults to dailyDir/date.md). */
  dailyPath?: string;
}

export function buildDailyDigest(input: BuildDailyDigestInput): DailyDigest {
  return assembleDailyDigest(input, (input.slots ?? []).map(digestPaperFromSlot));
}

/**
 * Group digest papers under the configured topics (empty topics are kept, and a
 * topic tag nobody configured still gets a section). Shared by digests built
 * from assembly slots and digests read back from a committed report.
 */
export function assembleDailyDigest(
  input: Omit<BuildDailyDigestInput, "slots">,
  papers: DigestPaper[],
): DailyDigest {
  const language = normalizeSummaryLanguage(input.output.summaryLanguage);
  const dailyPath =
    input.dailyPath?.trim() ||
    joinVaultPath(input.output.dailyDir, `${input.date}.md`);

  const papersByTopic = new Map<string, DigestPaper[]>();
  for (const topic of input.arxiv.topics) {
    papersByTopic.set(topic.tag, []);
  }

  for (const paper of papers) {
    const list = papersByTopic.get(paper.topicTag);
    if (list) {
      list.push(paper);
    } else {
      // Unknown topic tags still appear so body is not silently dropped.
      papersByTopic.set(paper.topicTag, [paper]);
    }
  }

  const topics: DigestTopic[] = input.arxiv.topics.map((topic) => ({
    tag: topic.tag,
    name: topic.name,
    papers: papersByTopic.get(topic.tag) ?? [],
  }));

  // Include any papers whose category was not in configured topics.
  for (const [tag, topicPapers] of papersByTopic) {
    if (topics.some((t) => t.tag === tag)) continue;
    topics.push({ tag, name: tag, papers: topicPapers });
  }

  const paperCount = topics.reduce((n, topic) => n + topic.papers.length, 0);

  return {
    date: input.date,
    summaryLanguage: language,
    categories: formatArxivCategories(input.arxiv),
    dailyPath,
    paperCount,
    topics,
  };
}

export function emptyDailyDigest(input: {
  date: string;
  arxiv: ArxivSettings;
  output: Pick<OutputSettings, "dailyDir" | "summaryLanguage">;
  dailyPath?: string;
}): DailyDigest {
  return buildDailyDigest({ ...input, slots: [] });
}

/** abs and PDF links of a paper, canonical when the ID is a modern arXiv one. */
export function arxivLinks(id: string): { absUrl: string; pdfUrl: string } {
  const resources = modernArxivResources(id);
  return {
    absUrl: resources?.absUrl ?? `https://arxiv.org/abs/${id}`,
    pdfUrl: resources?.pdfUrl ?? `https://arxiv.org/pdf/${id}`,
  };
}

function digestPaperFromSlot(slot: DailyPaperSlot): DigestPaper {
  const sourceSections = slot.paper.sourceSections?.trim() || undefined;
  const base = {
    id: slot.paper.id,
    title: slot.paper.title,
    authors: slot.paper.authors,
    topicTag: slot.paper.category,
    sourceSections,
    ...arxivLinks(slot.paper.id),
  };

  if (slot.result.kind === "structured") {
    const s = slot.result.summary;
    return {
      ...base,
      kind: "structured",
      fields: {
        coreProblem: s.coreProblem,
        keyMethod: s.keyMethod,
        mainResult: s.mainResult,
        whyRelevant: s.whyRelevant,
        limitations: s.limitations,
      },
    };
  }

  return {
    ...base,
    kind: "fallback",
    abstract: slot.result.originalAbstract,
  };
}

function joinVaultPath(dir: string, file: string): string {
  // Scan the trailing-slash boundary instead of /\/+$/, which forces the
  // regex engine to retry at every "/" in a long run of repeated slashes
  // (CodeQL js/polynomial-redos).
  let end = dir.length;
  while (end > 0 && dir[end - 1] === "/") end -= 1;
  const trimmed = dir.slice(0, end);
  return trimmed ? `${trimmed}/${file}` : file;
}

export function digestLanguage(digest: DailyDigest): SummaryLanguage {
  return normalizeSummaryLanguage(digest.summaryLanguage);
}
