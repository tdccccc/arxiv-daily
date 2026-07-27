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
  const language = normalizeSummaryLanguage(input.output.summaryLanguage);
  const slots = input.slots ?? [];
  const dailyPath =
    input.dailyPath?.trim() ||
    joinVaultPath(input.output.dailyDir, `${input.date}.md`);

  const papersByTopic = new Map<string, DigestPaper[]>();
  for (const topic of input.arxiv.topics) {
    papersByTopic.set(topic.tag, []);
  }

  for (const slot of slots) {
    const paper = digestPaperFromSlot(slot);
    const list = papersByTopic.get(slot.paper.category);
    if (list) {
      list.push(paper);
    } else {
      // Unknown topic tags still appear so body is not silently dropped.
      papersByTopic.set(slot.paper.category, [paper]);
    }
  }

  const topics: DigestTopic[] = input.arxiv.topics.map((topic) => ({
    tag: topic.tag,
    name: topic.name,
    papers: papersByTopic.get(topic.tag) ?? [],
  }));

  // Include any papers whose category was not in configured topics.
  for (const [tag, papers] of papersByTopic) {
    if (topics.some((t) => t.tag === tag)) continue;
    topics.push({ tag, name: tag, papers });
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

function digestPaperFromSlot(slot: DailyPaperSlot): DigestPaper {
  const resources = modernArxivResources(slot.paper.id);
  const absUrl = resources?.absUrl ?? `https://arxiv.org/abs/${slot.paper.id}`;
  const pdfUrl = resources?.pdfUrl ?? `https://arxiv.org/pdf/${slot.paper.id}`;
  const sourceSections = slot.paper.sourceSections?.trim() || undefined;
  const base = {
    id: slot.paper.id,
    title: slot.paper.title,
    authors: slot.paper.authors,
    topicTag: slot.paper.category,
    sourceSections,
    absUrl,
    pdfUrl,
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
  const trimmed = dir.replace(/\/+$/, "");
  return trimmed ? `${trimmed}/${file}` : file;
}

export function digestLanguage(digest: DailyDigest): SummaryLanguage {
  return normalizeSummaryLanguage(digest.summaryLanguage);
}
