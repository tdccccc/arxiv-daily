import { describe, expect, it } from "vitest";
import { buildDailyDigest } from "../../src/delivery/digest";
import { digestFromDailyReport } from "../../src/delivery/digest-from-report";
import {
  renderEmailHtml,
  renderEmailSubject,
  renderEmailText,
} from "../../src/delivery/email-render";
import type { DailyDigest } from "../../src/delivery/types";
import { appendGenerationMetrics } from "../../src/metrics/generation";
import {
  assembleDailySummary,
  assembleEmergencyDailySummary,
  type DailyPaperSlot,
} from "../../src/pipeline/daily-summary-assembler";
import { parseDailyReportPapers } from "../../src/pipeline/daily-summary-parser";
import type { ArxivSettings, SummaryLanguage } from "../../src/settings/types";

const arxiv: ArxivSettings = {
  category: "astro-ph",
  categories: ["astro-ph.GA", "astro-ph.CO"],
  timezone: "UTC",
  topics: [
    { id: "t1", name: "Photo-z", tag: "photo-z", description: "", directions: [], detail: true },
    { id: "t2", name: "Galaxy Cluster", tag: "galaxy-cluster", description: "", directions: [], detail: false },
    { id: "t3", name: "Quiet topic", tag: "quiet", description: "", directions: [], detail: false },
  ],
};

function slot(
  id: string,
  category: string,
  title: string,
  overrides: Partial<DailyPaperSlot["paper"]> = {},
): DailyPaperSlot {
  return {
    paper: {
      id,
      title,
      authors: `${title} Author, B. Coauthor`,
      category,
      sourceSections: "Abstract, Results",
      isDetail: false,
      ...overrides,
    },
    result: {
      kind: "structured",
      summary: {
        id,
        coreProblem: `Does $z<0.1$ hold for ${id}?`,
        keyMethod: `A neural network for ${id}.`,
        mainResult: `Scatter of $\\sigma_z=0.02$ on ${id}.`,
        whyRelevant: `Useful for surveys (${id}).`,
        limitations: `Small sample in ${id}.`,
      },
    },
  };
}

const slots: DailyPaperSlot[] = [
  slot("2605.08080", "photo-z", "Photometric redshifts <b>with</b> NN", {
    isDetail: true,
    paperPath: "papers/2605.08080.md",
    detailLink: "[[2605.08080]]",
  }),
  slot("2605.08081", "photo-z", "Second photo-z paper → with an arrow"),
  slot("2605.08068", "galaxy-cluster", "Cluster counts", {
    isDetail: true,
    paperPath: "papers/2605.08068.md",
    detailLink: "[2605.08068](../papers/2605.08068.md)",
  }),
  {
    paper: {
      id: "2605.08069",
      title: "A cluster paper the model could not summarize",
      authors: "C. Author",
      category: "galaxy-cluster",
      sourceSections: "Abstract",
      isDetail: false,
    },
    result: {
      kind: "fallback",
      reasonCode: "validation-exhausted",
      attempts: 3,
      originalAbstract: "We study the original abstract of the paper.",
    },
  },
];

function report(language: SummaryLanguage, extra = ""): string {
  const body = assembleDailySummary({
    slots,
    dateStr: "2026-07-27",
    arxivSettings: arxiv,
    summaryLanguage: language,
  });
  const withFrontmatter = `---\ndate: 2026-07-27\nweekday: Monday\ntags: [arxiv, daily]\n---\n\n${body}${extra}`;
  return appendGenerationMetrics(withFrontmatter, {
    llmCalls: 5,
    llmHttpAttempts: 6,
    llmDurationMs: 1200,
    pipelineElapsedMs: 4500,
    generatedAt: "2026-07-27T08:00:00.000Z",
  } as never);
}

const input = (markdown: string, language: SummaryLanguage = "zh") => ({
  markdown,
  date: "2026-07-27",
  arxiv,
  output: { dailyDir: "arxiv-daily/daily", summaryLanguage: language },
  dailyPath: "arxiv-daily/daily/2026-07-27.md",
});

/** What the digest of a fresh run would have been, built from the same slots. */
const fresh = (language: SummaryLanguage): DailyDigest =>
  buildDailyDigest({
    date: "2026-07-27",
    arxiv,
    output: { dailyDir: "arxiv-daily/daily", summaryLanguage: language },
    slots,
    dailyPath: "arxiv-daily/daily/2026-07-27.md",
  });

/** Source sections are vault noise that mail omits; the report keeps them only for structured papers. */
function withoutSourceSections(digest: DailyDigest): DailyDigest {
  return {
    ...digest,
    topics: digest.topics.map((topic) => ({
      ...topic,
      papers: topic.papers.map(({ sourceSections: _omit, ...paper }) => paper),
    })),
  };
}

describe("digestFromDailyReport", () => {
  it.each(["zh", "en"] as const)(
    "rebuilds the digest a fresh run would have mailed from its %s report",
    (language) => {
      const rebuilt = digestFromDailyReport(input(report(language), language));

      expect(rebuilt).toBeDefined();
      expect(withoutSourceSections(rebuilt!)).toEqual(withoutSourceSections(fresh(language)));
      // The reader sees the same mail, byte for byte.
      expect(renderEmailSubject(rebuilt!)).toBe(renderEmailSubject(fresh(language)));
      expect(renderEmailHtml(rebuilt!)).toBe(renderEmailHtml(fresh(language)));
      expect(renderEmailText(rebuilt!)).toBe(renderEmailText(fresh(language)));
    },
  );

  it("keeps the report's order, configured empty topics and fallback abstract", () => {
    const rebuilt = digestFromDailyReport(input(report("en"), "en"))!;

    expect(rebuilt.paperCount).toBe(4);
    expect(rebuilt.topics.map((topic) => [topic.tag, topic.papers.map((p) => p.id)])).toEqual([
      ["photo-z", ["2605.08080", "2605.08081"]],
      ["galaxy-cluster", ["2605.08068", "2605.08069"]],
      ["quiet", []],
    ]);
    const fallback = rebuilt.topics[1]!.papers[1]!;
    expect(fallback).toMatchObject({
      kind: "fallback",
      abstract: "We study the original abstract of the paper.",
    });
    expect(fallback.fields).toBeUndefined();
  });

  it("takes titles back out of the heading's detail link and the neutralized HTML", () => {
    const rebuilt = digestFromDailyReport(input(report("zh")))!;
    const titles = rebuilt.topics.flatMap((topic) => topic.papers.map((paper) => paper.title));

    expect(titles).toEqual([
      "Photometric redshifts <b>with</b> NN",
      "Second photo-z paper → with an arrow",
      "Cluster counts",
      "A cluster paper the model could not summarize",
    ]);
  });

  it("follows the language the report was written in, not the current setting", () => {
    const rebuilt = digestFromDailyReport(input(report("en"), "zh"))!;
    expect(rebuilt.summaryLanguage).toBe("en");
    expect(renderEmailText(rebuilt)).toContain("Research problem:");
    expect(digestFromDailyReport(input(report("zh"), "en"))!.summaryLanguage).toBe("zh");
  });

  it.each(["zh", "en"] as const)("keeps a CRLF %s report's language after settings change", language => {
    const other = language === "en" ? "zh" : "en";
    const rebuilt = digestFromDailyReport(input(report(language).replace(/\n/g, "\r\n"), other))!;
    expect(rebuilt.summaryLanguage).toBe(language);
    expect(renderEmailHtml(rebuilt)).toBe(renderEmailHtml(fresh(language)));
    expect(renderEmailText(rebuilt)).toBe(renderEmailText(fresh(language)));
  });

  it("keeps a paper under a section whose topic was renamed or removed since", () => {
    const renamed = report("zh").replace("## Photo-z", "## Photometric redshift");

    const rebuilt = digestFromDailyReport(input(renamed))!;

    expect(rebuilt.paperCount).toBe(4);
    const orphan = rebuilt.topics.find((topic) => topic.tag === "Photometric redshift")!;
    expect(orphan.name).toBe("Photometric redshift");
    expect(orphan.papers.map((paper) => paper.id)).toEqual(["2605.08080", "2605.08081"]);
    expect(rebuilt.topics.find((topic) => topic.tag === "photo-z")!.papers).toEqual([]);
  });

  it("reads a degraded emergency report and a Windows-line-ended file", () => {
    const emergency = assembleEmergencyDailySummary({
      slots,
      dateStr: "2026-07-27",
      arxivSettings: arxiv,
      summaryLanguage: "zh",
    });
    expect(emergency).toContain("arxiv-daily-emergency-report");
    const rebuilt = digestFromDailyReport(input(emergency))!;
    expect(withoutSourceSections(rebuilt)).toEqual(withoutSourceSections(fresh("zh")));

    const crlf = digestFromDailyReport(input(report("zh").replace(/\n/g, "\r\n")))!;
    expect(withoutSourceSections(crlf)).toEqual(withoutSourceSections(fresh("zh")));
  });

  it("returns nothing for a file with no paper it can vouch for, instead of a made-up mail", () => {
    expect(digestFromDailyReport(input(""))).toBeUndefined();
    expect(digestFromDailyReport(input("# arXiv astro-ph 每日追踪 2026-07-27\n\n今日未发现相关论文。\n"))).toBeUndefined();
    // A heading without any identity (no marker, arXiv link or legacy ID) is not a paper.
    expect(digestFromDailyReport(input("## Topic\n\n### Some note the user wrote\n\nText.\n"))).toBeUndefined();
  });

  it("reads papers of a plain legacy layout without sections", () => {
    const legacy = [
      "### Legacy paper",
      "- **arXiv**: [2609.00001](https://arxiv.org/abs/2609.00001)",
      "- **核心结果**: Recorded result.",
    ].join("\n");

    const rebuilt = digestFromDailyReport(input(legacy))!;

    expect(rebuilt.paperCount).toBe(1);
    const only = rebuilt.topics.find((topic) => topic.papers.length > 0)!;
    expect(only.name).toBe("论文");
    expect(only.papers[0]).toMatchObject({
      id: "2609.00001",
      title: "Legacy paper",
      absUrl: "https://arxiv.org/abs/2609.00001",
      pdfUrl: "https://arxiv.org/pdf/2609.00001",
      kind: "structured",
    });
    expect(only.papers[0]!.fields!.mainResult).toBe("Recorded result.");
  });

  it("parses a long, hostile report in linear time", () => {
    const hostileHeading = `### ${" ".repeat(50_000)}→${" ".repeat(50_000)}x\n- **arXiv**: [2609.00002](https://arxiv.org/abs/2609.00002)\n`;
    const markdown = `## Topic\n\n${hostileHeading.repeat(20)}`;
    const start = performance.now();
    const papers = parseDailyReportPapers(markdown);
    expect(performance.now() - start).toBeLessThan(500);
    expect(papers.length).toBe(20);
  });
});
