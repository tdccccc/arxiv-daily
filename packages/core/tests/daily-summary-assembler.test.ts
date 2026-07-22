import { describe, expect, it } from "vitest";
import {
  assembleDailySummary,
  type DailySummaryAssemblyInput,
  type DailySummaryAssemblyPaper,
  type StructuredPaperSummary,
} from "../src/pipeline/daily-summary-assembler";
import { extractPaperSummaries } from "../src/pipeline/daily-summary-parser";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const topics = [
  { id: "methods", name: "Methods", tag: "methods", description: "", detail: false },
  { id: "results", name: "Results", tag: "results", description: "", detail: true },
  { id: "empty", name: "Empty", tag: "empty", description: "", detail: false },
];

function paper(
  id: string,
  title: string,
  category: string,
  overrides: Partial<DailySummaryAssemblyPaper> = {},
): DailySummaryAssemblyPaper {
  return {
    id,
    title,
    authors: `${title} Author`,
    category,
    sourceSections: "Abstract, Results",
    isDetail: false,
    ...overrides,
  };
}

function summary(id: string, prefix = id): StructuredPaperSummary {
  return {
    id,
    coreProblem: `${prefix} problem`,
    keyMethod: `${prefix} method`,
    mainResult: `${prefix} result`,
    whyRelevant: `${prefix} value`,
    limitations: `${prefix} limits`,
  };
}

function input(
  overrides: Partial<DailySummaryAssemblyInput> = {},
): DailySummaryAssemblyInput {
  const papers = [
    paper("2607.00001", "First Results", "results", {
      isDetail: true,
      detailLink: "[[2607.00001]]",
    }),
    paper("2607.00002", "Methods Paper", "methods"),
    paper("2607.00003", "Second Results", "results", {
      paperPath: "arxiv-daily/papers/2607.00003.md",
      detailLink: "[2607.00003](../papers/2607.00003.md)",
    }),
  ];
  return {
    papers,
    summaries: papers.map((item) => summary(item.id)),
    dateStr: "2026-07-22",
    arxivSettings: {
      ...DEFAULT_SETTINGS.arxiv,
      categories: ["astro-ph", "cs.AI"],
      topics,
    },
    ...overrides,
  };
}

describe("assembleDailySummary", () => {
  it("renders Chinese metadata, topic/input order, links, counts, and empty topics", () => {
    const markdown = assembleDailySummary(input());

    expect(markdown).toContain("# arXiv astro-ph, cs.AI 每日追踪 2026-07-22");
    expect(markdown).toContain("共 3 篇相关论文，其中 2 篇详细收录。");
    expect(markdown.match(/^## Methods$/gm)).toHaveLength(1);
    expect(markdown.match(/^## Results$/gm)).toHaveLength(1);
    expect(markdown.match(/^## Empty$/gm)).toHaveLength(1);
    expect(markdown.indexOf("## Methods")).toBeLessThan(markdown.indexOf("## Results"));
    expect(markdown.indexOf("### First Results")).toBeLessThan(
      markdown.indexOf("### Second Results"),
    );
    expect(markdown).toContain("### First Results → [[2607.00001]]");
    expect(markdown).toContain(
      "### Second Results → [2607.00003](../papers/2607.00003.md)",
    );
    expect(markdown).toContain("### Methods Paper\n> 信息来源： Abstract, Results");
    expect(markdown).toContain("- **作者**: Methods Paper Author");
    expect(markdown).toContain(
      "- **arXiv**: [2607.00002](https://arxiv.org/abs/2607.00002)",
    );
    expect(markdown).toContain("- **研究问题**: 2607.00002 problem");
    expect(markdown).toContain("## Empty\n今日无相关论文更新。");
  });

  it("renders the English contract and omits untrusted detail links", () => {
    const base = input();
    base.papers[1] = {
      ...base.papers[1]!,
      detailLink: "[[2607.00002]]",
    };
    const markdown = assembleDailySummary({ ...base, summaryLanguage: "en" });

    expect(markdown).toContain("# arXiv astro-ph, cs.AI Daily Digest 2026-07-22");
    expect(markdown).toContain("3 relevant papers, including 2 with detail notes.");
    expect(markdown).toContain("> Source sections: Abstract, Results");
    expect(markdown).toContain("- **Authors**: Methods Paper Author");
    expect(markdown).toContain("- **Research problem**: 2607.00002 problem");
    expect(markdown).toContain("- **Scope and limits**: 2607.00002 limits");
    expect(markdown).toContain("## Empty\nNo relevant paper updates today.");
    expect(markdown).toContain("### Methods Paper\n");
    expect(markdown).not.toContain("### Methods Paper → [[2607.00002]]");
  });

  it("round-trips all five fields and trusted source sections through the parser", () => {
    const assemblyInput = input();
    const parsed = extractPaperSummaries(assembleDailySummary(assemblyInput));

    expect(Object.keys(parsed)).toEqual([
      "2607.00002",
      "2607.00001",
      "2607.00003",
    ]);
    for (const expected of assemblyInput.summaries) {
      expect(parsed[expected.id]).toEqual({
        sourceSections: "Abstract, Results",
        coreProblem: expected.coreProblem,
        keyMethod: expected.keyMethod,
        mainResult: expected.mainResult,
        whyRelevant: expected.whyRelevant,
        limitations: expected.limitations,
      });
    }
  });

  it("rejects duplicate configured topic names", () => {
    const assemblyInput = input();
    assemblyInput.arxivSettings = {
      ...assemblyInput.arxivSettings,
      topics: assemblyInput.arxivSettings.topics.map((topic, index) =>
        index === 1 ? { ...topic, name: "Methods" } : topic,
      ),
    };

    expect(() => assembleDailySummary(assemblyInput)).toThrow(
      "duplicate topic name: Methods",
    );
  });

  it.each([
    {
      name: "duplicate topic tags",
      modify: (value: DailySummaryAssemblyInput) => {
        value.arxivSettings = {
          ...value.arxivSettings,
          topics: value.arxivSettings.topics.map((topic, index) =>
            index === 1 ? { ...topic, tag: "methods" } : topic,
          ),
        };
      },
      message: "duplicate topic tag: methods",
    },
    {
      name: "duplicate input paper IDs",
      modify: (value: DailySummaryAssemblyInput) => {
        value.papers.push({ ...value.papers[0]! });
      },
      message: "duplicate input paper ID: 2607.00001",
    },
    {
      name: "duplicate summary IDs",
      modify: (value: DailySummaryAssemblyInput) => {
        value.summaries.push(summary("2607.00001", "duplicate"));
      },
      message: "duplicate summary ID: 2607.00001",
    },
    {
      name: "missing summary IDs",
      modify: (value: DailySummaryAssemblyInput) => {
        value.summaries = value.summaries.slice(0, 2);
      },
      message: "missing summary IDs: 2607.00003",
    },
    {
      name: "unknown summary IDs",
      modify: (value: DailySummaryAssemblyInput) => {
        value.summaries.push(summary("2607.99999"));
      },
      message: "unknown summary ID: 2607.99999",
    },
    {
      name: "unknown category tags",
      modify: (value: DailySummaryAssemblyInput) => {
        value.papers[0] = { ...value.papers[0]!, category: "unknown" };
      },
      message: "paper 2607.00001 has unknown category tag: unknown",
    },
  ])("rejects $name", ({ modify, message }) => {
    const assemblyInput = input();
    modify(assemblyInput);
    expect(() => assembleDailySummary(assemblyInput)).toThrow(message);
  });
});
