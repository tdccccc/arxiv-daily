import { describe, expect, it } from "vitest";
import {
  assembleDailySummary,
  assembleEmergencyDailySummary,
  DailySummaryAssemblyRuntimeError,
  type DailyPaperSlot,
  type DailySummaryAssemblyInput,
  type DailySummaryAssemblyPaper,
  preflightDailySummaryAssembly,
  type StructuredPaperSummary,
} from "../src/pipeline/daily-summary-assembler";
import {
  extractFallbackAbstracts,
  extractFallbackPaperIds,
  extractPaperSummaries,
  hasEmergencyDailySummaryMarker,
} from "../src/pipeline/daily-summary-parser";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeMarkdownLine } from "../src/pipeline/daily-summary-rendering";

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

function structuredSlot(paperValue: DailySummaryAssemblyPaper): DailyPaperSlot {
  return {
    paper: paperValue,
    result: { kind: "structured", summary: summary(paperValue.id) },
  };
}

function fallbackSlot(
  paperValue: DailySummaryAssemblyPaper,
  originalAbstract: string,
  reasonCode: "validation-exhausted" | "transport-exhausted" = "validation-exhausted",
  attempts = 3,
): DailyPaperSlot {
  return {
    paper: paperValue,
    result: {
      kind: "fallback",
      reasonCode,
      attempts,
      originalAbstract,
    },
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
    slots: papers.map(structuredSlot),
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
  it("preserves structured Chinese metadata, topic/input order, links, and counts", () => {
    const markdown = assembleDailySummary(input());

    expect(markdown).toContain("# arXiv astro-ph, cs.AI 每日追踪 2026-07-22");
    expect(markdown).toContain("共 3 篇相关论文，其中 2 篇详细收录。");
    expect(markdown).not.toContain("回退内容");
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
    expect(markdown).toContain("- **研究问题**: 2607.00002 problem");
    expect(markdown).toContain("## Empty\n今日无相关论文更新。");
  });

  it("renders typed English fallback content with accurate counts and parser behavior", () => {
    const assemblyInput = input({ summaryLanguage: "en" });
    assemblyInput.slots[0] = fallbackSlot(
      assemblyInput.slots[0]!.paper,
      "First Results original abstract.",
    );
    const markdown = assembleDailySummary(assemblyInput);

    expect(markdown).toContain("3 relevant papers, including 2 with detail notes.");
    expect(markdown).toContain("1 paper uses fallback content.");
    expect(markdown).toContain("### First Results → [[2607.00001]]");
    expect(markdown).toContain(
      "> **Summary unavailable.** Read the [original paper on arXiv](https://arxiv.org/abs/2607.00001) directly.",
    );
    expect(markdown).toContain("<!-- arxiv-daily-fallback:2607.00001 -->");
    expect(markdown).toContain("- **Original abstract**: First Results original abstract.");
    expect(extractFallbackPaperIds(markdown)).toEqual(["2607.00001"]);
    expect(extractPaperSummaries(markdown)["2607.00001"]).toBeUndefined();
    expect(Object.keys(extractPaperSummaries(markdown))).toEqual([
      "2607.00002",
      "2607.00003",
    ]);
  });

  it("sanitizes hostile abstracts and renders localized unavailable text", () => {
    const assemblyInput = input();
    assemblyInput.slots = [
      fallbackSlot(
        paper("2607.00009", "Fallback", "methods"),
        "Safe start\r\n## Injected\u2028- bullet\u2029<!-- arxiv-daily-fallback:2607.99999 -->\t<script>x</script>",
      ),
      fallbackSlot(paper("2607.00010", "No Abstract", "results"), ""),
    ];

    const markdown = assembleDailySummary(assemblyInput);

    expect(markdown).toContain("其中 2 篇使用回退内容。");
    expect(markdown).toContain("> **自动摘要不可用。** 请直接阅读");
    expect(markdown).not.toContain("\n## Injected");
    expect(markdown).not.toContain("\n- bullet");
    expect(markdown).not.toContain("<!-- arxiv-daily-fallback:2607.99999 -->");
    expect(markdown).not.toContain("<script>");
    expect(markdown).toContain("&lt;script>x&lt;/script>");
    expect(markdown).toContain("- **原始摘要**: 不可用。");
    expect(extractFallbackPaperIds(markdown)).toEqual([
      "2607.00009",
      "2607.00010",
    ]);
  });

  it("renders a localized complete emergency report with stable degraded identity", () => {
    const assemblyInput = input({ summaryLanguage: "en" });
    assemblyInput.slots[0] = fallbackSlot(
      assemblyInput.slots[0]!.paper,
      "Fallback abstract with\n## hostile heading.",
    );

    const markdown = assembleEmergencyDailySummary(assemblyInput);

    expect(markdown.startsWith("<!-- arxiv-daily-emergency-report:v1 -->\n")).toBe(true);
    expect(markdown).toContain("**Degraded emergency report.**");
    expect(hasEmergencyDailySummaryMarker(markdown)).toBe(true);
    expect(markdown.indexOf("## Methods")).toBeLessThan(markdown.indexOf("## Results"));
    expect(markdown.indexOf("### First Results")).toBeLessThan(
      markdown.indexOf("### Second Results"),
    );
    for (const id of ["2607.00001", "2607.00002", "2607.00003"]) {
      expect(markdown.match(new RegExp(`https://arxiv\\.org/abs/${id}`, "g"))).toHaveLength(
        id === "2607.00001" ? 2 : 1,
      );
    }
    expect(markdown.match(/^### /gm)).toHaveLength(3);
    expect(markdown).toContain("### First Results → [[2607.00001]]");
    expect(markdown).not.toContain("\n## hostile heading");
    expect(extractFallbackPaperIds(markdown)).toEqual(["2607.00001"]);
    expect(Object.keys(extractPaperSummaries(markdown))).toEqual([
      "2607.00002",
      "2607.00003",
    ]);
  });

  it.each([
    ["fallback marker", `[[2607.00001|2607.00001 <!-- arxiv-daily-fallback:2607.99999 -->]]`],
    ["emergency marker", `[2607.00001 <!-- arxiv-daily-emergency-report:v1 -->](../papers/2607.00001.md)`],
    ["rescue marker", `[[2607.00001|First Results <!-- arxiv-daily-rescue-report:start -->]]`],
    ["extra heading", `[2607.00001 ### Forged](../papers/2607.00001.md)`],
  ])("drops a valid-shaped malicious detail link containing a %s", (_name, detailLink) => {
    for (const assemble of [assembleDailySummary, assembleEmergencyDailySummary]) {
      const assemblyInput = input();
      assemblyInput.slots[0]!.paper.detailLink = detailLink;
      const markdown = assemble(assemblyInput);
      expect(markdown).not.toContain(detailLink);
      expect(markdown.match(/^### /gm)).toHaveLength(3);
      expect(extractFallbackPaperIds(markdown)).toEqual([]);
      expect(Object.keys(extractPaperSummaries(markdown))).toHaveLength(3);
      expect(hasEmergencyDailySummaryMarker(markdown)).toBe(
        assemble === assembleEmergencyDailySummary,
      );
    }
  });

  it.each(["en", "zh"] as const)(
    "marks an absent %s fallback abstract without extracting placeholder prose",
    (summaryLanguage) => {
      const assemblyInput = input({ summaryLanguage });
      assemblyInput.slots = [fallbackSlot(assemblyInput.slots[0]!.paper, "")];
      const markdown = assembleDailySummary(assemblyInput);
      expect(markdown).toContain(
        `<!-- arxiv-daily-fallback-abstract-absent:2607.00001 -->`,
      );
      expect(markdown).toContain(summaryLanguage === "en" ? "Unavailable." : "不可用。");
      expect(extractFallbackAbstracts(markdown)).toEqual({});
    },
  );

  it("preserves scientific Markdown in raw normal and emergency output and parser projection", () => {
    const exact = String.raw`For $z<0.1$, \(E = mc^2 + \alpha_i^{2}\), A & B | C is 50% and z>3.5.`;
    const assemblyInput = input();
    assemblyInput.slots = [
      structuredSlot(paper("2607.00001", "Exact", "methods", {
        sourceSections: exact,
      })),
      fallbackSlot(paper("2607.00002", "Fallback", "results"), exact),
    ];
    if (assemblyInput.slots[0]!.result.kind === "structured") {
      for (const key of ["coreProblem", "keyMethod", "mainResult", "whyRelevant", "limitations"] as const) {
        assemblyInput.slots[0]!.result.summary[key] = exact;
      }
    }
    for (const assemble of [assembleDailySummary, assembleEmergencyDailySummary]) {
      const markdown = assemble(assemblyInput);
      expect(markdown).toContain(`> 信息来源： ${exact}`);
      expect(markdown).toContain(`- **研究问题**: ${exact}`);
      expect(markdown).toContain(`- **原始摘要**: ${exact}`);
      expect(extractPaperSummaries(markdown)["2607.00001"]).toEqual({
        sourceSections: exact,
        coreProblem: exact,
        keyMethod: exact,
        mainResult: exact,
        whyRelevant: exact,
        limitations: exact,
      });
      expect(extractFallbackAbstracts(markdown)).toEqual({ "2607.00002": exact });
    }
  });

  it("normalizes CommonMark autolinks, math, code, and hostile HTML to exact output", () => {
    expect(normalizeMarkdownLine(
      "  [paper](url)  *emphasis*  `code <b>x</b>`  <https://arxiv.org/a> <mailto:user@example.org> <user@example.org> z<0.1/z>3.5 A & B <x =bad> <b class='x'>bold</b> <!-- c --> <!DOCTYPE html> <?pi?> <![CDATA[x]]>  ",
    )).toBe(
      "[paper](url) *emphasis* `code <b>x</b>` <https://arxiv.org/a> <mailto:user@example.org> <user@example.org> z<0.1/z>3.5 A & B <x =bad> &lt;b class='x'>bold&lt;/b> &lt;!-- c --> &lt;!DOCTYPE html> &lt;?pi?> &lt;![CDATA[x]]>",
    );
  });

  it("matches only equal-length maximal backtick runs and treats backslashes literally", () => {
    const cases = [
      ["``code``` then <b>outside</b>`", "``code``` then &lt;b>outside&lt;/b>`"],
      ["``shorter ` run and <b>inside</b>`` <i>outside</i>", "``shorter ` run and <b>inside</b>`` &lt;i>outside&lt;/i>"],
      ["`unmatched <u>outside</u>", "`unmatched &lt;u>outside&lt;/u>"],
      [String.raw`\`<b>inside</b>\` <i>outside</i>`, String.raw`\`<b>inside</b>\` &lt;i>outside&lt;/i>`],
      ["`<b>one</b>` ``<i>two</i>`` ```<u>three</u>``` <em>outside</em>", "`<b>one</b>` ``<i>two</i>`` ```<u>three</u>``` &lt;em>outside&lt;/em>"],
    ] as const;
    for (const [source, expected] of cases) {
      expect(normalizeMarkdownLine(source)).toBe(expected);
    }
    for (let length = 1; length <= 8; length += 1) {
      const delimiter = "`".repeat(length);
      expect(normalizeMarkdownLine(
        `${delimiter}<mark>code-${length}</mark>${delimiter} <b>outside</b>`,
      )).toBe(
        `${delimiter}<mark>code-${length}</mark>${delimiter} &lt;b>outside&lt;/b>`,
      );
    }
  });

  it("keeps code-span HTML exact but neutralizes adversarial outside HTML in normal and emergency raw output", () => {
    const reproducer = "``code``` then <b>outside</b>`";
    const shorterRun = "``shorter ` run and <i>inside</i>`` <u>outside</u>";
    const backslashDelimited = "\\`<mark>inside</mark>\\` <em>outside</em>";
    const unmatched = "`unmatched <strong>outside</strong>";
    const assemblyInput = input({ summaryLanguage: "en" });
    assemblyInput.slots = [fallbackSlot(
      paper("2607.00001", reproducer, "methods", {
        sourceSections: shorterRun,
        authors: backslashDelimited,
      }),
      unmatched,
    )];

    for (const assemble of [assembleDailySummary, assembleEmergencyDailySummary]) {
      const markdown = assemble(assemblyInput);
      expect(markdown).toContain("### ``code``` then &lt;b>outside&lt;/b>`");
      expect(markdown).toContain("> Source sections: ``shorter ` run and <i>inside</i>`` &lt;u>outside&lt;/u>");
      expect(markdown).toContain("- **Authors**: \\`<mark>inside</mark>\\` &lt;em>outside&lt;/em>");
      expect(markdown).toContain("- **Original abstract**: `unmatched &lt;strong>outside&lt;/strong>");
      expect(extractFallbackAbstracts(markdown)).toEqual({
        "2607.00001": "`unmatched &lt;strong>outside&lt;/strong>",
      });
    }
  });

  it("preserves CommonMark inline forms while neutralizing exact raw HTML constructs", () => {
    const inline = "[paper](https://arxiv.org/abs/2607.00001) *emphasis* \\(x<y & y>z\\) <https://arxiv.org/abs/2607.00001> <mailto:user@example.org> <user@example.org> z<0.1/z>3.5 `<b>code</b>`";
    const hostile = `${inline} <script data-x="1 > 0">alert(1)</script> <!-- comment --> <!DOCTYPE html> <?target value?> <![CDATA[raw <x>]]>`;
    const assemblyInput = input({ summaryLanguage: "en" });
    assemblyInput.slots = [fallbackSlot(
      paper("2607.00001", "Exact", "methods"),
      hostile,
    )];

    for (const assemble of [assembleDailySummary, assembleEmergencyDailySummary]) {
      const markdown = assemble(assemblyInput);
      expect(markdown).toContain(`- **Original abstract**: ${inline} &lt;script data-x="1 > 0">alert(1)&lt;/script> &lt;!-- comment --> &lt;!DOCTYPE html> &lt;?target value?> &lt;![CDATA[raw <x>]]>`);
      expect(markdown).not.toContain("<script");
      expect(markdown).not.toContain("<!-- comment -->");
      expect(extractFallbackAbstracts(markdown)).toEqual({
        "2607.00001": `${inline} &lt;script data-x="1 > 0">alert(1)&lt;/script> &lt;!-- comment --> &lt;!DOCTYPE html> &lt;?target value?> &lt;![CDATA[raw <x>]]>`,
      });
    }
  });

  it("wraps only post-preflight rendering failures in the typed runtime error", () => {
    const render = () => {
      throw new Error("unexpected renderer failure");
    };
    expect(() => assembleDailySummary(input(), { render })).toThrow(
      DailySummaryAssemblyRuntimeError,
    );

    const invalid = input();
    invalid.slots[0] = {
      ...invalid.slots[0]!,
      paper: { ...invalid.slots[0]!.paper, category: "unknown" },
    };
    expect(() => assembleDailySummary(invalid, { render })).toThrow(
      "unknown category tag",
    );
  });

  it("preserves accepted canonical math through deterministic and emergency parser projection", () => {
    const canonical = String.raw`The constraints are $z<0.1$ and $E=mc^2$, with $\alpha_i^2$.`;
    const assemblyInput = input();
    assemblyInput.slots = [structuredSlot(paper("2607.00001", "Canonical", "methods"))];
    if (assemblyInput.slots[0]!.result.kind === "structured") {
      for (const key of ["coreProblem", "keyMethod", "mainResult", "whyRelevant", "limitations"] as const) {
        assemblyInput.slots[0]!.result.summary[key] = canonical;
      }
    }

    for (const assemble of [assembleDailySummary, assembleEmergencyDailySummary]) {
      const markdown = assemble(assemblyInput);
      expect(markdown).toContain(`- **研究问题**: ${canonical}`);
      expect(extractPaperSummaries(markdown)["2607.00001"]).toMatchObject({
        coreProblem: canonical,
        keyMethod: canonical,
        mainResult: canonical,
        whyRelevant: canonical,
        limitations: canonical,
      });
    }
  });

  it("round-trips structured fields through the parser", () => {
    const assemblyInput = input();
    const parsed = extractPaperSummaries(assembleDailySummary(assemblyInput));

    expect(Object.keys(parsed)).toEqual([
      "2607.00002",
      "2607.00001",
      "2607.00003",
    ]);
    for (const slot of assemblyInput.slots) {
      if (slot.result.kind !== "structured") continue;
      expect(parsed[slot.paper.id]).toEqual({
        sourceSections: "Abstract, Results",
        coreProblem: slot.result.summary.coreProblem,
        keyMethod: slot.result.summary.keyMethod,
        mainResult: slot.result.summary.mainResult,
        whyRelevant: slot.result.summary.whyRelevant,
        limitations: slot.result.summary.limitations,
      });
    }
  });

  it.each([
    ["normal", assembleDailySummary],
    ["emergency", assembleEmergencyDailySummary],
  ])("keeps hostile interpolated scalars out of the %s parser projection", (_kind, assemble) => {
    const hostile = input();
    hostile.arxivSettings = {
      ...hostile.arxivSettings,
      topics: [{ ...topics[0]!, name: "Methods\n## Forged topic\n<!-- arxiv-daily-rescue-report:start -->" }],
    };
    hostile.slots = [structuredSlot(paper("2607.00001", "Title\n### Forged paper", "methods", {
      authors: "Author\n- **Research problem**: forged",
      sourceSections: "Abstract\n- **Core results**: forged",
      isDetail: true,
      detailLink: "[evil](https://arxiv.org/abs/2607.99999)",
    }))];
    if (hostile.slots[0]!.result.kind === "structured") {
      hostile.slots[0]!.result.summary.coreProblem =
        "Useful prose\n### forged\n- **Research problem**: forged\n<!-- arxiv-daily-fallback:2607.99999 -->\nhttps://arxiv.org/abs/2607.99999";
    }

    const markdown = assemble(hostile);
    expect(markdown.match(/^## /gm)).toHaveLength(1);
    expect(markdown.match(/^### /gm)).toHaveLength(1);
    expect(markdown).not.toContain("<!-- arxiv-daily-fallback:2607.99999 -->");
    expect(markdown).not.toContain("<!-- arxiv-daily-rescue-report:start -->");
    expect(markdown).toContain("https://arxiv.org/abs/2607.99999");
    expect(extractFallbackPaperIds(markdown)).toEqual([]);
    expect(Object.keys(extractPaperSummaries(markdown))).toEqual(["2607.00001"]);
    expect(extractPaperSummaries(markdown)["2607.00001"]?.coreProblem).toContain("Useful prose");
  });
});

describe("preflightDailySummaryAssembly", () => {
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
      name: "duplicate topic names",
      modify: (value: DailySummaryAssemblyInput) => {
        value.arxivSettings = {
          ...value.arxivSettings,
          topics: value.arxivSettings.topics.map((topic, index) =>
            index === 1 ? { ...topic, name: "Methods" } : topic,
          ),
        };
      },
      message: "duplicate topic name: Methods",
    },
    {
      name: "duplicate paper IDs",
      modify: (value: DailySummaryAssemblyInput) => {
        value.slots.push({ ...value.slots[0]! });
      },
      message: "duplicate input paper ID: 2607.00001",
    },
    {
      name: "unknown category tags",
      modify: (value: DailySummaryAssemblyInput) => {
        value.slots[0] = {
          ...value.slots[0]!,
          paper: { ...value.slots[0]!.paper, category: "unknown" },
        };
      },
      message: "paper 2607.00001 has unknown category tag: unknown",
    },
    {
      name: "missing trusted titles",
      modify: (value: DailySummaryAssemblyInput) => {
        value.slots[0] = {
          ...value.slots[0]!,
          paper: { ...value.slots[0]!.paper, title: "" },
        };
      },
      message: "missing trusted metadata: paper 2607.00001 title",
    },
    {
      name: "mismatched structured IDs",
      modify: (value: DailySummaryAssemblyInput) => {
        value.slots[0] = {
          ...value.slots[0]!,
          result: { kind: "structured", summary: summary("2607.99999") },
        };
      },
      message: "summary ID 2607.99999 does not match paper ID: 2607.00001",
    },
    {
      name: "fallback attempts above the logical-call limit",
      modify: (value: DailySummaryAssemblyInput) => {
        value.slots[0] = fallbackSlot(
          value.slots[0]!.paper,
          "trusted abstract",
          "validation-exhausted",
          4,
        );
      },
      message: "paper 2607.00001 has invalid fallback attempts",
    },
  ])("rejects $name before assembly", ({ modify, message }) => {
    const assemblyInput = input();
    modify(assemblyInput);
    expect(() => preflightDailySummaryAssembly(assemblyInput)).toThrow(message);
    expect(() => assembleDailySummary(assemblyInput)).toThrow(message);
  });
});
