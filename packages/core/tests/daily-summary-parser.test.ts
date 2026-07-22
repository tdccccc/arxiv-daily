import { describe, expect, it } from "vitest";
import {
  extractFallbackAbstracts,
  extractFallbackPaperIds,
  extractPaperSummaries,
  hasEmergencyDailySummaryMarker,
} from "../src/pipeline/daily-summary-parser";

describe("extractPaperSummaries", () => {
  it("extracts only the exact standalone stable emergency marker", () => {
    expect(
      hasEmergencyDailySummaryMarker(
        "<!-- arxiv-daily-emergency-report:v1 -->\n# report",
      ),
    ).toBe(true);
    for (const inline of [
      "arxiv-daily-emergency-report:v1",
      "prose <!-- arxiv-daily-emergency-report:v1 -->",
      "<!-- arxiv-daily-emergency-report:v1 --> prose",
    ]) {
      expect(hasEmergencyDailySummaryMarker(inline)).toBe(false);
    }
  });

  it("extracts structured summary fields from daily markdown", () => {
    const summaries = extractPaperSummaries(
      [
        "## Photo-z",
        "### Example Paper → [[2606.12345]]",
        "> 信息来源：Abstract, Conclusion",
        "- **作者**: A. Author et al.",
        "- **arXiv**: [2606.12345](https://arxiv.org/abs/2606.12345)",
        "- [x] 关注 <!-- arxiv-daily:2606.12345:watch -->",
        "- **核心问题**: What problem.",
        "- **关键方法**: Main method.",
        "- **主要结果**: Main result.",
        "- **为什么值得看**: Why relevant.",
        "- **局限或边界**: Boundary.",
        "",
        "### Another Paper",
        "- **arXiv**: [2606.54321](https://arxiv.org/abs/2606.54321v2)",
        "- **核心问题**: Other problem.",
      ].join("\n"),
    );

    expect(summaries["2606.12345"]).toEqual({
      sourceSections: "Abstract, Conclusion",
      coreProblem: "What problem.",
      keyMethod: "Main method.",
      mainResult: "Main result.",
      whyRelevant: "Why relevant.",
      limitations: "Boundary.",
    });
    expect(summaries["2606.54321"]).toEqual({
      coreProblem: "Other problem.",
    });
  });

  it("restores legacy H3 heading identity without accepting bracket-ID summary prose", () => {
    const summaries = extractPaperSummaries([
      "## Topic",
      "### Paper [2607.12345] — historical suffix",
      "- **Research problem**: historical heading identity",
      "### Arrow [2607.12346] → [[2607.12346]]",
      "- **Research problem**: arrow suffix identity",
      "### No heading identity",
      "- **Research problem**: prose contains [2607.99999], but must not identify the block",
    ].join("\n"));

    expect(summaries).toEqual({
      "2607.12345": { coreProblem: "historical heading identity" },
      "2607.12346": { coreProblem: "arrow suffix identity" },
    });
  });

  it("keeps canonical bullet and selection identity ahead of a conflicting legacy heading", () => {
    const markdown = [
      "### Conflicting [2607.99999]",
      "- [x] Watch <!--  arxiv-daily:2607.11111:selection:watch  -->\r",
      "- **arXiv**: [2607.22222](https://arxiv.org/abs/2607.22222)",
      "- **Research problem**: selected identity wins",
      "### Bullet [2607.99998]",
      "- **arXiv**: [2607.33333](https://arxiv.org/abs/2607.33333)",
      "- **Research problem**: bullet identity wins",
    ].join("\n");
    expect(extractPaperSummaries(markdown)).toEqual({
      "2607.11111": { coreProblem: "selected identity wins" },
      "2607.33333": { coreProblem: "bullet identity wins" },
    });
  });

  it("recognizes every historical selection-marker namespace and kind as block identity", () => {
    const markdown = [
      ["2607.12001", "watch", "<!--arxiv-daily:2607.12001:watch-->"],
      ["2607.12002", "highlight", "<!--  arxiv-daily:2607.12002:highlight  -->"],
      ["2607.12003", "selection watch", "<!--\tarxiv-daily:2607.12003:selection:watch\t-->"],
      ["2607.12004", "selection highlight", "<!-- arxiv-daily:2607.12004:selection:highlight -->"],
    ].flatMap(([id, label, marker]) => [
      `### ${label}`,
      `- [x] ${label} ${marker}\r`,
      `- **Research problem**: ${id}`,
    ]).join("\n");

    expect(extractPaperSummaries(markdown)).toEqual({
      "2607.12001": { coreProblem: "2607.12001" },
      "2607.12002": { coreProblem: "2607.12002" },
      "2607.12003": { coreProblem: "2607.12003" },
      "2607.12004": { coreProblem: "2607.12004" },
    });
  });

  it("does not classify inline marker-like prose as fallback, absent, or watch identity", () => {
    const markdown = [
      "## Topic",
      "### Inline prose",
      "> Source sections: Abstract",
      "- **arXiv**: [2606.40001](https://arxiv.org/abs/2606.40001)",
      "- **Research problem**: prose <!-- arxiv-daily-fallback:2606.40001 -->",
      "- **Original abstract**: prose <!-- arxiv-daily-fallback-abstract-absent:2606.40001 -->",
      "### Marker-only identity",
      "- **Research problem**: prose <!-- arxiv-daily:2606.49999:watch -->",
    ].join("\n");

    expect(extractFallbackPaperIds(markdown)).toEqual([]);
    expect(extractFallbackAbstracts(markdown)).toEqual({});
    expect(extractPaperSummaries(markdown)).toEqual({
      "2606.40001": {
        sourceSections: "Abstract",
        coreProblem: "prose <!-- arxiv-daily-fallback:2606.40001 -->",
      },
    });
  });

  it("skips fallback blocks and extracts stable fallback IDs", () => {
    const markdown = [
      "## Topic",
      "### Fallback Paper",
      "<!-- arxiv-daily-fallback:2606.33333 -->",
      "<!-- arxiv-daily-fallback:2606.33333 -->",
      "- **arXiv**: [2606.33333](https://arxiv.org/abs/2606.33333)",
      "- **研究问题**: Must not be indexed.",
      "",
      "### Structured Paper",
      "- **arXiv**: [2606.44444](https://arxiv.org/abs/2606.44444)",
      "- **研究问题**: Keep this.",
    ].join("\n");

    expect(extractFallbackPaperIds(markdown)).toEqual(["2606.33333"]);
    expect(extractPaperSummaries(markdown)).toEqual({
      "2606.44444": { coreProblem: "Keep this." },
    });
  });

  it("omits marker-confirmed absent abstracts while preserving similar real prose", () => {
    const markdown = [
      "## Topic",
      "### English Missing",
      "<!-- arxiv-daily-fallback:2606.50001 -->",
      "<!-- arxiv-daily-fallback-abstract-absent:2606.50001 -->",
      "- **arXiv**: [2606.50001](https://arxiv.org/abs/2606.50001)",
      "- **Original abstract**: Unavailable.",
      "### Chinese Missing",
      "<!-- arxiv-daily-fallback:2606.50002 -->",
      "<!-- arxiv-daily-fallback-abstract-absent:2606.50002 -->",
      "- **arXiv**: [2606.50002](https://arxiv.org/abs/2606.50002)",
      "- **原始摘要**: 不可用。",
      "### Real Abstract",
      "<!-- arxiv-daily-fallback:2606.50003 -->",
      "- **arXiv**: [2606.50003](https://arxiv.org/abs/2606.50003)",
      "- **Original abstract**: Availability is unavailable in one setting, but this is real prose.",
    ].join("\n");

    expect(extractFallbackAbstracts(markdown)).toEqual({
      "2606.50003": "Availability is unavailable in one setting, but this is real prose.",
    });
  });

  it("projects canonical inline math byte-for-byte without parser normalization", () => {
    const canonical = String.raw`Constraint $z<0.1$ with $\alpha_i^2$ and $E=mc^2$.`;
    const summaries = extractPaperSummaries([
      "### Canonical math",
      "- **arXiv**: [2607.12345](https://arxiv.org/abs/2607.12345)",
      `- **Research problem**: ${canonical}`,
      `- **Method design**: ${canonical}`,
      `- **Core results**: ${canonical}`,
      `- **Research value**: ${canonical}`,
      `- **Scope and limits**: ${canonical}`,
    ].join("\n"));

    expect(summaries["2607.12345"]).toEqual({
      coreProblem: canonical,
      keyMethod: canonical,
      mainResult: canonical,
      whyRelevant: canonical,
      limitations: canonical,
    });
  });

  it("extracts current Chinese and English summary fields", () => {
    const summaries = extractPaperSummaries(
      [
        "## Topic",
        "### Current Chinese Paper",
        "> 信息来源：Abstract",
        "- **arXiv**: [2606.11111](https://arxiv.org/abs/2606.11111)",
        "- **研究问题**: Chinese problem.",
        "- **方法设计**: Chinese method.",
        "- **核心结果**: Chinese result.",
        "- **研究价值**: Chinese value.",
        "- **适用边界**: Chinese limits.",
        "",
        "### English Paper",
        "> Source sections: Abstract, Results",
        "- **arXiv**: [2606.22222](https://arxiv.org/abs/2606.22222)",
        "- **Research problem**: English problem.",
        "- **Method design**: English method.",
        "- **Core results**: English result.",
        "- **Research value**: English value.",
        "- **Scope and limits**: English limits.",
      ].join("\n"),
    );

    expect(summaries["2606.11111"]).toEqual({
      sourceSections: "Abstract",
      coreProblem: "Chinese problem.",
      keyMethod: "Chinese method.",
      mainResult: "Chinese result.",
      whyRelevant: "Chinese value.",
      limitations: "Chinese limits.",
    });
    expect(summaries["2606.22222"]).toEqual({
      sourceSections: "Abstract, Results",
      coreProblem: "English problem.",
      keyMethod: "English method.",
      mainResult: "English result.",
      whyRelevant: "English value.",
      limitations: "English limits.",
    });
  });
});
