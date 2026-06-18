import { describe, expect, it } from "vitest";
import { extractPaperSummaries } from "../src/pipeline/daily-summary-parser";

describe("extractPaperSummaries", () => {
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
