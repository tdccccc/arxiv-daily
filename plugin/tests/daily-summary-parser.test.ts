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
});
