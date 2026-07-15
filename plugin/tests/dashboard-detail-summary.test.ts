import { describe, expect, it } from "vitest";
import { looksLikeDetailSummary } from "@arxiv-daily/core";

const longParagraph =
  "这段内容用于模拟详细总结中的解释、证据、边界和价值判断。" +
  "它包含足够长的正文，避免把只有标题或空模板的文件误判为详细总结。";

function repeatedText(times: number): string {
  return Array.from({ length: times }, () => longParagraph).join("");
}

describe("looksLikeDetailSummary", () => {
  it("detects generated detail summaries", () => {
    const markdown = [
      "---",
      "type: paper",
      "---",
      "",
      "# A Real Paper Title",
      "",
      "- **arXiv**: [2606.12345](https://arxiv.org/abs/2606.12345)",
      "",
      "## 研究问题",
      repeatedText(2),
      "",
      "## 方法设计",
      repeatedText(2),
      "",
      "## 关键证据",
      repeatedText(2),
      "",
      "## 主要结论",
      repeatedText(2),
      "",
      "## 贡献与创新点",
      repeatedText(2),
      "",
      "## 适用边界",
      repeatedText(2),
      "",
      "## 学术价值判断",
      repeatedText(1),
    ].join("\n");

    expect(looksLikeDetailSummary(markdown)).toBe(true);
  });

  it("rejects lightweight paper notes", () => {
    const markdown = [
      "---",
      "type: paper",
      "---",
      "",
      "# A Real Paper Title",
      "",
      "- **arXiv**: [2606.12345](https://arxiv.org/abs/2606.12345)",
      "- **PDF**: [PDF](https://arxiv.org/pdf/2606.12345)",
      "",
      "## Notes",
      "",
      repeatedText(8),
    ].join("\n");

    expect(looksLikeDetailSummary(markdown)).toBe(false);
  });

  it("rejects short or title-less markdown files", () => {
    expect(
      looksLikeDetailSummary("# A Real Paper Title\n\n## 研究问题\n太短"),
    ).toBe(false);
    expect(
      looksLikeDetailSummary(
        ["## 研究问题", repeatedText(2), "## 方法设计", repeatedText(2)].join(
          "\n",
        ),
      ),
    ).toBe(false);
  });
});
