import { describe, expect, it } from "vitest";
import { looksLikeDetailSummary } from "../src/dashboard/detail-summary";

function detailBody(headings: string[]): string {
  return [
    "# Verified detail",
    "",
    ...headings.flatMap((heading) => [`## ${heading}`, "a".repeat(150), ""]),
  ].join("\n");
}

describe("looksLikeDetailSummary", () => {
  it("recognizes a detail summary with 3+ matched Chinese headings", () => {
    expect(
      looksLikeDetailSummary(detailBody(["研究问题", "方法设计", "主要结论"])),
    ).toBe(true);
  });

  it("recognizes a detail summary with 4+ generic sections that are not a lightweight note", () => {
    expect(
      looksLikeDetailSummary(detailBody(["One", "Two", "Three", "Four"])),
    ).toBe(true);
  });

  it("rejects a short body", () => {
    expect(looksLikeDetailSummary("# Title\n\n## 研究问题\nshort")).toBe(false);
  });

  it("rejects a body without an H1", () => {
    expect(looksLikeDetailSummary(`## 研究问题\n${"a".repeat(500)}`)).toBe(false);
  });

  it("treats a single lightweight 'Notes' section as not a detail summary", () => {
    const body = ["# Title", "", "## Notes", "a".repeat(500)].join("\n");
    expect(looksLikeDetailSummary(body)).toBe(false);
  });

  it("ignores heading-like text embedded in frontmatter and generation metrics", () => {
    const body = [
      "---",
      "title: has ## not a heading",
      "---",
      detailBody(["研究问题", "方法设计", "主要结论"]),
    ].join("\n");
    expect(looksLikeDetailSummary(body)).toBe(true);
  });

  it("completes quickly on adversarial long-whitespace headings (CodeQL js/polynomial-redos)", () => {
    const adversarial = `# Title\n${"## " + " ".repeat(20_000) + "\n"}`.repeat(5);
    const start = performance.now();
    const result = looksLikeDetailSummary(adversarial + "a".repeat(500));
    const elapsedMs = performance.now() - start;
    expect(elapsedMs).toBeLessThan(200);
    expect(typeof result).toBe("boolean");
  });

  it("completes quickly on a single huge line with no newline (no heading content)", () => {
    const adversarial = `# Title\n## ${" ".repeat(50_000)}`;
    const start = performance.now();
    const result = looksLikeDetailSummary(adversarial);
    const elapsedMs = performance.now() - start;
    expect(elapsedMs).toBeLessThan(200);
    expect(result).toBe(false);
  });
});
