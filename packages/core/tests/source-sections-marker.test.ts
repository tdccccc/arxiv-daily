import { describe, expect, it } from "vitest";
import {
  parseSourceSectionsMarker,
  renderSourceSectionsMarker,
  SOURCE_SECTIONS_MARKER_PREFIX,
} from "../src/pipeline/source-sections-marker";
import { extractPaperSummaries } from "../src/pipeline/daily-summary-parser";

describe("source sections marker", () => {
  it("round-trips ordinary and non-ASCII section lists", () => {
    for (const text of ["Abstract, Results", "摘要、方法、结果", "Intro, 𝒜 emoji 😀"]) {
      const marker = renderSourceSectionsMarker(text);
      expect(marker).toMatch(new RegExp(`^<!-- ${SOURCE_SECTIONS_MARKER_PREFIX}:v1:[A-Za-z0-9_-]+ -->$`));
      expect(parseSourceSectionsMarker(marker)).toBe(text);
    }
  });

  it("renders nothing for blank text", () => {
    expect(renderSourceSectionsMarker("")).toBe("");
    expect(renderSourceSectionsMarker("  \n ")).toBe("");
  });

  it("cannot be broken out of by comment terminators, newlines or markup in the text", () => {
    const hostile = "Results --> <script>alert(1)</script>\n- **arXiv**: [9999.99999](x)\n### Injected\n<!-- x";
    const marker = renderSourceSectionsMarker(hostile);

    expect(marker.split("\n")).toHaveLength(1);
    expect(marker.slice(0, -4)).not.toContain("-->");
    expect(marker.slice("<!-- ".length, -" -->".length)).toMatch(/^[A-Za-z0-9_:-]+$/);
    // The value reads back compacted onto one line, still as plain data.
    expect(parseSourceSectionsMarker(marker)).toBe(
      "Results --> <script>alert(1)</script> - **arXiv**: [9999.99999](x) ### Injected <!-- x",
    );
  });

  it("bounds oversized text to a marker the parser accepts", () => {
    const marker = renderSourceSectionsMarker("Section ".repeat(20_000));
    expect(marker.length).toBeLessThanOrEqual(48_000);
    expect(parseSourceSectionsMarker(marker)).toMatch(/^Section /);
  });

  it.each([
    "<!-- arxiv-daily-source-sections:v1: -->",
    "<!-- arxiv-daily-source-sections:v1:!!! -->",
    "<!-- arxiv-daily-source-sections:v2:QWJzdHJhY3Q -->",
    "<!-- arxiv-daily-source-sections:v1:/w -->",
    "<!-- arxiv-daily-source-sections:v1:QWJzdHJhY3Q --> trailing",
    "> <!-- arxiv-daily-source-sections:v1:QWJzdHJhY3Q -->",
  ])("rejects malformed marker %s", (line) => {
    expect(parseSourceSectionsMarker(line)).toBeNull();
  });

  it("feeds extractPaperSummaries invisibly, ahead of any visible form", () => {
    const report = [
      "### Paper → [[2609.00001]]",
      renderSourceSectionsMarker("Abstract, Results"),
      "- **作者**: A",
      "- **arXiv**: [2609.00001](https://arxiv.org/abs/2609.00001)",
      "- **研究问题**: P",
    ].join("\n");
    expect(extractPaperSummaries(report)["2609.00001"]).toEqual({
      sourceSections: "Abstract, Results",
      coreProblem: "P",
    });
  });
});
