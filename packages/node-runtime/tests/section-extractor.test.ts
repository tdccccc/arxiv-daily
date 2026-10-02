import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { extractAbstractConclusion, extractSections } from "@arxiv-daily/core";
import { LinkedomMarkupParser } from "../src/markup-parser";

const html = readFileSync(
  new URL("../../core/tests/fixtures/arxiv-scientific-math.html", import.meta.url),
  "utf8",
);

describe("Node host scientific HTML extraction", () => {
  it("keeps one representation of headings, identifiers, and units through the CLI parser", () => {
    const parser = new LinkedomMarkupParser();
    const sections = extractSections(html, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
    }, parser);
    const abstract = extractAbstractConclusion(html, { sectionCharLimit: 8000 }, parser);

    expect(sections).toContain("## II.1 Model and $\\alpha_{M}$ parameter");
    expect(sections).toContain("## 2.1 $w$CDM DE models");
    expect(sections).toContain("$F_{\\rm peak}>0.75\\,\\rm{mJy}/\\rm{beam}$");
    expect(abstract).toContain("HE0435$-$1223, PG1115$+$080, and WFI2033$-$4723");
    expect(abstract).toContain("$6$–$11\\%$");
  });
});
