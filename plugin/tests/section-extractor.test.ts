import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { extractAbstractConclusion, extractSections } from "@arxiv-daily/core";
import { ObsidianMarkupParser } from "../src/hosts/obsidian/markup-parser";

const html = readFileSync(
  resolve(dirname(fileURLToPath(import.meta.url)), "../../packages/core/tests/fixtures/arxiv-scientific-math.html"),
  "utf8",
);

describe("Obsidian host scientific HTML extraction", () => {
  it("keeps one representation of headings, identifiers, and units through the plugin parser", () => {
    const parser = new ObsidianMarkupParser();
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
