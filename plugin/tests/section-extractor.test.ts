import { describe, it, expect } from "vitest";
import {
  classifySection,
  extractAbstractConclusion,
  extractSections,
} from "../src/pipeline/section-extractor";

const sample = `
<html><body>
<div class="ltx_abstract">This is the abstract content with key findings.</div>
<h2>Introduction</h2><p>intro text body here</p>
<h2>Methods</h2><p>methods body</p>
<h2>Conclusions</h2><p>final remarks summary</p>
<h2>References</h2><p>[1] paper</p>
<h2>Appendix A</h2><p>extra</p>
</body></html>
`;

describe("section-extractor", () => {
  it("classifies domain-specific non-standard section titles", () => {
    expect(classifySection("Photometric redshift inference")).toContain("method");
    expect(classifySection("The weak-lensing shear catalogue")).toContain("data");
    expect(classifySection("Cosmological constraints")).toContain("result");
  });

  it("uses body preview signals when titles are not explicit", () => {
    expect(
      classifySection("Cluster mass calibration", "We model the likelihood and calibrate the selection function."),
    ).toContain("method");
    expect(
      classifySection("Galaxy sample", "We find a 12 percent improvement over the baseline."),
    ).toContain("result");
  });

  it("extractAbstractConclusion finds abstract and conclusion sections", () => {
    const out = extractAbstractConclusion(sample, { sectionCharLimit: 8000 });
    expect(out).toContain("## Abstract");
    expect(out).toContain("abstract content");
    expect(out).toContain("## Conclusions");
    expect(out).toContain("final remarks");
  });

  it("extractSections includes priority + body, skips refs/appendix", () => {
    const out = extractSections(sample, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
      skipSections: ["reference", "appendix", "bibliography"],
      prioritySections: ["abstract", "conclusion", "summary"],
    });
    expect(out).toContain("## Introduction");
    expect(out).toContain("## Methods");
    expect(out).toContain("## Conclusions");
    expect(out).not.toContain("## References");
    expect(out).not.toContain("## Appendix");
  });

  it("extractSections returns null when no useful sections", () => {
    const out = extractSections("<html><body><p>no headings here</p></body></html>", {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
      skipSections: [],
      prioritySections: [],
    });
    expect(out).toBeNull();
  });

  it("extractSections truncates within section char limit", () => {
    const longBody = "x".repeat(20000);
    const html = `<html><body><h2>BigSection</h2><p>${longBody}</p></body></html>`;
    const out = extractSections(html, {
      sectionCharLimit: 1000,
      paperCharLimit: 50000,
      skipSections: [],
      prioritySections: [],
    });
    expect(out).toBeTruthy();
    expect(out!.length).toBeLessThan(2000);
  });

  it("prioritizes high-value classified sections when budget is tight", () => {
    const intro = "intro ".repeat(220);
    const method = "We model the likelihood and calibrate the selection function. ".repeat(4);
    const result = "We find improved constraints and report a 12 percent gain. ".repeat(4);
    const html = `<html><body>
      <h2>Introduction</h2><p>${intro}</p>
      <h2>Photometric redshift inference</h2><p>${method}</p>
      <h2>Cosmological constraints</h2><p>${result}</p>
    </body></html>`;
    const out = extractSections(html, {
      sectionCharLimit: 2000,
      paperCharLimit: 700,
      skipSections: [],
      prioritySections: [],
    });
    expect(out).toContain("## Photometric redshift inference");
    expect(out).toContain("## Cosmological constraints");
    expect(out).not.toContain("## Introduction");
  });

  it("preserves figure captions and compact table text", () => {
    const html = `<html><body>
      <h2>Results</h2>
      <p>result body</p>
      <figure><figcaption>Figure 1: posterior constraints improve at high redshift.</figcaption></figure>
      <table><caption>Table 1: benchmark metrics.</caption><tr><td>RMSE</td><td>0.12</td></tr></table>
    </body></html>`;
    const out = extractSections(html, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
      skipSections: [],
      prioritySections: [],
    });
    expect(out).toContain("Figure caption: Figure 1");
    expect(out).toContain("Table text: Table 1");
  });

  it("extractAbstractConclusion returns null when neither present", () => {
    const out = extractAbstractConclusion("<html><body><h2>Methods</h2><p>x</p></body></html>", {
      sectionCharLimit: 8000,
    });
    expect(out).toBeNull();
  });
});
