import { markupParser } from "./markup-parser";
import { readFileSync } from "node:fs";
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

const scientificMathSample = readFileSync(
  new URL("./fixtures/arxiv-scientific-math.html", import.meta.url),
  "utf8",
);

describe("section-extractor", () => {
  it("extracts one mathematical representation from public arXiv headings and body text", () => {
    const out = extractSections(scientificMathSample, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
    }, markupParser);

    expect(out).toContain("## II.1 Model and $\\alpha_{M}$ parameter");
    expect(out).toContain("## 2.1 $w$CDM DE models");
    expect(out).toContain("peak flux density of $F_{\\rm peak}>0.75\\,\\rm{mJy}/\\rm{beam}$");
    expect(out).toContain("$121\\,179$ objects satisfied this condition.");
  });

  it("preserves object identifiers and numerical ranges in abstract and conclusion MathML", () => {
    const out = extractAbstractConclusion(scientificMathSample, {
      sectionCharLimit: 8000,
    }, markupParser);

    expect(out).toContain("HE0435$-$1223, PG1115$+$080, and WFI2033$-$4723");
    expect(out).toContain("uncertainties of $6$–$11\\%$");
    expect(out).toContain("errors of only $\\sim 1.2\\%$");
  });

  it("preserves MathML structure before flattening figure and table captions", () => {
    const html = '<html><body><h2>Results</h2>'
      + '<figure><figcaption>Estimate <math><semantics><msub><mi>H</mi><mn>0</mn></msub>'
      + '<annotation encoding="application/x-tex">H_{0}</annotation></semantics></math>.</figcaption></figure>'
      + '<table><caption>Fraction <math><semantics><mfrac><mn>1</mn><mn>2</mn></mfrac>'
      + '<annotation encoding="application/x-tex">\\frac{1}{2}</annotation></semantics></math>.</caption></table>'
      + '</body></html>';

    const out = extractSections(html, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
    }, markupParser);

    expect(out).toContain("Figure caption: Estimate $H_{0}$.");
    expect(out).toContain("Table text: Fraction $\\frac{1}{2}$.");
  });

  it("uses LaTeXML alttext when the TeX annotation is unavailable", () => {
    const html = '<html><body><h2>Results</h2><p>Value '
      + '<math class="ltx_Math" alttext="x_{1}"><msub><mi>x</mi><mn>1</mn></msub></math>.'
      + '</p></body></html>';

    expect(extractSections(html, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
    }, markupParser)).toContain("Value $x_{1}$.");
  });

  it("excludes non-TeX auxiliary representations without deduplicating literal scientific text", () => {
    const html = '<html><body><h2>Results</h2><p>'
      + '<math alttext="energy sum"><semantics><mrow><mi>E</mi><mo>+</mo><mi>E</mi></mrow>'
      + '<annotation encoding="application/json">{"symbol":"E"}</annotation>'
      + '<annotation-xml encoding="MathML-Content"><apply><plus/><ci>E</ci><ci>E</ci></apply></annotation-xml>'
      + '</semantics></math>; C++ and ww remain literal.'
      + '</p></body></html>';

    expect(extractSections(html, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
    }, markupParser)).toBe("## Results\nE+E; C++ and ww remain literal.");
  });

  it("does not select TeX embedded inside an auxiliary XML representation", () => {
    const html = '<html><body><h2>Results</h2><p>'
      + '<math><semantics><mi>E</mi><annotation-xml encoding="application/xhtml+xml">'
      + '<math><semantics><mi>q</mi><annotation encoding="application/x-tex">q</annotation></semantics></math>'
      + '</annotation-xml></semantics></math>'
      + '</p></body></html>';

    expect(extractSections(html, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
    }, markupParser)).toBe("## Results\nE");
  });

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
    const out = extractAbstractConclusion(sample, { sectionCharLimit: 8000 }, markupParser);
    expect(out).toContain("## Abstract");
    expect(out).toContain("abstract content");
    expect(out).toContain("## Conclusions");
    expect(out).toContain("final remarks");
  });

  it("extractSections includes priority + body, skips refs/appendix", () => {
    const out = extractSections(sample, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
    }, markupParser);
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
    }, markupParser);
    expect(out).toBeNull();
  });

  it("extractSections truncates within section char limit", () => {
    const longBody = "x".repeat(20000);
    const html = `<html><body><h2>BigSection</h2><p>${longBody}</p></body></html>`;
    const out = extractSections(html, {
      sectionCharLimit: 1000,
      paperCharLimit: 50000,
    }, markupParser);
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
    }, markupParser);
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
    }, markupParser);
    expect(out).toContain("Figure caption: Figure 1");
    expect(out).toContain("Table text: Table 1");
  });

  it("extractAbstractConclusion returns null when neither present", () => {
    const out = extractAbstractConclusion("<html><body><h2>Methods</h2><p>x</p></body></html>", {
      sectionCharLimit: 8000,
    }, markupParser);
    expect(out).toBeNull();
  });

  it("ranks introduction above generic sections when budget is tight", () => {
    const intro = "Background and motivation text. ".repeat(40);
    const generic = "Notation conventions used throughout. ".repeat(40);
    const html = `<html><body>
      <h2>Introduction</h2><p>${intro}</p>
      <h2>Notation</h2><p>${generic}</p>
    </body></html>`;
    const out = extractSections(html, {
      sectionCharLimit: 2000,
      paperCharLimit: 900,
    }, markupParser);
    expect(out).toContain("## Introduction");
    expect(out).not.toContain("## Notation");
  });
});
