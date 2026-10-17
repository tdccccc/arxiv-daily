import { describe, expect, it } from "vitest";
import { gzipSync } from "node:zlib";
import {
  extractLatexSource,
  SOURCE_EXTRACTION_LIMITS,
  SourceBudgetExceededError,
} from "../src/pipeline/source-extractor";

const opts = {
  sectionCharLimit: 2000,
  paperCharLimit: 6000,
  skipSections: ["references", "appendix"],
  prioritySections: ["abstract", "conclusion", "results"],
};

function toArrayBuffer(buffer: Buffer): ArrayBuffer {
  const out = new Uint8Array(buffer.byteLength);
  out.set(buffer);
  return out.buffer;
}

function texBuffer(text: string): ArrayBuffer {
  return toArrayBuffer(Buffer.from(text, "utf8"));
}

function tarBuffer(files: Record<string, string>): Buffer {
  const chunks: Buffer[] = [];
  for (const [name, content] of Object.entries(files)) {
    const body = Buffer.from(content, "utf8");
    const header = Buffer.alloc(512);
    header.write(name, 0, 100, "utf8");
    header.write("0000644\0", 100, 8, "ascii");
    header.write("0000000\0", 108, 8, "ascii");
    header.write("0000000\0", 116, 8, "ascii");
    header.write(body.length.toString(8).padStart(11, "0") + "\0", 124, 12, "ascii");
    header.write("00000000000\0", 136, 12, "ascii");
    header.write("0", 156, 1, "ascii");
    chunks.push(header, body);
    const padding = (512 - (body.length % 512)) % 512;
    if (padding) chunks.push(Buffer.alloc(padding));
  }
  chunks.push(Buffer.alloc(1024));
  return Buffer.concat(chunks);
}

describe("extractLatexSource", () => {
  it("extracts useful sections from a single TeX source file", () => {
    const source = String.raw`
\documentclass{article}
\begin{document}
\begin{abstract}
We present a careful analysis of galaxy cluster selection with quantitative evidence and limitations.
\end{abstract}
\section{Introduction}
This introduction motivates the problem with enough natural language to be treated as useful paper text.
\section{Method}
We model the likelihood, calibrate the selection function, and compare against a baseline method in simulations.
\section{Results}
We find a 12 percent improvement, report the sample size, and describe the uncertainty budget.
\section{References}
Should not appear.
\end{document}
`;

    const result = extractLatexSource(texBuffer(source), opts);

    expect(result.mainFile).toBe("source.tex");
    expect(result.abstractConclusion).toContain("## Abstract");
    expect(result.fullSections).toContain("## Method");
    expect(result.fullSections).toContain("12 percent improvement");
    expect(result.fullSections).not.toContain("Should not appear");
  });

  it("expands input files from a gzipped tar source archive", () => {
    const archive = tarBuffer({
      "paper/main.tex": String.raw`
\documentclass{article}
\begin{document}
\begin{abstract}
This source archive includes a multi-file paper with a useful abstract for downstream summarization.
\end{abstract}
\section{Introduction}
The introduction explains the scientific motivation and the concrete problem being addressed.
\input{sections/method}
\section{Conclusion}
The conclusion summarizes the validated result and the conditions where it applies.
\end{document}
`,
      "paper/sections/method.tex": String.raw`
\section{Method}
The method section describes the data, model assumptions, inference procedure, and validation experiment.
`,
    });

    const result = extractLatexSource(toArrayBuffer(gzipSync(archive)), opts);

    expect(result.mainFile).toBe("paper/main.tex");
    expect(result.fullSections).toContain("## Method");
    expect(result.fullSections).toContain("inference procedure");
    expect(result.abstractConclusion).toContain("## Conclusion");
  });

  it("expands repeated input files without treating siblings as cycles", () => {
    const archive = tarBuffer({
      "paper/main.tex": String.raw`
\documentclass{article}
\begin{document}
\begin{abstract}
This source archive repeats one included section in two different places.
\end{abstract}
\input{sections/method}
\input{sections/method}
\end{document}
`,
      "paper/sections/method.tex": String.raw`
\section{Method}
The repeated method section describes data, calibration, validation, and uncertainty checks.
`,
    });

    const result = extractLatexSource(toArrayBuffer(gzipSync(archive)), opts);
    const fullSections = result.fullSections ?? "";

    expect(
      fullSections.match(/validation, and uncertainty checks/g),
    ).toHaveLength(2);
  });

  it("treats limitations as core evidence over introduction when budget is tight", () => {
    const intro = "Motivation and background context here. ".repeat(25);
    const limitation = "Caveats, systematics, and uncertainty budget. ".repeat(20);
    const source = String.raw`
\documentclass{article}
\begin{document}
\section{Introduction}
${intro}
\section{Limitations}
${limitation}
\end{document}
`;
    const result = extractLatexSource(texBuffer(source), {
      sectionCharLimit: 2000,
      paperCharLimit: 800,
    });
    const full = result.fullSections ?? "";
    expect(full).toContain("## Limitations");
    expect(full).not.toContain("## Introduction");
  });
  describe("extraction budgets", () => {
    it("rejects a gzip bomb without materialising the inflated data", () => {
      const bomb = gzipSync(Buffer.alloc(SOURCE_EXTRACTION_LIMITS.maxInflatedBytes + 1024 * 1024));
      expect(bomb.byteLength).toBeLessThan(1024 * 1024);
      expect(() => extractLatexSource(toArrayBuffer(bomb), opts)).toThrow(SourceBudgetExceededError);
    });

    it("bounds a deeply nested \\input fan-out in time and memory", () => {
      const depth = 12;
      const files: Record<string, string> = {};
      for (let level = 0; level < depth; level += 1) {
        const include = `\\input{f${level + 1}}\n`;
        const fan = level + 1 < depth ? include.repeat(3) : "";
        files[level === 0 ? "main.tex" : `f${level}.tex`] = level === 0
          ? String.raw`\documentclass{article}\begin{document}\section{Intro}` + "\n" + fan + String.raw`\end{document}`
          : "filler text for expansion. ".repeat(20) + "\n" + fan;
      }
      const archive = gzipSync(tarBuffer(files));
      expect(archive.byteLength).toBeLessThan(2000);
      const started = Date.now();
      expect(() => extractLatexSource(toArrayBuffer(archive), opts)).toThrow(SourceBudgetExceededError);
      expect(Date.now() - started).toBeLessThan(5000);
    });

    it("rejects archives with too many entries", () => {
      const files: Record<string, string> = {};
      for (let i = 0; i <= SOURCE_EXTRACTION_LIMITS.maxTarEntries; i += 1) files[`n${i}.txt`] = "x";
      expect(() => extractLatexSource(toArrayBuffer(tarBuffer(files)), opts)).toThrow(SourceBudgetExceededError);
    });

    it("rejects oversized archives before parsing", () => {
      const huge = new ArrayBuffer(SOURCE_EXTRACTION_LIMITS.maxArchiveBytes + 1);
      expect(() => extractLatexSource(huge, opts)).toThrow(SourceBudgetExceededError);
    });

    it("keeps a normal multi-file package unchanged", () => {
      const archive = gzipSync(tarBuffer({
        "main.tex": String.raw`\documentclass{article}\begin{document}\section{Introduction}\input{intro}\end{document}`,
        "intro.tex": "A sufficiently long introduction paragraph that explains the problem, the data and the approach in detail for readers.",
      }));
      const result = extractLatexSource(toArrayBuffer(archive), opts);
      expect(result.fullSections).toContain("sufficiently long introduction");
    });
  });
});
