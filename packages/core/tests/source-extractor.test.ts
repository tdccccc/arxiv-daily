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
  it.each([
    String.raw`$H_0 = 67.4 \pm 0.5$`,
    String.raw`$\Omega_m \leq 0.3$`,
    String.raw`$\frac{1}{3}$`,
    String.raw`\(\alpha = 0.31^{+0.04}_{-0.05}\)`,
    String.raw`\[4.2\sigma \approx \sqrt{2}\]`,
    String.raw`$$z \sim 0.5$$`,
    String.raw`\begin{equation}x = \frac{1}{3}\end{equation}`,
    String.raw`\begin{align}a &\leq b \\ c &= d\end{align}`,
  ])("keeps scientific meaning in source formula %s", formula => {
    const prose = `We report precise measurements and uncertainty estimates from a controlled scientific experiment: ${formula}.`;
    const source = String.raw`\documentclass{article}\begin{document}\begin{abstract}${prose}\end{abstract}\section{Results}${prose}\end{document}`;
    const result = extractLatexSource(texBuffer(source), opts);
    expect(result.abstractConclusion).toContain(formula);
    expect(result.fullSections).toContain(formula);
  });

  it("keeps escaped percent and dollar signs while cleaning references and formatting", () => {
    const source = String.raw`\documentclass{article}\begin{document}\section{Results}
The \textbf{confidence} is 95\% and the measured cost is \$5. We tested \emph{several datasets} with strong evidence. See \citep[section 2]{private-citation-key} and Fig.~\ref{private-figure-key}.
\begin{figure}not useful figure body\end{figure}
\end{document}`;
    const result = extractLatexSource(texBuffer(source), opts).fullSections!;
    expect(result).toContain("confidence is 95% and the measured cost is $5");
    expect(result).toContain("several datasets");
    expect(result).not.toMatch(/private-citation|private-figure|not useful figure|textbf|emph/);
  });

  it.each([String.raw`\begin{figure}`, String.raw`\cite[`, String.raw`\input{`, String.raw`\begin{abstract}`])(
    "bounds cleanup of repeated unclosed %s", token => {
      const source = String.raw`\documentclass{article}\begin{document}\section{Results}` + token.repeat(24_000) + String.raw`\end{document}`;
      const started = performance.now();
      extractLatexSource(texBuffer(source), opts);
      expect(performance.now() - started).toBeLessThan(1500);
    },
  );

  it("rejects more than one MiB of expanded source before synchronous cleanup", () => {
    const source = String.raw`\documentclass{article}\section{Results}` + "x".repeat(1024 * 1024);
    expect(() => extractLatexSource(texBuffer(source), opts)).toThrow(SourceBudgetExceededError);
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
