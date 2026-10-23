import { describe, expect, it } from "vitest";
import { buildDailyDigest } from "../../src/delivery/digest";
import {
  emailProse,
  escapeHtml,
  renderEmailHtml,
  renderEmailSubject,
  renderEmailText,
} from "../../src/delivery/email-render";
import type { DailyDigest } from "../../src/delivery/types";
import type { DailyPaperSlot } from "../../src/pipeline/daily-summary-assembler";
import type { ArxivSettings } from "../../src/settings/types";

const arxiv: ArxivSettings = {
  category: "astro-ph",
  categories: ["astro-ph"],
  timezone: "UTC",
  topics: [
    {
      id: "t1",
      name: "Photo-z",
      tag: "photo-z",
      description: "",
      detail: true,
    },
    {
      id: "t2",
      name: "Clusters",
      tag: "cluster",
      description: "",
      detail: false,
    },
  ],
};

function structuredSlot(id: string, topic: string): DailyPaperSlot {
  return {
    paper: {
      id,
      title: `Title <script> of ${id}`,
      authors: "A & B",
      category: topic,
      sourceSections: "Abstract, Results",
      isDetail: false,
    },
    result: {
      kind: "structured",
      summary: {
        id,
        coreProblem: `problem $E=mc^2$ for ${id}`,
        keyMethod: `method for ${id}`,
        mainResult: `result for ${id}`,
        whyRelevant: `value for ${id}`,
        limitations: `limits for ${id}`,
      },
    },
  };
}

describe("email render", () => {
  it("renders subject for N papers and zero day (zh/en)", () => {
    const zero: DailyDigest = {
      date: "2026-07-25",
      summaryLanguage: "zh",
      categories: "astro-ph",
      dailyPath: "arxiv-daily/daily/2026-07-25.md",
      paperCount: 0,
      topics: [],
    };
    expect(renderEmailSubject(zero)).toBe("arXiv Daily 2026-07-25 · 0 篇");
    expect(
      renderEmailSubject({ ...zero, summaryLanguage: "en", paperCount: 2 }),
    ).toBe("arXiv Daily 2026-07-25 · 2 papers");
    expect(
      renderEmailSubject({ ...zero, summaryLanguage: "en", paperCount: 1 }),
    ).toBe("arXiv Daily 2026-07-25 · 1 paper");
  });

  it("zero-day body includes lead line and empty topics", () => {
    const digest = buildDailyDigest({
      date: "2026-07-25",
      arxiv,
      output: {
        dailyDir: "arxiv-daily/daily",
        summaryLanguage: "zh",
      },
      slots: [],
    });
    expect(digest.paperCount).toBe(0);
    const html = renderEmailHtml(digest);
    const text = renderEmailText(digest);
    expect(html).toContain("今日无相关论文");
    expect(html).toContain("Photo-z");
    expect(html).toContain("Clusters");
    expect(html).toContain("今日无相关论文更新。");
    expect(text).toContain("今日无相关论文");
    expect(text).toContain("## Photo-z");
    expect(text).toContain("今日无相关论文更新。");
  });

  it("points to the daily report only when the day produced one", () => {
    const zero: DailyDigest = {
      date: "2026-07-27",
      summaryLanguage: "en",
      categories: "astro-ph",
      dailyPath: "arxiv-daily/daily/2026-07-27.md",
      paperCount: 0,
      topics: [],
    };
    for (const digest of [zero, { ...zero, summaryLanguage: "zh" as const }]) {
      const html = renderEmailHtml(digest);
      const text = renderEmailText(digest);
      expect(html).not.toContain("2026-07-27.md");
      expect(text).not.toContain("2026-07-27.md");
      expect(html).not.toMatch(/Full Markdown daily report|完整 Markdown 日报路径/);
      expect(text).not.toMatch(/Full Markdown daily report|完整 Markdown 日报路径/);
      // The signature still closes the mail.
      expect(text.trimEnd().endsWith("arXiv Daily")).toBe(true);
    }

    const written = { ...zero, paperCount: 1 };
    expect(renderEmailText(written)).toContain(
      "Full Markdown daily report: arxiv-daily/daily/2026-07-27.md",
    );
    expect(renderEmailHtml(written)).toContain(
      "Full Markdown daily report: arxiv-daily/daily/2026-07-27.md",
    );
    expect(renderEmailText({ ...written, summaryLanguage: "zh" })).toContain(
      "完整 Markdown 日报路径：arxiv-daily/daily/2026-07-27.md",
    );
  });

  it("agrees with the subject on singular and plural paper counts in the header", () => {
    const digest: DailyDigest = {
      date: "2026-07-27",
      summaryLanguage: "en",
      categories: "astro-ph",
      dailyPath: "arxiv-daily/daily/2026-07-27.md",
      paperCount: 1,
      topics: [],
    };
    const header = (n: number, summaryLanguage: DailyDigest["summaryLanguage"]) =>
      renderEmailText({ ...digest, paperCount: n, summaryLanguage }).split("\n")[1];
    expect(header(1, "en")).toBe("Date: 2026-07-27 · 1 paper · Categories: astro-ph");
    expect(header(0, "en")).toBe("Date: 2026-07-27 · 0 papers · Categories: astro-ph");
    expect(header(2, "en")).toBe("Date: 2026-07-27 · 2 papers · Categories: astro-ph");
    expect(header(1, "zh")).toBe("日期：2026-07-27 · 1 篇 · 分类：astro-ph");
    expect(renderEmailHtml(digest)).toContain("1 paper · Categories");
    expect(renderEmailSubject(digest)).toBe("arXiv Daily 2026-07-27 · 1 paper");
  });

  it("renders five fields, abs+pdf links, omits source sections, and softens math", () => {
    const digest = buildDailyDigest({
      date: "2026-07-26",
      arxiv,
      output: {
        dailyDir: "arxiv-daily/daily",
        summaryLanguage: "zh",
      },
      slots: [structuredSlot("2607.12345", "photo-z")],
    });
    const html = renderEmailHtml(digest);
    const text = renderEmailText(digest);
    expect(html).toContain("研究问题");
    expect(html).toContain("方法设计");
    expect(html).toContain("核心结果");
    expect(html).toContain("研究价值");
    expect(html).toContain("适用边界");
    expect(html).toContain('href="https://arxiv.org/abs/2607.12345"');
    expect(html).toContain('href="https://arxiv.org/pdf/2607.12345"');
    // Source-section lists are vault noise — omit from mail.
    expect(html).not.toContain("信息来源");
    expect(html).not.toContain("Source sections");
    expect(text).not.toContain("信息来源");
    // Math delimiters stripped for email readability.
    expect(html).toContain("E=mc^2");
    expect(html).not.toContain("$E=mc^2$");
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain("A &amp; B");
    expect(text).toContain("arXiv: https://arxiv.org/abs/2607.12345");
    expect(text).toContain("PDF: https://arxiv.org/pdf/2607.12345");
    expect(text).toContain("研究问题: problem E=mc^2 for 2607.12345");
  });

  it("emailProse simplifies common LaTeX for mail clients", () => {
    expect(emailProse("scale $N_{\\rm side}=2048$")).toContain("N_side=2048");
    expect(emailProse("ratio $\\frac{a}{b}$")).toContain("(a)/(b)");
    expect(emailProse("angle $\\theta \\leq \\pi$")).toMatch(/θ.*≤.*π/);
    // \b must not block commands before underscore
    expect(emailProse("$\\Omega_m$")).toBe("Ω_m");
    expect(emailProse("$\\sum_i x_i$")).toMatch(/∑_i x_i/);
    expect(
      emailProse("$H_0\\simeq 70\\,\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$"),
    ).toMatch(/H_0≃ 70 km s\^-1 Mpc\^-1/);
    expect(emailProse("$\\Delta\\chi^2$")).toMatch(/Δχ\^2/);
    expect(emailProse("$\\alpha\\approx 0.3$")).toMatch(/α≈\s*0\.3/);
  });

  it.each([
    // Symbols the table did not know used to lose their backslash and keep the name.
    ["$M_\\star > 10^{10}\\,M_\\odot$", "M_⋆ > 10^10 M_⊙"],
    ["$50^\\circ$", "50°"],
    ["$50^{\\circ}$", "50°"],
    ["$f \\circ g$", "f ∘ g"],
    ["$T_\\oplus$", "T_⊕"],
    ["$F \\ge 3\\sigma$", "F ≥ 3σ"],
    ["$F \\le 1$", "F ≤ 1"],
    ["$\\Omega_m \\ne 0.3$", "Ω_m ≠ 0.3"],
    ["$x \\ll 1$ and $y \\gg 1$", "x ≪ 1 and y ≫ 1"],
    ["$\\forall i \\exists j$", "∀ i ∃ j"],
    ["$\\langle x \\rangle$", "⟨x⟩"],
    ["$f\\colon X \\to Y$ and $A \\setminus B$", "f: X → Y and A ∖ B"],
    ["$\\mathbb{R}^d$ and $\\mathbb{N}$", "ℝ^d and ℕ"],
    // Typesetting-only commands vanish and leave their argument.
    ["$\\dot{M} \\ll 1$", "M ≪ 1"],
    ["$\\mathcal{O}(n \\log n)$", "O(n log n)"],
    ["$\\hat{\\theta}$, $\\bar{x}$, $\\vec{v}$, $\\tilde{y}$", "θ, x, v, y"],
    ["$\\vec v$ and $\\hat x$", "v and x"],
    ["$\\sqrt{d}$", "√(d)"],
    ["$\\sqrt[3]{x}$", "3√(x)"],
    ["$\\sqrt{\\frac{a}{b}}$", "√((a)/(b))"],
    ["$\\left( \\frac{a}{b} \\right)^2$", "((a)/(b))^2"],
    ["$\\left. x \\right|_0$", "x |_0"],
    ["$\\displaystyle \\sum_i x_i$", "∑_i x_i"],
    ["$\\sin\\theta + \\ln x$", "sinθ + ln x"],
    // What cannot be rendered stays recognisably TeX instead of becoming a word.
    ["$\\foo{x}$", "\\foo{x}"],
    ["$\\foo{a{b}c}_i$", "\\foo{a{b}c}_i"],
    ["$\\foo x + \\bar{y}$", "\\foo x + y"],
    [
      "$A=\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$",
      "A=\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}",
    ],
    ["\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}", "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}"],
    ["see C:\\Users\\arxiv", "see C:\\Users\\arxiv"],
  ])("emailProse renders %j as %j", (input, expected) => {
    expect(emailProse(input)).toBe(expected);
  });

  it("emailProse never glues a TeX command name onto its argument", () => {
    // Deterministic pseudo-random snippets built only from TeX tokens: a command
    // name fused with the next letter or digit is exactly the fake word the
    // reader should never see.
    let seed = 20261021;
    const next = (n: number) => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed % n;
    };
    const commands = [
      "mathcal", "mathbb", "mathfrak", "boldsymbol", "bm", "sqrt", "hat", "bar", "vec",
      "tilde", "widetilde", "dot", "ddot", "overline", "underline", "mathrm", "text",
    ];
    const atoms = ["x", "y", "v", "n", "d", "M", "O", "\\theta", "\\odot", "2"];
    const fused = /(mathcal|mathbb|mathfrak|boldsymbol|sqrt|widetilde|overline|underline|mathrm|left|right|tilde|ddot)[A-Za-z0-9(]|\b(?:hat|bar|vec|dot|bm)[A-Za-z0-9(]/;
    for (let i = 0; i < 300; i += 1) {
      const parts: string[] = [];
      for (let j = 0; j < 1 + next(4); j += 1) {
        const atom = atoms[next(atoms.length)]!;
        parts.push(
          next(3) === 0
            ? atom
            : `\\${commands[next(commands.length)]!}{${atom}}`,
          next(2) === 0 ? "^2" : "_i",
        );
        if (next(4) === 0) parts.push("\\left(", atom, "\\right)");
      }
      const snippet = parts.join(" ");
      const out = emailProse(`$${snippet}$`);
      expect(out, snippet).not.toMatch(fused);
      expect(out, snippet).not.toContain("\\mathcal");
    }
  });

  it("emailProse keeps handling common math after the symbol table grew", () => {
    // A command name still reads as a word when it is a function, and the
    // longer \leq / \leftarrow forms are not shadowed by \le / \left.
    expect(emailProse("$a \\leq b \\leftarrow c \\geq d \\neq e$")).toBe("a ≤ b ← c ≥ d ≠ e");
    expect(emailProse("$\\gg$ and $\\ggg$")).toContain("≫");
  });

  it.each([
    ["begin", "\\begin{pmatrix}"],
    ["sqrt", "\\sqrt["],
    ["unknown command", "\\foo{"],
  ])("emailProse stays fast on repeated unterminated %s openers", (_name, opener) => {
    const input = (opener + "a".repeat(20_000)).repeat(50);
    // Each adversarial input must finish within the UI budget. Summing three
    // independent calls couples this regression to the runner's CPU speed.
    const start = performance.now();
    const output = emailProse(input);
    const elapsed = performance.now() - start;
    expect(output.length).toBeGreaterThan(0);
    expect(elapsed).toBeLessThan(500);
  });

  it("emailProse simplifies display/block math delimiters ($$, \\[ \\], \\( \\))", () => {
    expect(emailProse("Energy: $$E=mc^2$$ done")).toBe("Energy: E=mc^2 done");
    expect(emailProse("Energy: \\[E=mc^2\\] done")).toBe("Energy: E=mc^2 done");
    expect(emailProse("Energy: \\(E=mc^2\\) done")).toBe("Energy: E=mc^2 done");
  });

  it("emailProse leaves an unterminated display-math delimiter's tail untouched", () => {
    expect(emailProse("before $$never closes")).toBe("before $$never closes");
    // simplifyLatex still unwraps a leftover, unterminated "\[" bracket
    // outside any delimiter span; replaceDelimited correctly left it alone.
    expect(emailProse("before \\[never closes")).toBe("before [never closes");
  });

  it("emailProse completes quickly on repeated unterminated delimiters (CodeQL js/polynomial-redos)", () => {
    const adversarialDollar = "$$" + "a".repeat(20_000);
    const adversarialBracket = "\\[" + "a".repeat(20_000);
    const start = performance.now();
    const resultDollar = emailProse(adversarialDollar.repeat(50));
    const resultBracket = emailProse(adversarialBracket.repeat(50));
    const elapsedMs = performance.now() - start;
    expect(elapsedMs).toBeLessThan(500);
    expect(resultDollar.length).toBeGreaterThan(0);
    expect(resultBracket.length).toBeGreaterThan(0);
  });

  it("escapeHtml encodes markup-sensitive characters", () => {
    expect(escapeHtml(`<script>alert("x")</script>`)).toBe(
      "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;",
    );
  });
});
