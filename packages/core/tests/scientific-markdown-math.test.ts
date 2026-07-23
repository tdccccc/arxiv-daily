import { describe, expect, it } from "vitest";
import {
  canonicalizeScientificMarkdownMath,
  SCIENTIFIC_MARKDOWN_MATH_POLICY,
  type ScientificMarkdownMathIssueCode,
} from "../src/pipeline/scientific-markdown-math";
import { normalizeMarkdownLine } from "../src/pipeline/daily-summary-rendering";

function expectValid(input: string, expected = input): void {
  expect(canonicalizeScientificMarkdownMath(input)).toEqual({
    ok: true,
    value: expected,
    issues: [],
  });
}

function issueCodes(input: string): ScientificMarkdownMathIssueCode[] {
  const result = canonicalizeScientificMarkdownMath(input);
  expect(result.ok).toBe(false);
  return result.issues.map(({ code }) => code);
}

describe("thin scientific math acceptance policy", () => {
  it("documents prompt-primary role and the only allowed rewrite class", () => {
    expect(SCIENTIFIC_MARKDOWN_MATH_POLICY).toEqual({
      role: "thin-acceptance-net",
      happyPath: "prompt-primary-$...$",
      allowedRewrites: ["\\(...\\)->$...$", "simple-one-line-\\[...\\]->$...$"],
      onFailure: "reject-with-diagnostics-then-retry-or-fallback",
    });
  });

  it("keeps accepted dollar math byte-stable", () => {
    const input = "Radius $r=800\\,\\mathrm{kpc}$ and uncertainty $24\\pm2$.";
    const result = canonicalizeScientificMarkdownMath(input);
    expect(result).toEqual({ ok: true, value: input, issues: [] });
  });

  it("rewrites only explicitly delimited parentheses and simple one-line brackets", () => {
    expectValid(
      "Radius \\(r=800\\,\\mathrm{kpc}\\), error \\(24\\pm2\\), and \\(\\sigma_{\\rm DM}/m=0\\).",
      "Radius $r=800\\,\\mathrm{kpc}$, error $24\\pm2$, and $\\sigma_{\\rm DM}/m=0$.",
    );
    expectValid("Result \\[x^2+y^2=z^2\\] follows.", "Result $x^2+y^2=z^2$ follows.");
  });

  it("never rewrites rejected input and returns original bytes", () => {
    const samples = [
      "Bare \\alpha outside math.",
      "before $$x=1$$ after",
      "Mass $(0.8$—$0.95)\\times10^{12}\\,M_\\odot$.",
      "Mean redshift $<z>$ is high.",
    ];
    for (const input of samples) {
      const result = canonicalizeScientificMarkdownMath(input);
      expect(result.ok).toBe(false);
      expect(result.value).toBe(input);
    }
  });

  it("is idempotent over accepted values", () => {
    const accepted = [
      "Radius $r=800\\,\\mathrm{kpc}$ and uncertainty $24\\pm2$.",
      "Use $5 x$.",
      "Mean redshift $\\langle z\\rangle$ is high.",
      "Constraint $z<0.5$ and $M>10^{12}$ hold.",
    ];
    for (const input of accepted) {
      const first = canonicalizeScientificMarkdownMath(input);
      expect(first.ok).toBe(true);
      if (!first.ok) continue;
      const second = canonicalizeScientificMarkdownMath(first.value);
      expect(second).toEqual(first);
      expect(second.value).toBe(first.value);
    }
  });
});

describe("scientific Markdown math canonicalization", () => {
  it("preserves valid dollar-delimited inline math", () => {
    expectValid("Radius $r=800\\,\\mathrm{kpc}$ and uncertainty $24\\pm2$.");
    expectValid("Constraint $\\sigma_{\\rm DM}/m=0$ is the baseline.");
  });

  it("canonicalizes parenthesized and simple bracketed TeX delimiters", () => {
    expectValid(
      "Radius \\(r=800\\,\\mathrm{kpc}\\), error \\(24\\pm2\\), and \\(\\sigma_{\\rm DM}/m=0\\).",
      "Radius $r=800\\,\\mathrm{kpc}$, error $24\\pm2$, and $\\sigma_{\\rm DM}/m=0$.",
    );
    expectValid("Result \\[x^2+y^2=z^2\\] follows.", "Result $x^2+y^2=z^2$ follows.");
  });

  it("is idempotent after successful canonicalization", () => {
    const first = canonicalizeScientificMarkdownMath("Use \\(24\\pm2\\) at \\[800\\,\\mathrm{kpc}\\].");
    expect(first.ok).toBe(true);
    if (!first.ok) return;
    expect(canonicalizeScientificMarkdownMath(first.value)).toEqual(first);
  });

  it("keeps numeric-leading canonical math valid on a second pass", () => {
    const first = canonicalizeScientificMarkdownMath(String.raw`Use \(5 x\).`);
    expect(first).toEqual({ ok: true, value: "Use $5 x$.", issues: [] });
    expect(canonicalizeScientificMarkdownMath(first.value)).toEqual(first);
  });
});

describe("scientific Markdown math rejection", () => {
  it.each([
    ["display dollars", "before $$x=1$$ after", "display-math"],
    ["empty dollars", "before $  $ after", "empty-math"],
    ["empty parentheses", "before \\( \\) after", "empty-math"],
    ["unclosed dollars", "before $x=1 after", "unclosed-delimiter"],
    ["unclosed parentheses", "before \\(x=1 after", "unclosed-delimiter"],
    ["unmatched close", "before \\) after", "unmatched-delimiter"],
    ["nested delimiter", "before \\(x+$y$\\) after", "nested-delimiter"],
    ["unbalanced braces", "before $x_{1$ after", "unbalanced-braces"],
    ["unbalanced parentheses", "before $f(x$ after", "unbalanced-parentheses"],
    ["unmatched left", "before $\\left(x$ after", "unmatched-left-right"],
    ["unmatched right", "before $x\\right)$ after", "unmatched-left-right"],
    ["display environment", "before \\begin{align}x&=1\\end{align} after", "display-environment"],
    ["display line break", String.raw`before \[a\\b\] after`, "display-line-break"],
    ["malformed optional argument", String.raw`before $\sqrt[3{x}$ after`, "unbalanced-optional-argument"],
    ["multiline", "before \\[x=1\ny=2\\] after", "multiline"],
  ] as const)("rejects %s", (_name, input, expectedCode) => {
    expect(issueCodes(input)).toContain(expectedCode);
  });

  it.each([
    "The radius is 800\\,\\mathrm{kpc}.",
    "The uncertainty is 24\\pm2.",
    "Assume \\sigma_{\\rm DM}/m=0.",
    "Use \\log x, \\ln x, and \\exp x.",
    "Compare \\sin x, \\cos x, and \\tanh x.",
    "Vectors \\bar x, \\vec v, \\overline{AB}, and \\hat n.",
    "Take \\min x, \\max y, and \\lim_{n \\to \\infty} x_n.",
    "For x \\in A, use x \\lesssim y or x \\gtrsim y.",
    String.raw`Map x \to y with \langle x,y \rangle at 90\degree.`,
    String.raw`Formatting \textbf{bold} and \textnormal{normal} is still a bare TeX command.`,
  ])("rejects high-confidence bare TeX: %s", (input) => {
    expect(issueCodes(input)).toContain("bare-tex");
  });

  it("does not guess a repair for the reduced malformed mass-range regression", () => {
    const input = "Mass $(0.8$—$0.95)\\times10^{12}\\,M_\\odot$.";
    const result = canonicalizeScientificMarkdownMath(input);
    expect(result.ok).toBe(false);
    expect(result.value).toBe(input);
    expect(result.issues.map(({ code }) => code)).toContain("unbalanced-parentheses");
  });

  it("returns deterministic positional diagnostics", () => {
    const input = "A $x_{1$ and \\pm bare";
    expect(canonicalizeScientificMarkdownMath(input)).toEqual(canonicalizeScientificMarkdownMath(input));
    const result = canonicalizeScientificMarkdownMath(input);
    expect(result.ok).toBe(false);
    expect(result.issues).toEqual([...result.issues].sort(
      (left, right) => left.offset - right.offset || left.code.localeCompare(right.code),
    ));
  });
});

describe("protected and non-math Markdown", () => {
  it.each([
    "Code `$x$ \\pm \\(y\\)` stays literal.",
    "Code ``a ` $x$ \\pm` b`` stays literal.",
    "See [query](https://example.org/find?q=$x$&tex=\\pm).",
    'See [query](https://example.org/path "title \\textbf{$x$}").',
    "See ![nested](https://example.org/a_(b)/$x$).",
    "See ![image](https://example.org/$x$ 'title \\textnormal{literal}').",
    "[paper]: https://example.org/$x$/\\pm",
    "[paper]: <https://example.org/$x$/\\pm> 'title \\textbf{literal}'",
    "Visit <https://example.org/$x$/\\pm>.",
    "Write <first.$tag@example.org>.",
    "An escaped dollar \\$5 and \\$x are literal.",
    "Escaped command \\\\left is literal.",
    String.raw`Windows C:\Users\alice\papers and path/to\archive stay prose.`,
    "The fee is $5 and the grant was $1,250.50.",
    "Costs $5 or $10.",
    "$1250 and $5000",
    "Range $(0,1]$ is half-open.",
    "Range $\\left(0,1\\right]$ is half-open.",
    "Label $\\text{case (a}$ is valid text.",
  ])("preserves protected content byte-for-byte: %s", (input) => {
    expectValid(input);
  });

  it.each([
    "Not a link (https://example.org/$x$/\\pm).",
    "Broken [link](https://example.org/$x$/\\pm extra",
    "Malformed <https://example.org/$x$/\\pm space>.",
    "[paper]: https://example.org/$x$/\\pm trailing junk",
  ])("does not let malformed Markdown bypass bare-TeX validation: %s", (input) => {
    expect(issueCodes(input)).toContain("bare-tex");
  });

  it("does not treat a malformed email autolink as protected", () => {
    expect(issueCodes("Malformed <first.$tag@example.org/extra>.")).toContain("unclosed-delimiter");
  });

  it("does not let escaped backticks create a protected code span", () => {
    expect(issueCodes("Escaped \\` \\pm ` remains prose.")).toContain("bare-tex");
  });

  it("rejects a non-interval mismatched parenthesis", () => {
    expect(issueCodes("Value $P(x,y]$ is malformed.")).toContain("unbalanced-parentheses");
  });

  it("does not guess an unclosed numeric dollar is math", () => {
    expectValid("The amount is $5");
  });

  it("still validates math around protected ranges", () => {
    expectValid("`$bad` then \\(x=1\\) and [link](https://x.test/$5)", "`$bad` then $x=1$ and [link](https://x.test/$5)");
  });
});

describe("raw HTML-shaped angle brackets in math", () => {
  it.each([
    "Mean redshift $<z>$ is high.",
    "The average $<v>$ grows and $<n>$ peaks.",
    "Chained $a<b>c$ comparison.",
  ])("rejects %s", (input) => {
    expect(issueCodes(input)).toContain("raw-html-in-math");
  });

  it.each([
    "Mean redshift $\\langle z\\rangle$ is high.",
    "Mean $\\left<z\\right>$ velocity.",
    "Constraint $z<0.5$ and $M>10^{12}$ hold.",
    "Range $a<b$ applies.",
  ])("accepts ordinary or TeX angle-bracket forms: %s", (input) => {
    expectValid(input);
  });

  it("keeps a subscripted angle-bracket average as an intentional boundary", () => {
    // <n_e> is not an HTML tag name, so the renderer leaves it untouched;
    // this is intentionally not hard-rejected by the scanner.
    expectValid("Subscripted average $<n_e>$ is intentionally not hard-rejected.");
  });

  it("rejects exactly when the renderer neutralizes the raw HTML shape", () => {
    expect(normalizeMarkdownLine("$<z>$")).not.toBe("$<z>$");
    expect(normalizeMarkdownLine("$\\left<z\\right>$")).toBe("$\\left<z\\right>$");
  });
});
