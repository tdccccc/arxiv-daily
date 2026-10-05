import { describe, expect, it } from "vitest";
import { describeMarkdown, renderInlineMarkdown, renderMarkdown } from "../src/workbench/markdown";

describe("workbench Markdown reading", () => {
  it("reads writer frontmatter without displaying it and preserves ordinary article formatting", () => {
    const result = renderMarkdown('---\ntitle: "A \\"quoted\\" title"\nauthors: "Ada, Bob"\narxiv_id: "2609.12345"\npublished: "[[arxiv-daily/daily/2026-10-01|2026-10-01]]"\ntags: [arxiv, paper]\n---\n\n# Article\n\n**Evidence** and `code`.\n\n| Method | Score |\n| --- | --- |\n| Ours | 42 |\n\n```js\nconst x = "<script>";\n```');
    expect(result.metadata).toMatchObject({ title: 'A "quoted" title', authors: "Ada, Bob", arxiv_id: "2609.12345", published: "[[arxiv-daily/daily/2026-10-01|2026-10-01]]" });
    expect(result.title).toBe('A "quoted" title');
    expect(result.html).not.toContain("tags:");
    expect(result.html).toContain("<strong>Evidence</strong>");
    expect(result.html).toContain("<table>");
    expect(result.html).toContain('class="language-js"');
    expect(result.html).toContain("&lt;script&gt;");
  });

  it("provides stable unique Unicode headings and an article title fallback", () => {
    const source = "# 研究问题\n\n## 方法 **设计**\n\n## 方法 **设计**\n\n## 方法 设计-2\n\n## !!!";
    const result = renderMarkdown(source);
    expect(result.title).toBe("研究问题");
    expect(result.headings.map(h => h.title)).toEqual(["研究问题", "方法 设计", "方法 设计", "方法 设计-2", "!!!"]);
    expect(new Set(result.headings.map(h => h.id)).size).toBe(5);
    expect(result.headings[0]).toMatchObject({ id: "研究问题", level: 1 });
    expect(renderMarkdown(source).headings).toEqual(result.headings);
    for (const heading of result.headings) expect(result.html).toContain(`id="${heading.id}"`);
  });

  it("renders inline and display scientific math while leaving code intact", () => {
    const result = renderMarkdown("# Science\n\nValue $x^2 + \\alpha$.\n\n$$\n\\frac{a}{b}\n$$\n\n`$not_math$`\n\n```text\n$$not_math$$\n```\n\nCost is $5 and $10.");
    expect(result.html).toContain('class="katex"');
    expect(result.html).toContain('class="katex-display"');
    expect(result.html).toContain("<code>$not_math$</code>");
    expect(result.html).toContain("$$not_math$$");
    expect(result.html).toContain("Cost is $5 and $10.");
    expect(result.html).toContain("<math");
  });

  it("renders standalone display environments across blank lines without Markdown interference", () => {
    for (const [open, close] of [["$$", "$$"], ["\\[", "\\]"]]) {
      const result = renderMarkdown(`Before\n${open}\n\\begin{aligned}\na &= b \\\\\n\n c &= \\begin{pmatrix}1 & 2 \\\\ 3 & 4\\end{pmatrix}\n\\end{aligned}\n${close}\nAfter`);
      expect(result.html.match(/class="katex-display"/g)).toHaveLength(1);
      expect(result.html).not.toContain("katex-error");
      expect(result.html).toContain("<annotation encoding=\"application/x-tex\">");
      expect(result.html).toMatch(/<p>Before<\/p>\s*<span class="katex-display"/);
      expect(result.html).toMatch(/<\/span>\s*<p>After<\/p>/);
    }
  });

  it("renders safe inline previews through the same math and link contract", () => {
    const html = renderInlineMarkdown(String.raw`**Mass** $M_\odot$ and \(x^2\) [web](https://example.com) <img src=x onerror=alert(1)>`, { resolveLink: () => null });
    expect(html).toContain("<strong>Mass</strong>");
    expect(html.match(/class="katex"/g)).toHaveLength(2);
    expect(html).not.toContain("<p>");
    expect(html).not.toContain("<a ");
    expect(html).not.toContain("<img");
  });

  it("preserves escaped delimiters, code, currency and unmatched math", () => {
    const source = String.raw`Escaped \$x\$ and \\(x\\), costs $5 and $10; unmatched $x.

` + "`$x$` and `\\(x\\)`\n\n```latex\n$$x$$\n```\n\n    $$x$$\n";
    const html = renderMarkdown(source).html;
    expect(html).not.toContain('class="katex"');
    expect(html).toContain("costs $5 and $10; unmatched $x.");
    expect(html).toContain("<code>$x$</code>");
    expect(html).toContain('class="language-latex"');
    expect(html).toContain("$$x$$");
  });

  it("supports display math inside lists and quotes without consuming following blocks", () => {
    const result = renderMarkdown("> $$\n> \\begin{aligned}x &= y \\\\\n>\n> y &= z\\end{aligned}\n> $$\n\n- Formula:\n\n  \\[\n  x^2\n\n  + y^2\n  \\]\n- Next item\n\nOutside");
    expect(result.html.match(/class="katex-display"/g)).toHaveLength(2);
    expect(result.html).not.toContain("katex-error");
    expect(result.html).toMatch(/<li>\s*<p>Next item<\/p>\s*<\/li>/);
    expect(result.html).toContain("<p>Outside</p>");
  });

  it("keeps unsupported and unclosed display syntax readable", () => {
    const html = renderMarkdown("$$\n\\unknowncommand{x}\n$$\n\nUnclosed:\n\\[\nx^2\n\nEnd").html;
    expect(html).toContain("unknowncommand");
    expect(html).toContain("x^2");
    expect(html).toContain("<p>End</p>");
    expect(html).not.toContain("undefined");
  });

  it("shares safe link and math policies in inline previews", () => {
    const html = renderInlineMarkdown(String.raw`[web](https://example.com) [bad](safe) $\href{https://example.com}{hidden}$ $\htmlClass{evil}{x}$`, { resolveLink: (target) => target === "safe" ? "javascript:alert(1)" : target });
    expect(html).toContain('href="https://example.com" target="_blank" rel="noopener noreferrer"');
    expect(html.match(/<a /g)).toHaveLength(1);
    expect(html).not.toContain('href="javascript:');
    expect(html).not.toContain('class="evil"');
  });

  it("keeps malformed math readable and does not enable KaTeX trusted links", () => {
    const result = renderMarkdown("$\\unknowncommand{x}$ and $\\href{javascript:alert(1)}{click}$");
    expect(result.html).not.toMatch(/href="javascript:/);
    expect(result.html).toContain("unknowncommand");
  });

  it("resolves writer wikilinks, aliases, relative links and images through the host", () => {
    const targets: Array<[string, string]> = [];
    const result = renderMarkdown("[[2609.12345]] · [[arxiv-daily/daily/2026-10-01|日报]] · [paper](../papers/2609.12345.md)\n\n![Result](figures/result.png)\n\n[[missing]]", { resolveLink(target, kind) {
      targets.push([target, kind]);
      if (target === "missing") return null;
      return kind === "image" ? "/cap/api/asset?id=result" : "/cap/?document=paper";
    } });
    expect(targets).toContainEqual(["2609.12345", "link"]);
    expect(targets).toContainEqual(["arxiv-daily/daily/2026-10-01", "link"]);
    expect(targets).toContainEqual(["../papers/2609.12345.md", "link"]);
    expect(targets).toContainEqual(["figures/result.png", "image"]);
    expect(result.html).toContain('href="/cap/?document=paper"');
    expect(result.html).toContain(">日报</a>");
    expect(result.html).toContain('src="/cap/api/asset?id=result"');
    expect(result.html).toContain("missing");
    expect(result.html).not.toContain('href="missing"');
  });

  it("makes unresolved local content inert and only links safe external destinations by default", () => {
    const result = renderMarkdown("[paper](../paper.md) [[paper]] ![Figure](./figure.png) [web](https://arxiv.org/abs/2609.12345) [mail](mailto:reader@example.com) [section](#results) [bad](javascript:alert(1)) [file](file:///etc/passwd) ![bad image](data:image/svg+xml;base64,PHN2Zz4=)");
    expect(result.html).toContain('href="https://arxiv.org/abs/2609.12345" target="_blank" rel="noopener noreferrer"');
    expect(result.html).toContain('href="mailto:reader@example.com"');
    expect(result.html).toContain('href="#results"');
    expect(result.html).not.toMatch(/(?:href|src)="(?:\.\.|\.\/|file:|javascript:|data:)/);
    expect(result.html).toContain("Figure");
    expect(result.html).not.toContain("<img");
  });

  it("does not trust resolver output or source HTML, and hides bookkeeping outside code", () => {
    const result = renderMarkdown('# <img src=x onerror=alert(1)>\n\n<script>alert(1)</script>\n\n<!-- arxiv-daily:generation-metrics -->\n\nText<br>next\n\n`<!-- visible -->`\n\n```html\n<!-- also visible -->\n```\n\n[bad](safe) ![bad](image) [[bad]]', { resolveLink: () => "javascript:alert(1)" });
    expect(result.html).not.toContain("<script>");
    expect(result.html).not.toMatch(/<img[^>]*onerror/);
    expect(result.html).not.toContain("arxiv-daily:generation-metrics");
    expect(result.html).toContain("&lt;!-- visible --&gt;");
    expect(result.html).toContain("&lt;!-- also visible --&gt;");
    expect(result.html).toMatch(/Text<br>\s*next/);
    expect(result.html).not.toMatch(/(?:href|src)="javascript:/);
  });

  it("describes frontmatter and formatted heading titles without rendering an article", () => {
    for (const source of ['---\r\ntitle: "Saved title"\r\ndate: 2026-10-01\r\n---\r\n# Heading', "## 方法 **设计**\n\n$\\alpha$", "plain text"]) {
      const rendered = renderMarkdown(source);
      expect(describeMarkdown(source)).toEqual({ title: rendered.title, metadata: rendered.metadata });
    }
  });
});

it("preserves heading math delimiters for rendered document titles and navigation", () => {
  const source=String.raw`# Expansion $H_0$

## Matter \(\Omega_m\)`;
  const result=renderMarkdown(source);
  expect(result.title).toBe(String.raw`Expansion $H_0$`);
  expect(describeMarkdown(source).title).toBe(result.title);
  expect(result.headings[1]?.title).toBe(String.raw`Matter $\Omega_m$`);
});

it("separates trailing summary sources and projects saved generation statistics", () => {
 const source = '# Paper\n\nMain result.\n\n## Summary sources\n\nAbstract and Conclusion.\n\n## Original abstract\n\nOriginal text.\n\n<!-- arxiv-daily:generation-metrics -->\n> [!info]- Generation metrics\n> - LLM calls: 1 logical, 1 HTTP attempt\n> - LLM duration: 2.5 s\n> - Pipeline wall time: 4.0 s\n> - Provider token usage: 100 input / 50 output / 150 total\n';
 const result = renderMarkdown(source);
 expect(result.html).toMatch(/<hr[^>]*class="reading-appendix-divider"[^>]*>\s*<h2[^>]*>Summary sources/);
 expect(result.html).not.toContain('Generation metrics');
 expect(result.generationMetrics).toMatchObject({ inputTokens:100,outputTokens:50,totalTokens:150,elapsedMs:2500,pipelineElapsedMs:4000 });
 expect(result.generationMetrics?.generatedAt).toBeUndefined();
 expect(result.html).toContain('Original text.');
});
