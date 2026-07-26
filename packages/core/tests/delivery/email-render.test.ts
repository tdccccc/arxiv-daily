import { describe, expect, it } from "vitest";
import { buildDailyDigest } from "../../src/delivery/digest";
import {
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
    expect(html).toContain("arxiv-daily/daily/2026-07-25.md");
    expect(text).toContain("今日无相关论文");
    expect(text).toContain("## Photo-z");
    expect(text).toContain("今日无相关论文更新。");
  });

  it("renders five fields, abs+pdf links, and HTML-escapes prose", () => {
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
    expect(html).toContain("$E=mc^2$");
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain("A &amp; B");
    expect(text).toContain("arXiv: https://arxiv.org/abs/2607.12345");
    expect(text).toContain("PDF: https://arxiv.org/pdf/2607.12345");
    expect(text).toContain("研究问题: problem $E=mc^2$ for 2607.12345");
  });

  it("escapeHtml encodes markup-sensitive characters", () => {
    expect(escapeHtml(`<script>alert("x")</script>`)).toBe(
      "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;",
    );
  });
});
