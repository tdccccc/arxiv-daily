import { describe, it, expect, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { MarkdownWriter } from "../src/pipeline/markdown-writer";
import { assembleDailySummary } from "../src/pipeline/daily-summary-assembler";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { OutputSettings } from "../src/settings/types";

function makeStorage(initialFiles: Record<string, string> = {}) {
  const files: Record<string, string> = { ...initialFiles };
  return {
    files,
    storage: {
      normalizePath(path: string) {
        return path.replace(/\\/g, "/");
      },
      async writeText(path: string, content: string) {
        files[path] = content;
      },
      async readText(path: string) {
        return files[path];
      },
      async exists(path: string) {
        return Object.prototype.hasOwnProperty.call(files, path);
      },
      async mkdir(_path: string) {},
      async rename(from: string, to: string) {
        files[to] = files[from];
        delete files[from];
      },
      async remove(path: string) {
        delete files[path];
      },
      async list(dir: string) {
        return Object.keys(files)
          .filter((path) => path.startsWith(`${dir}/`))
          .map((path) => ({ path, type: "file" as const }));
      },
    } satisfies StorageAdapter,
  };
}

function makeWriter(
  initialFiles: Record<string, string> = {},
  output: Partial<OutputSettings> = {},
) {
  const { files, storage } = makeStorage(initialFiles);
  const writer = new MarkdownWriter({
    storage,
    logger: new Logger("error"),
    arxiv: DEFAULT_SETTINGS.arxiv,
    output: { ...DEFAULT_SETTINGS.output, ...output },
  });
  return { files, writer };
}

describe("MarkdownWriter output boundaries", () => {
  it("rejects portable output directory collisions", () => {
    expect(() => makeWriter({}, {
      dailyDir: "Café/Notes",
      papersDir: "CAFE\u0301/notes",
    })).toThrow(/must be different/);
  });
});

describe("MarkdownWriter existence checks", () => {
  it("dailyExists returns false when daily missing", async () => {
    const { writer } = makeWriter();
    expect(await writer.dailyExists("2026-05-11")).toBe(false);
  });

  it("dailyExists returns true when daily present", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    expect(await writer.dailyExists("2026-05-11")).toBe(true);
  });

  it("paperDetailExists returns false when paper missing", async () => {
    const { writer } = makeWriter();
    expect(await writer.paperDetailExists("2605.06587")).toBe(false);
  });

  it("paperDetailExists returns true when paper present", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/papers/2605.06587.md": "x",
    });
    expect(await writer.paperDetailExists("2605.06587")).toBe(true);
  });
});

describe("MarkdownWriter link style", () => {
  it("uses Obsidian wikilinks by default", () => {
    const { writer } = makeWriter();
    expect(writer.paperDetailLink("2605.06587", "2026-05-11")).toBe(
      "[[2605.06587]]",
    );
  });

  it("uses standard relative markdown links when configured", () => {
    const { writer } = makeWriter(
      {},
      {
        dailyDir: "arxiv-daily/daily",
        papersDir: "arxiv-daily/papers",
        linkStyle: "relative",
      },
    );

    expect(writer.paperDetailLink("2605.06587", "2026-05-11")).toBe(
      "[2605.06587](../papers/2605.06587.md)",
    );
  });

  it("uses existing paper paths for relative links", () => {
    const { writer } = makeWriter({}, { linkStyle: "relative" });

    expect(
      writer.paperDetailLink(
        "2605.06587",
        "2026-05-11",
        "research notes/papers/2605.06587 (saved).md",
      ),
    ).toBe(
      "[2605.06587](../../research%20notes/papers/2605.06587%20%28saved%29.md)",
    );
  });
});

describe("MarkdownWriter strictness on existing files", () => {
  it("writeDaily throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    await expect(writer.writeDaily("2026-05-11", "new")).rejects.toThrow(
      /already exists/,
    );
  });

  it("writePaperDetail throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/papers/2605.06587.md": "x",
    });
    const paper = {
      id: "2605.06587",
      title: "T",
      authors: "A",
      abstract: "",
      category: "photo-z",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
    };
    await expect(writer.writePaperDetail(paper as any, "2026-05-11", "x"))
      .rejects.toThrow(/already exists/);
  });

  it("atomically replaces an existing paper only when explicitly requested", async () => {
    const path = "arxiv-daily/papers/2605.06587.md";
    const { files, writer } = makeWriter({ [path]: "original" });
    const paper = {
      id: "2605.06587",
      title: "T",
      authors: "A",
      abstract: "",
      category: "photo-z",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
    };

    await writer.writePaperDetail(
      paper as any,
      "2026-05-11",
      "replacement",
      undefined,
      { replaceExisting: true },
    );

    expect(files[path]).toContain("replacement");
    expect(files[`${path}.tmp`]).toBeUndefined();
    expect(files[`${path}.bak`]).toBeUndefined();
  });

  it("writeEmptyDaily throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    await expect(writer.writeEmptyDaily("2026-05-11")).rejects.toThrow(
      /already exists/,
    );
  });

  it("writeEmptyDaily uses the configured summary language", async () => {
    const { files, writer } = makeWriter(
      {},
      {
        summaryLanguage: "en",
      },
    );
    await writer.writeEmptyDaily("2026-05-11");

    const written = files["arxiv-daily/daily/2026-05-11.md"];
    expect(written).toContain("# arXiv astro-ph Daily Digest 2026-05-11");
    expect(written).toContain("No relevant papers found today.");
    expect(written).not.toContain("今日未发现相关论文");
  });

  it("writeDaily writes content (no bak file produced)", async () => {
    const { files, writer } = makeWriter();
    await writer.writeDaily("2026-05-11", "body");
    const written = files["arxiv-daily/daily/2026-05-11.md"];
    expect(written).toContain("date: 2026-05-11");
    expect(written).toContain("weekday: Monday");
    expect(written).toContain("body");
    expect(files["arxiv-daily/daily/2026-05-11.bak.md"]).toBeUndefined();
  });

  it("persists assembled scientific Markdown exactly between frontmatter and metrics", async () => {
    const { files, writer } = makeWriter();
    const scientific = {
      id: "2607.12345",
      coreProblem: String.raw`Measure $\mathrm{NMAD}$ and $\eta$ for PS1+WISE at z<0.1 and z>3.5.`,
      keyMethod: String.raw`Use \(r_{\rm cut}/R_{\rm vir}\) with M_\odot and \left|x\right|.`,
      mainResult: "A & B remain distinguishable.",
      whyRelevant: String.raw`The $\mathrm{NMAD}$ trend survives.`,
      limitations: "PS1+WISE coverage is finite.",
    };
    const body = assembleDailySummary({
      dateStr: "2026-05-11",
      arxivSettings: {
        ...DEFAULT_SETTINGS.arxiv,
        topics: [{ id: "science", name: "Science", tag: "science", description: "", detail: false }],
      },
      summaryLanguage: "en",
      slots: [{
        paper: {
          id: scientific.id,
          title: "Scientific report",
          authors: "A & B",
          category: "science",
          sourceSections: "Abstract, Results",
          isDetail: false,
        },
        result: { kind: "structured", summary: scientific },
      }],
    });

    await writer.writeDaily("2026-05-11", body, {
      metrics: { logicalCalls: 1, attempts: 1, elapsedMs: 25, usageComplete: false },
    });

    const path = "arxiv-daily/daily/2026-05-11.md";
    const written = files[path]!;
    const frontmatter = "---\ndate: 2026-05-11\nweekday: Monday\ntags: [arxiv, daily]\n---\n\n";
    const metricsStart = written.indexOf("\n\n<!-- arxiv-daily:generation-metrics -->");
    expect(metricsStart).toBeGreaterThan(0);
    expect(written.slice(frontmatter.length, metricsStart)).toBe(body);
    for (const exact of [
      String.raw`$\mathrm{NMAD}$`,
      String.raw`$\eta$`,
      String.raw`\(r_{\rm cut}/R_{\rm vir}\)`,
      String.raw`M_\odot`,
      String.raw`\left|x\right|`,
      "PS1+WISE",
      "z<0.1",
      "z>3.5",
      "A & B",
    ]) expect(written).toContain(exact);
    for (const corruption of ["\\\\mathrm", "\\\\eta", "&lt;0.1", "&gt;3.5", "&amp; B"]) {
      expect(written).not.toContain(corruption);
    }
    expect(files[`${path}.tmp`]).toBeUndefined();
    expect(files[`${path}.bak`]).toBeUndefined();
  });

  it("appends folded generation metrics at the absolute end without changing frontmatter", async () => {
    const { files, writer } = makeWriter();
    await writer.writeDaily("2026-05-11", "body", {
      metrics: {
        logicalCalls: 2,
        attempts: 3,
        elapsedMs: 2500,
        pipelineElapsedMs: 5000,
        usageComplete: true,
        inputTokens: 100,
        outputTokens: 25,
        totalTokens: 125,
      },
    });
    const written = files["arxiv-daily/daily/2026-05-11.md"]!;
    expect(written).toContain("date: 2026-05-11\nweekday: Monday\ntags: [arxiv, daily]");
    expect(written).toMatch(/<!-- arxiv-daily:generation-metrics -->\n> \[!info\]- Generation metrics/);
    expect(written).toMatch(/Provider token usage: 100 input \/ 25 output \/ 125 total\n$/);
  });

  it("writePaperDetail appends only supplied detail-call metrics", async () => {
    const { files, writer } = makeWriter();
    const paper = {
      id: "2605.06587", title: "T", authors: "A", abstract: "", category: "photo-z",
      isDetail: true, abstractConclusion: "", fullSections: "sections",
    };
    await writer.writePaperDetail(paper as any, "2026-05-11", "detail", undefined, {
      metrics: {
        logicalCalls: 1, attempts: 2, elapsedMs: 800, usageComplete: false,
      },
    });
    const written = files["arxiv-daily/papers/2605.06587.md"]!;
    expect(written).toContain("LLM calls: 1 logical, 2 HTTP attempts");
    expect(written).not.toContain("Pipeline wall time");
    expect(written).toMatch(/Provider token usage: unavailable or incomplete\n$/);
  });

  it("writeDaily uses storage writeTextAtomic when available", async () => {
    const { files, storage } = makeStorage();
    const writeTextAtomic = vi.fn(async (path: string, content: string) => {
      files[path] = content;
    });
    const writer = new MarkdownWriter({
      storage: { ...storage, writeTextAtomic },
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      output: DEFAULT_SETTINGS.output,
    });

    await writer.writeDaily("2026-05-11", "body");

    expect(writeTextAtomic).toHaveBeenCalledWith(
      "arxiv-daily/daily/2026-05-11.md",
      expect.stringContaining("body"),
    );
  });

  it("cleanupTemporaryFiles removes stale markdown temp files from output dirs", async () => {
    const { files, writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md.tmp": "partial",
      "arxiv-daily/papers/2605.06587.md.tmp": "partial",
      "arxiv-daily/daily/2026-05-11.md": "ok",
    });

    await expect(writer.cleanupTemporaryFiles()).resolves.toEqual([
      "arxiv-daily/daily/2026-05-11.md.tmp",
      "arxiv-daily/papers/2605.06587.md.tmp",
    ]);

    expect(files["arxiv-daily/daily/2026-05-11.md.tmp"]).toBeUndefined();
    expect(files["arxiv-daily/papers/2605.06587.md.tmp"]).toBeUndefined();
    expect(files["arxiv-daily/daily/2026-05-11.md"]).toBe("ok");
  });

  it("writeDaily includes submitted-date fallback notes", async () => {
    const { files, writer } = makeWriter();
    await writer.writeDaily("2026-05-11", "body", {
      dateWindowNote: "submittedDate fallback",
    });

    const written = files["arxiv-daily/daily/2026-05-11.md"];
    expect(written).toContain("> submittedDate fallback");
    expect(written.indexOf("> submittedDate fallback")).toBeLessThan(
      written.indexOf("body"),
    );
  });

  it("writePaperDetail writes published date as a daily report link when that report exists", async () => {
    const { files, writer } = makeWriter({
      "arxiv-daily/daily/2026-06-09.md": "daily",
    });
    const paper = {
      id: "2605.06587",
      title: "T",
      authors: "A",
      abstract: "",
      category: "photo-z",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
      published: "2026-06-09T02:28:06Z",
    };
    await writer.writePaperDetail(paper as any, "2026-06-10", "detail");
    const written = files["arxiv-daily/papers/2605.06587.md"];
    expect(written).toContain('arxiv_id: "2605.06587"');
    expect(written).toContain(
      'published: "[[arxiv-daily/daily/2026-06-09|2026-06-09]]"',
    );
    expect(written).not.toContain("daily_report:");
    expect(written).not.toContain("type: paper");
    expect(written).not.toContain("source: arxiv");
    expect(written).not.toContain('arxiv: "2605.06587"');
    expect(written).not.toContain("date: 2026-06-10");
    expect(written).not.toContain("weekday: Wednesday");
    expect(written).not.toContain("status: inbox");
    expect(written).not.toContain("priority: normal");
    expect(written).not.toContain("seen_dates");
    expect(written).toContain("detail");
  });

  it("writePaperDetail writes plain published date when no matching daily report exists", async () => {
    const { files, writer } = makeWriter();
    const paper = {
      id: "2605.06587",
      title: "T",
      authors: "A",
      abstract: "",
      category: "photo-z",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
      published: "2026-06-09T02:28:06Z",
    };
    await writer.writePaperDetail(paper as any, "2026-06-10", "detail");
    const written = files["arxiv-daily/papers/2605.06587.md"];
    expect(written).toContain("published: 2026-06-09");
    expect(written).not.toContain("[[arxiv-daily/daily/2026-06-09");
    expect(written).not.toContain("daily_report:");
  });

  it("writePaperDetail prefers the daily report date over stale Atom published dates", async () => {
    const { files, writer } = makeWriter({
      "arxiv-daily/daily/2026-06-12.md": "daily",
    });
    const paper = {
      id: "2606.12938",
      title: "Cluster Mass Inference from Galaxy Kinematics",
      authors: "Bonny Y. Wang et al.",
      abstract: "",
      category: "galaxy-cluster",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
      published: "2026-06-11",
    };
    await writer.writePaperDetail(paper as any, "2026-06-16", "detail", {
      arxivId: "2606.12938",
      source: "arxiv",
      title: "Cluster Mass Inference from Galaxy Kinematics",
      authors: ["Bonny Y. Wang et al."],
      published: "2026-06-11",
      updated: "2026-06-11",
      category: "astro-ph.CO",
      topics: ["galaxy-cluster"],
      primaryTopic: "galaxy-cluster",
      detail: true,
      status: "inbox",
      priority: "high",
      seenDates: ["2026-06-12"],
      dailyReports: ["arxiv-daily/daily/2026-06-12.md"],
      paperPath: null,
      arxivUrl: "https://arxiv.org/abs/2606.12938",
      pdfUrl: "https://arxiv.org/pdf/2606.12938",
      pdfPath: "",
      zoteroKey: "",
      zoteroUri: "",
      citationKey: "",
      projects: [],
    });
    const written = files["arxiv-daily/papers/2606.12938.md"];
    expect(written).toContain(
      'published: "[[arxiv-daily/daily/2026-06-12|2026-06-12]]"',
    );
    expect(written).not.toContain("2026-06-11");
  });

  it("writePaperDetail does not mirror paper index state into properties", async () => {
    const { files, writer } = makeWriter();
    const paper = {
      id: "2605.06587",
      title: "T",
      authors: "A",
      abstract: "",
      category: "photo-z",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
    };
    await writer.writePaperDetail(paper as any, "2026-06-10", "detail", {
      arxivId: "2605.06587",
      source: "arxiv",
      title: "T",
      authors: ["A"],
      published: "2026-06-09",
      updated: "2026-06-10",
      category: "astro-ph",
      topics: ["photo-z"],
      primaryTopic: "photo-z",
      detail: true,
      status: "saved",
      priority: "high",
      seenDates: ["2026-06-09", "2026-06-10"],
      dailyReports: [],
      paperPath: null,
      arxivUrl: "https://arxiv.org/abs/2605.06587",
      pdfUrl: "https://arxiv.org/pdf/2605.06587",
      pdfPath: "",
      zoteroKey: "ZOTERO",
      zoteroUri: "zotero://select/items/ABC123",
      citationKey: "cite",
      projects: [],
    });
    const written = files["arxiv-daily/papers/2605.06587.md"];
    expect(written).toContain("primary_topic: photo-z");
    expect(written).toContain("published: 2026-06-09");
    expect(written).not.toContain("status: saved");
    expect(written).not.toContain("priority: high");
    expect(written).not.toContain('  - "2026-06-09"');
    expect(written).not.toContain("zotero_key");
    expect(written).not.toContain("zotero_uri");
    expect(written).not.toContain("citation_key");
  });

  it("writePaperNote creates a lightweight note from an index entry", async () => {
    const { files, writer } = makeWriter();
    await writer.writePaperNote({
      arxivId: "2605.06587",
      source: "arxiv",
      title: "T",
      authors: ["A"],
      published: "2026-06-09",
      updated: "2026-06-10",
      category: "astro-ph",
      topics: ["photo-z"],
      primaryTopic: "photo-z",
      detail: false,
      status: "saved",
      priority: "normal",
      seenDates: ["2026-06-10"],
      dailyReports: ["arxiv-daily/daily/2026-06-10.md"],
      paperPath: null,
      arxivUrl: "javascript:alert(1)",
      pdfUrl: "https://evil.test/file.pdf",
      pdfPath: "",
      zoteroKey: "",
      zoteroUri: "",
      citationKey: "",
      projects: [],
    });
    const written = files["arxiv-daily/papers/2605.06587.md"];
    expect(written).toContain("published: 2026-06-10");
    expect(written).toContain("https://arxiv.org/abs/2605.06587");
    expect(written).toContain("https://arxiv.org/pdf/2605.06587");
    expect(written).not.toContain("evil.test");
    expect(written).not.toContain("javascript:");
    expect(written).not.toContain("daily_report:");
    expect(written).not.toContain("status: saved");
    expect(written).not.toContain("priority: normal");
    expect(written).toContain("- **arXiv**: [2605.06587]");
    expect(written).toContain("## Notes");
  });

  it("refreshPaperNoteFrontmatter preserves body and uses the daily report date", async () => {
    const { files, writer } = makeWriter({
      "arxiv-daily/daily/2026-06-12.md": "daily",
      "arxiv-daily/papers/2606.12938.md": [
        "---",
        'published: "[[arxiv-daily/daily/2026-06-11|2026-06-11]]"',
        "---",
        "",
        "# Cluster Mass Inference from Galaxy Kinematics",
        "",
        "body",
      ].join("\n"),
    });

    await writer.refreshPaperNoteFrontmatter({
      arxivId: "2606.12938",
      source: "arxiv",
      title: "Cluster Mass Inference from Galaxy Kinematics",
      authors: ["Bonny Y. Wang et al."],
      published: "2026-06-11",
      updated: "2026-06-11",
      category: "astro-ph.CO",
      topics: ["galaxy-cluster"],
      primaryTopic: "galaxy-cluster",
      detail: true,
      status: "inbox",
      priority: "high",
      seenDates: ["2026-06-12"],
      dailyReports: ["arxiv-daily/daily/2026-06-12.md"],
      paperPath: "arxiv-daily/papers/2606.12938.md",
      arxivUrl: "https://arxiv.org/abs/2606.12938",
      pdfUrl: "https://arxiv.org/pdf/2606.12938",
      pdfPath: "",
      zoteroKey: "",
      zoteroUri: "",
      citationKey: "",
      projects: [],
    });

    const written = files["arxiv-daily/papers/2606.12938.md"];
    expect(written).toContain(
      'published: "[[arxiv-daily/daily/2026-06-12|2026-06-12]]"',
    );
    expect(written).not.toContain("2026-06-11");
    expect(written).toContain("# Cluster Mass Inference from Galaxy Kinematics");
    expect(written).toContain("body");
  });
});
