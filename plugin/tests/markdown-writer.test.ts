import { describe, it, expect } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { MarkdownWriter } from "../src/pipeline/markdown-writer";
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

  it("writeEmptyDaily throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    await expect(writer.writeEmptyDaily("2026-05-11")).rejects.toThrow(
      /already exists/,
    );
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

  it("writeDaily appends collapsed missed papers", async () => {
    const { files, writer } = makeWriter();
    await writer.writeDaily("2026-05-11", "body\n", {
      missedPapers: [
        {
          id: "2605.11111",
          title: " Missed   Paper ",
          authors: "A. Author et al.",
        },
        {
          id: "2605.22222",
          title: "",
          authors: "",
        },
      ],
    });
    const written = files["arxiv-daily/daily/2026-05-11.md"];
    expect(written).toContain("<details>");
    expect(written).toContain("未入选论文（可能漏报） · 2 篇");
    expect(written).toContain(
      "- [2605.11111](https://arxiv.org/abs/2605.11111) — Missed Paper（A. Author et al.）",
    );
    expect(written).toContain(
      "- [2605.22222](https://arxiv.org/abs/2605.22222) — 2605.22222",
    );
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

  it("writePaperDetail writes date and weekday properties", async () => {
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
    await writer.writePaperDetail(paper as any, "2026-06-10", "detail");
    const written = files["arxiv-daily/papers/2605.06587.md"];
    expect(written).toContain("type: paper");
    expect(written).toContain('arxiv_id: "2605.06587"');
    expect(written).toContain("status: inbox");
    expect(written).toContain("priority: normal");
    expect(written).toContain("date: 2026-06-10");
    expect(written).toContain("weekday: Wednesday");
    expect(written).toContain("detail");
  });

  it("writePaperDetail uses paper index fields when provided", async () => {
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
    expect(written).toContain("status: saved");
    expect(written).toContain("priority: high");
    expect(written).toContain('  - "2026-06-09"');
    expect(written).toContain('zotero_key: "ZOTERO"');
    expect(written).toContain('zotero_uri: "zotero://select/items/ABC123"');
    expect(written).toContain('citation_key: "cite"');
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
      dailyReports: [],
      paperPath: null,
      arxivUrl: "https://arxiv.org/abs/2605.06587",
      pdfUrl: "https://arxiv.org/pdf/2605.06587",
      pdfPath: "",
      zoteroKey: "",
      zoteroUri: "",
      citationKey: "",
      projects: [],
    });
    const written = files["arxiv-daily/papers/2605.06587.md"];
    expect(written).toContain("status: saved");
    expect(written).toContain("- **arXiv**: [2605.06587]");
    expect(written).toContain("## Notes");
  });
});
