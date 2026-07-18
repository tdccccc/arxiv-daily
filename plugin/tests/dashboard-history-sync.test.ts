import { describe, expect, it, vi } from "vitest";
import { syncDashboardHistory } from "@arxiv-daily/core";
import { PaperIndexStore } from "@arxiv-daily/core";
import type { PaperIndexEntry } from "@arxiv-daily/core";
import type { StorageAdapter } from "@arxiv-daily/core";
import type { OutputSettings } from "@arxiv-daily/core";

const output: OutputSettings = {
  dailyDir: "arxiv/daily",
  papersDir: "arxiv/papers",
};

const topics = [
  {
    id: "photo-z",
    name: "Photo-z",
    tag: "photo-z",
    description: "",
    detail: true,
  },
  {
    id: "cluster",
    name: "Galaxy Cluster",
    tag: "galaxy-cluster",
    description: "",
    detail: true,
  },
];

function makeStorage(initialFiles: Record<string, string> = {}) {
  const files: Record<string, string> = { ...initialFiles };
  const dirs = new Set<string>();
  const storage = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/");
    },
    async readText(path: string) {
      if (!(path in files)) throw new Error(`missing ${path}`);
      return files[path];
    },
    async writeText(path: string, content: string) {
      files[path] = content;
    },
    async exists(path: string) {
      return path in files || dirs.has(path);
    },
    async mkdir(path: string) {
      dirs.add(path);
    },
    async rename(from: string, to: string) {
      if (!(from in files)) throw new Error(`missing ${from}`);
      files[to] = files[from];
      delete files[from];
    },
    async remove(path: string) {
      delete files[path];
      dirs.delete(path);
    },
  } satisfies StorageAdapter;
  return { files, dirs, storage };
}

function makeVault(files: Record<string, string>) {
  return {
    getMarkdownFiles() {
      return Object.keys(files)
        .filter((path) => path.endsWith(".md"))
        .map((path) => ({ path }));
    },
    adapter: {
      async read(path: string) {
        if (!(path in files)) throw new Error(`missing ${path}`);
        return files[path];
      },
    },
  };
}

function detailMarkdown(id: string, title: string): string {
  const paragraph =
    "这里是详细总结正文，包含方法、数据、证据、结论和适用边界。" +
    "这段文字重复出现以保证内容长度足够，避免被当成空白笔记。";
  return [
    "---",
    `title: "${title}"`,
    `authors: "A. Author et al."`,
    `arxiv_id: "${id}"`,
    'daily_report: "[[arxiv/daily/2026-06-10|2026-06-10]]"',
    "tags: [arxiv, paper, photo-z]",
    "---",
    "",
    `# ${title}`,
    "",
    "## 研究问题",
    paragraph.repeat(2),
    "## 方法设计",
    paragraph.repeat(2),
    "## 关键证据",
    paragraph.repeat(2),
    "## 主要结论",
    paragraph.repeat(2),
    "## 适用边界",
    paragraph.repeat(2),
  ].join("\n");
}

function indexJson(
  papers: Record<string, Partial<PaperIndexEntry>>,
): string {
  const out: Record<string, PaperIndexEntry> = {};
  for (const [id, overrides] of Object.entries(papers)) {
    out[id] = indexedPaper(id, overrides);
  }
  return JSON.stringify({
    schemaVersion: 3,
    updatedAt: "2026-06-13T00:00:00.000Z",
    papers: out,
  });
}

function indexedPaper(
  id: string,
  overrides: Partial<PaperIndexEntry> = {},
): PaperIndexEntry {
  return {
    arxivId: id,
    source: "arxiv",
    title: `Paper ${id}`,
    authors: ["A. Author"],
    published: "2026-06-10",
    updated: "2026-06-10",
    category: "photo-z",
    categories: ["photo-z"],
    summary: undefined,
    topics: ["photo-z"],
    primaryTopic: "photo-z",
    detail: false,
    status: "inbox",
    priority: "normal",
    seenDates: ["2026-06-10"],
    dailyReports: [],
    paperPath: null,
    arxivUrl: `https://arxiv.org/abs/${id}`,
    pdfUrl: `https://arxiv.org/pdf/${id}`,
    pdfPath: "",
    zoteroKey: "",
    zoteroUri: "",
    citationKey: "",
    projects: [],
    ...overrides,
  };
}

describe("syncDashboardHistory", () => {
  it("backfills daily papers and detail summary files into the paper index", async () => {
    const { files, storage } = makeStorage({
      "arxiv/daily/2026-06-10.md": [
        "---",
        "date: 2026-06-10",
        "---",
        "",
        "# Daily",
        "",
        "## Photo-z",
        "### Detail Paper → [[2606.00001]]",
        "- **作者**: A. Author et al.",
        "- **arXiv**: [2606.00001](https://arxiv.org/abs/2606.00001)",
        "",
        "### Daily Only Paper",
        "- **作者**: B. Author",
        "- **arXiv**: [2606.00002](https://arxiv.org/abs/2606.00002)",
      ].join("\n"),
      "arxiv/papers/2606.00001.md": detailMarkdown(
        "2606.00001",
        "Detail Paper",
      ),
      "arxiv/papers/2606.00003.md": detailMarkdown(
        "2606.00003",
        "Orphan Detail Paper",
      ),
      "arxiv/papers/2606.00004.md": [
        "---",
        'arxiv: "2606.00004"',
        'title: "Lightweight Note"',
        "date: 2026-06-10",
        "---",
        "",
        "# Lightweight Note",
        "",
        "- **arXiv**: [2606.00004](https://arxiv.org/abs/2606.00004)",
        "",
        "## Notes",
      ].join("\n"),
    });
    const store = new PaperIndexStore(
      storage,
      output,
      () => new Date("2026-06-14T00:00:00.000Z"),
    );

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(Object.keys(index.papers).sort()).toEqual([
      "2606.00001",
      "2606.00002",
      "2606.00003",
    ]);
    expect(index.papers["2606.00001"]).toMatchObject({
      title: "Detail Paper",
      authors: ["A. Author et al."],
      primaryTopic: "photo-z",
      detail: true,
      paperPath: "arxiv/papers/2606.00001.md",
      dailyReports: ["arxiv/daily/2026-06-10.md"],
    });
    expect(index.papers["2606.00002"]).toMatchObject({
      title: "Daily Only Paper",
      authors: ["B. Author"],
      primaryTopic: "photo-z",
      detail: false,
      paperPath: null,
      dailyReports: ["arxiv/daily/2026-06-10.md"],
    });
    expect(index.papers["2606.00003"]).toMatchObject({
      title: "Orphan Detail Paper",
      detail: true,
      paperPath: "arxiv/papers/2606.00003.md",
      dailyReports: [],
    });
  });

  it("backfills structured summaries with newer non-empty fields winning", async () => {
    const daily = (id: string, fields: string[]) => [
      "# Daily",
      "",
      "## Photo-z",
      "### Summary Paper",
      `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
      ...fields,
    ].join("\n");
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        "2606.40000": {
          seenDates: ["2026-06-10", "2026-06-11"],
          dailyReports: [
            "arxiv/daily/2026-06-10.md",
            "arxiv/daily/2026-06-11.md",
          ],
          summary: { limitations: "Existing limitation." },
        },
      }),
      "arxiv/daily/2026-06-11.md": daily("2606.40000", [
        "- **核心问题**: New problem.",
        "- **核心结果**: New result.",
      ]),
      "arxiv/daily/2026-06-10.md": daily("2606.40000", [
        "- **核心问题**: Old problem.",
        "- **方法设计**: Old method.",
      ]),
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers["2606.40000"]?.summary).toEqual({
      coreProblem: "New problem.",
      keyMethod: "Old method.",
      mainResult: "New result.",
      limitations: "Existing limitation.",
    });
  });

  it("does not persist when historical summaries already match", async () => {
    const markdown = [
      "# Daily",
      "",
      "## Photo-z",
      "### Summary Paper",
      "- **arXiv**: [2606.40001](https://arxiv.org/abs/2606.40001)",
      "- **核心问题**: Same problem.",
    ].join("\n");
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        "2606.40001": {
          seenDates: ["2026-06-10"],
          dailyReports: ["arxiv/daily/2026-06-10.md"],
          summary: { coreProblem: "Same problem." },
        },
      }),
      "arxiv/daily/2026-06-10.md": markdown,
    });
    const store = new PaperIndexStore(storage, output);
    const write = vi.spyOn(storage, "writeText");

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers["2606.40001"]?.summary).toEqual({
      coreProblem: "Same problem.",
    });
    expect(write).not.toHaveBeenCalled();
  });

  it("clears deleted detail summaries while keeping daily report papers", async () => {
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        "2606.00001": {
          detail: true,
          paperPath: "arxiv/papers/2606.00001.md",
          dailyReports: ["arxiv/daily/2026-06-10.md"],
        },
      }),
      "arxiv/daily/2026-06-10.md": [
        "# Daily",
        "",
        "## Photo-z",
        "### Deleted Detail Paper -> [[2606.00001]]",
        "- **Authors**: A. Author",
        "- **arXiv**: [2606.00001](https://arxiv.org/abs/2606.00001)",
      ].join("\n"),
    });
    const store = new PaperIndexStore(
      storage,
      output,
      () => new Date("2026-06-14T00:00:00.000Z"),
    );

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers["2606.00001"]).toMatchObject({
      detail: false,
      paperPath: null,
      dailyReports: ["arxiv/daily/2026-06-10.md"],
    });
  });

  it("removes orphan detail entries when their paper file is gone", async () => {
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        "2606.00003": {
          title: "Orphan Detail Paper",
          detail: true,
          paperPath: "arxiv/papers/2606.00003.md",
          dailyReports: [],
        },
      }),
    });
    const store = new PaperIndexStore(
      storage,
      output,
      () => new Date("2026-06-14T00:00:00.000Z"),
    );

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers["2606.00003"]).toBeUndefined();
  });

  it("prunes deleted daily report references from the paper index", async () => {
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        "2606.10000": {
          title: "Deleted Daily Only",
          seenDates: ["2026-06-15"],
          dailyReports: ["arxiv/daily/2026-06-15.md"],
        },
        "2606.10001": {
          title: "Still Reported",
          seenDates: ["2026-06-14", "2026-06-15"],
          dailyReports: [
            "arxiv/daily/2026-06-14.md",
            "arxiv/daily/2026-06-15.md",
          ],
        },
        "2606.10002": {
          title: "Removed From Existing Daily",
          seenDates: ["2026-06-14"],
          dailyReports: ["arxiv/daily/2026-06-14.md"],
        },
      }),
      "arxiv/daily/2026-06-14.md": [
        "# Daily",
        "",
        "## Photo-z",
        "### Still Reported",
        "- **Authors**: A. Author",
        "- **arXiv**: [2606.10001](https://arxiv.org/abs/2606.10001)",
      ].join("\n"),
    });
    const store = new PaperIndexStore(
      storage,
      output,
      () => new Date("2026-06-16T00:00:00.000Z"),
    );

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers["2606.10000"]).toBeUndefined();
    expect(index.papers["2606.10002"]).toBeUndefined();
    expect(index.papers["2606.10001"]).toMatchObject({
      seenDates: ["2026-06-14"],
      dailyReports: ["arxiv/daily/2026-06-14.md"],
    });
  });

  it("keeps indexed daily papers when an existing daily file cannot be read", async () => {
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        "2606.15000": {
          title: "Unreadable Daily Paper",
          seenDates: ["2026-06-15"],
          dailyReports: ["arxiv/daily/2026-06-15.md"],
        },
      }),
      "arxiv/daily/2026-06-15.md": "# temporarily unavailable",
    });
    const store = new PaperIndexStore(
      storage,
      output,
      () => new Date("2026-06-16T00:00:00.000Z"),
    );
    const vault = makeVault(files);
    vault.adapter.read = async (path: string) => {
      if (path === "arxiv/daily/2026-06-15.md") {
        throw new Error("sync in progress");
      }
      return files[path];
    };

    const index = await syncDashboardHistory({
      vault,
      store,
      output,
      topics,
    });

    expect(index.papers["2606.15000"]).toMatchObject({
      seenDates: ["2026-06-15"],
      dailyReports: ["arxiv/daily/2026-06-15.md"],
    });
  });

  it("does not backfill missing daily reports from detail frontmatter", async () => {
    const { files, storage } = makeStorage({
      "arxiv/papers/2606.20000.md": detailMarkdown(
        "2606.20000",
        "Manual Detail",
      ),
    });
    const store = new PaperIndexStore(
      storage,
      output,
      () => new Date("2026-06-16T00:00:00.000Z"),
    );

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers["2606.20000"]).toMatchObject({
      detail: true,
      paperPath: "arxiv/papers/2606.20000.md",
      dailyReports: [],
    });
  });

  it("can reuse a pre-scanned markdown file list instead of repeatedly querying the vault", async () => {
    const { files, storage } = makeStorage({
      "arxiv/daily/2026-06-10.md": [
        "# Daily",
        "",
        "## Photo-z",
        "### Fast Paper",
        "- **Authors**: A. Author",
        "- **arXiv**: [2606.30000](https://arxiv.org/abs/2606.30000)",
      ].join("\n"),
      "arxiv/papers/2606.30001.md": detailMarkdown(
        "2606.30001",
        "Fast Detail",
      ),
    });
    const store = new PaperIndexStore(
      storage,
      output,
      () => new Date("2026-06-16T00:00:00.000Z"),
    );
    const vault = makeVault(files);
    const markdownFiles = vault.getMarkdownFiles();
    vault.getMarkdownFiles = vi.fn(() => {
      throw new Error("sync should use the pre-scanned markdown file list");
    });

    const index = await syncDashboardHistory({
      vault,
      store,
      output,
      topics,
      markdownFiles,
    });

    expect(vault.getMarkdownFiles).not.toHaveBeenCalled();
    expect(Object.keys(index.papers).sort()).toEqual([
      "2606.30000",
      "2606.30001",
    ]);
  });
});
