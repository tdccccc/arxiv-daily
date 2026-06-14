import { describe, expect, it } from "vitest";
import { syncDashboardHistory } from "../src/dashboard/history-sync";
import { PaperIndexStore } from "../src/services/paper-index";
import type { StorageAdapter } from "../src/core/adapters";
import type { OutputSettings } from "../src/settings/types";

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
    `arxiv: "${id}"`,
    "date: 2026-06-10",
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
});
