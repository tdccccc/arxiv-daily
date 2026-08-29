import { describe, expect, it, vi } from "vitest";
import {
  PaperSearchIndex,
  queryDashboard,
  syncDashboardHistory,
  renderDiscoveryProvenanceMarker,
  renderPersonalNoveltyMarker,
} from "@arxiv-daily/core";
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
    discoveryProvenanceByReport: {},
    noveltyByReport: {},
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
  it("rebuilds occurrence provenance and removes it only for no-longer-managed reports", async () => {
    const provenance = {
      manualTopicTags: ["photo-z"],
      directions: [{
        id: "direction-1", name: "Library direction", representatives: [{
          paperKey: "arxiv:2501.00001", title: "Prior", evidenceDepth: "metadata-and-abstract" as const,
        }],
      }],
    };
    const marker = renderDiscoveryProvenanceMarker(provenance, "2606.00001", "2026-06-10");
    const { files, storage } = makeStorage({
      "arxiv/daily/2026-06-10.md": [
        "## Photo-z", "### Paper", marker,
        "- **Authors**: A", "- **arXiv**: [2606.00001](https://arxiv.org/abs/2606.00001)",
      ].join("\n"),
      "arxiv/.index/papers.json": indexJson({
        "2606.00001": {
          dailyReports: ["arxiv/daily/2026-06-09.md", "external/daily.md"],
          discoveryProvenanceByReport: {
            "arxiv/daily/2026-06-09.md": provenance,
            "external/daily.md": provenance,
          },
        },
      }),
    });
    const store = new PaperIndexStore(storage, output);
    const index = await syncDashboardHistory({ vault: makeVault(files), store, output, topics });
    expect(index.papers["arxiv:2606.00001"]!.discoveryProvenanceByReport).toEqual({
      "external/daily.md": provenance,
      "arxiv/daily/2026-06-10.md": provenance,
    });

    files["arxiv/daily/2026-06-10.md"] = files["arxiv/daily/2026-06-10.md"]!.replace(
      marker,
      `${marker}\n${marker}`,
    );
    const afterInvalidEdit = await syncDashboardHistory({
      vault: makeVault(files), store, output, topics,
    });
    expect(afterInvalidEdit.papers["arxiv:2606.00001"]!.discoveryProvenanceByReport)
      .toEqual({
        "external/daily.md": provenance,
        "arxiv/daily/2026-06-10.md": provenance,
      });
  });
  it("rebuilds occurrence novelty and removes it only for no-longer-managed reports", async () => {
    const novelty = {
      differenceType: "new-method",
      comparisonBasis: ["arxiv:2501.00001"],
      evidenceDepth: "metadata-and-abstract",
      explanation: "New method explanation.",
    };
    const marker = renderPersonalNoveltyMarker(novelty, "2606.00001", "2026-06-10");
    const { files, storage } = makeStorage({
      "arxiv/daily/2026-06-10.md": [
        "## Photo-z", "### Paper", marker,
        "- **Authors**: A", "- **arXiv**: [2606.00001](https://arxiv.org/abs/2606.00001)",
      ].join("\n"),
      "arxiv/.index/papers.json": indexJson({
        "2606.00001": {
          dailyReports: ["arxiv/daily/2026-06-09.md", "external/daily.md"],
          discoveryProvenanceByReport: {},
          noveltyByReport: {
            "arxiv/daily/2026-06-09.md": novelty,
            "external/daily.md": novelty,
          },
        },
      }),
    });
    const store = new PaperIndexStore(storage, output);
    const index = await syncDashboardHistory({ vault: makeVault(files), store, output, topics });
    expect(index.papers["arxiv:2606.00001"]!.noveltyByReport).toEqual({
      "external/daily.md": novelty,
      "arxiv/daily/2026-06-10.md": novelty,
    });

    files["arxiv/daily/2026-06-10.md"] = files["arxiv/daily/2026-06-10.md"]!.replace(
      marker,
      `${marker}\n${marker}`,
    );
    const afterInvalidEdit = await syncDashboardHistory({
      vault: makeVault(files), store, output, topics,
    });
    expect(afterInvalidEdit.papers["arxiv:2606.00001"]!.noveltyByReport)
      .toEqual({
        "external/daily.md": novelty,
        "arxiv/daily/2026-06-10.md": novelty,
      });
  });
  it("still rebuilds discovery provenance when novelty markers are invalid", async () => {
    const provenance = {
      manualTopicTags: ["photo-z"],
      directions: [{
        id: "direction-1", name: "Library direction", representatives: [{
          paperKey: "arxiv:2501.00001", title: "Prior", evidenceDepth: "metadata-and-abstract" as const,
        }],
      }],
    };
    const novelty = {
      differenceType: "new-method",
      comparisonBasis: ["arxiv:2501.00001"],
      evidenceDepth: "metadata-and-abstract",
      explanation: "New method explanation.",
    };
    const provMarker = renderDiscoveryProvenanceMarker(provenance, "2606.00001", "2026-06-10");
    const novMarker = renderPersonalNoveltyMarker(novelty, "2606.00001", "2026-06-10");
    const { files, storage } = makeStorage({
      "arxiv/daily/2026-06-10.md": [
        "## Photo-z", "### Paper", provMarker, novMarker, novMarker,
        "- **Authors**: A", "- **arXiv**: [2606.00001](https://arxiv.org/abs/2606.00001)",
      ].join("\n"),
      "arxiv/.index/papers.json": indexJson({
        "2606.00001": {
          dailyReports: ["arxiv/daily/2026-06-09.md", "external/daily.md"],
          discoveryProvenanceByReport: {
            "arxiv/daily/2026-06-09.md": provenance,
            "external/daily.md": provenance,
          },
          noveltyByReport: {
            "arxiv/daily/2026-06-09.md": novelty,
            "external/daily.md": novelty,
          },
        },
      }),
    });
    const store = new PaperIndexStore(storage, output);
    const index = await syncDashboardHistory({ vault: makeVault(files), store, output, topics });
    expect(index.papers["arxiv:2606.00001"]!.discoveryProvenanceByReport).toEqual({
      "external/daily.md": provenance,
      "arxiv/daily/2026-06-10.md": provenance,
    });
    // Invalid novelty markers disable only the novelty projection: the stale
    // managed-report novelty is removed, the current report is not written,
    // and unmanaged novelty survives.
    expect(index.papers["arxiv:2606.00001"]!.noveltyByReport).toEqual({
      "external/daily.md": novelty,
    });
  });
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
      "arxiv:2606.00001",
      "arxiv:2606.00002",
      "arxiv:2606.00003",
    ]);
    expect(index.papers["arxiv:2606.00001"]).toMatchObject({
      title: "Detail Paper",
      authors: ["A. Author et al."],
      primaryTopic: "photo-z",
      detail: true,
      paperPath: "arxiv/papers/2606.00001.md",
      dailyReports: ["arxiv/daily/2026-06-10.md"],
    });
    expect(index.papers["arxiv:2606.00002"]).toMatchObject({
      title: "Daily Only Paper",
      authors: ["B. Author"],
      primaryTopic: "photo-z",
      detail: false,
      paperPath: null,
      dailyReports: ["arxiv/daily/2026-06-10.md"],
    });
    expect(index.papers["arxiv:2606.00003"]).toMatchObject({
      title: "Orphan Detail Paper",
      detail: true,
      paperPath: "arxiv/papers/2606.00003.md",
      dailyReports: [],
    });
  });

  it("does not index a detail file whose frontmatter identity mismatches its managed path", async () => {
    const { files, storage } = makeStorage({
      "arxiv/papers/2606.00010.md": detailMarkdown(
        "2606.99999",
        "Mismatched Detail Paper",
      ),
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers["arxiv:2606.00010"]).toBeUndefined();
    expect(index.papers["arxiv:2606.99999"]).toBeUndefined();
  });

  it("re-scans persisted mixed scientific reports without decoding or fallback misclassification", async () => {
    const structuredId = "2606.30001";
    const fallbackId = "2606.30002";
    const structuredMath = String.raw`$\mathrm{NMAD}$ and $\eta$ at z<0.1`;
    const fallbackMath = String.raw`\(r_{\rm cut}/R_{\rm vir}\) with M_\odot and \left|x\right| at z>3.5`;
    const { files, storage } = makeStorage({
      "arxiv/daily/2026-06-10.md": [
        "<!-- arxiv-daily-emergency-report:v1 -->",
        "## Photo-z",
        "### Structured Paper",
        `- **arXiv**: [${structuredId}](https://arxiv.org/abs/${structuredId})`,
        `- **研究问题**: ${structuredMath}; PS1+WISE A & B.`,
        "### Fallback Paper",
        `<!-- arxiv-daily-fallback:${fallbackId} -->`,
        `- **arXiv**: [${fallbackId}](https://arxiv.org/abs/${fallbackId})`,
        `- **原始摘要**: ${fallbackMath}; Trusted original abstract.`,
      ].join("\n"),
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(Object.keys(index.papers).sort()).toEqual([`arxiv:${structuredId}`, `arxiv:${fallbackId}`].sort());
    expect(index.papers[`arxiv:${structuredId}`]?.summary).toEqual({
      coreProblem: `${structuredMath}; PS1+WISE A & B.`,
    });
    expect(index.papers[`arxiv:${fallbackId}`]?.summary).toBeUndefined();
    expect(index.papers[`arxiv:${fallbackId}`]?.abstract).toBe(
      `${fallbackMath}; Trusted original abstract.`,
    );
    expect(index.papers[`arxiv:${fallbackId}`]?.dailyReports).toEqual([
      "arxiv/daily/2026-06-10.md",
    ]);

    const entries = Object.values(index.papers);
    expect(queryDashboard(entries, {
      tab: "all",
      search: "Trusted original abstract",
    }, {
      searchIndex: new PaperSearchIndex(entries),
    }).rows.map((row) => row.arxivId)).toEqual([fallbackId]);
    expect(queryDashboard(entries, {
      tab: "all",
      search: "Trusted original abstract",
    }, {
      searchIndex: null,
    }).rows.map((row) => row.arxivId)).toEqual([fallbackId]);
    for (const searchIndex of [new PaperSearchIndex(entries), null]) {
      expect(queryDashboard(entries, {
        tab: "all",
        search: String.raw`r_{\rm cut}`,
      }, { searchIndex }).rows.map((row) => row.arxivId)).toEqual([fallbackId]);
      expect(queryDashboard(entries, {
        tab: "all",
        search: String.raw`\mathrm{NMAD}`,
      }, { searchIndex }).rows.map((row) => row.arxivId)).toEqual([structuredId]);
    }
    expect(index.papers[`arxiv:${structuredId}`]?.summary?.coreProblem).not.toContain("&lt;");
    expect(index.papers[`arxiv:${fallbackId}`]?.abstract).not.toContain("\\\\");
  });

  it("fills only missing abstracts from the earliest fallback without downgrading canonical data", async () => {
    const canonicalId = "2606.30501";
    const englishId = "2606.30502";
    const chineseId = "2606.30503";
    const canonicalAbstract = [
      "Canonical first line from Atom.",
      "Second line keeps <b>raw HTML</b> and <!-- upstream comment --> exactly.",
    ].join("\n");
    const englishRecovered = "English fallback with distinctive lensing tomography evidence.";
    const chineseRecovered = "中文回退摘要包含独特的星系团透镜证据。";
    const daily = (
      date: string,
      papers: Array<{ id: string; title: string; label: string; abstract: string }>,
    ) => [
      `# Daily ${date}`,
      "",
      "## Photo-z",
      ...papers.flatMap((paper) => [
        `### ${paper.title}`,
        `<!-- arxiv-daily-fallback:${paper.id} -->`,
        `- **arXiv**: [${paper.id}](https://arxiv.org/abs/${paper.id})`,
        `- **${paper.label}**: ${paper.abstract}`,
      ]),
    ].join("\n");
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [canonicalId]: {
          abstract: canonicalAbstract,
          dailyReports: [],
        },
        [englishId]: {
          abstract: undefined,
          dailyReports: [],
        },
        [chineseId]: {
          abstract: "  \t  ",
          dailyReports: [],
        },
      }),
      "arxiv/daily/2026-06-10.md": daily("2026-06-10", [
        {
          id: canonicalId,
          title: "Canonical Paper",
          label: "Original abstract",
          abstract: "Normalized display fallback with &lt;b>neutralized HTML&lt;/b>.",
        },
        {
          id: englishId,
          title: "English Recovered",
          label: "Original abstract",
          abstract: englishRecovered,
        },
        {
          id: chineseId,
          title: "Chinese Recovered",
          label: "原始摘要",
          abstract: chineseRecovered,
        },
      ]),
      "arxiv/daily/2026-06-11.md": daily("2026-06-11", [
        {
          id: englishId,
          title: "English Recovered Again",
          label: "Original abstract",
          abstract: "Later shorter fallback must not replace recovered text.",
        },
        {
          id: chineseId,
          title: "Chinese Recovered Again",
          label: "原始摘要",
          abstract: "后续回退摘要不得覆盖先前恢复的内容。",
        },
      ]),
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers[`arxiv:${canonicalId}`]?.abstract).toBe(canonicalAbstract);
    expect(index.papers[`arxiv:${englishId}`]?.abstract).toBe(englishRecovered);
    expect(index.papers[`arxiv:${chineseId}`]?.abstract).toBe(chineseRecovered);
    expect(index.papers[`arxiv:${canonicalId}`]?.summary).toBeUndefined();
    expect(index.papers[`arxiv:${englishId}`]?.summary).toBeUndefined();
    expect(index.papers[`arxiv:${chineseId}`]?.summary).toBeUndefined();

    const entries = Object.values(index.papers);
    for (const searchIndex of [new PaperSearchIndex(entries), null]) {
      expect(queryDashboard(entries, {
        tab: "all",
        search: "lensing tomography",
      }, { searchIndex }).rows.map((row) => row.arxivId)).toEqual([englishId]);
      expect(queryDashboard(entries, {
        tab: "all",
        search: "星系团透镜",
      }, { searchIndex }).rows.map((row) => row.arxivId)).toEqual([chineseId]);
      expect(queryDashboard(entries, {
        tab: "all",
        search: "upstream comment",
      }, { searchIndex }).rows.map((row) => row.arxivId)).toEqual([canonicalId]);
    }
  });

  it("does not persist or search localized unavailable abstract placeholders", async () => {
    const englishId = "2606.31001";
    const chineseId = "2606.31002";
    const realId = "2606.31003";
    const { files, storage } = makeStorage({
      "arxiv/daily/2026-06-10.md": [
        "## Photo-z",
        "### English Missing",
        `<!-- arxiv-daily-fallback:${englishId} -->`,
        `<!-- arxiv-daily-fallback-abstract-absent:${englishId} -->`,
        `- **arXiv**: [${englishId}](https://arxiv.org/abs/${englishId})`,
        "- **Original abstract**: Unavailable.",
        "### Chinese Missing",
        `<!-- arxiv-daily-fallback:${chineseId} -->`,
        `<!-- arxiv-daily-fallback-abstract-absent:${chineseId} -->`,
        `- **arXiv**: [${chineseId}](https://arxiv.org/abs/${chineseId})`,
        "- **原始摘要**: 不可用。",
        "### Real Similar Prose",
        `<!-- arxiv-daily-fallback:${realId} -->`,
        `- **arXiv**: [${realId}](https://arxiv.org/abs/${realId})`,
        "- **Original abstract**: Availability is unavailable for one instrument; this is real prose.",
      ].join("\n"),
    });
    const store = new PaperIndexStore(storage, output);
    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers[`arxiv:${englishId}`]?.abstract).toBeUndefined();
    expect(index.papers[`arxiv:${chineseId}`]?.abstract).toBeUndefined();
    expect(index.papers[`arxiv:${realId}`]?.abstract).toBe(
      "Availability is unavailable for one instrument; this is real prose.",
    );
    const entries = Object.values(index.papers);
    for (const searchIndex of [new PaperSearchIndex(entries), null]) {
      const englishMatches = queryDashboard(entries, {
        tab: "all",
        search: "Unavailable.",
      }, { searchIndex }).rows.map((row) => row.arxivId);
      expect(englishMatches).not.toContain(englishId);
      expect(englishMatches).not.toContain(chineseId);
      expect(queryDashboard(entries, { tab: "all", search: "不可用。" }, {
        searchIndex,
      }).rows).toEqual([]);
    }
    expect(queryDashboard(entries, { tab: "all", search: "one instrument" }, {
      searchIndex: new PaperSearchIndex(entries),
    }).rows.map((row) => row.arxivId)).toEqual([realId]);
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

    expect(index.papers["arxiv:2606.40000"]?.summary).toEqual({
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

    expect(index.papers["arxiv:2606.40001"]?.summary).toEqual({
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

    expect(index.papers["arxiv:2606.00001"]).toMatchObject({
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

    expect(index.papers["arxiv:2606.00003"]).toBeUndefined();
  });

  it("preserves concurrent status changes and newly indexed papers while pruning history", async () => {
    const primaryPath = "arxiv/.index/papers.json";
    const { files, storage } = makeStorage({
      [primaryPath]: indexJson({
        "2606.09001": {
          title: "Existing paper",
          seenDates: ["2026-06-10", "2026-06-11"],
          dailyReports: [
            "arxiv/daily/2026-06-10.md",
            "arxiv/daily/2026-06-11.md",
          ],
        },
      }),
      "arxiv/daily/2026-06-10.md": [
        "## Photo-z",
        "### Existing paper",
        "- **Authors**: A. Author",
        "- **arXiv**: [2606.09001](https://arxiv.org/abs/2606.09001)",
      ].join("\n"),
      "arxiv/daily/2026-06-11.md": "# Report without the indexed paper",
    });
    const store = new PaperIndexStore(storage, output);
    const vault = makeVault(files);
    let releaseRead!: () => void;
    let markRead!: () => void;
    const readStarted = new Promise<void>((resolve) => { markRead = resolve; });
    const readRelease = new Promise<void>((resolve) => { releaseRead = resolve; });
    let delayed = false;
    const originalRead = vault.adapter.read;
    vault.adapter.read = async (path: string) => {
      const content = await originalRead(path);
      if (!delayed && path.startsWith("arxiv/daily/")) {
        delayed = true;
        markRead();
        await readRelease;
      }
      return content;
    };

    const syncing = syncDashboardHistory({ vault, store, output, topics });
    await readStarted;
    await store.setStatus("2606.09001", "saved");
    await store.upsertFromDailyPaper({
      arxivId: "2606.09002",
      title: "Concurrent paper",
      authors: "B. Author",
      date: "2026-06-12",
      arxivCategory: "astro-ph",
      primaryTopic: "galaxy-cluster",
      detail: false,
    });
    releaseRead();

    const index = await syncing;

    expect(index.papers["arxiv:2606.09001"]).toMatchObject({
      status: "saved",
      seenDates: ["2026-06-10", "2026-06-11"],
      dailyReports: [
        "arxiv/daily/2026-06-10.md",
        "arxiv/daily/2026-06-11.md",
      ],
    });
    expect(index.papers["arxiv:2606.09002"]?.title).toBe("Concurrent paper");
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

    expect(index.papers["arxiv:2606.10000"]).toBeUndefined();
    expect(index.papers["arxiv:2606.10002"]).toBeUndefined();
    expect(index.papers["arxiv:2606.10001"]).toMatchObject({
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

    expect(index.papers["arxiv:2606.15000"]).toMatchObject({
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

    expect(index.papers["arxiv:2606.20000"]).toMatchObject({
      detail: true,
      paperPath: "arxiv/papers/2606.20000.md",
      dailyReports: [],
    });
  });

  it("preserves unrelated index-only papers across repeated history syncs", async () => {
    const id = "2606.21001";
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [id]: {
          title: "Unrelated index-only paper",
          seenDates: [],
          dailyReports: [],
          detail: false,
          paperPath: null,
        },
      }),
    });
    const store = new PaperIndexStore(storage, output);
    const vault = makeVault(files);

    await syncDashboardHistory({ vault, store, output, topics });
    const index = await syncDashboardHistory({ vault, store, output, topics });

    expect(index.papers[`arxiv:${id}`]).toMatchObject({
      title: "Unrelated index-only paper",
      status: "inbox",
      detail: false,
      paperPath: null,
    });
  });

  it("preserves saved and ignored index-only papers", async () => {
    const savedId = "2606.21002";
    const ignoredId = "2606.21003";
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [savedId]: {
          status: "saved",
          seenDates: [],
          dailyReports: [],
        },
        [ignoredId]: {
          status: "ignored",
          seenDates: [],
          dailyReports: [],
        },
      }),
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers[`arxiv:${savedId}`]?.status).toBe("saved");
    expect(index.papers[`arxiv:${ignoredId}`]?.status).toBe("ignored");
  });

  it("preserves concurrent paper path, daily report, and status changes during a scan", async () => {
    const id = "2606.21004";
    const primaryPath = "arxiv/.index/papers.json";
    const scanPath = "arxiv/daily/2026-06-20.md";
    const concurrentPath = "notes/manual-paper.md";
    const concurrentReport = "arxiv/daily/2026-06-21.md";
    const { files, storage } = makeStorage({
      [primaryPath]: indexJson({
        [id]: {
          detail: true,
          paperPath: `arxiv/papers/${id}.md`,
          seenDates: ["2026-06-19"],
          dailyReports: ["arxiv/daily/2026-06-19.md"],
        },
      }),
      [scanPath]: "# Temporarily unrelated report",
    });
    const store = new PaperIndexStore(storage, output);
    const vault = makeVault(files);
    let releaseRead!: () => void;
    let markRead!: () => void;
    const readStarted = new Promise<void>((resolve) => { markRead = resolve; });
    const readRelease = new Promise<void>((resolve) => { releaseRead = resolve; });
    const originalRead = vault.adapter.read;
    vault.adapter.read = async (path: string) => {
      const content = await originalRead(path);
      if (path === scanPath) {
        markRead();
        await readRelease;
      }
      return content;
    };

    const syncing = syncDashboardHistory({ vault, store, output, topics });
    await readStarted;
    await store.setPaperPath(id, concurrentPath);
    await store.addDailyReports([id], concurrentReport);
    await store.setStatus(id, "saved");
    releaseRead();

    const index = await syncing;

    expect(index.papers[`arxiv:${id}`]).toMatchObject({
      status: "saved",
      detail: true,
      paperPath: concurrentPath,
      dailyReports: ["arxiv/daily/2026-06-19.md", concurrentReport],
      seenDates: ["2026-06-19"],
    });
  });

  it("does not overwrite a concurrent paper path with a valid scanned detail path", async () => {
    const id = "2606.21006";
    const scannedPath = `arxiv/papers/${id}.md`;
    const concurrentPath = "notes/curated-paper.md";
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [id]: {
          detail: true,
          paperPath: scannedPath,
          dailyReports: [],
        },
      }),
      [scannedPath]: detailMarkdown(id, "Scanned managed detail"),
    });
    const store = new PaperIndexStore(storage, output);
    const vault = makeVault(files);
    let releaseRead!: () => void;
    let markRead!: () => void;
    const readStarted = new Promise<void>((resolve) => { markRead = resolve; });
    const readRelease = new Promise<void>((resolve) => { releaseRead = resolve; });
    const originalRead = vault.adapter.read;
    vault.adapter.read = async (path: string) => {
      const content = await originalRead(path);
      if (path === scannedPath) {
        markRead();
        await readRelease;
      }
      return content;
    };

    const syncing = syncDashboardHistory({ vault, store, output, topics });
    await readStarted;
    await store.setPaperPath(id, concurrentPath);
    releaseRead();

    const index = await syncing;

    expect(index.papers[`arxiv:${id}`]).toMatchObject({
      detail: true,
      paperPath: concurrentPath,
    });
  });

  it.each([
    { label: "without independent daily evidence", withDailyEvidence: false },
    { label: "with independent daily evidence", withDailyEvidence: true },
  ])(
    "does not revive a concurrently deleted scanned detail $label",
    async ({ withDailyEvidence }) => {
      const id = "2606.21007";
      const scannedPath = `arxiv/papers/${id}.md`;
      const dailyPath = "arxiv/daily/2026-06-24.md";
      const initialFiles: Record<string, string> = {
        "arxiv/.index/papers.json": indexJson({
          [id]: {
            detail: true,
            paperPath: scannedPath,
            dailyReports: [],
          },
        }),
        [scannedPath]: detailMarkdown(id, "Concurrently deleted detail"),
      };
      if (withDailyEvidence) {
        initialFiles[dailyPath] = [
          "## Photo-z",
          "### Independently reported paper",
          "- **Authors**: A. Author",
          `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
        ].join("\n");
      }
      const { files, storage } = makeStorage(initialFiles);
      const store = new PaperIndexStore(storage, output);
      const vault = makeVault(files);
      let releaseRead!: () => void;
      let markRead!: () => void;
      const readStarted = new Promise<void>((resolve) => { markRead = resolve; });
      const readRelease = new Promise<void>((resolve) => { releaseRead = resolve; });
      const originalRead = vault.adapter.read;
      let paperScanReads = 0;
      vault.adapter.read = async (path: string) => {
        const content = await originalRead(path);
        if (path === scannedPath) {
          paperScanReads += 1;
          markRead();
          await readRelease;
        }
        return content;
      };

      const syncing = syncDashboardHistory({ vault, store, output, topics });
      await readStarted;
      await expect(store.removePaperDetailsAtPath(
        id,
        scannedPath,
        async () => { delete files[scannedPath]; },
      )).resolves.toMatchObject({ kind: "removed" });
      releaseRead();

      const index = await syncing;

      expect(paperScanReads).toBe(1);
      if (!withDailyEvidence) {
        expect(index.papers[`arxiv:${id}`]).toBeUndefined();
      } else {
        expect(index.papers[`arxiv:${id}`]).toMatchObject({
          detail: false,
          paperPath: null,
          seenDates: ["2026-06-24"],
          dailyReports: [dailyPath],
        });
      }
    },
  );

  it("still prunes an unchanged stale managed daily reference", async () => {
    const id = "2606.21005";
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [id]: {
          seenDates: ["2026-06-22"],
          dailyReports: ["arxiv/daily/2026-06-22.md"],
          detail: false,
          paperPath: null,
        },
      }),
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers[`arxiv:${id}`]).toBeUndefined();
  });

  it("keeps an unreadable filename-owned detail path when a second note claims the same ID", async () => {
    const id = "2606.12345";
    const baselinePath = `arxiv/papers/${id}.md`;
    const alternatePath = "arxiv/papers/readable-alternate.md";
    const dailyPath = "arxiv/daily/2026-06-23.md";
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [id]: {
          detail: true,
          paperPath: baselinePath,
          seenDates: ["2026-06-10"],
          dailyReports: [],
        },
      }),
      [baselinePath]: detailMarkdown(id, "Unreadable baseline detail"),
      [alternatePath]: detailMarkdown(id, "Readable alternate detail"),
      [dailyPath]: [
        "## Photo-z",
        "### Daily data remains safe",
        "- **Authors**: A. Author",
        `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
      ].join("\n"),
    });
    const store = new PaperIndexStore(storage, output);
    const vault = makeVault(files);
    const originalRead = vault.adapter.read;
    vault.adapter.read = async (path: string) => {
      if (path === baselinePath) throw new Error("permission denied");
      return originalRead(path);
    };

    const index = await syncDashboardHistory({ vault, store, output, topics });

    expect(index.papers[`arxiv:${id}`]).toMatchObject({
      detail: true,
      paperPath: baselinePath,
      seenDates: ["2026-06-10", "2026-06-23"],
      dailyReports: [dailyPath],
    });
  });

  it("protects an indexed detail when its existing note is unreadable", async () => {
    const id = "2606.22001";
    const paperPath = `arxiv/papers/${id}.md`;
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [id]: { detail: true, paperPath, dailyReports: [] },
      }),
      [paperPath]: detailMarkdown(id, "Unreadable detail"),
    });
    const store = new PaperIndexStore(storage, output);
    const vault = makeVault(files);
    vault.adapter.read = async (path: string) => {
      if (path === paperPath) throw new Error("permission denied");
      return files[path];
    };

    const index = await syncDashboardHistory({ vault, store, output, topics });

    expect(index.papers[`arxiv:${id}`]).toMatchObject({
      detail: true,
      paperPath,
    });
  });

  it("protects an indexed detail when its note identity is malformed", async () => {
    const id = "2606.22002";
    const paperPath = `arxiv/papers/${id}.md`;
    const malformed = detailMarkdown(id, "Malformed detail").replace(
      `arxiv_id: "${id}"`,
      'arxiv_id: "not-an-arxiv-id"',
    );
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [id]: { detail: true, paperPath, dailyReports: [] },
      }),
      [paperPath]: malformed,
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers[`arxiv:${id}`]).toMatchObject({
      detail: true,
      paperPath,
    });
  });

  it("protects every top-level duplicate identity while unrelated history still reconciles", async () => {
    const firstId = "2606.22004";
    const secondId = "2606.22005";
    const unrelatedId = "2606.22006";
    const firstBaselinePath = `arxiv/papers/${firstId}.md`;
    const secondBaselinePath = `arxiv/papers/${secondId}.md`;
    const unrelatedBaselinePath = `arxiv/papers/${unrelatedId}.md`;
    const conflictingPath = "arxiv/papers/duplicate-identities.md";
    const dailyPath = "arxiv/daily/2026-06-25.md";
    const conflictingNote = detailMarkdown(firstId, "Duplicate identities")
      .replace(
        `arxiv_id: "${firstId}"`,
        [
          `arxiv_id: "${firstId}"`,
          `arxiv_id: "${secondId}"`,
          "metadata:",
          `  arxiv_id: "${unrelatedId}"`,
          "aliases:",
          `  - arxiv: "${unrelatedId}"`,
          `arxiv: ["${unrelatedId}"]`,
        ].join("\n"),
      )
      .concat(`\n\narxiv_id: "${unrelatedId}"`);
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [firstId]: {
          detail: true,
          paperPath: firstBaselinePath,
          dailyReports: [],
        },
        [secondId]: {
          detail: true,
          paperPath: secondBaselinePath,
          dailyReports: [],
        },
        [unrelatedId]: {
          detail: true,
          paperPath: unrelatedBaselinePath,
          dailyReports: [],
        },
      }),
      [conflictingPath]: conflictingNote,
      [dailyPath]: [
        "## Photo-z",
        "### Unrelated daily evidence",
        "- **Authors**: C. Author",
        `- **arXiv**: [${unrelatedId}](https://arxiv.org/abs/${unrelatedId})`,
      ].join("\n"),
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers[`arxiv:${firstId}`]).toMatchObject({
      detail: true,
      paperPath: firstBaselinePath,
    });
    expect(index.papers[`arxiv:${secondId}`]).toMatchObject({
      detail: true,
      paperPath: secondBaselinePath,
    });
    expect(index.papers[`arxiv:${unrelatedId}`]).toMatchObject({
      detail: false,
      paperPath: null,
      seenDates: ["2026-06-10", "2026-06-25"],
      dailyReports: [dailyPath],
    });
  });

  it("does not let a duplicate arXiv identity overwrite or clear a valid detail", async () => {
    const id = "2606.22003";
    const validPath = "arxiv/papers/verified-note.md";
    const duplicatePath = "arxiv/papers/duplicate-note.md";
    const { files, storage } = makeStorage({
      "arxiv/.index/papers.json": indexJson({
        [id]: { detail: true, paperPath: validPath, dailyReports: [] },
      }),
      [validPath]: detailMarkdown(id, "Verified duplicate detail"),
      [duplicatePath]: [
        "---",
        `arxiv_id: "${id}"`,
        "---",
      ].join("\n"),
    });
    const store = new PaperIndexStore(storage, output);

    const index = await syncDashboardHistory({
      vault: makeVault(files),
      store,
      output,
      topics,
    });

    expect(index.papers[`arxiv:${id}`]).toMatchObject({
      detail: true,
      paperPath: validPath,
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
      "arxiv:2606.30000",
      "arxiv:2606.30001",
    ]);
  });
});
