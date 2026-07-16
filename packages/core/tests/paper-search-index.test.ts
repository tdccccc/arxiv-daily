import { describe, expect, it } from "vitest";
import {
  PaperSearchIndex,
  normalizeArxivSearchId,
  tokenizePaperSearchText,
  type PaperIndexEntry,
} from "../src";

function paper(id: string, overrides: Partial<PaperIndexEntry> = {}): PaperIndexEntry {
  return {
    arxivId: id,
    source: "arxiv",
    title: `Paper ${id}`,
    authors: ["A. Author"],
    published: "2026-07-01",
    updated: "2026-07-01",
    category: "cs.LG",
    categories: ["cs.LG"],
    topics: ["machine learning"],
    primaryTopic: "machine learning",
    detail: false,
    status: "inbox",
    priority: "normal",
    seenDates: ["2026-07-01"],
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

describe("paper search tokenization", () => {
  it("normalizes NFKC/case and retains technical compounds plus components", () => {
    expect(tokenizePaperSearchText("Ｔｒａｎｓｆｏｒｍｅｒ Photo-Z SELF-SUPERVISED")).toEqual([
      "transformer", "photo-z", "photo", "z", "self-supervised", "self", "supervised",
    ]);
  });

  it("emits deterministic Han bigrams and short full tokens in mixed text", () => {
    expect(tokenizePaperSearchText("星系红移 photo-z 星系")).toEqual([
      "星系", "系红", "红移", "photo-z", "photo", "z", "星系",
    ]);
  });
});

describe("arXiv ID normalization", () => {
  it.each([
    ["2607.01234", "2607.01234"],
    ["arXiv:2607.01234v3", "2607.01234"],
    ["https://arxiv.org/abs/2607.01234v2", "2607.01234"],
    ["https://arxiv.org/pdf/2607.01234v9.pdf?download=1", "2607.01234"],
  ])("normalizes %s", (input, expected) => {
    expect(normalizeArxivSearchId(input)).toBe(expected);
  });

  it("rejects non-modern and malformed IDs", () => {
    expect(normalizeArxivSearchId("hep-th/9901001")).toBeNull();
    expect(normalizeArxivSearchId("2607.12")).toBeNull();
  });
});

describe("PaperSearchIndex", () => {
  const entries = [
    paper("2607.00001", {
      title: "Photometric-redshift calibration with transformer",
      authors: ["Ada Lovelace"],
      primaryTopic: "photo-z",
      topics: ["photo-z", "photo-z"],
      category: "astro-ph.CO",
      categories: ["astro-ph.CO", "astro-ph.CO"],
      summary: { keyMethod: "contrastive learning", limitations: "small sample" },
    }),
    paper("2607.00002", {
      title: "A catalog for galaxy clusters",
      authors: ["Grace Hopper"],
      primaryTopic: "galaxy clusters",
      topics: ["galaxy clusters"],
      category: "astro-ph.GA",
      categories: ["astro-ph.GA"],
      summary: { coreProblem: "photometric redshift calibration", keyMethod: "linear fit" },
    }),
    paper("2607.00003", {
      title: "星系红移的深度学习方法",
      primaryTopic: "星系演化",
      topics: ["星系演化"],
      summary: { mainResult: "混合 language benchmark" },
    }),
  ];

  it("uses AND clauses while treating hyphen compound/components as one clause", () => {
    const index = new PaperSearchIndex(entries);
    expect(index.search("photo-z transformer").map((result) => result.entry.arxivId)).toEqual(["2607.00001"]);
    expect(index.search("photo calibration").map((result) => result.entry.arxivId)).toEqual(["2607.00001"]);
    expect(index.search("photo nonexistent")).toEqual([]);
  });

  it("weights title above summaries and deduplicates topic/category values", () => {
    const index = new PaperSearchIndex(entries);
    const results = index.search("calibration");
    expect(results.map((result) => result.entry.arxivId)).toEqual(["2607.00001", "2607.00002"]);
    expect(results[0].reasons[0].text).toContain("title");
    expect(Number.isFinite(new PaperSearchIndex([entries[0], { ...entries[0], arxivId: "2607.99999", topics: ["photo-z"] }]).search("photo-z")[0].score)).toBe(true);
  });

  it("prioritizes exact and partial canonical IDs deterministically", () => {
    const index = new PaperSearchIndex(entries);
    expect(index.search("https://arxiv.org/pdf/2607.00002v4.pdf")[0].entry.arxivId).toBe("2607.00002");
    expect(index.search("2607.000").map((result) => result.entry.arxivId)).toEqual([
      "2607.00001", "2607.00002", "2607.00003",
    ]);
  });

  it("supports Han and mixed-language AND searches", () => {
    const index = new PaperSearchIndex(entries);
    expect(index.search("星系 language").map((result) => result.entry.arxivId)).toEqual(["2607.00003"]);
  });

  it("finds similar local papers with bounded OR terms and deterministic reasons", () => {
    const similar = new PaperSearchIndex([
      ...entries,
      paper("2607.00004", { title: "Transformer calibration", status: "ignored" }),
    ]).similar(entries[0]);
    expect(similar.map((result) => result.entry.arxivId)).toContain("2607.00002");
    expect(similar.map((result) => result.entry.arxivId)).not.toContain("2607.00001");
    expect(similar.map((result) => result.entry.arxivId)).not.toContain("2607.00004");
    expect(similar[0].reasons.length).toBeGreaterThan(0);
  });
});
