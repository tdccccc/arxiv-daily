import { describe, expect, it } from "vitest";
import {
  PaperSearchIndex,
  normalizeArxivSearchId,
  queryDashboard,
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
  it("finds a fallback paper through its separately stored abstract in indexed and legacy modes", () => {
    const fallback = paper("2607.00999", {
      title: "Unrelated title",
      abstract: "Distinctive emergency fallback quasar tomography evidence",
      summary: undefined,
    });
    const query = { tab: "all" as const, search: "tomography evidence" };

    expect(queryDashboard([fallback], query).rows.map((row) => row.arxivId)).toEqual([
      fallback.arxivId,
    ]);
    expect(
      queryDashboard([fallback], query, { searchIndex: null }).rows.map(
        (row) => row.arxivId,
      ),
    ).toEqual([fallback.arxivId]);
    expect(fallback.summary).toBeUndefined();
  });

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

  it("indexes abstracts and favors weighted coverage across multiple fields", () => {
    const source = paper("2607.10000", {
      title: "Neural retrieval calibration",
      authors: ["Source Author"],
      primaryTopic: "robust retrieval",
      topics: ["robust retrieval"],
      summary: { keyMethod: "contrastive calibration" },
      abstract: "dense embeddings for robust search",
    });
    const broad = paper("2607.10001", {
      title: "Neural retrieval for robust search",
      authors: ["Other Author"],
      primaryTopic: "robust retrieval",
      topics: ["robust retrieval"],
      summary: { keyMethod: "contrastive calibration" },
    });
    const narrow = paper("2607.10002", {
      title: "Neural retrieval calibration",
      authors: ["Other Author"],
      primaryTopic: "unrelated",
      topics: ["unrelated"],
    });
    const abstractOnly = paper("2607.10003", {
      title: "A different application",
      authors: ["Other Author"],
      abstract: "dense embeddings robust search",
    });
    const index = new PaperSearchIndex([source, narrow, abstractOnly, broad]);

    expect(index.search("embeddings search").map((result) => result.entry.arxivId)).toContain("2607.10000");
    expect(index.similar(source).map((result) => result.entry.arxivId)).toEqual([
      "2607.10001", "2607.10002", "2607.10003",
    ]);
  });

  it("suppresses weak one-token matches when enough covered candidates exist but falls back for sparse indexes", () => {
    const source = paper("2607.20000", {
      title: "Quantum graph retrieval",
      authors: ["Unique Source"],
      topics: ["spectral methods"], primaryTopic: "spectral methods",
      category: "quant-ph", categories: ["quant-ph"],
    });
    const strong = [1, 2, 3].map((value) => paper(`2607.2000${value}`, {
      title: `Quantum graph study ${value}`,
      authors: [`Author ${value}`],
      category: "quant-ph", categories: ["quant-ph"],
    }));
    const weak = paper("2607.20004", {
      title: "Quantum biology", authors: ["Other"], category: "q-bio.NC", categories: ["q-bio.NC"],
      topics: ["biology"], primaryTopic: "biology",
    });
    expect(new PaperSearchIndex([source, ...strong, weak]).similar(source, { limit: 4 })
      .map((result) => result.entry.arxivId)).not.toContain(weak.arxivId);

    expect(new PaperSearchIndex([source, weak]).similar(source, { limit: 4 })
      .map((result) => result.entry.arxivId)).toEqual([weak.arxivId]);
  });

  it("keeps author-only matches behind semantic candidates and uses them only as sparse fallback", () => {
    const source = paper("2607.25000", {
      title: "Quasar tomography",
      authors: ["Alice Unique"],
      topics: ["intergalactic mapping"], primaryTopic: "intergalactic mapping",
      category: "astro-ph.CO", categories: ["astro-ph.CO"],
    });
    const authorOnly = paper("2607.25001", {
      title: "Unrelated compiler verification",
      authors: ["Alice Unique"],
      topics: ["program analysis"], primaryTopic: "program analysis",
      category: "cs.PL", categories: ["cs.PL"],
    });
    const semantic = paper("2607.25002", {
      title: "Quasar census",
      authors: ["Bob Independent"],
      topics: ["active galaxies"], primaryTopic: "active galaxies",
      category: "astro-ph.GA", categories: ["astro-ph.GA"],
    });

    const ranked = new PaperSearchIndex([source, authorOnly, semantic])
      .similar(source, { limit: 2 });
    expect(ranked.map((result) => result.entry.arxivId)).toEqual([
      semantic.arxivId,
      authorOnly.arxivId,
    ]);
    expect(ranked[1].score).toBeLessThan(ranked[0].score * 0.5);
    expect(new PaperSearchIndex([source, authorOnly]).similar(source))
      .toEqual([expect.objectContaining({ entry: authorOnly })]);
  });

  it("does not count author-only source terms toward strong semantic coverage", () => {
    const source = paper("2607.26000", {
      title: "Quantum graph retrieval",
      authors: ["Alice Smith"],
      topics: ["spectral methods"], primaryTopic: "spectral methods",
      category: "quant-ph", categories: ["quant-ph"],
    });
    const strong = [1, 2, 3].map((value) => paper(`2607.2600${value}`, {
      title: `Quantum graph analysis ${value}`,
      authors: [`Independent ${value}`],
      category: "quant-ph", categories: ["quant-ph"],
    }));
    const authorPlusOneSemantic = paper("2607.26004", {
      title: "Quantum chemistry",
      authors: ["Alice Smith"],
      topics: ["molecules"], primaryTopic: "molecules",
      category: "cs.PL", categories: ["cs.PL"],
    });

    expect(new PaperSearchIndex([source, ...strong, authorPlusOneSemantic])
      .similar(source, { limit: 4 }).map((result) => result.entry.arxivId))
      .not.toContain(authorPlusOneSemantic.arxivId);
  });

  it("caps shared-author domination, backfills when needed, and ties by arXiv ID", () => {
    const source = paper("2607.30000", {
      title: "Graph retrieval calibration",
      authors: ["Alice Smith"],
      topics: ["graph retrieval"], primaryTopic: "graph retrieval",
    });
    const collaborators = [5, 4, 3].map((value) => paper(`2607.3000${value}`, {
      title: `Graph retrieval calibration ${value}`,
      authors: ["Alice Smith"],
    }));
    const independent = [2, 1].map((value) => paper(`2607.3000${value}`, {
      title: "Graph retrieval calibration",
      authors: [`Independent ${value}`],
    }));
    const results = new PaperSearchIndex([source, ...collaborators, ...independent])
      .similar(source, { limit: 5 });
    expect(results.slice(0, 4).filter((result) => result.entry.authors.includes("Alice Smith"))).toHaveLength(2);
    expect(results).toHaveLength(5);

    const tied = new PaperSearchIndex([source, independent[0], independent[1]])
      .similar(source).map((result) => result.entry.arxivId);
    expect(tied).toEqual(["2607.30001", "2607.30002"]);
  });
});
