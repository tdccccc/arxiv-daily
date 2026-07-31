import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import {
  ArxivSourceAdapter,
  legacyContentFromNormalized,
  mapLegacyPaperContent,
  paperMetaFromSourcePaper,
} from "../../src/sources";
import type { SourcePaperMeta } from "../../src/sources";
import { Logger } from "../../src/services/logger";
import { markupParser } from "../markup-parser";
import { parseRecent } from "../../src/pipeline/arxiv-parser";
import {
  ArxivHttpError,
  ArxivRetryDeferredError,
} from "../../src/pipeline/arxiv-fetcher";

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(
  resolve(here, "../fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

describe("mapLegacyPaperContent / legacyContentFromNormalized", () => {
  it("maps abstract + sections markdown into normalized content", () => {
    const normalized = mapLegacyPaperContent(
      {
        abstractConclusion: "## Abstract\nHello world.",
        fullSections: "## Intro\nBody one.\n\n## Methods\nBody two.",
        fullTextSource: "arxiv-html",
      },
      "https://arxiv.org/abs/2606.12345",
      "2606.12345",
    );
    expect(normalized.abstract).toBe("Hello world.");
    expect(normalized.sections).toEqual([
      { heading: "Intro", text: "Body one." },
      { heading: "Methods", text: "Body two." },
    ]);
    expect(normalized.quality).toBe("full");
    expect(normalized.canonicalUrl).toBe("https://arxiv.org/abs/2606.12345");
    expect(normalized.fullTextSource).toBe("arxiv-html");
  });

  it("round-trips enough for summarizer string fields", () => {
    const normalized = mapLegacyPaperContent(
      {
        abstractConclusion: "## Abstract\nAbs.",
        fullSections: "## S\nT",
        fullTextSource: "arxiv-source",
      },
      "https://arxiv.org/abs/2606.1",
      "2606.1",
    );
    const legacy = legacyContentFromNormalized(normalized);
    expect(legacy.abstractConclusion).toContain("Abs.");
    expect(legacy.fullSections).toContain("## S");
    expect(legacy.fullTextSource).toBe("arxiv-source");
  });

  it("marks unavailable when empty", () => {
    const normalized = mapLegacyPaperContent(
      {
        abstractConclusion: "",
        fullSections: null,
        fullTextFailure: "no content",
      },
      "",
      "2606.12345",
    );
    expect(normalized.quality).toBe("unavailable");
    expect(normalized.fullTextFailure).toBe("no content");
  });
});

describe("paperMetaFromSourcePaper", () => {
  it("exposes externalId as filter id and categories as arxivCategories", () => {
    const paper: SourcePaperMeta = {
      paperKey: "arxiv:2606.12345",
      source: "arxiv",
      externalId: "2606.12345",
      title: "T",
      authors: "A",
      abstract: "Abs",
      canonicalUrl: "https://arxiv.org/abs/2606.12345",
      pdfUrl: "https://arxiv.org/pdf/2606.12345",
      categories: ["astro-ph", "cs.LG"],
      published: "2026-07-26",
    };
    const meta = paperMetaFromSourcePaper(paper);
    expect(meta.id).toBe("2606.12345");
    expect(meta.arxivCategories).toEqual(["astro-ph", "cs.LG"]);
    expect(meta.abstract).toBe("Abs");
  });
});

describe("ArxivSourceAdapter", () => {
  it("listForDate maps fixture /recent and enriches abstracts", async () => {
    const buckets = parseRecent(recentHtml, markupParser);
    const date = buckets[0]!.announceDate;
    const sampleId = buckets[0]!.papers[0]!.id;
    const fetcher = {
      fetchRecent: vi.fn().mockResolvedValue(recentHtml),
      fetchMetadataByIds: vi.fn().mockResolvedValue(
        new Map([
          [
            sampleId,
            {
              id: sampleId,
              abstract: "atom abstract",
              updated: "2026-06-15T00:00:00Z",
              categories: ["astro-ph.CO"],
            },
          ],
        ]),
      ),
    };
    const adapter = new ArxivSourceAdapter({
      fetcher: fetcher as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger(),
      defaultCategories: ["astro-ph"],
    });

    const result = await adapter.listForDate(date);
    expect(result.kind).toBe("ok");
    if (result.kind !== "ok") return;
    expect(result.channels).toEqual(["astro-ph"]);
    expect(result.papers.length).toBeGreaterThan(0);
    const hit = result.papers.find((p) => p.externalId === sampleId);
    expect(hit).toMatchObject({
      paperKey: `arxiv:${sampleId}`,
      source: "arxiv",
      abstract: "atom abstract",
      updated: "2026-06-15",
    });
    expect(hit?.categories).toEqual(expect.arrayContaining(["astro-ph"]));
    expect(fetcher.fetchMetadataByIds).toHaveBeenCalled();
  });

  it("fetchContent delegates to PaperContentFetcher and normalizes", async () => {
    const paperFetcher = {
      fetch: vi.fn().mockResolvedValue({
        abstractConclusion: "## Abstract\nFrom html",
        fullSections: "## Body\nText",
        fullTextSource: "arxiv-html",
      }),
    };
    const adapter = new ArxivSourceAdapter({
      fetcher: { fetchRecent: vi.fn(), fetchMetadataByIds: vi.fn() } as any,
      paperFetcher: paperFetcher as any,
      markupParser,
      logger: new Logger(),
      defaultCategories: ["astro-ph"],
    });

    const content = await adapter.fetchContent("2605.08080", {
      wantFullText: true,
      sectionCharLimit: 1000,
      paperCharLimit: 5000,
    });
    expect(paperFetcher.fetch).toHaveBeenCalledWith(
      "2605.08080",
      expect.objectContaining({ isDetail: true }),
    );
    expect(content.abstract).toBe("From html");
    expect(content.sections[0]?.heading).toBe("Body");
    expect(content.quality).toBe("full");
    expect(content.canonicalUrl).toBe("https://arxiv.org/abs/2605.08080");
  });

  it("rejects partial multi-category input before Atom enrichment and refetches all categories", async () => {
    const buckets = parseRecent(recentHtml, markupParser);
    const date = buckets[0]!.announceDate;
    const fetcher = {
      fetchRecent: vi.fn()
        .mockResolvedValueOnce(recentHtml)
        .mockRejectedValueOnce(new ArxivHttpError(503, "https://arxiv.org/list/cs.LG/recent"))
        .mockResolvedValueOnce(recentHtml)
        .mockResolvedValueOnce(recentHtml),
      fetchMetadataByIds: vi.fn().mockResolvedValue(new Map()),
    };
    const adapter = new ArxivSourceAdapter({
      fetcher: fetcher as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger("error"),
      defaultCategories: ["astro-ph", "cs.LG"],
    });

    await expect(adapter.listForDate(date)).resolves.toMatchObject({
      kind: "error",
      failureKind: "failed_transient",
    });
    expect(fetcher.fetchMetadataByIds).not.toHaveBeenCalled();

    const retry = await adapter.listForDate(date);
    expect(retry.kind).toBe("ok");
    if (retry.kind === "ok") {
      expect(retry.channels).toEqual(["astro-ph", "cs.LG"]);
      expect(new Set(retry.papers.map((paper) => paper.externalId)).size).toBe(
        retry.papers.length,
      );
      expect(retry.papers.every((paper) =>
        paper.categories.includes("astro-ph") && paper.categories.includes("cs.LG")
      )).toBe(true);
    }
    expect(fetcher.fetchRecent.mock.calls.map(([category]) => category)).toEqual([
      "astro-ph", "cs.LG", "astro-ph", "cs.LG",
    ]);
    expect(fetcher.fetchMetadataByIds).toHaveBeenCalledTimes(1);
  });

  it("makes a mixed transient/permanent category failure transient", async () => {
    const fetcher = {
      fetchRecent: vi.fn()
        .mockRejectedValueOnce(new ArxivHttpError(404, "https://arxiv.org/list/bad/recent"))
        .mockRejectedValueOnce(new ArxivHttpError(503, "https://arxiv.org/list/slow/recent")),
      fetchMetadataByIds: vi.fn(),
    };
    const adapter = new ArxivSourceAdapter({
      fetcher: fetcher as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger("error"),
      defaultCategories: ["bad", "slow"],
    });

    await expect(adapter.listForDate("2026-05-11")).resolves.toMatchObject({
      kind: "error",
      failureKind: "failed_transient",
    });
  });

  it("listForDate returns collapsed error when all categories fail fetch", async () => {
    const fetcher = {
      fetchRecent: vi.fn().mockRejectedValue(new Error("network down")),
      fetchMetadataByIds: vi.fn(),
    };
    const adapter = new ArxivSourceAdapter({
      fetcher: fetcher as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger(),
      defaultCategories: ["astro-ph", "cs.LG"],
    });
    const result = await adapter.listForDate("2026-05-11");
    expect(result.kind).toBe("error");
    if (result.kind === "error") {
      expect(result.failureKind).toBe("failed_transient");
      expect(result.reason).toContain("all arXiv categories failed");
    }
  });

  it.each([
    [429, "failed_transient", "rate-limiting"],
    [503, "failed_transient", "temporarily unavailable"],
    [404, "failed_permanent", "rejected the request"],
  ])("classifies and formats typed HTTP %i failures", async (status, failureKind, text) => {
    const fetcher = {
      fetchRecent: vi.fn().mockRejectedValue(
        new ArxivHttpError(status, "https://arxiv.org/list/astro-ph/recent?skip=0"),
      ),
      fetchMetadataByIds: vi.fn(),
    };
    const adapter = new ArxivSourceAdapter({
      fetcher: fetcher as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger("error"),
      defaultCategories: ["astro-ph"],
    });

    const result = await adapter.listForDate("2026-05-11");

    expect(result).toMatchObject({ kind: "error", failureKind });
    expect(result.kind === "error" ? result.reason : "").toContain(text);
    expect(result.kind === "error" ? result.reason : "").not.toContain("skip=0");
  });

  it("keeps long Retry-After deferral transient with actionable timing", async () => {
    const fetcher = {
      fetchRecent: vi.fn().mockRejectedValue(
        new ArxivRetryDeferredError(new Date("2026-06-25T12:00:00.000Z"), 7_200_000),
      ),
      fetchMetadataByIds: vi.fn(),
    };
    const adapter = new ArxivSourceAdapter({
      fetcher: fetcher as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger("error"),
      defaultCategories: ["astro-ph"],
    });

    const result = await adapter.listForDate("2026-05-11");

    expect(result).toMatchObject({ kind: "error", failureKind: "failed_transient" });
    expect(result.kind === "error" ? result.reason : "").toContain(
      "retry at 2026-06-25T12:00:00.000Z (2h remaining)",
    );
  });

  it("rejects invalid externalId in fetchContent as unavailable", async () => {
    const adapter = new ArxivSourceAdapter({
      fetcher: { fetchRecent: vi.fn(), fetchMetadataByIds: vi.fn() } as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger(),
      defaultCategories: ["astro-ph"],
    });
    const content = await adapter.fetchContent("not-an-id", {
      wantFullText: false,
      sectionCharLimit: 100,
      paperCharLimit: 100,
    });
    expect(content.quality).toBe("unavailable");
    expect(content.fullTextFailure).toMatch(/invalid arXiv id/);
  });
});
