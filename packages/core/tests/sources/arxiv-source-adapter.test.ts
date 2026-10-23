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
import { listingIds, recentListingWorld, type WorldDay } from "../recent-listing-world";
import { parseRecent } from "../../src/pipeline/arxiv-parser";
import {
  ArxivFetcher,
  ArxivHttpError,
  ArxivRequestCoordinator,
  ArxivRetryDeferredError,
} from "../../src/pipeline/arxiv-fetcher";
import type { HttpClient } from "../../src/core/adapters";

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(
  resolve(here, "../fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

describe("mapLegacyPaperContent / legacyContentFromNormalized", () => {
  it.each([
    ["## Abstract\nFirst line.\nSecond line.\n## Methods\nDetails.", "First line.\nSecond line."],
    ["## Abstract\r\nFirst line.\r\n## Methods\r\nDetails.", "First line."],
    ["##aBsTrAcT\nBody text.", "Body text."],
    ["##\nAbstract\nBody text.", "Body text."],
    ["No abstract heading\nBody text.", "No abstract heading\nBody text."],
    ["## Abstract Inline body.", "Inline body."],
    ["## Abstract", ""],
    ["## Abstract\n\n## Conclusion\nConclusion.", "## Conclusion\nConclusion."],
    [" ".repeat(16_000), ""],
    ["## Abstract\n" + " ".repeat(16_000) + "payload", "payload"],
    ["##" + " ".repeat(16_000) + "no heading", "##" + " ".repeat(16_000) + "no heading"],
  ])("preserves normalized abstract text (case %#)", (abstractConclusion, expected) => {
    expect(mapLegacyPaperContent({ abstractConclusion, fullSections: null }, "", "2606.12345").abstract).toBe(expected);
  });

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

describe("content that could not be fetched", () => {
  const FAILED = "[获取失败] arXiv ID: 2606.12345";

  it("stays a bare placeholder instead of becoming an Abstract section", () => {
    const normalized = mapLegacyPaperContent(
      { abstractConclusion: FAILED, fullSections: null, fullTextFailure: "html down; abs down" },
      "",
      "2606.12345",
    );

    expect(normalized.quality).toBe("unavailable");
    const legacy = legacyContentFromNormalized(normalized);
    expect(legacy.abstractConclusion).toBe(FAILED);
    expect(legacy.abstractConclusion).not.toContain("## Abstract");
    expect(legacy.fullTextFailure).toBe("html down; abs down");
  });

  it("does not turn the failure reasons into paper text", () => {
    const legacy = legacyContentFromNormalized(mapLegacyPaperContent(
      { abstractConclusion: "", fullSections: null, fullTextFailure: "rendered HTML unavailable (HTTP 503)" },
      "",
      "2606.12345",
    ));

    expect(legacy.abstractConclusion).toBe("");
    expect(legacy.fullTextFailure).toBe("rendered HTML unavailable (HTTP 503)");
  });

  it("still wraps a real abstract and uses full text when the abstract is missing", () => {
    expect(legacyContentFromNormalized(mapLegacyPaperContent(
      { abstractConclusion: "Plain abstract.", fullSections: null }, "", "2606.12345",
    )).abstractConclusion).toBe("## Abstract\nPlain abstract.");
    expect(legacyContentFromNormalized(mapLegacyPaperContent(
      { abstractConclusion: "", fullSections: "## Intro\nBody." }, "", "2606.12345",
    )).abstractConclusion).toBe("## Intro\nBody.");
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
    // The Atom map covers one paper; the others fall back to their abs pages.
    const fetchListingAbstract = vi.fn(async (id: string) => ({
      abstract: `abs-page ${id}`,
      fromNetwork: true,
    }));
    const adapter = new ArxivSourceAdapter({
      fetcher: fetcher as any,
      paperFetcher: { fetch: vi.fn(), fetchListingAbstract } as any,
      markupParser,
      logger: new Logger(),
      defaultCategories: ["astro-ph"],
    });

    const result = await adapter.listForDate(date);
    expect(result.kind).toBe("ok");
    if (result.kind !== "ok") return;
    expect(fetchListingAbstract).not.toHaveBeenCalledWith(sampleId, undefined);
    expect(result.papers.filter((p) => p.externalId !== sampleId).every((p) => p.abstract === `abs-page ${p.externalId}`)).toBe(true);
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

  it.each([
    ["2026-10-03", false, "pending", "awaiting_announcement"],
    ["2026-10-02", false, "ok", "no_updates"],
    ["2026-10-03", true, "error", undefined],
  ])("classifies announcement availability for %s (network failure: %s)", async (date, fails, kind, outcome) => {
    const html = '<dl id="articles"><h3>Fri, 2 Oct 2026 (showing 0 of 0 entries)</h3></dl>';
    const fetchRecent = vi.fn().mockResolvedValue(html);
    if (fails) fetchRecent.mockRejectedValueOnce(new Error("network down"));
    const enrich = vi.fn();
    const adapter = new ArxivSourceAdapter({
      fetcher: { fetchRecent, fetchMetadataByIds: enrich } as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser, logger: new Logger("error"),
      defaultCategories: ["astro-ph.CO", "astro-ph.GA"],
    });
    const result = await adapter.listForDate(date);
    expect(result).toMatchObject({ kind, ...(outcome ? { outcome } : {}) });
    expect(enrich).not.toHaveBeenCalled();
    if (kind === "error") expect(result).toMatchObject({ failureKind: "failed_transient", reason: expect.stringContaining("network down") });
  });

  it.each([
    ['<dl id="articles"><h3>Thu, 1 Oct 2026</h3></dl>', "pending"],
    ['<html>upstream maintenance</html>', "error"],
    ['<dl id="articles"><h3>Thu, 1 Oct 2026</h3></dl><dl id="articles"><h3>Mon, 5 Oct 2026</h3></dl>', "error"],
  ])("does not finalize partial, malformed or unexplained missing announcements", async (other, kind) => {
    const enrich = vi.fn();
    const adapter = new ArxivSourceAdapter({
      fetcher: {
        fetchRecent: vi.fn().mockResolvedValueOnce('<dl id="articles"><h3>Fri, 2 Oct 2026</h3></dl>').mockResolvedValueOnce(other),
        fetchMetadataByIds: enrich,
      } as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser, logger: new Logger("error"), defaultCategories: ["astro-ph.CO", "astro-ph.GA"],
    });
    expect(await adapter.listForDate("2026-10-02")).toMatchObject({ kind });
    expect(enrich).not.toHaveBeenCalled();
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
      // Empty Atom map: every paper reaches the abs-page fallback, which finds nothing here.
      paperFetcher: {
        fetch: vi.fn(),
        fetchListingAbstract: vi.fn(async () => ({ abstract: "", fromNetwork: false })),
      } as any,
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

  it("fails permanently for a date that has scrolled out of /recent", async () => {
    const dates = parseRecent(recentHtml, markupParser)
      .map((bucket) => bucket.announceDate)
      .sort();
    const shift = (date: string, days: number) => {
      const next = new Date(`${date}T00:00:00Z`);
      next.setUTCDate(next.getUTCDate() + days);
      return next.toISOString().slice(0, 10);
    };
    const adapter = new ArxivSourceAdapter({
      fetcher: {
        fetchRecent: vi.fn().mockResolvedValue(recentHtml),
        fetchMetadataByIds: vi.fn(),
      } as any,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger("error"),
      defaultCategories: ["astro-ph"],
    });

    await expect(adapter.listForDate(shift(dates[0]!, -7))).resolves.toMatchObject({
      kind: "error",
      failureKind: "failed_permanent",
    });
    await expect(adapter.listForDate(shift(dates.at(-1)!, 1))).resolves.toMatchObject({
      kind: "pending",
      outcome: "awaiting_announcement",
    });
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

  describe("abs-page abstract fallback", () => {
    const buckets = parseRecent(recentHtml, markupParser);
    const date = buckets[0]!.announceDate;
    const ids = buckets[0]!.papers.map((p) => p.id);

    type Listing = { abstract: string; fromNetwork: boolean };
    function setup(options: {
      metadata?: Map<string, unknown> | Error;
      listing?: (id: string, signal?: AbortSignal) => Promise<Listing>;
      limit?: number;
    } = {}) {
      const logger = new Logger("error");
      const warn = vi.spyOn(logger, "warn");
      const info = vi.spyOn(logger, "info");
      const metadata = options.metadata ?? new Error("query API down");
      const fetcher = {
        fetchRecent: vi.fn().mockResolvedValue(recentHtml),
        fetchMetadataByIds: vi.fn(async () => {
          if (metadata instanceof Error) throw metadata;
          return metadata;
        }),
      };
      const fetchListingAbstract = vi.fn(
        options.listing ?? (async (id: string) => ({ abstract: `abs-page ${id}`, fromNetwork: true })),
      );
      const adapter = new ArxivSourceAdapter({
        fetcher: fetcher as any,
        paperFetcher: { fetch: vi.fn(), fetchListingAbstract } as any,
        markupParser,
        logger,
        defaultCategories: ["astro-ph"],
        ...(options.limit === undefined ? {} : { absFallbackNetworkLimit: options.limit }),
      });
      return { adapter, fetcher, fetchListingAbstract, warn, info };
    }
    const okPapers = async (adapter: ArxivSourceAdapter, signal?: AbortSignal) => {
      const result = await adapter.listForDate(date, signal ? { signal } : {});
      expect(result.kind).toBe("ok");
      return result.kind === "ok" ? result.papers : [];
    };
    const allMessages = (spy: { mock: { calls: unknown[][] } }) =>
      spy.mock.calls.map((call) => String(call[0]));

    it("fills abstracts from abs pages when the query API throws, without failing the listing", async () => {
      const h = setup({ metadata: new ArxivHttpError(429, "https://export.arxiv.org/api/query") });
      const papers = await okPapers(h.adapter);

      expect(papers.length).toBe(ids.length);
      for (const paper of papers) expect(paper.abstract).toBe(`abs-page ${paper.externalId}`);
      expect(h.fetchListingAbstract).toHaveBeenCalledTimes(ids.length);
      const warnings = allMessages(h.warn);
      expect(warnings.some((m) => m.includes("abstract enrichment failed") && m.includes("429"))).toBe(true);
      expect(warnings.some((m) => /continuing with titles only/.test(m))).toBe(false);
      expect(allMessages(h.info).some((m) => m.includes("abs-page fallback") && m.includes(`${ids.length}/${ids.length}`))).toBe(true);
    });

    it("passes the run signal through to every abs-page fetch", async () => {
      const h = setup();
      const controller = new AbortController();
      await okPapers(h.adapter, controller.signal);
      for (const call of h.fetchListingAbstract.mock.calls) expect(call[1]).toBe(controller.signal);
    });

    it("only fetches the papers the query API did not cover, and warns about the partial result", async () => {
      const covered = new Set(ids.slice(0, 5));
      const metadata = new Map<string, unknown>(
        [...covered].map((id) => [id, { id, abstract: "atom abstract", updated: "2026-06-15T00:00:00Z", categories: [] }]),
      );
      // One id is returned without an abstract: it must fall back too.
      const emptyId = ids[5]!;
      metadata.set(emptyId, { id: emptyId, abstract: "", updated: "", categories: [] });
      const h = setup({ metadata });
      const papers = await okPapers(h.adapter);

      const fetched = h.fetchListingAbstract.mock.calls.map((call) => call[0]);
      expect(fetched).toEqual(ids.slice(5));
      for (const paper of papers) {
        expect(paper.abstract).toBe(covered.has(paper.externalId) ? "atom abstract" : `abs-page ${paper.externalId}`);
      }
      expect(allMessages(h.warn).some((m) => m.includes(`enriched 6/${ids.length}`) && m.includes("Atom API"))).toBe(true);
    });

    it("does not call the abs-page fallback when the query API covers every paper", async () => {
      const metadata = new Map<string, unknown>(
        ids.map((id) => [id, { id, abstract: "atom abstract", updated: "", categories: [] }]),
      );
      const h = setup({ metadata });
      await okPapers(h.adapter);
      expect(h.fetchListingAbstract).not.toHaveBeenCalled();
      expect(h.warn).not.toHaveBeenCalled();
    });

    it("does not count cache hits toward the network cap", async () => {
      const cached = new Set(ids.slice(0, 4));
      const h = setup({
        limit: 2,
        listing: async (id) => ({ abstract: `abs-page ${id}`, fromNetwork: !cached.has(id) }),
      });
      const papers = await okPapers(h.adapter);

      // 4 cache hits + 2 network fetches = 6 papers filled; the rest stay empty.
      expect(h.fetchListingAbstract).toHaveBeenCalledTimes(6);
      expect(papers.filter((p) => p.abstract).map((p) => p.externalId)).toEqual(ids.slice(0, 6));
      expect(papers.slice(6).every((p) => p.abstract === "")).toBe(true);
      const summary = allMessages(h.warn).find((m) => m.includes("abs-page fallback"));
      expect(summary).toContain(`${ids.length - 6}`);
      expect(summary).toMatch(/cap|limit/i);
    });

    it("stops after 3 consecutive failures; a success in between resets the counter", async () => {
      const failing = new Set([ids[0], ids[1], ids[3], ids[4], ids[5], ids[6]]);
      const h = setup({
        listing: async (id) => {
          if (failing.has(id)) throw new ArxivHttpError(503, `https://arxiv.org/abs/${id}`);
          return { abstract: `abs-page ${id}`, fromNetwork: true };
        },
      });
      const papers = await okPapers(h.adapter);

      // ids[0],[1] fail; [2] ok resets; [3],[4],[5] fail -> stop. [6]+ never tried.
      expect(h.fetchListingAbstract.mock.calls.map((call) => call[0])).toEqual(ids.slice(0, 6));
      expect(papers.filter((p) => p.abstract).map((p) => p.externalId)).toEqual([ids[2]]);
      const summary = allMessages(h.warn).find((m) => m.includes("abs-page fallback"));
      expect(summary).toMatch(/3 consecutive failures/);
      expect(summary).toContain(`${ids.length - 1}`);
    });

    it("counts an active cooldown (retry deferred) as a failure", async () => {
      const h = setup({
        listing: async () => { throw new ArxivRetryDeferredError(new Date("2026-06-25T12:00:00Z"), 600_000); },
      });
      const papers = await okPapers(h.adapter);
      expect(h.fetchListingAbstract).toHaveBeenCalledTimes(3);
      expect(papers.every((p) => p.abstract === "")).toBe(true);
      expect(allMessages(h.warn).some((m) => /3 consecutive failures/.test(m))).toBe(true);
    });

    it("treats a page without an abstract as a success, not a failure", async () => {
      const h = setup({ listing: async () => ({ abstract: "", fromNetwork: true }) });
      const papers = await okPapers(h.adapter);
      expect(h.fetchListingAbstract).toHaveBeenCalledTimes(ids.length);
      expect(papers.every((p) => p.abstract === "")).toBe(true);
      const summary = allMessages(h.warn).find((m) => m.includes("abs-page fallback"));
      expect(summary).toContain(`${ids.length}`);
      expect(summary).not.toMatch(/consecutive/);
    });

    it("rejects with a cancellation error mid-fallback and stops fetching", async () => {
      const controller = new AbortController();
      const h = setup({
        listing: async (id) => {
          if (id === ids[2]) controller.abort("user cancelled");
          return { abstract: `abs-page ${id}`, fromNetwork: true };
        },
      });

      await expect(h.adapter.listForDate(date, { signal: controller.signal }))
        .rejects.toMatchObject({ name: "RunCancelledError" });
      expect(h.fetchListingAbstract).toHaveBeenCalledTimes(3);
    });

    it("rethrows a cancellation thrown by the abs-page fetch instead of counting it as a failure", async () => {
      const h = setup({
        listing: async () => { throw Object.assign(new Error("aborted"), { name: "AbortError" }); },
      });
      await expect(h.adapter.listForDate(date)).rejects.toMatchObject({ name: "AbortError" });
      expect(h.fetchListingAbstract).toHaveBeenCalledTimes(1);
    });

    it("still rethrows cancellation from the query API step", async () => {
      const controller = new AbortController();
      const h = setup({ metadata: Object.assign(new Error("aborted"), { name: "AbortError" }) });
      await expect(h.adapter.listForDate(date, { signal: controller.signal }))
        .rejects.toMatchObject({ name: "AbortError" });
      expect(h.fetchListingAbstract).not.toHaveBeenCalled();
    });
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

describe("ArxivSourceAdapter with a listing longer than one page", () => {
  const D1 = "2026-05-11";
  const D2 = "2026-05-08";
  const D3 = "2026-05-07";
  const D4 = "2026-05-06";
  const OLDER = "2026-05-05";
  // arXiv serves 2000 entries per page: page 1 holds D1 and 900 of D2's 1300.
  const days: WorldDay[] = [
    { date: D1, ids: listingIds("2605", 10000, 1100) },
    { date: D2, ids: listingIds("2605", 20000, 1300) },
    { date: D3, ids: listingIds("2605", 30000, 900) },
    { date: D4, ids: listingIds("2605", 40000, 300) },
  ];

  function adapterOver(
    fetchRecent: (category: string, signal?: AbortSignal, options?: { skip?: number; show?: number }) => Promise<string>,
    categories = ["cs.LG"],
  ) {
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const adapter = new ArxivSourceAdapter({
      fetcher: { fetchRecent: vi.fn(fetchRecent), fetchMetadataByIds: vi.fn(async () => new Map()) } as any,
      paperFetcher: { fetch: vi.fn(), fetchListingAbstract: vi.fn(async () => ({ abstract: "listed", fromNetwork: false })) } as any,
      markupParser,
      logger,
      defaultCategories: categories,
    });
    return { adapter, warn };
  }

  it("takes a day that lies beyond the first page from the offset arXiv gives for it", async () => {
    const world = recentListingWorld(days);
    const { adapter } = adapterOver(world.fetchRecent);

    const result = await adapter.listForDate(D4);

    expect(result.kind).toBe("ok");
    if (result.kind !== "ok") return;
    expect(result.papers.map((paper) => paper.externalId)).toEqual(days[3]!.ids);
    expect(world.requests).toEqual([{ skip: 0, show: 2000 }, { skip: 3300, show: 300 }]);
  });

  it("completes a day that the first page cuts off", async () => {
    const world = recentListingWorld(days);
    const { adapter } = adapterOver(world.fetchRecent);

    const result = await adapter.listForDate(D2);

    expect(result.kind).toBe("ok");
    if (result.kind !== "ok") return;
    expect(result.papers).toHaveLength(1300);
    expect(new Set(result.papers.map((paper) => paper.externalId))).toEqual(new Set(days[1]!.ids));
    expect(world.requests).toEqual([{ skip: 0, show: 2000 }, { skip: 1100, show: 1300 }]);
  });

  it("asks for nothing more when the first page holds the day whole", async () => {
    const world = recentListingWorld(days);
    const { adapter } = adapterOver(world.fetchRecent);

    const result = await adapter.listForDate(D1);

    expect(result.kind).toBe("ok");
    if (result.kind === "ok") expect(result.papers).toHaveLength(1100);
    expect(world.requests).toEqual([{ skip: 0, show: 2000 }]);
  });

  it("calls a date permanently gone only when no page can list it", async () => {
    const world = recentListingWorld(days);
    const { adapter } = adapterOver(world.fetchRecent);

    const result = await adapter.listForDate(OLDER);

    expect(result).toMatchObject({ kind: "error", failureKind: "failed_permanent" });
    expect(world.requests).toEqual([{ skip: 0, show: 2000 }]);
  });

  it("still waits for an announcement newer than every listed day", async () => {
    const world = recentListingWorld(days);
    const { adapter } = adapterOver(world.fetchRecent);

    await expect(adapter.listForDate("2026-05-12")).resolves.toMatchObject({
      kind: "pending",
      outcome: "awaiting_announcement",
    });
    expect(world.requests).toHaveLength(1);
  });

  it("reads the rest of a day longer than one page from the page after it", async () => {
    const big: WorldDay[] = [
      { date: D1, ids: listingIds("2605", 10000, 2300) },
      { date: D2, ids: listingIds("2605", 20000, 100) },
    ];
    const world = recentListingWorld(big);
    const { adapter } = adapterOver(world.fetchRecent);

    const result = await adapter.listForDate(D1);

    expect(result.kind).toBe("ok");
    if (result.kind === "ok") expect(result.papers.map((paper) => paper.externalId)).toEqual(big[0]!.ids);
    expect(world.requests).toEqual([{ skip: 0, show: 2000 }, { skip: 2000, show: 300 }]);
  });

  it("starts a longer-than-a-page day at its own offset when it is not the first day", async () => {
    const big: WorldDay[] = [
      { date: D1, ids: listingIds("2605", 10000, 500) },
      { date: D2, ids: listingIds("2605", 20000, 2100) },
    ];
    const world = recentListingWorld(big);
    const { adapter } = adapterOver(world.fetchRecent);

    const result = await adapter.listForDate(D2);

    expect(result.kind).toBe("ok");
    if (result.kind === "ok") expect(result.papers.map((paper) => paper.externalId)).toEqual(big[1]!.ids);
    expect(world.requests).toEqual([
      { skip: 0, show: 2000 }, { skip: 500, show: 2000 }, { skip: 2500, show: 100 },
    ]);
  });

  it("fails transiently, never with a part of the day, when arXiv keeps serving the same page", async () => {
    const world = recentListingWorld(days);
    const { adapter, warn } = adapterOver(async (category, signal, options) => {
      await world.fetchRecent(category, signal, options);
      return world.page(0, 2000);
    });

    const result = await adapter.listForDate(D2);

    expect(result).toMatchObject({ kind: "error", failureKind: "failed_transient" });
    if (result.kind === "error") expect(result.reason).toMatch(/cs\.LG.*2026-05-08.*900 of 1300/);
    expect(world.requests.length).toBeLessThanOrEqual(6);
    expect(warn).toHaveBeenCalled();
  });

  it("fails transiently when a day is cut off and the page has no link to the rest", async () => {
    const world = recentListingWorld(days);
    const asked: unknown[] = [];
    const { adapter } = adapterOver(async (_category, _signal, options) => {
      asked.push(options);
      return world.page(0, 2000).replace(/<li>[\s\S]*?<\/li>/g, "");
    });

    const result = await adapter.listForDate(D2);

    expect(result).toMatchObject({ kind: "error", failureKind: "failed_transient" });
    if (result.kind === "error") expect(result.reason).toMatch(/cs\.LG.*2026-05-08.*900 of 1300/);
    expect(asked).toEqual([undefined]);
  });

  it("does not call a missing date permanent while the page may simply stop before it", async () => {
    const world = recentListingWorld(days);
    const { adapter } = adapterOver(async () => world.page(0, 2000).replace(/<li>[\s\S]*?<\/li>/g, ""));

    // Without day links nothing proves the date is gone: the first page ends at entry 2000 of 3600.
    const result = await adapter.listForDate(D4);

    expect(result).toMatchObject({ kind: "error", failureKind: "failed_transient" });
    if (result.kind === "error") expect(result.reason).toContain("longer than one page");
  });

  it("keeps calling a date permanent when a short page without day links ends before it", async () => {
    const short = recentListingWorld([{ date: D3, ids: listingIds("2605", 30000, 5) }]);
    const { adapter } = adapterOver(async () => short.page(0, 2000).replace(/<li>[\s\S]*?<\/li>/g, ""));

    await expect(adapter.listForDate(D4)).resolves.toMatchObject({ kind: "error", failureKind: "failed_permanent" });
  });

  it.each([
    ["an HTTP error", () => new ArxivHttpError(503, "https://arxiv.org/list/cs.LG/recent?skip=3300&show=300"), "failed_transient"],
    ["a refusal", () => new ArxivHttpError(404, "https://arxiv.org/list/cs.LG/recent?skip=3300&show=300"), "failed_permanent"],
  ])("classifies %s on a later page like one on the first", async (_name, makeError, kind) => {
    const world = recentListingWorld(days);
    const { adapter } = adapterOver(async (category, signal, options) => {
      if (options?.skip) throw makeError();
      return world.fetchRecent(category, signal, options);
    });

    const result = await adapter.listForDate(D4);

    expect(result).toMatchObject({ kind: "error", failureKind: kind });
    if (result.kind === "error") expect(result.reason).toContain("cs.LG");
  });

  it("lets a cancellation on a later page through", async () => {
    const world = recentListingWorld(days);
    const controller = new AbortController();
    const { adapter } = adapterOver(async (category, signal, options) => {
      if (options?.skip) controller.abort(new Error("stop"));
      return world.fetchRecent(category, signal, options);
    });

    await expect(adapter.listForDate(D4, { signal: controller.signal })).rejects.toThrow();
  });

  it("reads a later page through the real fetcher, at the right URL and under the shared spacing", async () => {
    const world = recentListingWorld(days);
    let monotonic = 0;
    const waits: number[] = [];
    const coordinator = new ArxivRequestCoordinator({
      monotonicNow: () => monotonic,
      wallNow: () => 0,
      sleep: async (ms) => { waits.push(ms); monotonic += ms; },
    });
    const urls: string[] = [];
    const http: HttpClient = {
      request: vi.fn(async (req) => {
        urls.push(req.url);
        const query = new URL(req.url).searchParams;
        return { status: 200, headers: {}, bodyText: world.page(Number(query.get("skip")), Number(query.get("show"))) };
      }),
    };
    const fetcher = new ArxivFetcher({
      categories: ["cs.LG"], markupParser, http, logger: new Logger("error"),
      requestDelayMs: 0, requestCoordinator: coordinator,
    });
    fetcher.fetchMetadataByIds = vi.fn(async () => new Map());
    const adapter = new ArxivSourceAdapter({
      fetcher,
      paperFetcher: { fetch: vi.fn(), fetchListingAbstract: vi.fn(async () => ({ abstract: "listed", fromNetwork: false })) } as any,
      markupParser,
      logger: new Logger("error"),
      defaultCategories: ["cs.LG"],
    });

    const result = await adapter.listForDate(D4);

    expect(result.kind).toBe("ok");
    if (result.kind === "ok") expect(result.papers.map((paper) => paper.externalId)).toEqual(days[3]!.ids);
    expect(urls).toEqual([
      "https://arxiv.org/list/cs.LG/recent?skip=0&show=2000",
      "https://arxiv.org/list/cs.LG/recent?skip=3300&show=300",
    ]);
    expect(waits).toEqual([3000]);
  });

  it("starts the host's cooldown when a later page is throttled, so no further page is asked for", async () => {
    const world = recentListingWorld(days);
    let monotonic = 0;
    const coordinator = new ArxivRequestCoordinator({
      monotonicNow: () => monotonic,
      wallNow: () => 0,
      sleep: async (ms) => { monotonic += ms; },
    });
    const urls: string[] = [];
    const http: HttpClient = {
      request: vi.fn(async (req) => {
        urls.push(req.url);
        const query = new URL(req.url).searchParams;
        if (query.get("skip") !== "0") return { status: 429, headers: { "Retry-After": "7200" }, bodyText: "slow down" };
        return { status: 200, headers: {}, bodyText: world.page(0, 2000) };
      }),
    };
    const fetcher = new ArxivFetcher({
      categories: ["cs.LG"], markupParser, http, logger: new Logger("error"),
      requestDelayMs: 0, requestCoordinator: coordinator,
    });
    const adapter = new ArxivSourceAdapter({
      fetcher,
      paperFetcher: { fetch: vi.fn() } as any,
      markupParser,
      logger: new Logger("error"),
      defaultCategories: ["cs.LG"],
    });

    const result = await adapter.listForDate(D4);
    expect(result).toMatchObject({ kind: "error", failureKind: "failed_transient" });
    // A second attempt inside the cooldown does not reach arxiv.org at all.
    const again = await adapter.listForDate(D4);
    expect(again).toMatchObject({ kind: "error", failureKind: "failed_transient" });
    expect(urls.filter((url) => url.includes("skip=3300"))).toHaveLength(1);
  });

  it("handles each category on its own", async () => {
    const small = recentListingWorld([{ date: D4, ids: listingIds("2606", 50000, 20) }], "astro-ph");
    const large = recentListingWorld(days);
    const { adapter } = adapterOver(
      async (category, signal, options) => (category === "astro-ph" ? small : large).fetchRecent(category, signal, options),
      ["astro-ph", "cs.LG"],
    );

    const result = await adapter.listForDate(D4);

    expect(result.kind).toBe("ok");
    if (result.kind === "ok") {
      expect(result.channels).toEqual(["astro-ph", "cs.LG"]);
      expect(result.papers).toHaveLength(320);
    }
    expect(small.requests).toEqual([{ skip: 0, show: 2000 }]);
    expect(large.requests).toEqual([{ skip: 0, show: 2000 }, { skip: 3300, show: 300 }]);
  });
});
