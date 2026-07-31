import { markupParser } from "./markup-parser";
import { describe, it, expect, vi } from "vitest";
import {
  ManualFetchService,
  normalizeArxivId,
} from "../src/services/manual-fetch";
import { Logger } from "../src/services/logger";
import { PaperIndexStore } from "../src/services/paper-index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { StorageAdapter } from "../src/core/adapters";
import { ArxivFetcher, ArxivHttpError } from "../src/pipeline/arxiv-fetcher";
import { AtomMetadataCache } from "../src/pipeline/atom-metadata-cache";

const atomFor = (
  id: string,
  opts: Partial<{
    title: string;
    authors: string[];
    primary: string;
    abstract: string;
    published: string;
    updated: string;
  }> = {},
) => {
  const title = opts.title ?? "Test paper title";
  const abstract = opts.abstract ?? "Abstract body.";
  const primary = opts.primary ?? "astro-ph.CO";
  const authors = opts.authors ?? ["Foo Bar", "Baz Qux"];
  const published = opts.published ?? "2026-02-02T02:28:06Z";
  const updated = opts.updated ?? "2026-06-15T02:34:08Z";
  const authorXml = authors.map((a) => `<author><name>${a}</name></author>`).join("");
  return `<?xml version='1.0'?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    <entry>
      <id>http://arxiv.org/abs/${id}v1</id>
      <published>${published}</published>
      <updated>${updated}</updated>
      <title>${title}</title>
      <summary>${abstract}</summary>
      <arxiv:primary_category term="${primary}"/>
      <category term="${primary}"/>
      ${authorXml}
    </entry>
  </feed>`;
};

function makeDeps(overrides: Partial<{
  exists: boolean;
  atom: string | null;
  content: { abstractConclusion: string; fullSections: string | null } | null;
  llmText: string;
}> = {}) {
  const files: Record<string, string> = {};
  if (overrides.exists) {
    files["arxiv-daily/papers/2605.08080.md"] = [
      "---",
      'arxiv_id: "2605.08080"',
      "---",
      "# Existing detail",
      "## Research question",
      "A".repeat(150),
      "## Method",
      "B".repeat(150),
      "## Evidence",
      "C".repeat(150),
      "## Limitations",
      "D".repeat(150),
    ].join("\n");
  }
  const dirs = new Set<string>();
  const vault = {
    adapter: {
      read: vi.fn(async (path: string) => files[path]),
      exists: vi.fn(async (path: string) => {
        if (overrides.exists !== undefined && path.endsWith(".md")) {
          return overrides.exists;
        }
        return path in files || dirs.has(path);
      }),
      write: vi.fn(async (path: string, content: string) => {
        files[path] = content;
      }),
      mkdir: vi.fn(async (path: string) => {
        dirs.add(path);
      }),
      rename: vi.fn(async (from: string, to: string) => {
        files[to] = files[from];
        delete files[from];
      }),
      remove: vi.fn(async (path: string) => {
        delete files[path];
        dirs.delete(path);
      }),
    },
  };
  const storage = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/");
    },
    readText: vault.adapter.read,
    writeText: vault.adapter.write,
    exists: vault.adapter.exists,
    mkdir: vault.adapter.mkdir,
    rename: vault.adapter.rename,
    remove: vault.adapter.remove,
  } satisfies StorageAdapter;
  const fetcher = {
    fetchMetadataByIds: vi.fn(async () => {
      if (overrides.atom?.includes("<feed") && !overrides.atom.includes("<entry")) {
        return new Map();
      }
      return new Map([["2605.08080", {
        id: "2605.08080",
        title: "Test paper title",
        authors: "Foo Bar et al.",
        abstract: "Abstract body.",
        published: "2026-02-02T02:28:06Z",
        updated: "2026-06-15T02:34:08Z",
        primaryCategory: "astro-ph.CO",
        categories: ["astro-ph.CO"],
      }]]);
    }),
  };
  const paperFetcher = {
    fetch: vi.fn(async () =>
      overrides.content === null
        ? { abstractConclusion: "## Abstract\nstub", fullSections: null }
        : overrides.content ?? { abstractConclusion: "## Abstract\nstub", fullSections: "## Methods\n..." },
    ),
  };
  const writer = {
    writePaperDetail: vi.fn(async (p: any) => `papers/${p.id}.md`),
    refreshPaperNoteFrontmatter: vi.fn(async (_entry: any, path: string) => path),
    writeDaily: vi.fn(),
    writeEmptyDaily: vi.fn(),
  };
  const llm = {
    call: vi.fn(async () => overrides.llmText ?? "# Summary\n\nbody"),
  };
  return { files, vault, storage, fetcher, paperFetcher, writer, llm };
}

describe("normalizeArxivId", () => {
  it("accepts bare id", () => {
    expect(normalizeArxivId("2605.08080")).toBe("2605.08080");
  });
  it("strips version suffix", () => {
    expect(normalizeArxivId("2605.08080v2")).toBe("2605.08080");
  });
  it("strips arXiv: prefix", () => {
    expect(normalizeArxivId("arXiv:2605.08080")).toBe("2605.08080");
    expect(normalizeArxivId("arxiv: 2605.08080v1")).toBe("2605.08080");
  });
  it("accepts abs/pdf/html URLs", () => {
    expect(normalizeArxivId("https://arxiv.org/abs/2605.08080")).toBe("2605.08080");
    expect(normalizeArxivId("https://arxiv.org/pdf/2605.08080v1")).toBe("2605.08080");
    expect(normalizeArxivId("http://www.arxiv.org/html/2605.08080v3")).toBe("2605.08080");
    expect(normalizeArxivId("https://arxiv.org/pdf/2605.08080.pdf")).toBe("2605.08080");
  });
  it("rejects malformed input", () => {
    expect(normalizeArxivId("")).toBeNull();
    expect(normalizeArxivId("xyz")).toBeNull();
    expect(normalizeArxivId("2605.08")).toBeNull();
  });
});

describe("ManualFetchService", () => {
  const baseDeps = (overrides: any = {}) => {
    const d = makeDeps(overrides);
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    return { svc, ...d };
  };

  it("returns error for invalid id", async () => {
    const { svc } = baseDeps();
    const r = await svc.fetchAndSummarize("xyz", "2026-05-12");
    expect(r.kind).toBe("error");
  });

  it("returns already_exists when target file present", async () => {
    const { svc, paperFetcher } = baseDeps({ exists: true });
    const r = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(r.kind).toBe("already_exists");
    expect(paperFetcher.fetch).not.toHaveBeenCalled();
  });

  it("regenerates an existing frontmatter-only paper note", async () => {
    const d = makeDeps();
    d.files["arxiv-daily/papers/2605.08080.md"] = [
      "---",
      'title: "Test paper title"',
      'arxiv_id: "2605.08080"',
      "---",
      "",
    ].join("\n");
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const r = await svc.fetchAndSummarize("2605.08080", "2026-05-12");

    expect(r.kind).toBe("done");
    expect(d.paperFetcher.fetch).toHaveBeenCalled();
    expect(d.llm.call).toHaveBeenCalled();
    expect(d.vault.adapter.remove).not.toHaveBeenCalledWith(
      "arxiv-daily/papers/2605.08080.md",
    );
    expect(d.writer.writePaperDetail).toHaveBeenCalledWith(
      expect.anything(),
      "2026-05-12",
      expect.any(String),
      undefined,
      expect.objectContaining({ replaceExisting: true }),
    );
  });

  it("regenerates only the exact generated empty Notes stub", async () => {
    const d = makeDeps();
    d.files["arxiv-daily/papers/2605.08080.md"] = [
      "---",
      'arxiv_id: "2605.08080"',
      "---",
      "# Test paper title",
      "",
      "- **arXiv**: [2605.08080](https://arxiv.org/abs/2605.08080)",
      "- **PDF**: [PDF](https://arxiv.org/pdf/2605.08080)",
      "",
      "## Notes",
      "",
    ].join("\n");
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    await expect(svc.fetchAndSummarize("2605.08080", "2026-05-12")).resolves.toMatchObject({
      kind: "done",
    });
    expect(d.writer.writePaperDetail).toHaveBeenCalledWith(
      expect.anything(),
      "2026-05-12",
      expect.any(String),
      undefined,
      expect.objectContaining({ replaceExisting: true }),
    );
  });

  it("protects a replaceable note changed while summary work is pending", async () => {
    const d = makeDeps();
    const target = "arxiv-daily/papers/2605.08080.md";
    d.files[target] = ["---", 'arxiv_id: "2605.08080"', "---", ""].join("\n");
    let finishLlm!: (value: string) => void;
    d.llm.call.mockImplementationOnce(() => new Promise<string>((resolve) => {
      finishLlm = resolve;
    }));
    const paperIndex = {
      get: vi.fn(async () => null),
      upsertFromDailyPaper: vi.fn(),
      setStatus: vi.fn(),
      setPaperPath: vi.fn(),
    };
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const pending = svc.fetchAndSummarize("2605.08080", "2026-05-12");
    await vi.waitFor(() => expect(d.llm.call).toHaveBeenCalled());
    const handwritten = "---\narxiv_id: \"2605.08080\"\n---\n# My notes\nDo not overwrite\n";
    d.files[target] = handwritten;
    finishLlm("# Summary\n\nbody");

    await expect(pending).resolves.toMatchObject({ kind: "note_conflict", path: target });
    expect(d.files[target]).toBe(handwritten);
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
    expect(paperIndex.upsertFromDailyPaper).not.toHaveBeenCalled();
    expect(paperIndex.setStatus).not.toHaveBeenCalled();
    expect(paperIndex.setPaperPath).not.toHaveBeenCalled();
  });

  it("keeps frontmatter-only replacement coherent when cancellation races the commit", async () => {
    const d = makeDeps();
    const target = "arxiv-daily/papers/2605.08080.md";
    const original = ["---", 'arxiv_id: "2605.08080"', "---", ""].join("\n");
    d.files[target] = original;
    const controller = new AbortController();
    d.writer.writePaperDetail.mockImplementationOnce(async (_paper: any, _date: string, _summary: string, _entry: any, options: any) => {
      expect(options.replaceExisting).toBe(true);
      controller.abort("cancelled during replacement commit");
      d.files[target] = "---\narxiv_id: 2605.08080\n---\n\n# Summary\n";
      return target;
    });
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await svc.fetchAndSummarize("2605.08080", "2026-05-12", controller.signal);

    expect(result).toEqual({ kind: "done", path: target });
    expect(d.vault.adapter.remove).not.toHaveBeenCalledWith(target);
    expect(d.files[target]).toContain("# Summary");
  });

  it("reconciles verified already_exists with zero network or LLM work", async () => {
    const d = makeDeps({ exists: true });
    const existing = {
      arxivId: "2605.08080",
      status: "reading",
      priority: "high",
      detail: false,
      paperPath: null,
    };
    const paperIndex = {
      get: vi.fn(async () => existing),
      upsertFromDailyPaper: vi.fn(),
      setStatus: vi.fn(),
      setPaperPath: vi.fn(),
      reconcileManualDetail: vi.fn(async () => ({
        wasNew: false,
        entry: { ...existing, detail: true, paperPath: "arxiv-daily/papers/2605.08080.md" },
      })),
    };
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    await expect(svc.fetchAndSummarize("2605.08080", "2026-05-12")).resolves.toEqual({
      kind: "already_exists",
      path: "arxiv-daily/papers/2605.08080.md",
    });
    expect(d.fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
    expect(d.paperFetcher.fetch).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
    expect(paperIndex.upsertFromDailyPaper).not.toHaveBeenCalled();
    expect(paperIndex.setStatus).not.toHaveBeenCalled();
    expect(paperIndex.reconcileManualDetail).toHaveBeenCalledWith(
      expect.objectContaining({ arxivId: "2605.08080", detail: true }),
      "arxiv-daily/papers/2605.08080.md",
      "saved",
    );
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
    expect(d.writer.refreshPaperNoteFrontmatter).toHaveBeenCalledWith(
      expect.objectContaining({ status: "reading", priority: "high", detail: true }),
      "arxiv-daily/papers/2605.08080.md",
    );
  });

  it("recreates a missing index from sufficient verified frontmatter without network work", async () => {
    const d = makeDeps();
    const path = "arxiv-daily/papers/2605.08080.md";
    d.files[path] = [
      "---",
      'title: "Recovered title"',
      'authors: "A. Author, B. Author"',
      'arxiv_id: "2605.08080"',
      "primary_topic: photo-z",
      "published: 2026-05-12",
      "---",
      "# Existing detail",
      "## Research question", "A".repeat(150),
      "## Method", "B".repeat(150),
      "## Evidence", "C".repeat(150),
      "## Limitations", "D".repeat(150),
    ].join("\n");
    const paperIndex = {
      get: vi.fn(async () => null),
      upsertFromDailyPaper: vi.fn(async (input: any) => ({
        wasNew: true,
        entry: { ...input, status: "inbox", priority: "normal", detail: true },
      })),
      setStatus: vi.fn(),
      setPaperPath: vi.fn(),
      reconcileManualDetail: vi.fn(async (input: any, paperPath: string) => ({
        wasNew: true,
        entry: {
          ...input, arxivId: "2605.08080", status: "saved", priority: "normal",
          detail: true, paperPath,
        },
      })),
    };
    const svc = new ManualFetchService({
      markupParser, storage: d.storage, fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any, writer: d.writer as any,
      paperIndex: paperIndex as any, llm: d.llm as any, logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv, advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output, llmSettings: DEFAULT_SETTINGS.llm,
    });

    await expect(svc.fetchAndSummarize("2605.08080", "2026-07-30")).resolves.toEqual({
      kind: "already_exists", path,
    });
    expect(paperIndex.reconcileManualDetail).toHaveBeenCalledWith(expect.objectContaining({
      title: "Recovered title",
      authors: "A. Author, B. Author",
      date: "2026-05-12",
      primaryTopic: "photo-z",
      detail: true,
    }), path, "saved");
    expect(paperIndex.upsertFromDailyPaper).not.toHaveBeenCalled();
    expect(paperIndex.setStatus).not.toHaveBeenCalled();
    expect(d.fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
    expect(d.paperFetcher.fetch).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
  });

  it("returns an explicit safe error when a verified note cannot recreate a missing index", async () => {
    const d = makeDeps({ exists: true });
    const paperIndex = {
      get: vi.fn(async () => null), upsertFromDailyPaper: vi.fn(),
      setStatus: vi.fn(), setPaperPath: vi.fn(),
    };
    const svc = new ManualFetchService({
      markupParser, storage: d.storage, fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any, writer: d.writer as any,
      paperIndex: paperIndex as any, llm: d.llm as any, logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv, advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output, llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(result).toMatchObject({ kind: "error", reason: expect.stringContaining("cannot safely recreate") });
    expect(paperIndex.upsertFromDailyPaper).not.toHaveBeenCalled();
    expect(d.fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
  });

  it.each([
    ["handwritten", "---\narxiv_id: \"2605.08080\"\n---\n# My notes\nImportant"],
    ["mismatched", "---\narxiv_id: \"2605.99999\"\n---\n"],
  ])("protects %s notes with zero downstream work", async (_label, markdown) => {
    const d = makeDeps();
    d.files["arxiv-daily/papers/2605.08080.md"] = markdown;
    const paperIndex = { upsertFromDailyPaper: vi.fn(), setStatus: vi.fn(), setPaperPath: vi.fn() };
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(result.kind).toBe("note_conflict");
    expect(d.fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
    expect(d.paperFetcher.fetch).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
    expect(paperIndex.upsertFromDailyPaper).not.toHaveBeenCalled();
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
  });

  it("returns error when an existing note cannot be read", async () => {
    const d = makeDeps({ exists: true });
    d.storage.readText = vi.fn(async () => { throw new Error("permission denied"); });
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(result).toMatchObject({ kind: "error", reason: expect.stringContaining("permission denied") });
    expect(d.fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
  });

  it("formats typed arXiv HTTP failures without exposing the query URL", async () => {
    const { svc, fetcher } = baseDeps();
    fetcher.fetchMetadataByIds.mockRejectedValueOnce(
      new ArxivHttpError(429, "https://export.arxiv.org/api/query?id_list=2605.08080"),
    );

    const result = await svc.fetchAndSummarize("2605.08080", "2026-05-12");

    expect(result).toMatchObject({ kind: "error", reason: expect.stringContaining("rate-limiting") });
    expect(result.kind === "error" ? result.reason : "").not.toContain("id_list");
  });

  it("returns not_found when Atom has no entry", async () => {
    const { svc } = baseDeps({
      atom: `<?xml version='1.0'?><feed xmlns="http://www.w3.org/2005/Atom"></feed>`,
    });
    const r = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(r.kind).toBe("not_found");
  });

  it("returns no_html when fullSections cannot be extracted", async () => {
    const { svc } = baseDeps({
      content: { abstractConclusion: "## Abstract\nstub", fullSections: null },
    });
    const r = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(r.kind).toBe("no_html");
  });

  it("reuses Atom metadata cached by a prior daily-style fetch", async () => {
    const d = makeDeps();
    const http = { request: vi.fn(async () => ({ status: 200, headers: {}, bodyText: atomFor("2605.08080") })) };
    const metadataCache = new AtomMetadataCache({
      rootDir: "cache",
      expiryDays: 7,
      storage: d.storage,
    });
    const firstFetcher = new ArxivFetcher({
      categories: ["astro-ph"], http, markupParser, logger: new Logger("error"),
      requestDelayMs: 0, metadataCache,
    });
    await firstFetcher.fetchMetadataByIds(["2605.08080"]);
    const manualFetcher = new ArxivFetcher({
      categories: ["astro-ph"], http, markupParser, logger: new Logger("error"),
      requestDelayMs: 0,
      metadataCache: new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage: d.storage }),
    });
    const svc = new ManualFetchService({
      markupParser, storage: d.storage, fetcher: manualFetcher,
      paperFetcher: d.paperFetcher as any, writer: d.writer as any, llm: d.llm as any,
      logger: new Logger("error"), arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced, output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    expect((await svc.fetchAndSummarize("2605.08080", "2026-05-12")).kind).toBe("done");
    expect(http.request).toHaveBeenCalledOnce();
  });

  it("happy path: writes paper file and returns done", async () => {
    const { svc, writer, llm, fetcher } = baseDeps();
    const r = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(r.kind).toBe("done");
    expect((r as any).path).toBe("papers/2605.08080.md");
    expect(fetcher.fetchMetadataByIds).toHaveBeenCalledWith(["2605.08080"], undefined);
    expect(llm.call).toHaveBeenCalled();
    expect(writer.writePaperDetail).toHaveBeenCalledTimes(1);
    const paperArg = writer.writePaperDetail.mock.calls[0][0];
    expect(paperArg.id).toBe("2605.08080");
    expect(paperArg.title).toContain("Test paper");
    expect(paperArg.isDetail).toBe(true);
  });

  it("leaves an existing paper index unchanged when detail writing fails", async () => {
    const d = makeDeps();
    const paperIndex = new PaperIndexStore(
      d.storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-11T01:30:00.000Z"),
    );
    await paperIndex.upsertFromDailyPaper({
      arxivId: "2605.08080",
      title: "Indexed paper",
      authors: "A. Author",
      date: "2026-05-10",
      published: "2026-05-10",
      updated: "2026-05-10",
      arxivCategory: "astro-ph.CO",
      primaryTopic: "existing-topic",
      detail: false,
      dailyReport: "arxiv-daily/daily/2026-05-10.md",
    });
    const before = d.files["arxiv-daily/.index/papers.json"];
    d.writer.writePaperDetail.mockRejectedValueOnce(new Error("disk full"));
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    await expect(svc.fetchAndSummarize("2605.08080", "2026-05-12")).rejects.toThrow("disk full");

    expect(d.files["arxiv-daily/.index/papers.json"]).toBe(before);
    await expect(paperIndex.get("2605.08080")).resolves.toMatchObject({
      detail: false,
      status: "inbox",
      paperPath: null,
      seenDates: ["2026-05-10"],
      primaryTopic: "existing-topic",
    });
    expect(d.writer.refreshPaperNoteFrontmatter).not.toHaveBeenCalled();
  });

  it("does not create saved index state when the first detail write fails", async () => {
    const d = makeDeps();
    const paperIndex = new PaperIndexStore(d.storage, DEFAULT_SETTINGS.output);
    d.writer.writePaperDetail.mockRejectedValueOnce(new Error("disk full"));
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    await expect(svc.fetchAndSummarize("2605.08080", "2026-05-12")).rejects.toThrow("disk full");

    expect(d.files["arxiv-daily/.index/papers.json"]).toBeUndefined();
    await expect(paperIndex.get("2605.08080")).resolves.toBeNull();
  });

  it("recovers a verified note after a post-write index failure without refetching", async () => {
    const d = makeDeps();
    const path = "arxiv-daily/papers/2605.08080.md";
    d.writer.writePaperDetail.mockImplementationOnce(async () => {
      d.files[path] = [
        "---", 'title: "Test paper title"', 'authors: "Foo Bar et al."',
        'arxiv_id: "2605.08080"', "primary_topic: astro-ph.CO",
        "published: 2026-02-02", "---", "# Detail",
        "## Research question", "A".repeat(150), "## Method", "B".repeat(150),
        "## Evidence", "C".repeat(150), "## Limitations", "D".repeat(150),
      ].join("\n");
      return path;
    });
    const entry = {
      arxivId: "2605.08080", status: "reading", priority: "high",
      detail: false, paperPath: null, dailyReports: [], published: "2026-02-02",
    };
    let fail = true;
    const paperIndex = {
      get: vi.fn(async () => entry),
      upsertFromDailyPaper: vi.fn(async () => {
        if (fail) { fail = false; throw new Error("index disk full"); }
        return { entry, wasNew: false };
      }),
      setStatus: vi.fn(),
      setPaperPath: vi.fn(),
      reconcileManualDetail: vi.fn(async (_input: any, paperPath: string) => {
        if (fail) { fail = false; throw new Error("index disk full"); }
        return { wasNew: false, entry: { ...entry, detail: true, paperPath } };
      }),
    };
    const build = () => new ManualFetchService({
      markupParser, storage: d.storage, fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any, writer: d.writer as any,
      paperIndex: paperIndex as any, llm: d.llm as any, logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv, advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output, llmSettings: DEFAULT_SETTINGS.llm,
    });

    await expect(build().fetchAndSummarize("2605.08080", "2026-05-12")).resolves.toMatchObject({
      kind: "error", reason: expect.stringContaining("index disk full"),
    });
    d.fetcher.fetchMetadataByIds.mockClear();
    d.paperFetcher.fetch.mockClear();
    d.llm.call.mockClear();

    await expect(build().fetchAndSummarize("2605.08080", "2026-05-12")).resolves.toEqual({
      kind: "already_exists", path,
    });
    expect(d.fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
    expect(d.paperFetcher.fetch).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
    expect(paperIndex.setStatus).not.toHaveBeenCalled();
    expect(paperIndex.reconcileManualDetail).toHaveBeenLastCalledWith(
      expect.objectContaining({ arxivId: "2605.08080", detail: true }), path, "saved",
    );
    expect(d.writer.refreshPaperNoteFrontmatter).toHaveBeenCalledWith(
      expect.objectContaining({ status: "reading", priority: "high", detail: true }), path,
    );
  });

  it("keeps an atomic index coherent when frontmatter refresh fails and retries refresh next invocation", async () => {
    const d = makeDeps();
    const path = "arxiv-daily/papers/2605.08080.md";
    d.writer.writePaperDetail.mockImplementationOnce(async () => {
      d.files[path] = [
        "---", 'title: "Test paper title"', 'authors: "Foo Bar et al."',
        'arxiv_id: "2605.08080"', "primary_topic: astro-ph.CO",
        "published: 2026-02-02", "---", "# Detail",
        "## Research question", "A".repeat(150), "## Method", "B".repeat(150),
        "## Evidence", "C".repeat(150), "## Limitations", "D".repeat(150),
      ].join("\n");
      return path;
    });
    d.writer.refreshPaperNoteFrontmatter
      .mockRejectedValueOnce(new Error("frontmatter disk full"))
      .mockResolvedValue(path);
    const paperIndex = new PaperIndexStore(d.storage, DEFAULT_SETTINGS.output);
    const build = () => new ManualFetchService({
      markupParser, storage: d.storage, fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any, writer: d.writer as any,
      paperIndex, llm: d.llm as any, logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv, advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output, llmSettings: DEFAULT_SETTINGS.llm,
    });

    await expect(build().fetchAndSummarize("2605.08080", "2026-05-12")).resolves.toMatchObject({
      kind: "error", reason: expect.stringContaining("frontmatter disk full"),
    });
    await expect(paperIndex.get("2605.08080")).resolves.toMatchObject({
      status: "saved", detail: true, paperPath: path,
    });
    d.fetcher.fetchMetadataByIds.mockClear();
    d.paperFetcher.fetch.mockClear();
    d.llm.call.mockClear();

    await expect(build().fetchAndSummarize("2605.08080", "2026-05-12")).resolves.toEqual({
      kind: "already_exists", path,
    });
    expect(d.writer.refreshPaperNoteFrontmatter).toHaveBeenCalledTimes(2);
    expect(d.fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
    expect(d.paperFetcher.fetch).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
  });

  it("updates the paper index when a manual detail note is created", async () => {
    const d = makeDeps();
    const paperIndex = new PaperIndexStore(
      d.storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-11T01:30:00.000Z"),
    );
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const r = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(r.kind).toBe("done");
    const index = JSON.parse(d.files["arxiv-daily/.index/papers.json"]);
    expect(index.papers["arxiv:2605.08080"].status).toBe("saved");
    expect(index.papers["arxiv:2605.08080"].detail).toBe(true);
    expect(index.papers["arxiv:2605.08080"].paperPath).toBe("papers/2605.08080.md");
    expect(index.papers["arxiv:2605.08080"].published).toBe("2026-02-02");
    expect(index.papers["arxiv:2605.08080"].updated).toBe("2026-06-15");
    expect(index.papers["arxiv:2605.08080"].abstract).toBe("Abstract body.");
    expect(index.papers["arxiv:2605.08080"].seenDates).toEqual(["2026-05-12"]);
    expect(d.writer.refreshPaperNoteFrontmatter).toHaveBeenCalledWith(
      expect.objectContaining({
        detail: true,
        status: "saved",
        paperPath: "papers/2605.08080.md",
      }),
      "papers/2605.08080.md",
    );
  });

  it("uses an existing daily report date over a stale Atom published date", async () => {
    const d = makeDeps();
    const paperIndex = new PaperIndexStore(
      d.storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-11T01:30:00.000Z"),
    );
    await paperIndex.upsertFromDailyPaper({
      arxivId: "2605.08080",
      title: "Indexed paper",
      authors: "A. Author",
      date: "2026-06-12",
      published: "2026-06-11",
      updated: "2026-06-11",
      arxivCategory: "astro-ph.CO",
      primaryTopic: "photo-z",
      detail: false,
      dailyReport: "arxiv-daily/daily/2026-06-12.md",
    });
    const svc = new ManualFetchService({
      markupParser,
      storage: d.storage,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex,
      llm: d.llm as any,
      logger: new Logger("error"),
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const r = await svc.fetchAndSummarize("2605.08080", "2026-06-17");

    expect(r.kind).toBe("done");
    const index = JSON.parse(d.files["arxiv-daily/.index/papers.json"]);
    expect(index.papers["arxiv:2605.08080"].published).toBe("2026-06-12");
    expect(index.papers["arxiv:2605.08080"].updated).toBe("2026-06-15");
    expect(index.papers["arxiv:2605.08080"].seenDates).toEqual([
      "2026-06-12",
      "2026-06-17",
    ]);
    expect(index.papers["arxiv:2605.08080"].dailyReports).toEqual([
      "arxiv-daily/daily/2026-06-12.md",
    ]);
  });
});
