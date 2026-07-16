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
    fetchAtomEntry: vi.fn(async () => overrides.atom ?? atomFor("2605.08080")),
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

  it("refreshes the paper index when the target file already exists", async () => {
    const d = makeDeps({ exists: true });
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

    expect(r.kind).toBe("already_exists");
    expect(d.paperFetcher.fetch).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
    expect(d.writer.refreshPaperNoteFrontmatter).toHaveBeenCalledWith(
      expect.objectContaining({
        arxivId: "2605.08080",
        published: "2026-02-02",
      }),
      "arxiv-daily/papers/2605.08080.md",
    );
    const index = JSON.parse(d.files["arxiv-daily/.index/papers.json"]);
    expect(index.papers["2605.08080"].published).toBe("2026-02-02");
    expect(index.papers["2605.08080"].updated).toBe("2026-06-15");
    expect(index.papers["2605.08080"].seenDates).toEqual(["2026-05-12"]);
    expect(index.papers["2605.08080"].paperPath).toBe(
      "arxiv-daily/papers/2605.08080.md",
    );
  });

  it("refreshes existing note frontmatter with the daily report date", async () => {
    const d = makeDeps({ exists: true });
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
      paperPath: "arxiv-daily/papers/2605.08080.md",
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

    expect(r.kind).toBe("already_exists");
    expect(d.writer.refreshPaperNoteFrontmatter).toHaveBeenCalledWith(
      expect.objectContaining({
        arxivId: "2605.08080",
        published: "2026-06-12",
      }),
      "arxiv-daily/papers/2605.08080.md",
    );
    const index = JSON.parse(d.files["arxiv-daily/.index/papers.json"]);
    expect(index.papers["2605.08080"].published).toBe("2026-06-12");
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

  it("happy path: writes paper file and returns done", async () => {
    const { svc, writer, llm } = baseDeps();
    const r = await svc.fetchAndSummarize("2605.08080", "2026-05-12");
    expect(r.kind).toBe("done");
    expect((r as any).path).toBe("papers/2605.08080.md");
    expect(llm.call).toHaveBeenCalled();
    expect(writer.writePaperDetail).toHaveBeenCalledTimes(1);
    const paperArg = writer.writePaperDetail.mock.calls[0][0];
    expect(paperArg.id).toBe("2605.08080");
    expect(paperArg.title).toContain("Test paper");
    expect(paperArg.isDetail).toBe(true);
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
    expect(index.papers["2605.08080"].status).toBe("saved");
    expect(index.papers["2605.08080"].detail).toBe(true);
    expect(index.papers["2605.08080"].paperPath).toBe("papers/2605.08080.md");
    expect(index.papers["2605.08080"].published).toBe("2026-02-02");
    expect(index.papers["2605.08080"].updated).toBe("2026-06-15");
    expect(index.papers["2605.08080"].seenDates).toEqual(["2026-05-12"]);
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
    expect(index.papers["2605.08080"].published).toBe("2026-06-12");
    expect(index.papers["2605.08080"].updated).toBe("2026-06-15");
    expect(index.papers["2605.08080"].seenDates).toEqual([
      "2026-06-12",
      "2026-06-17",
    ]);
    expect(index.papers["2605.08080"].dailyReports).toEqual([
      "arxiv-daily/daily/2026-06-12.md",
    ]);
  });
});
