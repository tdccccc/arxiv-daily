import { describe, it, expect, vi } from "vitest";
import {
  ManualFetchService,
  normalizeArxivId,
} from "../src/services/manual-fetch";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const atomFor = (id: string, opts: Partial<{ title: string; authors: string[]; primary: string; abstract: string }> = {}) => {
  const title = opts.title ?? "Test paper title";
  const abstract = opts.abstract ?? "Abstract body.";
  const primary = opts.primary ?? "astro-ph.CO";
  const authors = opts.authors ?? ["Foo Bar", "Baz Qux"];
  const authorXml = authors.map((a) => `<author><name>${a}</name></author>`).join("");
  return `<?xml version='1.0'?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    <entry>
      <id>http://arxiv.org/abs/${id}v1</id>
      <title>${title}</title>
      <summary>${abstract}</summary>
      <arxiv:primary_category term="${primary}"/>
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
  const vault = {
    adapter: {
      exists: vi.fn(async () => overrides.exists ?? false),
      write: vi.fn(async () => undefined),
      mkdir: vi.fn(async () => undefined),
      rename: vi.fn(async () => undefined),
      remove: vi.fn(async () => undefined),
    },
  };
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
    writeDaily: vi.fn(),
    writeEmptyDaily: vi.fn(),
  };
  const llm = {
    call: vi.fn(async () => overrides.llmText ?? "# Summary\n\nbody"),
  };
  return { vault, fetcher, paperFetcher, writer, llm };
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
      vault: d.vault as any,
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
});
