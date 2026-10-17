import { markupParser } from "./markup-parser";
import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { describe, expect, it, vi } from "vitest";
import {
  cleanupSourceCache,
  PaperContentFetcher,
} from "../src/pipeline/paper-content";
import { Logger } from "../src/services/logger";
import type { StorageAdapter } from "../src/core/adapters";
import { HtmlCache } from "../src/pipeline/html-cache";
import { SOURCE_EXTRACTION_LIMITS } from "../src/pipeline/source-extractor";

const opts = {
  isDetail: true,
  sectionCharLimit: 2000,
  paperCharLimit: 6000,
  skipSections: ["references", "appendix"],
  prioritySections: ["abstract", "conclusion", "results"],
};

function toArrayBuffer(text: string): ArrayBuffer {
  const buffer = Buffer.from(text, "utf8");
  const out = new Uint8Array(buffer.byteLength);
  out.set(buffer);
  return out.buffer;
}

function makeStorage() {
  const files = new Map<string, string | ArrayBuffer>();
  const dirs = new Set<string>();
  const storage = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/");
    },
    async readText(path: string) {
      const content = files.get(path);
      if (typeof content !== "string") throw new Error(`missing ${path}`);
      return content;
    },
    async writeText(path: string, content: string) {
      files.set(path, content);
    },
    async exists(path: string) {
      return files.has(path) || dirs.has(path);
    },
    async mkdir(path: string) {
      dirs.add(path);
    },
    async remove(path: string) {
      files.delete(path);
      dirs.delete(path);
    },
    async rename(from: string, to: string) {
      const content = files.get(from);
      if (!content) throw new Error(`missing ${from}`);
      files.set(to, content);
      files.delete(from);
    },
    async readBinary(path: string) {
      const content = files.get(path);
      if (!(content instanceof ArrayBuffer)) throw new Error(`missing ${path}`);
      return content;
    },
    async writeBinary(path: string, content: ArrayBuffer) {
      files.set(path, content);
    },
    async list(dir: string) {
      const prefix = `${dir.replace(/\/+$/g, "")}/`;
      const out: Array<{ path: string; type: "file" | "folder" }> = [];
      for (const path of files.keys()) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        if (rest && !rest.includes("/")) out.push({ path, type: "file" });
      }
      for (const path of dirs) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        if (rest && !rest.includes("/")) out.push({ path, type: "folder" });
      }
      return out;
    },
  } satisfies StorageAdapter;
  return { files, dirs, storage };
}

function makeCache() {
  const values = new Map<string, string>();
  return {
    get: vi.fn(async (key: string, kind: "html" | "abs") => {
      return values.get(`${kind}:${key}`) ?? null;
    }),
    set: vi.fn(async (key: string, kind: "html" | "abs", value: string) => {
      values.set(`${kind}:${key}`, value);
    }),
  };
}

describe("PaperContentFetcher source fallback", () => {
  it.each(["network", "cache"])(
    "preserves one mathematical representation in unstructured HTML from %s",
    async (source) => {
      const fixture = readFileSync(
        new URL("./fixtures/arxiv-scientific-math.html", import.meta.url),
        "utf8",
      );
      const doc = markupParser.parseFromString(fixture, "text/html");
      const paragraph = doc.getElementById("S2.I1.i3.p1.1")!.outerHTML;
      const html = "<html><body>" + paragraph + "<p>Further measurements.</p></body></html>";
      const { storage } = makeStorage();
      const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage });
      if (source === "cache") await cache.set("html/2609.03779", "html", html);
      const fetcher = {
        fetchPaperHtml: vi.fn(async () => ({ ok: true, status: 200, body: html })),
        fetchSource: vi.fn(async () => { throw new Error("unexpected source request"); }),
        fetchPaperAbsPage: vi.fn(async () => { throw new Error("unexpected abs request"); }),
      };
      const paperFetcher = new PaperContentFetcher(
        fetcher as any, cache, new Logger("error"), markupParser,
      );

      const result = await paperFetcher.fetch("2609.03779", { ...opts, isDetail: false });

      expect(result.abstractConclusion).toContain(
        "peak flux density of $F_{\\rm peak}>0.75\\,\\rm{mJy}/\\rm{beam}$; $121\\,179$ objects",
      );
      expect(result.abstractConclusion).toContain("condition. Further measurements.");
      expect(result.fullSections).toBeNull();
      expect(await cache.get("html/2609.03779", "html")).toBe(html);
      expect(fetcher.fetchPaperHtml).toHaveBeenCalledTimes(source === "cache" ? 0 : 1);
      expect(fetcher.fetchPaperAbsPage).not.toHaveBeenCalled();
    },
  );

  it("reports N/A when the abs-page fallback finds no abstract block", async () => {
    const { storage } = makeStorage();
    const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage });
    const fetcher = {
      fetchPaperHtml: vi.fn(async () => ({ ok: false, status: 404 })),
      fetchSource: vi.fn(async () => { throw new Error("unexpected source request"); }),
      fetchPaperAbsPage: vi.fn(async () => "<html><body><p>withdrawn</p></body></html>"),
    };
    const paperFetcher = new PaperContentFetcher(
      fetcher as any, cache, new Logger("error"), markupParser,
    );

    const result = await paperFetcher.fetch("2609.03089", { ...opts, isDetail: false });

    expect(result.abstractConclusion).toBe("## Abstract\nN/A");
  });

  it.each(["network", "cache"])(
    "preserves mathematical object identifiers in the abs-page fallback from %s",
    async (source) => {
      const fixture = readFileSync(
        new URL("./fixtures/arxiv-scientific-math.html", import.meta.url),
        "utf8",
      );
      const doc = markupParser.parseFromString(fixture, "text/html");
      const abstract = doc.querySelector(".ltx_abstract")!.innerHTML;
      const html = '<html><body><blockquote class="abstract">Abstract: '
        + abstract + "</blockquote></body></html>";
      const { storage } = makeStorage();
      const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage });
      if (source === "cache") await cache.set("abs/2609.03089", "abs", html);
      const fetcher = {
        fetchPaperHtml: vi.fn(async () => ({ ok: false, status: 404 })),
        fetchSource: vi.fn(async () => { throw new Error("unexpected source request"); }),
        fetchPaperAbsPage: vi.fn(async () => html),
      };
      const paperFetcher = new PaperContentFetcher(
        fetcher as any, cache, new Logger("error"), markupParser,
      );

      const result = await paperFetcher.fetch("2609.03089", { ...opts, isDetail: false });

      expect(result.abstractConclusion).toContain("HE0435$-$1223, PG1115$+$080, and WFI2033$-$4723");
      expect(result.abstractConclusion).toMatch(/^## Abstract\nWe extract stellar kinematics/);
      expect(result.fullSections).toBeNull();
      expect(await cache.get("abs/2609.03089", "abs")).toBe(html);
      expect(fetcher.fetchPaperAbsPage).toHaveBeenCalledTimes(source === "cache" ? 0 : 1);
    },
  );

  it("re-extracts mathematical text from cached raw HTML without rewriting it", async () => {
    const html = readFileSync(
      new URL("./fixtures/arxiv-scientific-math.html", import.meta.url),
      "utf8",
    );
    const { storage } = makeStorage();
    const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage });
    await cache.set("html/2609.03089", "html", html);
    const fetcher = {
      fetchPaperHtml: vi.fn(async () => { throw new Error("unexpected network request"); }),
      fetchSource: vi.fn(async () => { throw new Error("unexpected source request"); }),
      fetchPaperAbsPage: vi.fn(async () => { throw new Error("unexpected abs request"); }),
    };
    const paperFetcher = new PaperContentFetcher(
      fetcher as any,
      cache,
      new Logger("error"),
      markupParser,
    );

    const result = await paperFetcher.fetch("2609.03089", opts);

    expect(result.abstractConclusion).toContain("HE0435$-$1223, PG1115$+$080");
    expect(result.fullSections).toContain("## II.1 Model and $\\alpha_{M}$ parameter");
    expect(result.fullTextSource).toBe("arxiv-html");
    expect(await cache.get("html/2609.03089", "html")).toBe(html);
    expect(fetcher.fetchPaperHtml).not.toHaveBeenCalled();
  });

  it("uses arXiv source when rendered HTML has no full sections", async () => {
    const tex = String.raw`
\documentclass{article}
\begin{document}
\begin{abstract}
This paper has source text available even though rendered HTML is unavailable.
\end{abstract}
\section{Method}
We describe the data, model assumptions, calibration strategy, and validation experiment in detail.
\section{Results}
The results include a measured improvement, uncertainty estimates, and a comparison with the baseline.
\end{document}
`;
    const fetcher = {
      fetchPaperHtml: vi.fn(async () => ({ ok: false, status: 404 })),
      fetchSource: vi.fn(async () => ({
        ok: true,
        body: toArrayBuffer(tex),
      })),
      fetchPaperAbsPage: vi.fn(),
    };
    const cache = makeCache();
    const { files, storage } = makeStorage();
    const paperFetcher = new PaperContentFetcher(
      fetcher as any,
      cache as any,
      new Logger("error"),
      markupParser,
      {
        storage,
        cacheDir: ".obsidian/plugins/arxiv-daily/.cache/source",
        expiryDays: 7,
      },
    );

    const result = await paperFetcher.fetch("2606.13359", opts);

    expect(result.fullTextSource).toBe("arxiv-source");
    expect(result.fullSections).toContain("## Method");
    expect(result.fullSections).toContain("validation experiment");
    expect(fetcher.fetchPaperAbsPage).not.toHaveBeenCalled();
    expect(
      files.has(".obsidian/plugins/arxiv-daily/.cache/source/2606.13359/source"),
    ).toBe(true);
    expect(
      files.has(
        ".obsidian/plugins/arxiv-daily/.cache/source/2606.13359/source.meta.json",
      ),
    ).toBe(true);
  });

  it.each(["html cache", "html request", "html cache write"])(
    "continues to source after a thrown %s failure",
    async (failurePoint) => {
      const tex = String.raw`
\documentclass{article}
\begin{document}
\begin{abstract}
Fallback abstract from source.
\end{abstract}
\section{Method}
We describe the data, model assumptions, calibration strategy, and validation experiment in detail.
\section{Results}
The results include a measured improvement, uncertainty estimates, and a comparison with the baseline.
\end{document}
`;
      const fetcher = {
        fetchPaperHtml: vi.fn(async () => {
          if (failurePoint === "html request") throw new Error("html network failed");
          return { ok: true, status: 200, body: "<html>unused</html>" };
        }),
        fetchSource: vi.fn(async () => ({ ok: true, body: toArrayBuffer(tex) })),
        fetchPaperAbsPage: vi.fn(),
      };
      const cache = makeCache();
      if (failurePoint === "html cache") cache.get.mockRejectedValueOnce(new Error("cache read failed"));
      if (failurePoint === "html cache write") cache.set.mockRejectedValueOnce(new Error("cache write failed"));
      const paperFetcher = new PaperContentFetcher(
        fetcher as any,
        cache as any,
        new Logger("error"),
        markupParser,
      );

      const result = await paperFetcher.fetch("2606.13359", opts);

      expect(result.fullTextSource).toBe("arxiv-source");
      expect(fetcher.fetchSource).toHaveBeenCalledOnce();
      if (failurePoint === "html cache write") {
        expect(result.abstractConclusion).not.toContain("Fallback abstract from source");
        expect(fetcher.fetchPaperAbsPage).not.toHaveBeenCalled();
      } else {
        expect(fetcher.fetchPaperAbsPage).toHaveBeenCalledOnce();
      }
    },
  );

  it("treats an over-budget source package as unavailable and falls back to the abs page", async () => {
    const bomb = gzipSync(Buffer.alloc(SOURCE_EXTRACTION_LIMITS.maxInflatedBytes + 1024 * 1024));
    const fetcher = {
      fetchPaperHtml: vi.fn(async () => ({ ok: false, status: 404 })),
      fetchSource: vi.fn(async () => ({ ok: true, body: toArrayBuffer(bomb) })),
      fetchPaperAbsPage: vi.fn(async () => "<html><blockquote class=\"abstract\">Abstract: A short abstract.</blockquote></html>"),
    };
    const paperFetcher = new PaperContentFetcher(
      fetcher as any,
      makeCache() as any,
      new Logger("error"),
      markupParser,
    );

    const result = await paperFetcher.fetch("2606.13359", opts);

    expect(result.fullSections).toBeNull();
    expect(result.fullTextFailure).toContain("exceeds the extraction budget");
  });

  it("aggregates HTML, source, and abs failure reasons", async () => {
    const fetcher = {
      fetchPaperHtml: vi.fn(async () => { throw new Error("html down"); }),
      fetchSource: vi.fn(async () => { throw new Error("source down"); }),
      fetchPaperAbsPage: vi.fn(async () => { throw new Error("abs down"); }),
    };
    const paperFetcher = new PaperContentFetcher(
      fetcher as any,
      makeCache() as any,
      new Logger("error"),
      markupParser,
    );

    const result = await paperFetcher.fetch("2606.13359", opts);

    expect(result.abstractConclusion).toContain("获取失败");
    expect(result.fullTextFailure).toContain("html down");
    expect(result.fullTextFailure).toContain("source down");
    expect(result.fullTextFailure).toContain("abs down");
  });

  it("propagates cancellation from the rendered HTML path without fallbacks", async () => {
    const abort = Object.assign(new Error("cancelled"), { name: "AbortError" });
    const fetcher = {
      fetchPaperHtml: vi.fn(async () => { throw abort; }),
      fetchSource: vi.fn(),
      fetchPaperAbsPage: vi.fn(),
    };
    const paperFetcher = new PaperContentFetcher(
      fetcher as any,
      makeCache() as any,
      new Logger("error"),
      markupParser,
    );

    await expect(paperFetcher.fetch("2606.13359", opts)).rejects.toBe(abort);
    expect(fetcher.fetchSource).not.toHaveBeenCalled();
    expect(fetcher.fetchPaperAbsPage).not.toHaveBeenCalled();
  });

  it("returns a clear full-text failure when source is unavailable", async () => {
    const fetcher = {
      fetchPaperHtml: vi.fn(async () => ({ ok: false, status: 404 })),
      fetchSource: vi.fn(async () => ({ ok: false, status: 404 })),
      fetchPaperAbsPage: vi.fn(async () =>
        `<html><body><blockquote class="abstract">Abstract: fallback abstract.</blockquote></body></html>`,
      ),
    };
    const cache = makeCache();
    const { storage } = makeStorage();
    const paperFetcher = new PaperContentFetcher(
      fetcher as any,
      cache as any,
      new Logger("error"),
      markupParser,
      {
        storage,
        cacheDir: ".obsidian/plugins/arxiv-daily/.cache/source",
        expiryDays: 7,
      },
    );

    const result = await paperFetcher.fetch("2606.13359", opts);

    expect(result.fullSections).toBeNull();
    expect(result.abstractConclusion).toContain("fallback abstract");
    expect(result.fullTextFailure).toContain("no arXiv source");
  });

  it("cleans expired source cache metadata entries", async () => {
    const { files, dirs, storage } = makeStorage();
    const root = ".obsidian/plugins/arxiv-daily/.cache/source";
    dirs.add(root);
    dirs.add(`${root}/2606.13359`);
    files.set(`${root}/2606.13359/source`, toArrayBuffer("source"));
    files.set(
      `${root}/2606.13359/source.meta.json`,
      JSON.stringify({
        schemaVersion: 1,
        cachedAt: new Date(Date.now() - 8 * 86_400_000).toISOString(),
      }),
    );

    const removed = await cleanupSourceCache({
      storage,
      cacheDir: root,
      expiryDays: 7,
    });

    expect(removed).toBe(2);
    expect(files.has(`${root}/2606.13359/source`)).toBe(false);
    expect(files.has(`${root}/2606.13359/source.meta.json`)).toBe(false);
    expect(dirs.has(`${root}/2606.13359`)).toBe(false);
  });
});

describe("PaperContentFetcher.fetchListingAbstract", () => {
  const absPage = (text: string) =>
    `<html><body><blockquote class="abstract"><span>Abstract:</span> ${text}</blockquote></body></html>`;

  function build(fetchPaperAbsPage: (...args: any[]) => Promise<string>, cache = makeCache()) {
    const fetcher = {
      fetchPaperHtml: vi.fn(async () => { throw new Error("unexpected html request"); }),
      fetchSource: vi.fn(async () => { throw new Error("unexpected source request"); }),
      fetchPaperAbsPage: vi.fn(fetchPaperAbsPage),
    };
    const paperFetcher = new PaperContentFetcher(
      fetcher as any, cache as any, new Logger("error"), markupParser,
    );
    return { fetcher, cache, paperFetcher };
  }

  it("uses the cached abs page without touching the network", async () => {
    const cache = makeCache();
    await cache.set("abs/2609.00001", "abs", absPage("Cached text."));
    cache.set.mockClear();
    const { fetcher, paperFetcher } = build(async () => "unused", cache);

    const result = await paperFetcher.fetchListingAbstract("2609.00001");

    expect(result).toEqual({ abstract: "Cached text.", fromNetwork: false });
    expect(fetcher.fetchPaperAbsPage).not.toHaveBeenCalled();
    expect(cache.set).not.toHaveBeenCalled();
  });

  it("fetches and caches the abs page on a cache miss, passing the signal", async () => {
    const { fetcher, cache, paperFetcher } = build(async () => absPage("Fresh text."));
    const signal = new AbortController().signal;

    const result = await paperFetcher.fetchListingAbstract("2609.00002", signal);

    expect(result).toEqual({ abstract: "Fresh text.", fromNetwork: true });
    expect(fetcher.fetchPaperAbsPage).toHaveBeenCalledWith("2609.00002", signal);
    expect(cache.set).toHaveBeenCalledWith("abs/2609.00002", "abs", absPage("Fresh text."));
  });

  it("returns an empty abstract (not an error) for a page without an abstract block", async () => {
    const { paperFetcher } = build(async () => "<html><body><p>withdrawn</p></body></html>");

    await expect(paperFetcher.fetchListingAbstract("2609.00003"))
      .resolves.toEqual({ abstract: "", fromNetwork: true });
  });

  it("does not fail the call when writing the cache fails", async () => {
    const cache = makeCache();
    cache.set.mockRejectedValue(new Error("disk full"));
    const { paperFetcher } = build(async () => absPage("Still returned."), cache);

    await expect(paperFetcher.fetchListingAbstract("2609.00004"))
      .resolves.toEqual({ abstract: "Still returned.", fromNetwork: true });
  });

  it("propagates fetch failures and cancellation to the caller", async () => {
    const { paperFetcher } = build(async () => { throw new Error("abs down"); });
    await expect(paperFetcher.fetchListingAbstract("2609.00005")).rejects.toThrow("abs down");

    const controller = new AbortController();
    controller.abort("stop");
    const second = build(async () => absPage("x"));
    await expect(second.paperFetcher.fetchListingAbstract("2609.00006", controller.signal))
      .rejects.toMatchObject({ name: "RunCancelledError" });
    expect(second.fetcher.fetchPaperAbsPage).not.toHaveBeenCalled();
  });
});
