import { describe, expect, it, vi } from "vitest";
import {
  cleanupSourceCache,
  PaperContentFetcher,
} from "../src/pipeline/paper-content";
import { Logger } from "../src/services/logger";
import type { StorageAdapter } from "../src/core/adapters";

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
