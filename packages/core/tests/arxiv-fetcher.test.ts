import { markupParser } from "./markup-parser";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ArxivFetcher,
  ArxivHttpError,
  formatArxivHttpError,
  isArxivHttpError,
  isRetryableArxivError,
  resetArxivRequestCoordinatorForTests,
} from "../src/pipeline/arxiv-fetcher";
import type { HttpClient, HttpRequest, StorageAdapter } from "../src/core/adapters";
import { Logger } from "../src/services/logger";
import { AtomMetadataCache } from "../src/pipeline/atom-metadata-cache";

function makeFetcher(http: HttpClient, metadataCache?: AtomMetadataCache): ArxivFetcher {
  return new ArxivFetcher({
    categories: ["astro-ph"],
    markupParser,
    http,
    logger: new Logger("error"),
    requestDelayMs: 0,
    metadataCache,
  });
}

function atomFor(ids: string[]): string {
  return `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${ids.map((id) => `<entry><id>http://arxiv.org/abs/${id}v1</id><title>Paper ${id}</title><author><name>A. Author</name></author><summary>Abstract ${id}.</summary><published>2026-06-13T00:00:00Z</published><updated>2026-06-14T00:00:00Z</updated><arxiv:primary_category term="astro-ph"/><category term="astro-ph"/></entry>`).join("")}</feed>`;
}

function memoryStorage(): StorageAdapter {
  const files = new Map<string, string>();
  const dirs = new Set<string>();
  return {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    readText: async (path) => { const value = files.get(path); if (value == null) throw new Error("missing"); return value; },
    writeText: async (path, content) => { files.set(path, content); },
    exists: async (path) => files.has(path) || dirs.has(path),
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { files.delete(path); dirs.delete(path); },
    rename: async (from, to) => { const value = files.get(from); if (value != null) files.set(to, value); files.delete(from); },
    list: async (dir) => [...files.keys()].filter((path) => path.startsWith(`${dir}/`)).map((path) => ({ path, type: "file" as const })),
  };
}

describe("ArxivFetcher", () => {
  beforeEach(async () => {
    await resetArxivRequestCoordinatorForTests();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("uses the injected HTTP client for arXiv requests", async () => {
    const requests: HttpRequest[] = [];
    const http: HttpClient = {
      request: vi.fn(async (req) => {
        requests.push(req);
        return {
          status: 200,
          headers: {},
          bodyText: "<html>recent</html>",
        };
      }),
    };

    const html = await makeFetcher(http).fetchRecent("astro-ph");

    expect(html).toContain("recent");
    expect(requests).toEqual([
      {
        url: "https://arxiv.org/list/astro-ph/recent?skip=0&show=2000",
        method: "GET",
        headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
      },
    ]);
  });

  it("passes abort signals to recent and metadata HTTP requests", async () => {
    const requests: HttpRequest[] = [];
    const controller = new AbortController();
    const http: HttpClient = {
      request: vi.fn(async (req) => {
        requests.push(req);
        return {
          status: 200,
          headers: {},
          bodyText: req.url.includes("export.arxiv.org")
            ? `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>`
            : "<html>recent</html>",
        };
      }),
    };
    const fetcher = makeFetcher(http);

    await fetcher.fetchRecent("astro-ph", controller.signal);
    await fetcher.fetchMetadataByIds(["2606.12345"], controller.signal);

    expect(requests).toHaveLength(2);
    expect(requests[0].signal).toBe(controller.signal);
    expect(requests[1].signal).toBe(controller.signal);
  });

  it("serves full and partial metadata cache hits while persisting positive misses", async () => {
    const storage = memoryStorage();
    const cache = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage });
    const cachedPaper = {
      id: "2606.11111", title: "Cached", authors: "A. Author", abstract: "Cached abstract.",
      published: "2026-06-01T00:00:00Z", updated: "2026-06-02T00:00:00Z",
      primaryCategory: "astro-ph", categories: ["astro-ph"],
    };
    await cache.set(cachedPaper.id, cachedPaper);
    const http: HttpClient = { request: vi.fn(async () => ({ status: 200, headers: {}, bodyText: atomFor(["2606.22222"]) })) };
    const fetcher = makeFetcher(http, cache);

    const full = await fetcher.fetchMetadataByIds(["2606.11111v2", "2606.11111"]);
    expect([...full.keys()]).toEqual(["2606.11111"]);
    expect(http.request).not.toHaveBeenCalled();

    const partial = await fetcher.fetchMetadataByIds(["2606.11111", "2606.22222", "bad"]);
    expect([...partial.keys()]).toEqual(["2606.11111", "2606.22222"]);
    expect(http.request).toHaveBeenCalledOnce();
    expect((http.request as any).mock.calls[0][0].url).toContain("id_list=2606.22222");
    expect(await cache.get("2606.22222")).toMatchObject({ id: "2606.22222" });
  });

  it("deduplicates canonical IDs and does not negatively cache omitted entries", async () => {
    const storage = memoryStorage();
    const cache = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage });
    const http: HttpClient = { request: vi.fn(async () => ({ status: 200, headers: {}, bodyText: atomFor([]) })) };
    const fetcher = makeFetcher(http, cache);

    expect(await fetcher.fetchMetadataByIds(["2606.33333", "2606.33333v2"])).toEqual(new Map());
    expect(await fetcher.fetchMetadataByIds(["2606.33333"])).toEqual(new Map());
    expect(http.request).toHaveBeenCalledTimes(2);
    expect((http.request as any).mock.calls[0][0].url).toContain("id_list=2606.33333&max_results=1");
  });

  it("reuses metadata across fetcher instances sharing persistent storage", async () => {
    const storage = memoryStorage();
    const http: HttpClient = { request: vi.fn(async () => ({ status: 200, headers: {}, bodyText: atomFor(["2606.44444"]) })) };
    const first = makeFetcher(http, new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage }));
    await first.fetchMetadataByIds(["2606.44444"]);
    const second = makeFetcher(http, new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage }));
    expect(await second.fetchMetadataByIds(["2606.44444"])).toHaveProperty("size", 1);
    expect(http.request).toHaveBeenCalledOnce();
  });

  it("caps canonical metadata miss batches at 200", async () => {
    vi.useFakeTimers();
    const ids = Array.from({ length: 201 }, (_, index) => `2606.${String(index).padStart(5, "0")}`);
    const http: HttpClient = { request: vi.fn(async (req) => {
      const requested = new URL(req.url).searchParams.get("id_list")?.split(",") ?? [];
      return { status: 200, headers: {}, bodyText: atomFor(requested) };
    }) };
    const result = makeFetcher(http).fetchMetadataByIds(ids);
    await vi.advanceTimersByTimeAsync(3000);
    await expect(result).resolves.toHaveProperty("size", 201);
    const urls = (http.request as any).mock.calls.map((call: any[]) => new URL(call[0].url));
    expect(urls.map((url: URL) => Number(url.searchParams.get("max_results")))).toEqual([200, 1]);
  });

  it("returns a not-found result for missing HTML papers", async () => {
    const http: HttpClient = {
      request: vi.fn(async () => ({
        status: 404,
        headers: {},
        bodyText: "not found",
      })),
    };

    const result = await makeFetcher(http).fetchPaperHtml("2606.12345");

    expect(result).toEqual({ ok: false, status: 404 });
    expect(http.request).toHaveBeenCalledTimes(1);
  });

  it("fetches submitted-date fallback papers from the export API", async () => {
    const requests: HttpRequest[] = [];
    const http: HttpClient = {
      request: vi.fn(async (req) => {
        requests.push(req);
        return {
          status: 200,
          headers: {},
          bodyText: `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2606.12345v1</id><title>Fallback paper</title><author><name>A. Author</name></author><summary>Abstract.</summary><published>2026-06-13T00:00:00Z</published><updated>2026-06-13T00:00:00Z</updated><category term="astro-ph"/></entry></feed>`,
        };
      }),
    };

    const papers = await makeFetcher(http).fetchBySubmittedDate(
      "astro-ph",
      "2026-06-13",
    );

    expect(papers).toHaveLength(1);
    expect(papers[0]).toMatchObject({
      id: "2606.12345",
      title: "Fallback paper",
      abstract: "Abstract.",
    });
    expect(requests[0].url).toContain("https://export.arxiv.org/api/query?");
    expect(decodeURIComponent(requests[0].url)).toContain(
      "cat:astro-ph AND submittedDate:[202606130000 TO 202606132359]",
    );
  });

  it("fetches PDF bytes with an arrayBuffer response", async () => {
    const requests: HttpRequest[] = [];
    const bodyBuffer = new Uint8Array([1, 2, 3]).buffer;
    const http: HttpClient = {
      request: vi.fn(async (req) => {
        requests.push(req);
        return {
          status: 200,
          headers: {},
          bodyText: "",
          bodyBuffer,
        };
      }),
    };

    const pdf = await makeFetcher(http).fetchPdf("2606.12345");

    expect(Array.from(new Uint8Array(pdf))).toEqual([1, 2, 3]);
    expect(requests[0]).toMatchObject({
      url: "https://arxiv.org/pdf/2606.12345",
      responseType: "arrayBuffer",
    });
  });

  it("uses Retry-After seconds for 429 backoff before retrying", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const logger = { debug: vi.fn(), warn: vi.fn() };
    const http: HttpClient = {
      request: vi
        .fn()
        .mockResolvedValueOnce({
          status: 429,
          headers: { "retry-after": "5" },
          bodyText: "too many requests",
        })
        .mockResolvedValueOnce({
          status: 200,
          headers: {},
          bodyText: "<html>recent</html>",
        }),
    };
    const fetcher = new ArxivFetcher({
      categories: ["astro-ph"],
      markupParser,
      http,
      logger: logger as any,
      requestDelayMs: 0,
    });

    const result = fetcher.fetchRecent("astro-ph");
    await vi.advanceTimersByTimeAsync(4999);
    expect(http.request).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);

    await expect(result).resolves.toContain("recent");
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("after 5000ms"),
    );
  });

  it("caps oversized numeric Retry-After for retries and shared cooldown", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-25T10:00:00.000Z"));
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const starts: number[] = [];
    const logger = { debug: vi.fn(), warn: vi.fn() };
    const http: HttpClient = {
      request: vi.fn(async () => {
        starts.push(Date.now());
        if (starts.length === 1) {
          return {
            status: 429,
            headers: { "Retry-After": "9".repeat(400) },
            bodyText: "too many requests",
          };
        }
        return { status: 200, headers: {}, bodyText: "ok" };
      }),
    };
    const opts = {
      categories: ["astro-ph"],
      markupParser,
      http,
      logger: logger as any,
      requestDelayMs: 0,
    };

    const first = new ArxivFetcher(opts).fetchRecent("astro-ph");
    await vi.advanceTimersByTimeAsync(1);
    const subsequent = new ArxivFetcher(opts).fetchRecent("cs.CL");
    await vi.advanceTimersByTimeAsync(30 * 60 * 1000 - 2);
    expect(http.request).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);

    await expect(subsequent).resolves.toBe("ok");
    expect(starts[1] - starts[0]).toBe(30 * 60 * 1000);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("after 1800000ms"),
    );
    await vi.runAllTimersAsync();
    await first;
  });

  it("enforces the three-second runtime floor across fetcher instances", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-25T10:00:00.000Z"));
    const starts: number[] = [];
    const http: HttpClient = {
      request: vi.fn(async () => {
        starts.push(Date.now());
        return {
          status: 200,
          headers: {},
          bodyText: "<html>recent</html>",
        };
      }),
    };
    const logger = { debug: vi.fn(), warn: vi.fn() };
    const opts = {
      categories: ["astro-ph"],
      markupParser,
      http,
      logger: logger as any,
      requestDelayMs: 1000,
    };

    await new ArxivFetcher(opts).fetchRecent("astro-ph");
    const second = new ArxivFetcher(opts).fetchRecent("cs.CL");
    await vi.advanceTimersByTimeAsync(2999);
    expect(http.request).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);

    await expect(second).resolves.toContain("recent");
    expect(starts[1] - starts[0]).toBe(3000);
  });

  it("serializes the HTTP operation, not only attempt starts", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-25T10:00:00.000Z"));
    let finishFirst!: () => void;
    const firstFinished = new Promise<void>((resolve) => {
      finishFirst = resolve;
    });
    let active = 0;
    let maxActive = 0;
    const http: HttpClient = {
      request: vi.fn(async () => {
        active += 1;
        maxActive = Math.max(maxActive, active);
        if (active === 1) await firstFinished;
        active -= 1;
        return { status: 200, headers: {}, bodyText: "ok" };
      }),
    };
    const fetcher = makeFetcher(http);

    const first = fetcher.fetchRecent("astro-ph");
    const second = fetcher.fetchRecent("cs.CL");
    await vi.advanceTimersByTimeAsync(10_000);
    expect(http.request).toHaveBeenCalledTimes(1);
    finishFirst();
    await first;
    await vi.advanceTimersByTimeAsync(1);
    await second;

    expect(maxActive).toBe(1);
  });

  it("shares 503 Retry-After cooldown and never shortens it", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-25T10:00:00.000Z"));
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const starts: number[] = [];
    const http: HttpClient = {
      request: vi.fn(async () => {
        starts.push(Date.now());
        if (starts.length === 1) {
          return { status: 503, headers: { "Retry-After": "10" }, bodyText: "down" };
        }
        return { status: 200, headers: {}, bodyText: "ok" };
      }),
    };
    const first = makeFetcher(http).fetchRecent("astro-ph");
    await vi.advanceTimersByTimeAsync(1);
    const second = makeFetcher(http).fetchRecent("cs.CL");
    await vi.advanceTimersByTimeAsync(9998);
    expect(http.request).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await second;
    await vi.runAllTimersAsync();
    await first;

    expect(starts[1] - starts[0]).toBeGreaterThanOrEqual(10_000);
  });

  it("retries only network, 408, 429, and 5xx failures", () => {
    const network = new Error("socket closed");
    expect(isRetryableArxivError(network)).toBe(true);
    for (const status of [408, 429, 500, 503, 599]) {
      expect(isRetryableArxivError(new ArxivHttpError(status, "https://arxiv.org"))).toBe(true);
    }
    for (const status of [400, 401, 404, 409, 600]) {
      expect(isRetryableArxivError(new ArxivHttpError(status, "https://arxiv.org"))).toBe(false);
    }
  });

  it("does not retry a permanent HTTP response", async () => {
    const http: HttpClient = {
      request: vi.fn(async () => ({ status: 403, headers: {}, bodyText: "forbidden" })),
    };

    await expect(makeFetcher(http).fetchRecent()).rejects.toMatchObject({ status: 403 });
    expect(http.request).toHaveBeenCalledTimes(1);
  });

  it("provides structural HTTP error detection and actionable formatting", () => {
    const foreignError = {
      name: "ArxivHttpError",
      message: "HTTP 429",
      status: 429,
      url: "https://arxiv.org",
      headers: { "retry-after": "10" },
    };
    expect(isArxivHttpError(foreignError)).toBe(true);
    expect(formatArxivHttpError(foreignError)).toContain("rate-limiting");
    expect(formatArxivHttpError(new ArxivHttpError(503, "https://arxiv.org"))).toContain(
      "temporarily unavailable",
    );
  });

  it("cancels while queued without starting another HTTP attempt", async () => {
    vi.useFakeTimers();
    let finishFirst!: () => void;
    const firstFinished = new Promise<void>((resolve) => {
      finishFirst = resolve;
    });
    const http: HttpClient = {
      request: vi.fn(async () => {
        if (http.request.mock.calls.length === 1) await firstFinished;
        return { status: 200, headers: {}, bodyText: "ok" };
      }) as any,
    };
    const fetcher = makeFetcher(http);
    const first = fetcher.fetchRecent("astro-ph");
    const controller = new AbortController();
    const queued = fetcher.fetchRecent("cs.CL", controller.signal);
    controller.abort("stop waiting");

    await expect(queued).rejects.toThrow("stop waiting");
    finishFirst();
    await first;
    await vi.runAllTimersAsync();
    expect(http.request).toHaveBeenCalledTimes(1);
  });
});
