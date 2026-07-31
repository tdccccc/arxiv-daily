import { markupParser } from "./markup-parser";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ArxivFetcher,
  ArxivHttpError,
  ArxivRequestCoordinator,
  ArxivRetryDeferredError,
  formatArxivHttpError,
  isArxivHttpError,
  isArxivRetryDeferredError,
  isRetryableArxivError,
  parseRetryAfterMs,
  resetArxivRequestCoordinatorForTests,
} from "../src/pipeline/arxiv-fetcher";
import {
  HttpTransportError,
  type HttpClient,
  type HttpRequest,
  type StorageAdapter,
} from "../src/core/adapters";
import { Logger } from "../src/services/logger";
import { AtomMetadataCache } from "../src/pipeline/atom-metadata-cache";

function makeFetcher(
  http: HttpClient,
  metadataCache?: AtomMetadataCache,
  timeouts?: { textTimeoutMs?: number; binaryTimeoutMs?: number },
): ArxivFetcher {
  return new ArxivFetcher({
    categories: ["astro-ph"],
    markupParser,
    http,
    logger: new Logger("error"),
    requestDelayMs: 0,
    metadataCache,
    ...timeouts,
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
        timeoutMs: 60_000,
        signal: undefined,
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

  it.each(["without cache", "with cache"])(
    "single-flights overlapping [A,B]/[B,C] sets %s across fetchers",
    async (mode) => {
      vi.useFakeTimers();
      const ids = ["2606.10001", "2606.10002", "2606.10003"];
      let releaseFirst!: () => void;
      const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve; });
      const requests: string[][] = [];
      const http: HttpClient = { request: vi.fn(async (req) => {
        const requested = new URL(req.url).searchParams.get("id_list")?.split(",") ?? [];
        requests.push(requested);
        if (requests.length === 1) await firstGate;
        return { status: 200, headers: {}, bodyText: atomFor(requested) };
      }) };
      const storage = memoryStorage();
      const cache = () => mode === "with cache"
        ? new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage })
        : undefined;
      const first = makeFetcher(http, cache()).fetchMetadataByIds(ids.slice(0, 2));
      await vi.waitFor(() => expect(http.request).toHaveBeenCalledOnce());
      const second = makeFetcher(http, cache()).fetchMetadataByIds(ids.slice(1));
      releaseFirst();
      await vi.advanceTimersByTimeAsync(3_000);

      await expect(first).resolves.toHaveProperty("size", 2);
      await expect(second).resolves.toHaveProperty("size", 2);
      expect(requests).toHaveLength(2);
      expect(requests.flat().filter((id) => id === ids[1])).toHaveLength(1);
      expect(new Set(requests.flat())).toEqual(new Set(ids));
    },
  );

  it("shares flights across distinct cache roots and best-effort persists joined positives", async () => {
    const storage = memoryStorage();
    const firstCache = new AtomMetadataCache({ rootDir: "first", expiryDays: 7, storage });
    const joinedCache = new AtomMetadataCache({ rootDir: "joined", expiryDays: 7, storage });
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const http: HttpClient = { request: vi.fn(async () => {
      await gate;
      return { status: 200, headers: {}, bodyText: atomFor(["2606.10004"]) };
    }) };

    const owner = makeFetcher(http, firstCache).fetchMetadataByIds(["2606.10004"]);
    await vi.waitFor(() => expect(http.request).toHaveBeenCalledOnce());
    const joined = makeFetcher(http, joinedCache).fetchMetadataByIds(["2606.10004v2"]);
    release();

    await expect(owner).resolves.toHaveProperty("size", 1);
    await expect(joined).resolves.toHaveProperty("size", 1);
    expect(http.request).toHaveBeenCalledOnce();
    await expect(firstCache.get("2606.10004")).resolves.toMatchObject({ id: "2606.10004" });
    await expect(joinedCache.get("2606.10004")).resolves.toMatchObject({ id: "2606.10004" });
  });

  it("settles omitted metadata as retriable absence without negative caching or poisoned flights", async () => {
    vi.useFakeTimers();
    const http: HttpClient = { request: vi.fn(async () => ({ status: 200, headers: {}, bodyText: atomFor([]) })) };
    const first = makeFetcher(http).fetchMetadataByIds(["2606.10005"]);
    const joined = makeFetcher(http).fetchMetadataByIds(["2606.10005v3"]);
    await expect(first).resolves.toEqual(new Map());
    await expect(joined).resolves.toEqual(new Map());
    expect(http.request).toHaveBeenCalledOnce();

    const retry = makeFetcher(http).fetchMetadataByIds(["2606.10005"]);
    await vi.advanceTimersByTimeAsync(3_000);
    await expect(retry).resolves.toEqual(new Map());
    expect(http.request).toHaveBeenCalledTimes(2);
  });

  it("cleans failed metadata flights so later callers can retry", async () => {
    vi.useFakeTimers();
    const http: HttpClient = { request: vi.fn()
      .mockResolvedValueOnce({ status: 403, headers: {}, bodyText: "forbidden" })
      .mockResolvedValueOnce({ status: 200, headers: {}, bodyText: atomFor(["2606.10006"]) }) };
    const first = makeFetcher(http).fetchMetadataByIds(["2606.10006"]);
    const joined = makeFetcher(http).fetchMetadataByIds(["2606.10006"]);
    await expect(first).rejects.toMatchObject({ status: 403 });
    await expect(joined).rejects.toMatchObject({ status: 403 });
    expect(http.request).toHaveBeenCalledOnce();

    const retry = makeFetcher(http).fetchMetadataByIds(["2606.10006"]);
    await vi.advanceTimersByTimeAsync(3_000);
    await expect(retry).resolves.toHaveProperty("size", 1);
    expect(http.request).toHaveBeenCalledTimes(2);
  });

  it("defines owner cancellation to fail joined waiters and permits a clean retry", async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const http: HttpClient = { request: vi.fn((req) => {
      if (http.request.mock.calls.length === 1) return new Promise(() => {});
      return Promise.resolve({ status: 200, headers: {}, bodyText: atomFor(["2606.10007"]) });
    }) as any };
    const owner = makeFetcher(http, undefined, { textTimeoutMs: 60_000 })
      .fetchMetadataByIds(["2606.10007"], controller.signal);
    await vi.waitFor(() => expect(http.request).toHaveBeenCalledOnce());
    const joined = makeFetcher(http).fetchMetadataByIds(["2606.10007"]);
    controller.abort("metadata owner cancelled");

    await expect(owner).rejects.toThrow("metadata owner cancelled");
    await expect(joined).rejects.toThrow("metadata owner cancelled");
    const retry = makeFetcher(http).fetchMetadataByIds(["2606.10007"]);
    await vi.advanceTimersByTimeAsync(3_000);
    await expect(retry).resolves.toHaveProperty("size", 1);
    expect(http.request).toHaveBeenCalledTimes(2);
  });

  it("rejects and removes owned flights immediately when cancellation occurs in cache wait", async () => {
    const controller = new AbortController();
    const cache = {
      get: vi.fn()
        .mockResolvedValueOnce(null)
        .mockImplementationOnce(() => new Promise(() => {})),
      set: vi.fn(),
    } as unknown as AtomMetadataCache;
    const http: HttpClient = { request: vi.fn(async () => ({
      status: 200, headers: {}, bodyText: atomFor(["2606.10008"]),
    })) };
    const owner = makeFetcher(http, cache).fetchMetadataByIds(
      ["2606.10008"], controller.signal,
    );
    await vi.waitFor(() => expect((cache.get as any)).toHaveBeenCalledTimes(2));
    const joined = makeFetcher(http).fetchMetadataByIds(["2606.10008"]);
    controller.abort("cancel cache owner");

    await expect(owner).rejects.toThrow("cancel cache owner");
    await expect(joined).rejects.toThrow("cancel cache owner");
    await expect(makeFetcher(http).fetchMetadataByIds(["2606.10008"]))
      .resolves.toHaveProperty("size", 1);
    expect(http.request).toHaveBeenCalledOnce();
  });

  it("preserves first-batch results and joined waiters when a 201-ID second batch fails", async () => {
    vi.useFakeTimers();
    const ids = Array.from({ length: 201 }, (_, index) => `2606.${String(index + 11000).padStart(5, "0")}`);
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve; });
    const ownerStorage = memoryStorage();
    const joinedStorage = memoryStorage();
    const ownerCache = new AtomMetadataCache({ rootDir: "owner", expiryDays: 7, storage: ownerStorage });
    const joinedCache = new AtomMetadataCache({ rootDir: "joined", expiryDays: 7, storage: joinedStorage });
    const http: HttpClient = { request: vi.fn(async (req) => {
      const requested = new URL(req.url).searchParams.get("id_list")?.split(",") ?? [];
      if (http.request.mock.calls.length === 1) {
        await firstGate;
        return { status: 200, headers: {}, bodyText: atomFor(requested) };
      }
      return { status: 403, headers: {}, bodyText: "forbidden" };
    }) as any };

    const owner = makeFetcher(http, ownerCache).fetchMetadataByIds(ids);
    const ownerFailure = expect(owner).rejects.toMatchObject({ status: 403 });
    await vi.waitFor(() => expect(http.request).toHaveBeenCalledOnce());
    const joined = makeFetcher(http, joinedCache).fetchMetadataByIds([ids[0]!]);
    releaseFirst();
    await expect(joined).resolves.toHaveProperty("size", 1);
    await vi.advanceTimersByTimeAsync(3_000);
    await ownerFailure;

    expect(http.request).toHaveBeenCalledTimes(2);
    await expect(ownerCache.get(ids[0]!)).resolves.toMatchObject({ id: ids[0] });
    await expect(ownerCache.get(ids[199]!)).resolves.toMatchObject({ id: ids[199] });
    await expect(ownerCache.get(ids[200]!)).resolves.toBeNull();
    await expect(joinedCache.get(ids[0]!)).resolves.toMatchObject({ id: ids[0] });
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
      expect.stringContaining("after 2000ms"),
    );
  });

  it("safely parses full Retry-After delta and receipt-time HTTP dates", () => {
    const received = Date.parse("2026-06-25T10:00:00.000Z");
    expect(parseRetryAfterMs({ "Retry-After": "0" }, received)).toBe(0);
    expect(parseRetryAfterMs({ "Retry-After": "7200" }, received)).toBe(7_200_000);
    expect(parseRetryAfterMs({ "Retry-After": "0.5" }, received)).toBeNull();
    expect(parseRetryAfterMs({ "Retry-After": "invalid" }, received)).toBeNull();
    expect(parseRetryAfterMs({ "Retry-After": "Thursday, 25-Jun-26 12:00:00 GMT" }, received))
      .toBeNull();
    expect(parseRetryAfterMs({ "Retry-After": "Thu Jun 25 12:00:00 2026" }, received))
      .toBeNull();
    expect(parseRetryAfterMs({ "Retry-After": "Thu, 25 Jun 2026 12:00:00 UTC" }, received))
      .toBeNull();
    expect(parseRetryAfterMs({ "Retry-After": "Thu, 25 Jun 2026 12:00:00 GMT extra" }, received))
      .toBeNull();
    expect(parseRetryAfterMs({ "Retry-After": "Thu, 31 Feb 2026 12:00:00 GMT" }, received))
      .toBeNull();
    expect(parseRetryAfterMs({ "Retry-After": "Fri, 25 Jun 2026 12:00:00 GMT" }, received))
      .toBeNull();
    expect(parseRetryAfterMs({ "Retry-After": "Thu, 25 Jun 2026 12:00:00 GMT" }, received))
      .toBe(7_200_000);
    expect(parseRetryAfterMs({ "Retry-After": "Thu, 25 Jun 2026 09:00:00 GMT" }, received))
      .toBe(0);
    expect(parseRetryAfterMs({ "Retry-After": "9".repeat(400) }, received))
      .toBeGreaterThan(7_200_000);
  });

  it("defers long cooldowns immediately, fails active calls fast, and proceeds after monotonic expiry", async () => {
    let monotonic = 100;
    let wall = Date.parse("2026-06-25T10:00:00.000Z");
    const sleep = vi.fn(async (ms: number) => { monotonic += ms; });
    const coordinator = new ArxivRequestCoordinator({
      monotonicNow: () => monotonic,
      wallNow: () => wall,
      sleep,
    });
    const http: HttpClient = {
      request: vi.fn()
        .mockResolvedValueOnce({ status: 429, headers: { "Retry-After": "7200" }, bodyText: "slow" })
        .mockResolvedValue({ status: 200, headers: {}, bodyText: "ok" }),
    };
    const fetcher = new ArxivFetcher({
      categories: ["astro-ph"], markupParser, http, logger: new Logger("error"),
      requestDelayMs: 0, requestCoordinator: coordinator,
    });

    const first = await fetcher.fetchRecent().catch((error) => error);
    expect(isArxivRetryDeferredError(first)).toBe(true);
    expect(first).toMatchObject({ remainingMs: 7_200_000 });
    expect(first.message).toContain("2026-06-25T12:00:00.000Z");
    expect(sleep).not.toHaveBeenCalled();
    await expect(fetcher.fetchRecent()).rejects.toBeInstanceOf(ArxivRetryDeferredError);
    expect(http.request).toHaveBeenCalledTimes(1);

    wall -= 24 * 60 * 60 * 1000;
    monotonic += 7_200_000;
    await expect(fetcher.fetchRecent()).resolves.toBe("ok");
    expect(http.request).toHaveBeenCalledTimes(2);
  });

  it("waits short cooldowns in cancellable chunks", async () => {
    let monotonic = 0;
    const sleeps: number[] = [];
    const coordinator = new ArxivRequestCoordinator({
      monotonicNow: () => monotonic,
      wallNow: () => Date.parse("2026-06-25T10:00:00.000Z"),
      sleep: async (ms, signal) => {
        if (signal?.aborted) throw new Error(String(signal.reason));
        sleeps.push(ms);
        monotonic += ms;
      },
    });
    await expect(coordinator.coordinate(
      async () => { throw new ArxivHttpError(503, "https://arxiv.org", { "Retry-After": "120" }); },
      0,
    )).rejects.toMatchObject({ status: 503 });

    await expect(coordinator.coordinate(async () => "ok", 0)).resolves.toBe("ok");
    expect(sleeps).toEqual([30_000, 30_000, 30_000, 30_000]);
  });

  it("lets a later longer cooldown extend queued work without shortening the deadline", async () => {
    let monotonic = 0;
    const starts: number[] = [];
    const coordinator = new ArxivRequestCoordinator({
      monotonicNow: () => monotonic,
      wallNow: () => Date.parse("2026-06-25T10:00:00.000Z") + monotonic,
      sleep: async (ms) => { monotonic += ms; },
    });
    const first = coordinator.coordinate(async () => {
      starts.push(monotonic);
      throw new ArxivHttpError(503, "https://arxiv.org", { "Retry-After": "10" });
    }, 0).catch(() => undefined);
    const second = coordinator.coordinate(async () => {
      starts.push(monotonic);
      throw new ArxivHttpError(503, "https://arxiv.org", { "Retry-After": "60" });
    }, 0).catch(() => undefined);
    const third = coordinator.coordinate(async () => {
      starts.push(monotonic);
      return "ok";
    }, 0);

    await Promise.all([first, second]);
    await expect(third).resolves.toBe("ok");
    expect(starts).toEqual([0, 10_000, 70_000]);
  });

  it("cancels a chunked cooldown wait and releases queued work after expiry", async () => {
    vi.useFakeTimers();
    let monotonic = 0;
    const coordinator = new ArxivRequestCoordinator({
      monotonicNow: () => monotonic,
      wallNow: () => Date.parse("2026-06-25T10:00:00.000Z") + monotonic,
      sleep: (ms, signal) => new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => { cleanup(); monotonic += ms; resolve(); }, ms);
        const onAbort = () => { clearTimeout(timer); cleanup(); reject(new Error(String(signal?.reason))); };
        const cleanup = () => signal?.removeEventListener("abort", onAbort);
        signal?.addEventListener("abort", onAbort, { once: true });
      }),
    });
    await expect(coordinator.coordinate(
      async () => { throw new ArxivHttpError(503, "https://arxiv.org", { "Retry-After": "20" }); },
      0,
    )).rejects.toMatchObject({ status: 503 });
    const controller = new AbortController();
    const cancelled = coordinator.coordinate(async () => "unexpected", 0, controller.signal);
    await vi.advanceTimersByTimeAsync(1_000);
    controller.abort("stop cooldown");
    await expect(cancelled).rejects.toThrow("stop cooldown");
    expect(vi.getTimerCount()).toBe(0);

    monotonic = 20_000;
    await expect(coordinator.coordinate(async () => "ok", 0)).resolves.toBe("ok");
  });

  it("uses monotonic spacing despite wall-clock rollback and leap", async () => {
    let monotonic = 0;
    let wall = Date.parse("2026-06-25T10:00:00.000Z");
    const sleeps: number[] = [];
    const coordinator = new ArxivRequestCoordinator({
      monotonicNow: () => monotonic,
      wallNow: () => wall,
      sleep: async (ms) => { sleeps.push(ms); monotonic += ms; },
    });
    const http: HttpClient = { request: vi.fn(async () => ({ status: 200, headers: {}, bodyText: "ok" })) };
    const fetcher = new ArxivFetcher({
      categories: ["astro-ph"], markupParser, http, logger: new Logger("error"),
      requestDelayMs: 0, requestCoordinator: coordinator,
    });

    await fetcher.fetchRecent();
    wall -= 365 * 24 * 60 * 60 * 1000;
    await fetcher.fetchRecent();
    wall += 730 * 24 * 60 * 60 * 1000;
    await fetcher.fetchRecent();

    expect(sleeps).toEqual([3_000, 3_000]);
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

  it("passes bounded text and binary deadlines to HTTP adapters", async () => {
    const requests: HttpRequest[] = [];
    const http: HttpClient = { request: vi.fn(async (req) => {
      requests.push(req);
      return req.responseType === "arrayBuffer"
        ? { status: 200, headers: {}, bodyText: "", bodyBuffer: new ArrayBuffer(1) }
        : { status: 200, headers: {}, bodyText: "ok" };
    }) };
    const fetcher = makeFetcher(http, undefined, {
      textTimeoutMs: Number.POSITIVE_INFINITY,
      binaryTimeoutMs: 999_999_999,
    });

    await fetcher.fetchRecent();
    await fetcher.fetchPdf("2606.12345");

    expect(requests[0].timeoutMs).toBe(60_000);
    expect(requests[1].timeoutMs).toBe(30 * 60 * 1000);
  });

  it("releases the queue when a nonconforming client never settles", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const http: HttpClient = { request: vi.fn((req) => {
      if (http.request.mock.calls.length === 1) return new Promise(() => {});
      return Promise.resolve({ status: 200, headers: {}, bodyText: req.url });
    }) as any };
    const fetcher = makeFetcher(http, undefined, { textTimeoutMs: 10 });

    const stalled = fetcher.fetchRecent("astro-ph");
    const stalledAssertion = expect(stalled).rejects.toMatchObject({
      kind: "timeout",
      retryableAttempt: false,
    });
    const next = fetcher.fetchRecent("cs.CL");
    await vi.advanceTimersByTimeAsync(3_000);
    await expect(next).resolves.toContain("cs.CL");
    await stalledAssertion;
    expect(http.request).toHaveBeenCalledTimes(2);
  });

  it("releases an in-flight cancelled attempt without retrying it", async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const http: HttpClient = { request: vi.fn((req) => {
      if (http.request.mock.calls.length === 1) return new Promise(() => {});
      return Promise.resolve({ status: 200, headers: {}, bodyText: req.url });
    }) as any };
    const fetcher = makeFetcher(http, undefined, { textTimeoutMs: 60_000 });

    const cancelled = fetcher.fetchRecent("astro-ph", controller.signal);
    await vi.advanceTimersByTimeAsync(1);
    controller.abort("stop in flight");
    const next = fetcher.fetchRecent("cs.CL");
    await expect(cancelled).rejects.toThrow("stop in flight");
    await vi.advanceTimersByTimeAsync(2_999);

    await expect(next).resolves.toContain("cs.CL");
    expect(http.request).toHaveBeenCalledTimes(2);
  });

  it("makes one attempt for plain local errors and empty binary responses", async () => {
    const localHttp: HttpClient = {
      request: vi.fn(async () => { throw new Error("local parse setup failed"); }),
    };
    await expect(makeFetcher(localHttp).fetchRecent()).rejects.toThrow("local parse setup failed");
    expect(localHttp.request).toHaveBeenCalledOnce();

    const emptyHttp: HttpClient = {
      request: vi.fn(async () => ({ status: 200, headers: {}, bodyText: "" })),
    };
    await expect(makeFetcher(emptyHttp).fetchPdf("2606.12345"))
      .rejects.toThrow("empty binary response");
    expect(emptyHttp.request).toHaveBeenCalledOnce();
  });

  it("retries typed network and timeout errors exactly three times", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    for (const kind of ["network", "timeout"] as const) {
      await resetArxivRequestCoordinatorForTests();
      const http: HttpClient = {
        request: vi.fn(async () => {
          throw new HttpTransportError(kind, kind, { retryableAttempt: true });
        }),
      };
      const result = makeFetcher(http).fetchRecent();
      const assertion = expect(result).rejects.toMatchObject({ kind });
      await vi.runAllTimersAsync();
      await assertion;
      expect(http.request).toHaveBeenCalledTimes(3);
    }
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

  it("retries only typed transport, 408, 429, and 5xx failures", () => {
    expect(isRetryableArxivError(new HttpTransportError(
      "network", "socket closed", { retryableAttempt: true },
    ))).toBe(true);
    expect(isRetryableArxivError(new HttpTransportError(
      "timeout", "deadline", { retryableAttempt: true },
    ))).toBe(true);
    expect(isRetryableArxivError({
      name: "HttpTransportError",
      message: "foreign timeout",
      kind: "timeout",
      retryableAttempt: true,
    })).toBe(true);
    expect(isRetryableArxivError(new HttpTransportError(
      "timeout", "orphan may still run", { retryableAttempt: false },
    ))).toBe(false);
    expect(isRetryableArxivError(new Error("local failure"))).toBe(false);
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
