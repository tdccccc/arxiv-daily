import { afterEach, describe, expect, it, vi } from "vitest";
import { ArxivFetcher } from "../src/pipeline/arxiv-fetcher";
import type { HttpClient, HttpRequest } from "../src/core/adapters";
import { Logger } from "../src/services/logger";

function makeFetcher(http: HttpClient): ArxivFetcher {
  return new ArxivFetcher({
    categories: ["astro-ph"],
    http,
    logger: new Logger("error"),
    requestDelayMs: 0,
  });
}

describe("ArxivFetcher", () => {
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

  it("shares request throttling across fetcher instances", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-25T10:00:00.000Z"));
    const http: HttpClient = {
      request: vi.fn(async () => ({
        status: 200,
        headers: {},
        bodyText: "<html>recent</html>",
      })),
    };
    const logger = { debug: vi.fn(), warn: vi.fn() };
    const opts = {
      categories: ["astro-ph"],
      http,
      logger: logger as any,
      requestDelayMs: 1000,
    };

    await new ArxivFetcher(opts).fetchRecent("astro-ph");
    const second = new ArxivFetcher(opts).fetchRecent("cs.CL");
    await vi.advanceTimersByTimeAsync(999);
    expect(http.request).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);

    await expect(second).resolves.toContain("recent");
    expect(http.request).toHaveBeenCalledTimes(2);
  });
});
