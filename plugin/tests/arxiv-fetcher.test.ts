import { describe, expect, it, vi } from "vitest";
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
  it("uses the injected HTTP client for arXiv requests", async () => {
    const requests: HttpRequest[] = [];
    const http: HttpClient = {
      request: vi.fn(async (req) => {
        requests.push(req);
        return {
          status: 200,
          headers: {},
          bodyText: "@misc{Key2026,\n}\n",
        };
      }),
    };

    const bibtex = await makeFetcher(http).fetchBibtex("2606.12345");

    expect(bibtex).toContain("Key2026");
    expect(requests).toEqual([
      {
        url: "https://arxiv.org/bibtex/2606.12345",
        method: "GET",
        headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
      },
    ]);
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
});
