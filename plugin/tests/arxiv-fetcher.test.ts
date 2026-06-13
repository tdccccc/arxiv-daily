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
});
