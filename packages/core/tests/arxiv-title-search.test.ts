import { describe, expect, it, vi } from "vitest";
import { matchBestTitle, searchArxivTitle } from "../src/library/arxiv-title-search";
import type { HttpClient, HttpResponse } from "../src/core/adapters";

function atom(entries: Array<{ id: string; title: string }>): string {
  return `<feed xmlns="http://www.w3.org/2005/Atom">${entries.map((entry) =>
    `<entry><id>http://arxiv.org/abs/${entry.id}</id><title>${entry.title}</title></entry>`,
  ).join("")}</feed>`;
}

describe("arxiv title search matching", () => {
  it("accepts a unique high-overlap match", () => {
    const xml = atom([{
      id: "2403.19236v2",
      title: "Measuring the baryon fraction using galaxy clustering",
    }]);
    expect(matchBestTitle("Measuring the baryon fraction using galaxy clustering", xml))
      .toEqual({
        arxivId: "2403.19236",
        matchedTitle: "Measuring the baryon fraction using galaxy clustering",
      });
  });

  it("normalizes case and punctuation before comparing", () => {
    const xml = atom([{
      id: "2309.11425",
      title: "Astronomy with your camera: bright star images",
    }]);
    expect(matchBestTitle("Astronomy with your camera: Bright Star Images!", xml).arxivId)
      .toBe("2309.11425");
  });

  it("rejects low-overlap matches", () => {
    const xml = atom([{
      id: "1234.5678",
      title: "Something completely different about frogs",
    }]);
    expect(matchBestTitle("Measuring baryon fractions in galaxy clusters", xml).arxivId)
      .toBeNull();
  });

  it("rejects ambiguous matches with two similar candidates", () => {
    const xml = atom([
      { id: "2403.19236", title: "Measuring the baryon fraction using galaxy clustering" },
      { id: "2403.19237", title: "Measuring the baryon fraction using galaxy clusters" },
    ]);
    expect(matchBestTitle("Measuring the baryon fraction using galaxy clustering", xml).arxivId)
      .toBeNull();
  });

  it("returns nothing for empty or too-short queries and empty feeds", () => {
    expect(matchBestTitle("", atom([{ id: "1.1", title: "x" }])).arxivId).toBeNull();
    expect(matchBestTitle("abc", atom([{ id: "1.1", title: "abc" }])).arxivId).toBeNull();
    expect(matchBestTitle("A real title here", "").arxivId).toBeNull();
  });
});

describe("arxiv title search over HTTP", () => {
  it("issues a title query and returns the matched canonical ID", async () => {
    const http: HttpClient = {
      request: vi.fn(async (request): Promise<HttpResponse> => {
        expect(request.url).toContain("search_query=ti:");
        expect(request.url).toContain("max_results=5");
        return {
          status: 200,
          headers: { "content-type": "application/atom+xml" },
          bodyText: atom([{ id: "2403.19236v2", title: "Measuring the baryon fraction using galaxy clustering" }]),
        };
      }),
    };
    const result = await searchArxivTitle(
      http,
      "Measuring the baryon fraction using galaxy clustering",
    );
    expect(result.arxivId).toBe("2403.19236");
  });

  it("propagates HTTP failures without inventing identity", async () => {
    const http: HttpClient = {
      request: vi.fn(async (): Promise<HttpResponse> => ({ status: 503, headers: {}, bodyText: "" })),
    };
    await expect(searchArxivTitle(http, "Some paper title")).rejects.toThrow(/HTTP 503/);
  });
});
