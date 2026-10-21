import { describe, expect, it, vi } from "vitest";
import { matchBestTitle, searchArxivTitle } from "../src/library/arxiv-title-search";
import type { HttpClient, HttpResponse } from "../src/core/adapters";
import {
  ArxivRequestCoordinator,
  isArxivHttpError,
  isArxivRetryDeferredError,
} from "../src/pipeline/arxiv-fetcher";
import { isCancellationError } from "../src/services/cancellation";

/** A coordinator on a fake clock: waiting advances the clock instead of taking time. */
function fakeClock() {
  const state = { monotonic: 0 };
  const sleeps: number[] = [];
  const coordinator = new ArxivRequestCoordinator({
    monotonicNow: () => state.monotonic,
    wallNow: () => Date.parse("2026-10-21T10:00:00.000Z") + state.monotonic,
    sleep: async (ms) => { sleeps.push(ms); state.monotonic += ms; },
  });
  return { state, sleeps, coordinator };
}

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

  it("reads Atom titles with attributes without a nested regular expression", () => {
    const xml = `<feed><entry><id>http://arxiv.org/abs/2403.19236</id><title type="html">Measuring the baryon fraction using galaxy clustering</title></entry></feed>`;
    expect(matchBestTitle("Measuring the baryon fraction using galaxy clustering", xml).arxivId)
      .toBe("2403.19236");
  });
});

/**
 * An entry shaped like arXiv's own: the id is the first element, and the fields
 * that authors write (summary, comment, journal reference) come after the title.
 */
function arxivEntry(input: {
  id: string;
  title: string;
  summary?: string;
  comment?: string;
  journalRef?: string;
}): string {
  return `<entry>
    <id>${input.id}</id>
    <updated>2024-04-01T00:00:00Z</updated>
    <published>2024-03-28T00:00:00Z</published>
    <title>${input.title}</title>
    <summary>${input.summary ?? "A summary."}</summary>
    <author><name>A. Author</name></author>
    ${input.comment === undefined ? "" : `<arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">${input.comment}</arxiv:comment>`}
    ${input.journalRef === undefined ? "" : `<arxiv:journal_ref xmlns:arxiv="http://arxiv.org/schemas/atom">${input.journalRef}</arxiv:journal_ref>`}
    <link href="${input.id}" rel="alternate" type="text/html"/>
    <category term="astro-ph.CO" scheme="http://arxiv.org/schemas/atom"/>
  </entry>`;
}

function feedOf(...entries: string[]): string {
  return `<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">${entries.join("\n")}</feed>`;
}

describe("arxiv title search entry identity", () => {
  const title = "Measuring the baryon fraction using galaxy clustering";
  const own = "http://arxiv.org/abs/2403.19236v2";

  it.each([
    ["summary", { summary: "We extend the analysis of arXiv:2101.01234 to larger scales." }],
    ["comment", { comment: "12 pages; companion to arXiv:2101.01234" }],
    ["journal reference", { journalRef: "Phys. Rev. D 109 (2024), also arXiv:2101.01234" }],
    ["legacy-form citation", { summary: "Follows arXiv:astro-ph/0210215 closely." }],
    ["cited URL", { summary: "Data from https://arxiv.org/abs/2101.01234 are reused." }],
  ])("returns the entry's own id when its %s cites another paper", (_field, extra) => {
    const result = matchBestTitle(title, feedOf(arxivEntry({ id: own, title, ...extra })));
    expect(result).toEqual({ arxivId: "2403.19236", matchedTitle: title });
  });

  it("returns the entry's own id when the title itself cites another paper", () => {
    const cited = "A reply to arXiv:2101.01234 on galaxy clustering";
    expect(matchBestTitle(cited, feedOf(arxivEntry({ id: own, title: cited }))).arxivId)
      .toBe("2403.19236");
  });

  it("does not take a cited paper for an entry whose own id is not a modern id", () => {
    const legacy = arxivEntry({
      id: "http://arxiv.org/abs/astro-ph/0210215v1",
      title,
      summary: "Revisits arXiv:2101.01234.",
    });
    expect(matchBestTitle(title, feedOf(legacy)).arxivId).toBeNull();
  });

  it("skips an entry that has no id of its own", () => {
    const noId = arxivEntry({ id: own, title, summary: "Cites arXiv:2101.01234." }).replace(/<id>[^<]*<\/id>/, "");
    expect(matchBestTitle(title, feedOf(noId)).arxivId).toBeNull();
  });

  it("reads an id element that has attributes or surrounding whitespace", () => {
    const spaced = arxivEntry({ id: own, title }).replace(`<id>${own}</id>`, `<id xml:lang="en">\n  ${own}\n</id>`);
    expect(matchBestTitle(title, feedOf(spaced)).arxivId).toBe("2403.19236");
  });

  it("keeps two entries with different own ids ambiguous even when both cite one paper", () => {
    const summary = "Builds on arXiv:2101.01234.";
    const result = matchBestTitle(title, feedOf(
      arxivEntry({ id: "http://arxiv.org/abs/2403.19236v1", title, summary }),
      arxivEntry({ id: "http://arxiv.org/abs/2403.19237v1", title, summary }),
    ));
    expect(result.arxivId).toBeNull();
  });
});

describe("arxiv title search over HTTP", () => {
  const title = "Measuring the baryon fraction using galaxy clustering";
  const answering = (status: number, bodyText: string, headers: Record<string, string> = {}) => ({
    request: vi.fn(async (): Promise<HttpResponse> => ({ status, headers, bodyText })),
  }) satisfies HttpClient;

  it("issues a title query and returns the matched canonical ID", async () => {
    const http: HttpClient = {
      request: vi.fn(async (request): Promise<HttpResponse> => {
        expect(request.url).toContain("search_query=ti:");
        expect(request.url).toContain("max_results=5");
        return {
          status: 200,
          headers: { "content-type": "application/atom+xml" },
          bodyText: atom([{ id: "2403.19236v2", title }]),
        };
      }),
    };
    const result = await searchArxivTitle(http, title, undefined, fakeClock().coordinator);
    expect(result.arxivId).toBe("2403.19236");
  });

  it("propagates HTTP failures without inventing identity", async () => {
    const http = answering(503, "");
    await expect(searchArxivTitle(http, "Some paper title", undefined, fakeClock().coordinator))
      .rejects.toThrow(/HTTP 503/);
  });

  it("reports a failed answer as an ArxivHttpError with its status and headers", async () => {
    const http = answering(429, "slow down", { "Retry-After": "120" });
    const error = await searchArxivTitle(http, title, undefined, fakeClock().coordinator).catch((caught) => caught);
    expect(isArxivHttpError(error)).toBe(true);
    expect(error).toMatchObject({ status: 429, headers: { "Retry-After": "120" } });
  });

  it("reports an empty result as no match, and a body that is not an Atom feed as no answer", async () => {
    const { coordinator } = fakeClock();
    await expect(searchArxivTitle(answering(200, atom([])), title, undefined, coordinator))
      .resolves.toEqual({ arxivId: null });
    await expect(searchArxivTitle(answering(200, "<html>Down for maintenance</html>"), title, undefined, coordinator))
      .rejects.toThrow(/Atom feed/);
    await expect(searchArxivTitle(answering(200, ""), title, undefined, coordinator))
      .rejects.toThrow(/Atom feed/);
  });

  it("spaces searches like every other arXiv request", async () => {
    const { sleeps, coordinator } = fakeClock();
    const http = answering(200, atom([{ id: "2403.19236v2", title }]));
    await searchArxivTitle(http, title, undefined, coordinator);
    await searchArxivTitle(http, title, undefined, coordinator);
    expect(http.request).toHaveBeenCalledTimes(2);
    expect(sleeps).toEqual([3_000]);
  });

  it("stops sending requests while the host cools down after a 429", async () => {
    const { state, coordinator } = fakeClock();
    const http = answering(429, "slow down");
    await expect(searchArxivTitle(http, title, undefined, coordinator)).rejects.toMatchObject({ status: 429 });

    const deferred = await searchArxivTitle(http, title, undefined, coordinator).catch((caught) => caught);
    expect(isArxivRetryDeferredError(deferred)).toBe(true);
    expect(http.request).toHaveBeenCalledTimes(1);

    state.monotonic += 10 * 60 * 1000;
    await expect(searchArxivTitle(http, title, undefined, coordinator)).rejects.toMatchObject({ status: 429 });
    expect(http.request).toHaveBeenCalledTimes(2);
  });

  it("does not send a request once cancelled", async () => {
    const controller = new AbortController();
    controller.abort();
    const http = answering(200, atom([]));
    await expect(searchArxivTitle(http, title, controller.signal, fakeClock().coordinator))
      .rejects.toSatisfy(isCancellationError);
    expect(http.request).not.toHaveBeenCalled();
  });
});
