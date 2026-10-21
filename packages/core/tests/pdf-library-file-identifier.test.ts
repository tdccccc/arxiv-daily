import { describe, expect, it, vi } from "vitest";
import { deflate } from "pako";
import type { HttpClient, HttpResponse } from "../src/core/adapters";
import { createPdfLibraryFileIdentifier } from "../src/library/pdf-library-file-identifier";
import type { ScopedLibrarySource } from "../src/library/scoped-library-source";
import { ArxivRequestCoordinator, isArxivHttpError, isArxivRetryDeferredError } from "../src/pipeline/arxiv-fetcher";
import { isCancellationError } from "../src/services/cancellation";

const TITLE = "Measuring the baryon fraction using galaxy clustering";
const PATH = "papers/renamed.pdf";

/** A PDF whose first content stream carries `header` and whose Info dict carries `info`. */
function pdfBytes(input: { header?: string; info?: string[] }): Uint8Array {
  const parts: Array<string | Uint8Array> = ["%PDF-1.4\n"];
  if (input.header !== undefined) {
    const compressed = deflate(input.header);
    parts.push(
      `1 0 obj\n<< /Length ${compressed.length} /Filter /FlateDecode >>\nstream\n`,
      compressed,
      "\nendstream\nendobj\n",
    );
  }
  parts.push(`2 0 obj\n<< ${(input.info ?? []).join(" ")} >>\nendobj\ntrailer\n<< /Info 2 0 R >>\n%%EOF\n`);
  const chunks = parts.map((part) => typeof part === "string" ? new TextEncoder().encode(part) : part);
  const out = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function entry(id: string, title: string, summary = "A summary."): string {
  return `<entry><id>http://arxiv.org/abs/${id}</id><title>${title}</title><summary>${summary}</summary></entry>`;
}

function feed(...entries: string[]): string {
  return `<feed xmlns="http://www.w3.org/2005/Atom">${entries.join("")}</feed>`;
}

function sourceOf(bytes: Uint8Array): ScopedLibrarySource {
  return {
    inventory: vi.fn(async () => { throw new Error("not used"); }),
    readBinary: vi.fn(async (_path, options) => bytes.slice(options?.start ?? 0, options?.end).buffer),
  };
}

function httpAnswering(bodyText: string, status = 200) {
  const request = vi.fn(async (): Promise<HttpResponse> => ({ status, headers: {}, bodyText }));
  return { http: { request } satisfies HttpClient, request };
}

/** A coordinator on a fake clock: waiting advances the clock instead of taking time. */
function fakeClock() {
  const state = { monotonic: 0 };
  const coordinator = new ArxivRequestCoordinator({
    monotonicNow: () => state.monotonic,
    wallNow: () => Date.parse("2026-10-21T10:00:00.000Z") + state.monotonic,
    sleep: async (ms) => { state.monotonic += ms; },
  });
  return { state, coordinator };
}

async function identify(bytes: Uint8Array, http: HttpClient): Promise<string | null> {
  return await createPdfLibraryFileIdentifier({
    source: sourceOf(bytes), http, requestCoordinator: fakeClock().coordinator,
  }).identify(PATH, undefined, bytes.length);
}

describe("createPdfLibraryFileIdentifier", () => {
  const streamHeader = (id: string) => `(arXiv:${id} [astro-ph.CO] 28 Mar 2024) Tj`;
  const titled = (info: string[] = []) => [`/Title (${TITLE})`, ...info];

  it("identifies a PDF by the entry's own id, not an id its summary cites", async () => {
    const { http } = httpAnswering(feed(
      entry("2403.19236v2", TITLE, "We extend the analysis of arXiv:2101.01234 to larger scales."),
    ));
    expect(await identify(pdfBytes({ info: titled() }), http)).toBe("2403.19236");
  });

  describe("when the PDF carries an arXiv id and the title search answers too", () => {
    it("keeps the id when the search names the same paper", async () => {
      const { http, request } = httpAnswering(feed(entry("2403.19236v2", TITLE)));
      const bytes = pdfBytes({ header: streamHeader("2403.19236v2"), info: titled() });
      expect(await identify(bytes, http)).toBe("2403.19236");
      expect(request).toHaveBeenCalledTimes(1);
    });

    // The title search is the independent witness: an arXiv record whose title
    // matches the document's own title strictly, and matches nothing else,
    // outranks an id scraped from stream text, which can be a citation (a
    // reference list sitting in a content-stream header).
    it("takes a strict, unambiguous title match over an id scraped from a stream header", async () => {
      const { http } = httpAnswering(feed(entry("2403.19236v2", TITLE)));
      const bytes = pdfBytes({ header: streamHeader("0912.0201"), info: titled() });
      expect(await identify(bytes, http)).toBe("2403.19236");
    });

    it("takes it over a declared Info dict /arXivID as well", async () => {
      const { http } = httpAnswering(feed(entry("2403.19236v2", TITLE)));
      const bytes = pdfBytes({ info: titled(["/arXivID (https://arxiv.org/abs/2610.00001v1)"]) });
      expect(await identify(bytes, http)).toBe("2403.19236");
    });

    it("keeps the id when the search finds nothing", async () => {
      const { http } = httpAnswering(feed());
      const bytes = pdfBytes({ header: streamHeader("2302.05010"), info: titled() });
      expect(await identify(bytes, http)).toBe("2302.05010");
    });

    it("keeps the id when the search is ambiguous", async () => {
      const { http } = httpAnswering(feed(
        entry("2403.19236v1", TITLE),
        entry("2403.19237v1", `${TITLE} revisited`),
      ));
      const bytes = pdfBytes({ header: streamHeader("2302.05010"), info: titled() });
      expect(await identify(bytes, http)).toBe("2302.05010");
    });

    it.each([
      ["answers with an HTTP error", httpAnswering("", 503).http],
      ["cannot be reached", { request: vi.fn(async () => { throw new Error("offline"); }) } satisfies HttpClient],
    ])("keeps the id when the search %s", async (_case, http) => {
      const bytes = pdfBytes({ header: streamHeader("2302.05010"), info: titled() });
      expect(await identify(bytes, http)).toBe("2302.05010");
    });

    it("does not search for a title that only restates the arXiv stamp", async () => {
      const { http, request } = httpAnswering(feed(entry("2403.19236v2", TITLE)));
      const bytes = pdfBytes({
        header: streamHeader("2302.05010"),
        info: ["/Title (arXiv:2302.05010v2 [astro-ph.CO] 10 Feb 2023)"],
      });
      expect(await identify(bytes, http)).toBe("2302.05010");
      expect(request).not.toHaveBeenCalled();
    });
  });

  describe("when the PDF names no paper itself and the title search cannot answer", () => {
    const untitled = pdfBytes({ header: "(Just some text) Tj" });

    it("answers null, a judgement, when arXiv was asked and nothing matched", async () => {
      const { http, request } = httpAnswering(feed());
      expect(await identify(pdfBytes({ info: titled() }), http)).toBeNull();
      expect(request).toHaveBeenCalledTimes(1);
    });

    it("answers null without a lookup when the PDF has no usable title", async () => {
      const { http, request } = httpAnswering(feed(entry("2403.19236v2", TITLE)));
      expect(await identify(untitled, http)).toBeNull();
      expect(request).not.toHaveBeenCalled();
    });

    it.each([
      ["an HTTP error", httpAnswering("", 503).http],
      ["a rate limit", httpAnswering("", 429).http],
      ["a page that is not a feed", httpAnswering("<html>maintenance</html>").http],
      ["no connection", { request: vi.fn(async () => { throw new Error("offline"); }) } satisfies HttpClient],
    ])("throws rather than answering null on %s, so the file is not judged", async (_case, http) => {
      await expect(identify(pdfBytes({ info: titled() }), http)).rejects.toThrow();
    });

    it("throws the HTTP error itself, naming the status", async () => {
      const error = await identify(pdfBytes({ info: titled() }), httpAnswering("", 503).http).catch((caught) => caught);
      expect(isArxivHttpError(error)).toBe(true);
      expect(error).toMatchObject({ status: 503 });
    });

    it("lets cancellation through instead of treating it as a failed lookup", async () => {
      const controller = new AbortController();
      const request = vi.fn(async (): Promise<HttpResponse> => {
        controller.abort();
        throw new Error("aborted");
      });
      const identifier = createPdfLibraryFileIdentifier({
        source: sourceOf(pdfBytes({ info: titled() })), http: { request }, requestCoordinator: fakeClock().coordinator,
      });
      await expect(identifier.identify(PATH, controller.signal)).rejects.toSatisfy(isCancellationError);
    });

    it("does not read the file once cancelled", async () => {
      const controller = new AbortController();
      controller.abort();
      const source = sourceOf(pdfBytes({ info: titled() }));
      const identifier = createPdfLibraryFileIdentifier({ source, http: httpAnswering(feed()).http });
      await expect(identifier.identify(PATH, controller.signal)).rejects.toSatisfy(isCancellationError);
      expect(source.readBinary).not.toHaveBeenCalled();
    });

    it("stops asking after a 429 until the cooldown has passed", async () => {
      const { state, coordinator } = fakeClock();
      const request = vi.fn(async (): Promise<HttpResponse> => ({ status: 429, headers: {}, bodyText: "" }));
      const bytes = pdfBytes({ info: titled() });
      const identifier = createPdfLibraryFileIdentifier({ source: sourceOf(bytes), http: { request }, requestCoordinator: coordinator });

      await expect(identifier.identify("a.pdf")).rejects.toMatchObject({ status: 429 });
      const deferred = await identifier.identify("b.pdf").catch((caught) => caught);
      expect(isArxivRetryDeferredError(deferred)).toBe(true);
      expect(request).toHaveBeenCalledTimes(1);

      state.monotonic += 10 * 60 * 1000;
      await expect(identifier.identify("c.pdf")).rejects.toMatchObject({ status: 429 });
      expect(request).toHaveBeenCalledTimes(2);
    });

    it("spaces the lookups of consecutive files", async () => {
      const { state, coordinator } = fakeClock();
      const { http, request } = httpAnswering(feed());
      const identifier = createPdfLibraryFileIdentifier({ source: sourceOf(pdfBytes({ info: titled() })), http, requestCoordinator: coordinator });
      await identifier.identify("a.pdf");
      const afterFirst = state.monotonic;
      await identifier.identify("b.pdf");
      expect(request).toHaveBeenCalledTimes(2);
      expect(state.monotonic - afterFirst).toBeGreaterThanOrEqual(3_000);
    });

    it("pauses its lookups for the rest of the scan after repeated failures", async () => {
      const request = vi.fn(async () => { throw new Error("offline"); });
      const identifier = createPdfLibraryFileIdentifier({
        source: sourceOf(pdfBytes({ info: titled() })), http: { request }, requestCoordinator: fakeClock().coordinator,
      });
      for (const name of ["a", "b", "c", "d", "e"]) {
        await expect(identifier.identify(`${name}.pdf`)).rejects.toThrow();
      }
      expect(request).toHaveBeenCalledTimes(3);
    });

    it("keeps looking up after a failure that is followed by an answer", async () => {
      const request = vi.fn()
        .mockRejectedValueOnce(new Error("offline"))
        .mockRejectedValueOnce(new Error("offline"))
        .mockResolvedValueOnce({ status: 200, headers: {}, bodyText: feed() })
        .mockRejectedValueOnce(new Error("offline"))
        .mockRejectedValueOnce(new Error("offline"))
        .mockResolvedValue({ status: 200, headers: {}, bodyText: feed(entry("2403.19236v2", TITLE)) });
      const identifier = createPdfLibraryFileIdentifier({
        source: sourceOf(pdfBytes({ info: titled() })), http: { request }, requestCoordinator: fakeClock().coordinator,
      });
      const outcomes: Array<string | null> = [];
      for (const name of ["a", "b", "c", "d", "e", "f"]) {
        outcomes.push(await identifier.identify(`${name}.pdf`).catch(() => "failed"));
      }
      expect(outcomes).toEqual(["failed", "failed", null, "failed", "failed", "2403.19236"]);
    });

    it("still answers from the PDF's own id while lookups are failing", async () => {
      const request = vi.fn(async () => { throw new Error("offline"); });
      const identifier = createPdfLibraryFileIdentifier({
        source: sourceOf(pdfBytes({ header: streamHeader("2302.05010"), info: titled() })),
        http: { request }, requestCoordinator: fakeClock().coordinator,
      });
      for (const name of ["a", "b", "c", "d", "e"]) {
        expect(await identifier.identify(`${name}.pdf`)).toBe("2302.05010");
      }
    });
  });
});
