import { describe, expect, it, vi } from "vitest";
import { deflate } from "pako";
import type { HttpClient, HttpResponse } from "../src/core/adapters";
import { createPdfLibraryFileIdentifier } from "../src/library/pdf-library-file-identifier";
import type { ScopedLibrarySource } from "../src/library/scoped-library-source";

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

async function identify(bytes: Uint8Array, http: HttpClient): Promise<string | null> {
  return await createPdfLibraryFileIdentifier({ source: sourceOf(bytes), http })
    .identify(PATH, undefined, bytes.length);
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
});
