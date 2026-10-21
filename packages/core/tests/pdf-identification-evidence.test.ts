import { describe, expect, it, vi } from "vitest";
import { deflate, Deflate, Inflate } from "pako";
import { extractPdfIdentificationEvidence } from "../src/library/pdf-identification-evidence";
import { extractArxivIdsFromText } from "../src/library/pdf-text-utils";

function bytesOf(parts: Array<string | Uint8Array>): Uint8Array {
  const chunks = parts.map((part) =>
    typeof part === "string" ? new TextEncoder().encode(part) : part);
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function pdfWith(
  streams: Array<{ text: string; raw?: boolean } | { compressed: Uint8Array }>,
  trailer: string[] = [],
): Uint8Array {
  const parts: Array<string | Uint8Array> = ["%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"];
  let object = 1;
  for (const stream of streams) {
    if ("compressed" in stream) {
      parts.push(`${object} 0 obj\n<< /Length ${stream.compressed.length} /Filter /FlateDecode >>\nstream\n`, stream.compressed, "\nendstream\nendobj\n");
    } else if (stream.raw) {
      parts.push(`${object} 0 obj\n<< /Length ${stream.text.length} >>\nstream\n`, stream.text, "\nendstream\nendobj\n");
    } else {
      const compressed = deflate(stream.text);
      parts.push(`${object} 0 obj\n<< /Length ${compressed.length} /Filter /FlateDecode >>\nstream\n`, compressed, "\nendstream\nendobj\n");
    }
    object += 1;
  }
  parts.push(`trailer\n<< /Info ${object} 0 R >>\n%%EOF\n`);
  parts.push(`${object} 0 obj\n<< ${trailer.join(" ")} >>\nendobj\n`);
  return bytesOf(parts);
}

describe("pdf identification evidence", () => {
  it("extracts a canonical modern arXiv ID from a content-stream header", () => {
    const pdf = pdfWith([{ text: "(arXiv:2302.05010v2 [astro-ph.CO] 10 Feb 2023) Tj\n(Some title) Tj" }]);
    expect(extractPdfIdentificationEvidence(pdf).arxivId).toBe("2302.05010");
  });

  it("extracts a legacy arXiv ID from a content-stream header", () => {
    const pdf = pdfWith([{ text: "(arXiv:astro-ph/0210215  9 Oct 2002) Tj\n(Title text) Tj" }]);
    expect(extractPdfIdentificationEvidence(pdf).arxivId).toBe("astro-ph/0210215");
  });

  it("does not treat reference-list IDs in later stream text as identity", () => {
    const header = "(arXiv:2302.05010 [astro-ph.CO]) Tj\n(Title) Tj";
    const references = "(References) Tj\n1. arXiv:2004.00574 2. arXiv:2410.04229";
    const pdf = pdfWith([{ text: header }, { text: references }]);
    expect(extractPdfIdentificationEvidence(pdf).arxivId).toBe("2302.05010");
  });

  it("takes the first header ID in stream order, not the first of a particular form", () => {
    // The page's own URL-form stamp comes first; a later stream cites another
    // paper in the "arXiv:" form, which a form-by-form search would prefer.
    const pdf = pdfWith([
      { text: "(Read online at https://arxiv.org/abs/2403.19236v2) Tj\n(Some title) Tj" },
      { text: "(References) Tj\n(see arXiv:2101.01234 for details) Tj" },
    ]);
    expect(extractPdfIdentificationEvidence(pdf).arxivId).toBe("2403.19236");
  });

  it("prefers the Info dict /arXivID over reference-list IDs in stream text", () => {
    // A reference-list DOI ("arXiv.0912.0201" in a content stream) must not
    // override the submission system's own identity claim (/arXivID): the
    // citing file is the 2512.16010 paper, not the cited one.
    const header = "(LSTM-MDNz: Estimating Quasar Photometric Redshifts) Tj";
    const references = "(References) Tj\nsee doi.org/10.48550/arXiv.0912.0201";
    const pdf = pdfWith(
      [{ text: header }, { text: references }],
      [
        "/Title (LSTM-MDNz: Estimating Quasar Photometric Redshifts with an LSTM-Augmented Mixture Density Network) ",
        "/arXivID (https://arxiv.org/abs/2512.16010v1) ",
      ],
    );
    expect(extractPdfIdentificationEvidence(pdf).arxivId).toBe("2512.16010");
  });

  it("falls back to stream headers when the Info dict has no /arXivID", () => {
    const pdf = pdfWith(
      [{ text: "(arXiv:2302.05010v2 [astro-ph.CO]) Tj\n(Title) Tj" }],
      ["/Title (Some Paper) "],
    );
    expect(extractPdfIdentificationEvidence(pdf).arxivId).toBe("2302.05010");
  });

  it("extracts XMP dc:identifier arXiv URLs", () => {
    const xmp = '<x:xmpmeta><rdf:Description><dc:identifier><rdf:li>https://arxiv.org/abs/2309.03258v3</rdf:li></dc:identifier></rdf:Description></x:xmpmeta>';
    const pdf = pdfWith([{ text: "No header id here, just text" }, { text: xmp }]);
    expect(extractPdfIdentificationEvidence(pdf).arxivId).toBe("2309.03258");
  });

  it("decodes literal /Title metadata", () => {
    const pdf = pdfWith([{ text: "Some body text" }], ["/Title (The Cluster Mass Calibration Project) "]);
    expect(extractPdfIdentificationEvidence(pdf).title).toBe("The Cluster Mass Calibration Project");
  });

  it("decodes UTF-16BE hex /Title metadata", () => {
    const title = "Dark Energy Survey Data Release";
    const utf16 = Array.from(title).flatMap((char) => {
      const code = char.charCodeAt(0);
      return [code >> 8, code & 0xff];
    });
    const hex = `FEFF${utf16.map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
    const pdf = pdfWith([{ text: "Body" }], [`/Title <${hex}>`]);
    expect(extractPdfIdentificationEvidence(pdf).title).toBe(title);
  });

  it("rejects unusable metadata titles", () => {
    for (const raw of [
      "/Title (pipeline_diagram)",
      "/Title (2.2. Cluster-Finding Algorithm)",
      "/Title (PGPLOT PostScript plot)",
      "/Title (Fig. 1 The full pipeline)",
      "/Title (12345)",
      "/Title (draft)",
    ]) {      const pdf = pdfWith([{ text: "Body" }], [raw]);
      expect(extractPdfIdentificationEvidence(pdf).title).toBeUndefined();
    }
  });

  it("returns nothing for non-PDF bytes and for PDFs without text or metadata", () => {
    expect(extractPdfIdentificationEvidence(new TextEncoder().encode("not a pdf"))).toEqual({});
    const empty = pdfWith([]);
    expect(extractPdfIdentificationEvidence(empty)).toEqual({});
  });

  it("is bounded: oversized inflated streams are skipped without throwing", () => {
    const big = `${"A".repeat(200)} ${"x".repeat(6000)}`.repeat(3000); // > 8 MiB inflated
    const pdf = pdfWith([{ text: big }, { text: "arXiv:2402.18634 [astro-ph.CO]\nTitle" }]);
    expect(extractPdfIdentificationEvidence(pdf).arxivId).toBeUndefined();
  });
});

const MIB = 1024 * 1024;

/** A FlateDecode stream that inflates to `mib` MiB of zeros from about a thousandth of that. */
function zeroBomb(mib: number): Uint8Array {
  const parts: Uint8Array[] = [];
  const deflater = new Deflate({ level: 9 });
  deflater.onData = (chunk: Uint8Array) => { parts.push(chunk); };
  const block = new Uint8Array(MIB);
  for (let index = 0; index < mib; index += 1) deflater.push(block, false);
  deflater.push(new Uint8Array(0), true);
  return bytesOf(parts);
}

/** Extract evidence and report how many bytes pako inflated doing it: the memory and time a stream can cost. */
function extractCountingInflation(pdf: Uint8Array) {
  let inflated = 0;
  const push = Inflate.prototype.push;
  const spy = vi.spyOn(Inflate.prototype, "push").mockImplementation(function (this: Inflate, data, mode) {
    const handler = this.onData;
    this.onData = (chunk) => {
      inflated += chunk.byteLength;
      handler.call(this, chunk);
    };
    return push.call(this, data, mode);
  });
  try {
    return { evidence: extractPdfIdentificationEvidence(pdf), inflated };
  } finally {
    spy.mockRestore();
  }
}

describe("pdf identification evidence inflation budget", () => {
  const title = "The Cluster Mass Calibration Project";
  const bomb = zeroBomb(64);
  const stamp = { text: "(arXiv:2302.05010v2 [astro-ph.CO] 10 Feb 2023) Tj\n(Some title) Tj" };

  it("stops inflating a bomb stream at a small limit and identifies nothing from it", () => {
    expect(bomb.length).toBeLessThan(0.1 * MIB);
    const { evidence, inflated } = extractCountingInflation(pdfWith([{ compressed: bomb }]));
    expect(evidence).toEqual({});
    expect(inflated).toBeLessThanOrEqual(4 * MIB);
  });

  it("still reads the Info title and the ordinary streams around a bomb", () => {
    const xmp = "<x:xmpmeta><rdf:Description><dc:identifier><rdf:li>https://arxiv.org/abs/2309.03258v3</rdf:li></dc:identifier></rdf:Description></x:xmpmeta>";
    const withStamp = extractCountingInflation(pdfWith(
      [{ compressed: bomb }, stamp, { compressed: bomb }, { text: xmp }],
      [`/Title (${title}) `],
    ));
    expect(withStamp.evidence).toEqual({ arxivId: "2302.05010", title });
    expect(withStamp.inflated).toBeLessThanOrEqual(2 * 4 * MIB);

    const xmpOnly = extractCountingInflation(pdfWith([{ compressed: bomb }, { text: xmp }]));
    expect(xmpOnly.evidence).toEqual({ arxivId: "2309.03258" });
  });

  it("counts a discarded stream against the total, so many bombs cannot keep the scan busy", () => {
    const smaller = zeroBomb(16);
    const { evidence, inflated } = extractCountingInflation(
      pdfWith(Array.from({ length: 24 }, () => ({ compressed: smaller }))),
    );
    expect(evidence).toEqual({});
    // 24 x 16 MiB would be 384 MiB; the total budget stops it far short of that.
    expect(inflated).toBeLessThanOrEqual(34 * MIB);
  });

  it("stops reading once the total of ordinary streams is spent", () => {
    const oneMiB = zeroBomb(1);
    const streams = [...Array.from({ length: 40 }, () => ({ compressed: oneMiB })), stamp];
    const { evidence, inflated } = extractCountingInflation(pdfWith(streams));
    expect(inflated).toBeLessThanOrEqual(34 * MIB);
    expect(evidence.arxivId).toBeUndefined();
  });

  it("leaves the evidence of an ordinary PDF unchanged", () => {
    const xmp = `<x:xmpmeta><rdf:Description><dc:identifier><rdf:li>https://arxiv.org/abs/2309.03258v3</rdf:li></dc:identifier><dc:title><rdf:li>${title}</rdf:li></dc:title></rdf:Description></x:xmpmeta>`;
    const filler = { text: "BT /F1 12 Tf (Body text of the first page) Tj ET\n".repeat(2000) };
    expect(extractPdfIdentificationEvidence(pdfWith([filler, { text: xmp }])))
      .toEqual({ arxivId: "2309.03258", title });
    expect(extractPdfIdentificationEvidence(pdfWith([filler, stamp], [`/Title (${title}) `])))
      .toEqual({ arxivId: "2302.05010", title });
  });
});

describe("pdf text arXiv ID extraction", () => {
  it("extracts canonical modern and legacy IDs and deduplicates", () => {
    expect(extractArxivIdsFromText("arXiv:2302.05010v1 arXiv:2302.05010 arXiv:astro-ph/0210215"))
      .toEqual([
        { canonicalId: "2302.05010", raw: "2302.05010v1" },
        { canonicalId: "astro-ph/0210215", raw: "astro-ph/0210215" },
      ]);
  });

  it("extracts arXiv URLs", () => {
    expect(extractArxivIdsFromText("see https://arxiv.org/abs/2410.02857v2 for details")[0]?.canonicalId)
      .toBe("2410.02857");
  });

  it("returns candidates in document order whatever their form", () => {
    const ids = (text: string) => extractArxivIdsFromText(text).map((item) => item.canonicalId);
    expect(ids("见 https://arxiv.org/abs/2403.19236 以及 arXiv:2101.01234"))
      .toEqual(["2403.19236", "2101.01234"]);
    expect(ids("arXiv:2101.01234 and then https://arxiv.org/abs/2403.19236"))
      .toEqual(["2101.01234", "2403.19236"]);
    expect(ids("arXiv:astro-ph/0210215 before arXiv:2302.05010 before https://arxiv.org/abs/2403.19236"))
      .toEqual(["astro-ph/0210215", "2302.05010", "2403.19236"]);
    expect(extractArxivIdsFromText("https://arxiv.org/pdf/2403.19236v2 arXiv:2403.19236v1")).toEqual([
      { canonicalId: "2403.19236", raw: "2403.19236" },
    ]);
  });

  it("does not invent IDs from plain numbers", () => {
    expect(extractArxivIdsFromText("version 2.3.4 of the catalog")).toEqual([]);
  });

  it("still matches spaced arXiv prefixes without unbounded whitespace", () => {
    expect(extractArxivIdsFromText("arXiv:  2302.05010").map((item) => item.canonicalId))
      .toEqual(["2302.05010"]);
  });
});
