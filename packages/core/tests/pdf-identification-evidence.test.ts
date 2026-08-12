import { describe, expect, it } from "vitest";
import { deflate } from "pako";
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

function pdfWith(streams: Array<{ text: string; raw?: boolean }>, trailer: string[] = []): Uint8Array {
  const parts: Array<string | Uint8Array> = ["%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"];
  let object = 1;
  for (const { text, raw } of streams) {
    if (raw) {
      parts.push(`${object} 0 obj\n<< /Length ${text.length} >>\nstream\n`, text, "\nendstream\nendobj\n");
    } else {
      const compressed = deflate(text);
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

  it("does not invent IDs from plain numbers", () => {
    expect(extractArxivIdsFromText("version 2.3.4 of the catalog")).toEqual([]);
  });
});
