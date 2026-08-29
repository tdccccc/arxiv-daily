import { describe, expect, it, vi } from "vitest";
import {
  ObsidianPdfTextExtractor,
  type PdfJsLib,
  type PdfJsTextItem,
} from "../src/hosts/obsidian/pdf-text-extractor";

function pdfJsWith(
  items: PdfJsTextItem[],
  view?: readonly [number, number, number, number],
  infoTitle?: string,
): PdfJsLib & { destroy: ReturnType<typeof vi.fn> } {
  const destroy = vi.fn(async () => {});
  return {
    destroy,
    getDocument: vi.fn(() => ({
      promise: Promise.resolve({
        numPages: 1,
        getPage: vi.fn(async () => ({
          getTextContent: vi.fn(async () => ({ items })),
          cleanup: vi.fn(),
          view,
        })),
        getMetadata: vi.fn(async () => ({ info: { Title: infoTitle } })),
      }),
      destroy,
    })),
  };
}

describe("ObsidianPdfTextExtractor", () => {
  it("preserves line breaks represented by empty pdf.js EOL markers", async () => {
    const pdfjs = pdfJsWith([
      { str: "Draft version January 31, 2019", hasEOL: false },
      { str: "", hasEOL: true },
      { str: "Preprint typeset using LATEX", hasEOL: false },
      { str: "", hasEOL: true },
      { str: "THE PAN-STARRS1 SURVEYS", hasEOL: false },
      { str: "", hasEOL: true },
      { str: "K. C. Chambers, E. A. Magnier", hasEOL: false },
    ]);

    const result = await new ObsidianPdfTextExtractor(pdfjs)
      .extractPdfText(new Uint8Array([1]));

    expect(result.pages).toEqual([
      "Draft version January 31, 2019\n"
        + "Preprint typeset using LATEX\n"
        + "THE PAN-STARRS1 SURVEYS\n"
        + "K. C. Chambers, E. A. Magnier",
    ]);
    expect(pdfjs.destroy).toHaveBeenCalledTimes(1);
  });

  it("reports typographic layout lines (text, font size, position) from item transforms", async () => {
    // MNRAS-style first page: small banner above a large-font title. The
    // transform matrix entries are [a, b, c, d, e, f]; the font size is the
    // scale (hypot of c/d) and the baseline is f. Page box is 612 x 782.
    const pdfjs = pdfJsWith([
      { str: "Advance Access publication 2021 January 21", hasEOL: true, transform: [1, 0, 0, 8.97, 0, 734.4] },
      { str: "A machine learning approach to galaxy properties", hasEOL: true, transform: [1, 0, 0, 15.94, 0, 699.2] },
      { str: "S. Mucesh", hasEOL: false, transform: [1, 0, 0, 11.96, 0, 651.4] },
    ], [0, 0, 612, 782]);

    const result = await new ObsidianPdfTextExtractor(pdfjs)
      .extractPdfText(new Uint8Array([1]));

    expect(result.pages).toEqual([
      "Advance Access publication 2021 January 21\n"
        + "A machine learning approach to galaxy properties\n"
        + "S. Mucesh",
    ]);
    expect(result.layout?.[0]).toHaveLength(3);
    const lines = result.layout![0]!;
    expect(lines[0]!.text).toBe("Advance Access publication 2021 January 21");
    expect(lines[0]!.fontSize).toBe(8.97);
    expect(lines[0]!.topFraction).toBeCloseTo(0.061, 3);
    expect(lines[1]!.text).toBe("A machine learning approach to galaxy properties");
    expect(lines[1]!.fontSize).toBe(15.94);
    expect(lines[1]!.topFraction).toBeCloseTo(0.106, 3);
    expect(lines[2]!.text).toBe("S. Mucesh");
    expect(lines[2]!.fontSize).toBe(11.96);
    expect(lines[2]!.topFraction).toBeCloseTo(0.167, 3);
  });

  it("reports zero font size and position when items carry no transforms", async () => {
    const pdfjs = pdfJsWith([
      { str: "Plain text line", hasEOL: true },
      { str: "Another line", hasEOL: false },
    ]);

    const result = await new ObsidianPdfTextExtractor(pdfjs)
      .extractPdfText(new Uint8Array([1]));

    expect(result.pages).toEqual(["Plain text line\nAnother line"]);
    expect(result.layout?.[0]).toEqual([
      { text: "Plain text line", fontSize: 0, topFraction: 0 },
      { text: "Another line", fontSize: 0, topFraction: 0 },
    ]);
  });

  it("reports the document metadata title and trims it", async () => {
    const pdfjs = pdfJsWith(
      [{ str: "A paper title", hasEOL: false }],
      undefined,
      "  Photometric redshifts for the Pan-STARRS1 survey  ",
    );

    const result = await new ObsidianPdfTextExtractor(pdfjs)
      .extractPdfText(new Uint8Array([1]));

    expect(result.metadataTitle).toBe("Photometric redshifts for the Pan-STARRS1 survey");
  });

  it("passes Obsidian's pdf.js asset URLs for non-embedded fonts", async () => {
    // Without `standardFontDataUrl`/`cMapUrl`, pdf.js throws on PDFs whose
    // fonts are not embedded (standard fonts) or use CID/CMap encodings,
    // failing the whole extraction (observed for 79 vault PDFs).
    const pdfjs = pdfJsWith([{ str: "A paper title", hasEOL: false }]);

    await new ObsidianPdfTextExtractor(pdfjs)
      .extractPdfText(new Uint8Array([1]));

    expect(pdfjs.getDocument).toHaveBeenCalledWith({
      data: new Uint8Array([1]),
      cMapUrl: "/lib/pdfjs/cmaps/",
      cMapPacked: true,
      standardFontDataUrl: "/lib/pdfjs/standard_fonts/",
    });
  });

  it("copies the bytes so pdf.js transferring the buffer cannot break metadata fallback", async () => {
    // pdf.js transfers (detaches) the buffer handed to getDocument. The raw
    // /Title fallback reads the file head after extraction, so a detach would
    // throw "Cannot perform Construct on a detached ArrayBuffer" and fail the
    // whole extraction (observed for 79 vault PDFs without a readable
    // getMetadata Title).
    const bytes = new TextEncoder().encode(
      "%PDF-1.4\n1 0 obj\n<< /Title (A Robust Title) >>\nendobj\n%%EOF\n",
    );
    const pdfjs = pdfJsWith([{ str: "page text", hasEOL: false }]);
    pdfjs.getDocument.mockImplementation((src) => {
      // Simulate pdf.js transferring the provided buffer to its worker.
      src.data.buffer.transfer();
      return {
        promise: Promise.resolve({
          numPages: 1,
          getPage: vi.fn(async () => ({
            getTextContent: vi.fn(async () => ({ items: [{ str: "page text", hasEOL: false }] })),
            cleanup: vi.fn(),
          })),
          // No readable metadata title: forces the raw /Title fallback.
          getMetadata: vi.fn(async () => ({ info: { Title: "" } })),
        }),
        destroy: pdfjs.destroy,
      };
    });

    const result = await new ObsidianPdfTextExtractor(pdfjs)
      .extractPdfText(bytes);

    expect(result.metadataTitle).toBe("A Robust Title");
  });

  it("degrades missing or failing metadata to undefined", async () => {
    const pdfjs = pdfJsWith([{ str: "A paper title", hasEOL: false }]);
    pdfjs.getDocument.mockReturnValueOnce({
      promise: Promise.resolve({
        numPages: 1,
        getPage: vi.fn(async () => ({
          getTextContent: vi.fn(async () => ({ items: [{ str: "A paper title", hasEOL: false }] })),
          cleanup: vi.fn(),
        })),
      }),
      destroy: pdfjs.destroy,
    });
    // No getMetadata on the document: extraction still works.
    const result = await new ObsidianPdfTextExtractor(pdfjs)
      .extractPdfText(new Uint8Array([1]));
    expect(result.metadataTitle).toBeUndefined();
    expect(result.pages).toEqual(["A paper title"]);
  });
});

  it("falls back to the raw Info /Title when pdf.js resolves it empty", async () => {
    // Obsidian's pdf.js can resolve a duplicate /Title key to an empty entry;
    // the host then reads the first literal /Title from the file head.
    const bytes = new TextEncoder().encode(
      "%PDF-1.4\n1 0 obj\n<< /Title (Redshift Assessment Infrastructure Layers \\(RAIL\\): Rubin-era photometric redshift stress-testing and at-scale production) >>\nendobj\ntrailer\n<< /Info 1 0 R >>\n%%EOF\n",
    );
    const pdfjs = pdfJsWith([{ str: "A paper title", hasEOL: false }]);
    pdfjs.getDocument.mockReturnValueOnce({
      promise: Promise.resolve({
        numPages: 1,
        getPage: vi.fn(async () => ({
          getTextContent: vi.fn(async () => ({ items: [{ str: "A paper title", hasEOL: false }] })),
          cleanup: vi.fn(),
        })),
        getMetadata: vi.fn(async () => ({ info: { Title: "" } })),
      }),
      destroy: pdfjs.destroy,
    });

    const result = await new ObsidianPdfTextExtractor(pdfjs)
      .extractPdfText(bytes);

    expect(result.metadataTitle)
      .toBe("Redshift Assessment Infrastructure Layers (RAIL): Rubin-era photometric redshift stress-testing and at-scale production");
  });
