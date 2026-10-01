import { describe, expect, it, vi } from "vitest";
import { PdfJsDocumentParser, type PdfJsLib } from "../src/documents/pdfjs-document-parser";

describe("shared PDF.js parser", () => {
  it("uses the supplied engine assets and provenance without owning host loading", async () => {
    const destroy = vi.fn(async () => {});
    const getDocument = vi.fn<PdfJsLib["getDocument"]>(() => ({
      promise: Promise.resolve({
        numPages: 1,
        getPage: async () => ({
          getTextContent: async () => ({ items: [{ str: "A calibrated survey", transform: [1, 0, 0, 16, 0, 90] }] }),
          view: [0, 0, 100, 100] as const,
        }),
        getMetadata: async () => ({ info: { Title: "A calibrated survey" } }),
      }),
      destroy,
    }));
    const parser = new PdfJsDocumentParser({ getDocument }, {
      provenance: { id: "node-pdfjs", version: "fixture-engine-version" },
      cMapUrl: "/runtime/cmaps/",
      cMapPacked: true,
      standardFontDataUrl: "/runtime/standard_fonts/",
    });
    const bytes = new Uint8Array([1, 2, 3]);
    const result = await parser.parse(bytes);
    expect(parser.provenance).toEqual({ id: "node-pdfjs", version: "fixture-engine-version" });
    expect(parser.capabilities).toEqual(["page-text", "text-layout", "document-metadata"]);
    expect(getDocument).toHaveBeenCalledWith({
      data: bytes, cMapUrl: "/runtime/cmaps/", cMapPacked: true,
      standardFontDataUrl: "/runtime/standard_fonts/",
    });
    expect(getDocument.mock.calls[0]![0].data).not.toBe(bytes);
    expect(result).toMatchObject({
      mediaType: "application/pdf", metadata: { title: "A calibrated survey" },
      blocks: [{ kind: "page", text: "A calibrated survey", locator: { page: 1, block: 0 }, layout: [{ fontSize: 16, topFraction: 0.1 }] }],
    });
    expect(destroy).toHaveBeenCalledOnce();
  });

  it("does not start engine work for a pre-cancelled parse", async () => {
    const lib: PdfJsLib = { getDocument: vi.fn(() => { throw new Error("engine must not start"); }) };
    const parser = new PdfJsDocumentParser(lib, { provenance: { id: "host", version: "1" } });
    const controller = new AbortController();
    controller.abort("stop before load");
    await expect(parser.parse(new Uint8Array([1]), { signal: controller.signal }))
      .rejects.toMatchObject({ name: "AbortError", message: "stop before load" });
    expect(lib.getDocument).not.toHaveBeenCalled();
  });
});
