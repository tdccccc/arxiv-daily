import { describe, expect, it, vi } from "vitest";
import {
  OBSIDIAN_PDF_PARSER_CAPABILITIES,
  ObsidianPdfDocumentParser,
  ObsidianPdfTextExtractor,
  type PdfJsLib,
} from "../src/hosts/obsidian/pdf-text-extractor";

function textEngine(text: string): PdfJsLib {
  return {
    getDocument: vi.fn(() => ({
      promise: Promise.resolve({
        numPages: 1,
        getPage: async () => ({ getTextContent: async () => ({ items: [{ str: text }] }) }),
      }),
      destroy: vi.fn(async () => {}),
    })),
  };
}

describe("ObsidianPdfDocumentParser", () => {
  it("uses PDF.js loaded after extractor construction and reads the current default engine", async () => {
    const host = window as unknown as { pdfjsLib?: PdfJsLib };
    const previous = host.pdfjsLib;
    try {
      delete host.pdfjsLib;
      const extractor = new ObsidianPdfTextExtractor();
      host.pdfjsLib = textEngine("Loaded after construction");
      expect((await extractor.extractPdfText(new Uint8Array([1]))).pages)
        .toEqual(["Loaded after construction"]);
      host.pdfjsLib = textEngine("Current default engine");
      expect((await extractor.extractPdfText(new Uint8Array([1]))).pages)
        .toEqual(["Current default engine"]);
    } finally {
      if (previous === undefined) delete host.pdfjsLib;
      else host.pdfjsLib = previous;
    }
  });

  it("keeps explicit injection ahead of the current window PDF.js engine", async () => {
    const host = window as unknown as { pdfjsLib?: PdfJsLib };
    const previous = host.pdfjsLib;
    try {
      const injected = textEngine("Explicit engine");
      const parser = new ObsidianPdfDocumentParser(injected);
      host.pdfjsLib = textEngine("Window engine");
      expect((await parser.parse(new Uint8Array([1]))).blocks[0]?.text).toBe("Explicit engine");
      expect(host.pdfjsLib.getDocument).not.toHaveBeenCalled();
      expect(injected.getDocument).toHaveBeenCalledOnce();
    } finally {
      if (previous === undefined) delete host.pdfjsLib;
      else host.pdfjsLib = previous;
    }
  });

  it("cancels an abandoned page promise and destroys the loading task", async () => {
    const pageStarted = vi.fn();
    const destroy = vi.fn(async () => {});
    const pdfjs: PdfJsLib = {
      getDocument: () => ({
        promise: Promise.resolve({
          numPages: 1,
          getPage: () => { pageStarted(); return new Promise(() => {}); },
        }),
        destroy,
      }),
    };
    const controller = new AbortController();
    const pending = new ObsidianPdfDocumentParser(pdfjs).parse(new Uint8Array([1]), { signal: controller.signal });
    const rejected = expect(pending).rejects.toMatchObject({ name: "AbortError", message: "stop parsing" });
    await vi.waitFor(() => expect(pageStarted).toHaveBeenCalledOnce());
    controller.abort("stop parsing");
    await rejected;
    expect(destroy).toHaveBeenCalled();
  });

  it("emits page-aligned structured blocks and preserves a failed page", async () => {
    const destroy = vi.fn(async () => {});
    const pdfjs: PdfJsLib = {
      getDocument: vi.fn(() => ({
        promise: Promise.resolve({
          numPages: 2,
          getPage: vi.fn(async (pageNumber: number) => {
            if (pageNumber === 2) throw new Error("malformed page");
            return {
              getTextContent: vi.fn(async () => ({
                items: [{
                  str: "First page",
                  hasEOL: false,
                  transform: [1, 0, 0, 12, 0, 90],
                }],
              })),
              cleanup: vi.fn(),
              view: [0, 0, 100, 100] as const,
            };
          }),
          getMetadata: vi.fn(async () => ({
            info: { Title: "  Structured paper  " },
          })),
        }),
        destroy,
      })),
    };

    const parser = new ObsidianPdfDocumentParser(pdfjs);
    const result = await parser.parse(new Uint8Array([1]));

    expect(OBSIDIAN_PDF_PARSER_CAPABILITIES).toEqual([
      "page-text",
      "text-layout",
      "document-metadata",
    ]);
    expect(parser.capabilities).toBe(OBSIDIAN_PDF_PARSER_CAPABILITIES);
    expect(result).toEqual({
      mediaType: "application/pdf",
      blocks: [
        {
          kind: "page",
          text: "First page",
          locator: { page: 1, block: 0 },
          layout: [
            { text: "First page", fontSize: 12, topFraction: 0.1 },
          ],
        },
        {
          kind: "page",
          text: "",
          locator: { page: 2, block: 1 },
          layout: [],
        },
      ],
      metadata: { title: "Structured paper" },
    });
    expect(destroy).toHaveBeenCalledTimes(1);
  });
});
