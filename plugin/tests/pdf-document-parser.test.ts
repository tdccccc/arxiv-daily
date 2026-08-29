import { describe, expect, it, vi } from "vitest";
import {
  OBSIDIAN_PDF_PARSER_CAPABILITIES,
  ObsidianPdfDocumentParser,
  type PdfJsLib,
} from "../src/hosts/obsidian/pdf-text-extractor";

describe("ObsidianPdfDocumentParser", () => {
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
