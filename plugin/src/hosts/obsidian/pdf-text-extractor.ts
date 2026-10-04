/** Obsidian supplies its bundled PDF.js engine and renderer asset locations. */
import {
  PdfJsDocumentParser,
  PDFJS_DOCUMENT_PARSER_CAPABILITIES,
  parsedDocumentToPdfExtractionResult,
  type DocumentParser,
  type ParseDocumentOptions,
  type ParsedDocument,
  type PdfExtractionOptions,
  type PdfExtractionResult,
  type PdfTextExtractor,
  type PdfJsLib,
} from "@arxiv-daily/core";

export type {
  PdfJsLib,
  PdfJsLoadingTask,
  PdfJsDocument,
  PdfJsPage,
  PdfJsTextContent,
  PdfJsTextItem,
} from "@arxiv-daily/core";

export const OBSIDIAN_PDF_PARSER_CAPABILITIES = PDFJS_DOCUMENT_PARSER_CAPABILITIES;

export class ObsidianPdfDocumentParser implements DocumentParser {
  readonly capabilities = OBSIDIAN_PDF_PARSER_CAPABILITIES;
  readonly provenance = { id: "obsidian-pdfjs", version: "1" } as const;

  constructor(private readonly pdfjsLib?: PdfJsLib) {}

  async parse(bytes: Uint8Array, options?: ParseDocumentOptions): Promise<ParsedDocument> {
    // Obsidian may finish loadPdfJs() after this wrapper was constructed.
    const lib = this.pdfjsLib ?? defaultPdfJsLib();
    if (!lib) {
      throw new Error(
        "Obsidian's built-in pdf.js is not available. Call `loadPdfJs()` " +
          "(from the \"obsidian\" module) and wait for it to resolve before " +
          "extracting, or inject a pdf.js library into ObsidianPdfTextExtractor.",
      );
    }
    return new PdfJsDocumentParser(lib, {
      provenance: this.provenance,
      cMapUrl: "/lib/pdfjs/cmaps/",
      cMapPacked: true,
      standardFontDataUrl: "/lib/pdfjs/standard_fonts/",
    }).parse(bytes, options);
  }
}

export class ObsidianPdfTextExtractor implements PdfTextExtractor {
  private readonly parser: ObsidianPdfDocumentParser;

  get provenance() { return this.parser.provenance; }

  constructor(pdfjsLib?: PdfJsLib) {
    this.parser = new ObsidianPdfDocumentParser(pdfjsLib);
  }

  async extractPdfText(bytes: Uint8Array, options?: PdfExtractionOptions): Promise<PdfExtractionResult> {
    return parsedDocumentToPdfExtractionResult(
      await this.parser.parse(bytes, options),
      this.parser.capabilities,
    );
  }
}

function defaultPdfJsLib(): PdfJsLib | undefined {
  if (typeof window === "undefined") return undefined;
  return (window as unknown as { pdfjsLib?: PdfJsLib }).pdfjsLib;
}
