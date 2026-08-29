import { describe, expect, it } from "vitest";
import {
  parsedDocumentToPdfExtractionResult,
  type ParsedDocument,
  type ParserCapability,
} from "../src/index";

const PDF_CAPABILITIES = [
  "page-text",
  "text-layout",
  "document-metadata",
] as const satisfies readonly ParserCapability[];

function parsedDocument(): ParsedDocument {
  return {
    mediaType: "application/pdf",
    blocks: [
      {
        kind: "page",
        text: "  First page\nkeeps whitespace  ",
        locator: { page: 1, block: 0 },
        layout: [
          { text: "First page", fontSize: 14, topFraction: 0.1 },
        ],
      },
      {
        kind: "page",
        text: "",
        locator: { page: 2, block: 1 },
        layout: [],
      },
    ],
    metadata: { title: "Example title" },
  };
}

describe("parsed document PDF text compatibility", () => {
  it("preserves page order, empty pages, text, layout, and metadata title", () => {
    expect(parsedDocumentToPdfExtractionResult(
      parsedDocument(),
      PDF_CAPABILITIES,
    )).toEqual({
      pages: ["  First page\nkeeps whitespace  ", ""],
      layout: [
        [{ text: "First page", fontSize: 14, topFraction: 0.1 }],
        [],
      ],
      metadataTitle: "Example title",
    });
  });

  it("omits layout when the parser does not declare layout capability", () => {
    const capabilities = [
      "page-text",
      "document-metadata",
    ] as const satisfies readonly ParserCapability[];

    expect(parsedDocumentToPdfExtractionResult(
      parsedDocument(),
      capabilities,
    )).toEqual({
      pages: ["  First page\nkeeps whitespace  ", ""],
      metadataTitle: "Example title",
    });
  });

  it("rejects non-page blocks because the legacy contract is page-aligned", () => {
    const document: ParsedDocument = {
      mediaType: "application/pdf",
      blocks: [
        {
          kind: "paragraph",
          text: "Detached paragraph",
          locator: { page: 1, block: 0 },
        },
      ],
    };

    expect(() => parsedDocumentToPdfExtractionResult(
      document,
      PDF_CAPABILITIES,
    )).toThrow("page blocks");
  });

  it("rejects missing or out-of-order page locators", () => {
    const document: ParsedDocument = {
      mediaType: "application/pdf",
      blocks: [
        { kind: "page", text: "Second", locator: { page: 2, block: 0 } },
      ],
    };

    expect(() => parsedDocumentToPdfExtractionResult(
      document,
      PDF_CAPABILITIES,
    )).toThrow("page 1");
  });
});
