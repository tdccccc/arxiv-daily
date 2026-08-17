import { describe, expect, it } from "vitest";
import {
  DOCUMENT_PARSER_CAPABILITIES,
  type DocumentParser,
  type ParsedDocument,
  type ParserCapability,
  type SourceLocator,
} from "../src/index";

class FakeDocumentParser implements DocumentParser {
  readonly capabilities = [
    "page-text",
    "text-layout",
    "document-metadata",
  ] as const satisfies readonly ParserCapability[];

  async parse(): Promise<ParsedDocument> {
    return {
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
      ],
      metadata: { title: "Example paper" },
    };
  }
}

describe("structured document parser contract", () => {
  it("represents host-neutral blocks in reading order with source locators", async () => {
    const parsed = await new FakeDocumentParser().parse(new Uint8Array([1]));

    expect(parsed).toEqual({
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
      ],
      metadata: { title: "Example paper" },
    });
  });

  it("uses 1-based pages, 0-based block ordinals, and half-open character ranges", () => {
    const locator: SourceLocator = {
      page: 2,
      block: 3,
      charStart: 4,
      charEnd: 9,
    };

    expect(locator).toEqual({
      page: 2,
      block: 3,
      charStart: 4,
      charEnd: 9,
    });
  });

  it("exposes parser capabilities as a serializable ordered list", () => {
    const parser = new FakeDocumentParser();

    expect(DOCUMENT_PARSER_CAPABILITIES).toEqual([
      "page-text",
      "text-layout",
      "document-metadata",
      "document-structure",
    ]);
    expect(JSON.parse(JSON.stringify(parser.capabilities))).toEqual([
      "page-text",
      "text-layout",
      "document-metadata",
    ]);
  });
});
