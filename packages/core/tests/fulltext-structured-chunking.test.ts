import { describe, expect, it } from "vitest";
import type { ParsedDocument } from "../src/documents/parsed-document";
import { chunkFullText, chunkParsedDocument } from "../src/library/fulltext/chunking";

const parser = { id: "fixture-parser", version: "1.0.0" } as const;

describe("structured full-text chunking", () => {
  it("keeps the page-only branch legacy-equivalent for index, page, text, and embedding text", () => {
    const pages = [
      "First paragraph has enough text to remain.\n\nSecond paragraph also remains.",
      "A later page paragraph has enough text.",
    ];
    const document: ParsedDocument = {
      mediaType: "application/pdf",
      blocks: pages.map((text, index) => ({ kind: "page", text, locator: { page: index + 1, block: index } })),
    };
    const legacy = chunkFullText(pages, { targetTokens: 16, overlapTokens: 2 });
    const structured = chunkParsedDocument(document, ["page-text"], parser, { targetTokens: 16, overlapTokens: 2 });
    expect(structured.map(({ index, page, text }) => ({ index, page, text }))).toEqual(legacy);
    expect(structured.map((chunk) => chunk.text)).toEqual(legacy.map((chunk) => chunk.text));
    expect(structured.every((chunk) => chunk.headings.length === 0)).toBe(true);
    expect(structured.every((chunk) => chunk.locator.pageEnd === undefined)).toBe(true);
  });

  it("compresses skipped heading levels without sparse heading arrays", () => {
    const document: ParsedDocument = {
      mediaType: "application/pdf",
      blocks: [
        { kind: "heading", text: "Deep Topic", headingLevel: 3, locator: { page: 1, block: 0 } },
        { kind: "paragraph", text: "Evidence beneath a skipped heading level.", locator: { page: 1, block: 1 } },
      ],
    };
    const chunks = chunkParsedDocument(document, ["document-structure"], parser);
    expect(chunks[0]?.headings).toEqual(["Deep Topic"]);
    expect(Object.keys(chunks[0]!.headings)).toEqual(["0"]);
  });

  it("merges short blocks, hard-splits long blocks, and keeps overlap inside a section", () => {
    const document: ParsedDocument = {
      mediaType: "application/pdf",
      blocks: [
        { kind: "heading", text: "Methods", headingLevel: 1, locator: { page: 1, block: 0 } },
        { kind: "paragraph", text: "small one", locator: { page: 1, block: 1 } },
        { kind: "paragraph", text: "small two", locator: { page: 1, block: 2 } },
        { kind: "paragraph", text: "A".repeat(70), locator: { page: 2, block: 3 } },
      ],
    };
    const chunks = chunkParsedDocument(document, ["document-structure"], parser, {
      targetTokens: 8,
      overlapTokens: 2,
      minChunkChars: 0,
    });
    expect(chunks[0]?.text).toBe("small one\nsmall two");
    expect(chunks.every((chunk) => chunk.text.length <= 32)).toBe(true);
    expect(chunks.every((chunk) => chunk.headings.join("/") === "Methods")).toBe(true);
    expect(chunks.some((chunk, index) => index > 0 && chunk.text.startsWith("small two"))).toBe(true);
  });

  it("never carries overlap across a heading section boundary", () => {
    const document: ParsedDocument = {
      mediaType: "application/pdf",
      blocks: [
        { kind: "heading", text: "First", headingLevel: 1, locator: { page: 1, block: 0 } },
        { kind: "paragraph", text: "first-section-tail", locator: { page: 1, block: 1 } },
        { kind: "heading", text: "Second", headingLevel: 1, locator: { page: 2, block: 2 } },
        { kind: "paragraph", text: "second-section-body", locator: { page: 2, block: 3 } },
      ],
    };
    const chunks = chunkParsedDocument(document, ["document-structure"], parser, {
      targetTokens: 6, overlapTokens: 4, minChunkChars: 0,
    });
    const second = chunks.filter((chunk) => chunk.headings[0] === "Second");
    expect(second[0]?.text).toBe("second-section-body");
    expect(second.every((chunk) => !chunk.text.includes("first-section-tail"))).toBe(true);
  });

  it("uses headings as context and hard section boundaries only with document-structure", () => {
    const document: ParsedDocument = {
      mediaType: "application/pdf",
      blocks: [
        { kind: "heading", text: "Methods", headingLevel: 1, locator: { page: 2, block: 0 } },
        { kind: "paragraph", text: "Method details with enough text for evidence.", locator: { page: 2, block: 1 } },
        { kind: "heading", text: "Evaluation", headingLevel: 2, locator: { page: 3, block: 2 } },
        { kind: "paragraph", text: "Evaluation details with enough text for evidence.", locator: { page: 3, block: 3 } },
        { kind: "heading", text: "Results", headingLevel: 1, locator: { page: 4, block: 4 } },
        { kind: "paragraph", text: "Results details with enough text for evidence.", locator: { page: 4, block: 5 } },
      ],
    };
    const chunks = chunkParsedDocument(document, ["page-text", "document-structure"], parser, {
      targetTokens: 512,
      overlapTokens: 0,
    });
    expect(chunks.map((chunk) => chunk.headings)).toEqual([
      ["Methods"],
      ["Methods", "Evaluation"],
      ["Results"],
    ]);
    expect(chunks.map((chunk) => chunk.text)).toEqual([
      "Method details with enough text for evidence.",
      "Evaluation details with enough text for evidence.",
      "Results details with enough text for evidence.",
    ]);
    expect(chunks.map((chunk) => chunk.locator)).toEqual([
      { pageStart: 2, pageEnd: 2, blockStart: 1, blockEnd: 1 },
      { pageStart: 3, pageEnd: 3, blockStart: 3, blockEnd: 3 },
      { pageStart: 4, pageEnd: 4, blockStart: 5, blockEnd: 5 },
    ]);
  });
});
