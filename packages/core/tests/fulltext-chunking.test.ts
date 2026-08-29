import { describe, expect, it } from "vitest";
import { chunkFullText } from "../src/library/fulltext/chunking";
import type { FullTextChunk } from "../src/library/fulltext/knowledge-base";

const estimateTokens = (text: string): number => Math.ceil(text.length / 4);

/** A deterministic ~63-char sentence used to build test paragraphs. */
const sentence = (index: number): string =>
  `Sentence number ${index} used to build deterministic test paragraphs.`;

/** A paragraph of `count` sentences (~63 chars each), well above the noise threshold. */
const paragraph = (count: number, seed: number): string =>
  Array.from({ length: count }, (_, k) => sentence(seed * 100 + k)).join(" ");

/** `pageCount` pages, each with `perPage` separate paragraphs. */
const buildPages = (pageCount: number, perPage: number): string[] =>
  Array.from({ length: pageCount }, (_, pageIndex) =>
    Array.from({ length: perPage }, (_, k) => paragraph(5, pageIndex * 100 + k)).join("\n\n"));

describe("full-text chunking", () => {
  it("is deterministic: identical input yields identical output", () => {
    const pages = buildPages(4, 10);
    expect(chunkFullText(pages)).toEqual(chunkFullText(pages));
    expect(chunkFullText(pages, { targetTokens: 96, overlapTokens: 12 }))
      .toEqual(chunkFullText(pages, { targetTokens: 96, overlapTokens: 12 }));
  });

  it("returns a single chunk for short text", () => {
    expect(chunkFullText(["This is a short paper with one paragraph of text."]))
      .toEqual([{ index: 0, page: 1, text: "This is a short paper with one paragraph of text." }]);
  });

  it("returns an empty array for empty input", () => {
    expect(chunkFullText([])).toEqual([]);
    expect(chunkFullText([""])).toEqual([]);
    expect(chunkFullText(["   \n  "])).toEqual([]);
  });

  it("produces multiple chunks with continuous indexes and within the token cap", () => {
    const chunks = chunkFullText(buildPages(6, 10), { targetTokens: 96 });
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((chunk, index) => expect(chunk.index).toBe(index));
    for (const chunk of chunks) {
      expect(estimateTokens(chunk.text)).toBeLessThanOrEqual(96);
    }
  });

  it("preserves every original paragraph across chunks (overlap may duplicate)", () => {
    const paragraphTexts: string[] = [];
    const pages: string[] = [];
    for (let pageIndex = 0; pageIndex < 4; pageIndex += 1) {
      const pageParagraphs: string[] = [];
      for (let k = 0; k < 10; k += 1) {
        const para = [
          `Page ${pageIndex + 1} paragraph ${k} first line with meaningful content.`,
          `Page ${pageIndex + 1} paragraph ${k} second line with more detail.`,
        ].join("\n");
        paragraphTexts.push(para);
        pageParagraphs.push(para);
      }
      pages.push(pageParagraphs.join("\n\n"));
    }
    const chunks = chunkFullText(pages);
    expect(chunks.length).toBeGreaterThan(1);
    const joined = chunks.map((chunk) => chunk.text).join("");
    for (const paragraphText of paragraphTexts) {
      expect(joined).toContain(paragraphText);
    }
  });

  it("assigns each chunk the one-based page of its first character", () => {
    const pageCount = 3;
    const pages: string[] = [];
    for (let pageIndex = 0; pageIndex < pageCount; pageIndex += 1) {
      const lines: string[] = [];
      for (let k = 0; k < 14; k += 1) {
        lines.push(`PAGE${pageIndex + 1}: paragraph ${k} filler text for the page marker test.`);
      }
      pages.push(lines.join("\n\n"));
    }
    const chunks = chunkFullText(pages, { targetTokens: 64 });
    expect(chunks.length).toBeGreaterThan(2);
    for (const chunk of chunks) {
      const marker = /PAGE(\d+):/.exec(chunk.text);
      expect(marker).not.toBeNull();
      expect(chunk.page).toBe(Number(marker![1]));
    }
    // At least one chunk spans a page boundary and keeps its start page.
    expect(chunks.some((chunk) => new Set(chunk.text.match(/PAGE\d+/g)).size > 1)).toBe(true);
  });

  it("uses the start page for a chunk spanning pages", () => {
    const chunks = chunkFullText([
      "Introduction paragraph on the first page of the document.",
      "Second page paragraph with more content for the paper.",
    ]);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.page).toBe(1);
  });

  it("hard-splits an oversized single paragraph without exceeding the token cap", () => {
    const longParagraph = Array.from({ length: 200 }, (_, i) => `Token word ${i} of the oversized paragraph.`).join(" ");
    const targetTokens = 32;
    const chunks = chunkFullText([longParagraph], { targetTokens, overlapTokens: 0 });
    expect(chunks.length).toBeGreaterThan(10);
    for (const chunk of chunks) {
      expect(estimateTokens(chunk.text)).toBeLessThanOrEqual(targetTokens);
    }
    // Pieces are cut after whitespace boundaries, so joining chunk texts
    // reproduces the original paragraph exactly.
    expect(chunks.map((chunk) => chunk.text).join("")).toBe(longParagraph);
  });

  it("carries a tail overlap of roughly overlapTokens between consecutive chunks", () => {
    const targetTokens = 256;
    const overlapTokens = 26; // ~10%
    const shortParagraphs = Array.from({ length: 90 }, (_, i) => `P${String(i).padStart(4, "0")} filler filler filler`);
    const chunks = chunkFullText([shortParagraphs.join("\n\n")], { targetTokens, overlapTokens });
    expect(chunks.length).toBeGreaterThan(1);
    const maxUnitLength = Math.max(...shortParagraphs.map((text) => text.length));
    let overlapLength = 0;
    for (let i = 1; i < chunks.length; i += 1) {
      const prev = chunks[i - 1]!.text;
      const cur = chunks[i]!.text;
      const max = Math.min(prev.length, cur.length);
      let found = 0;
      for (let k = 1; k <= max; k += 1) {
        if (cur.startsWith(prev.slice(-k))) found = k;
      }
      if (found > 0) {
        overlapLength = found;
        break;
      }
    }
    // Meaningful overlap exists and stays within budget plus one paragraph.
    expect(overlapLength).toBeGreaterThanOrEqual((overlapTokens * 4) / 2);
    expect(overlapLength).toBeLessThanOrEqual(overlapTokens * 4 + maxUnitLength);
  });

  it("normalizes whitespace before chunking", () => {
    const chunks = chunkFullText(["  First   line\t\tof the paper here  \n\n  Second line of the paper  "]);
    expect(chunks).toEqual([{ index: 0, page: 1, text: "First line of the paper here\nSecond line of the paper" }]);
  });

  it("drops short noise lines but keeps meaningful paragraphs", () => {
    const chunks = chunkFullText(["1\n\n2\n\nIntroduction text for the paper here."]);
    expect(chunks).toEqual([{ index: 0, page: 1, text: "Introduction text for the paper here." }]);
  });

  it("honors a custom minChunkChars noise threshold", () => {
    const pages = ["tiny\n\nThis paragraph has enough text for the noise threshold."];
    expect(chunkFullText(pages, { minChunkChars: 40 }).map((chunk) => chunk.text))
      .toEqual(["This paragraph has enough text for the noise threshold."]);
    expect(chunkFullText(pages, { minChunkChars: 0 }).map((chunk) => chunk.text))
      .toEqual(["tiny\nThis paragraph has enough text for the noise threshold."]);
  });

  it("rejects invalid options", () => {
    expect(() => chunkFullText(["text"], { targetTokens: 0 })).toThrow();
    expect(() => chunkFullText(["text"], { targetTokens: 1.5 })).toThrow();
    expect(() => chunkFullText(["text"], { overlapTokens: -1 })).toThrow();
    expect(() => chunkFullText(["text"], { minChunkChars: -3 })).toThrow();
  });
});

describe("full-text chunking invariants", () => {
  it("emits only well-formed chunks for any input", () => {
    const chunks: FullTextChunk[] = chunkFullText([
      "alpha paragraph with enough text to survive the noise filter.\n\nbeta line one.\ngamma line two.",
      "",
      "  delta paragraph with enough text as well.  ",
    ]);
    expect(chunks.length).toBeGreaterThan(0);
    chunks.forEach((chunk, index) => {
      expect(chunk.index).toBe(index);
      expect(Number.isSafeInteger(chunk.page) && chunk.page > 0).toBe(true);
      expect(chunk.text.length).toBeGreaterThan(0);
    });
  });
});
