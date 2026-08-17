import { describe, expect, it } from "vitest";
import {
  CHUNK_DERIVATION_VERSIONS,
  createEvidenceChunkId,
  type EvidenceChunk,
  type NormalizedBoundingBox,
  type ParserProvenance,
} from "../src/library/fulltext/evidence-chunk";

describe("evidence chunk contract", () => {
  it("describes parser provenance, derivation versions, headings, and source location", () => {
    const parser: ParserProvenance = { id: "fixture-structured-parser", version: "2.1.0" };
    const bbox: NormalizedBoundingBox = { left: 0.1, top: 0.2, right: 0.8, bottom: 0.4 };
    const chunk: EvidenceChunk = {
      id: createEvidenceChunkId({
        text: "Stable evidence",
        headings: ["Methods"],
        locator: { pageStart: 3, pageEnd: 4, blockStart: 7, blockEnd: 9, bbox },
        derivation: { parser, ...CHUNK_DERIVATION_VERSIONS },
      }),
      index: 0,
      page: 3,
      text: "Stable evidence",
      headings: ["Methods"],
      locator: { pageStart: 3, pageEnd: 4, blockStart: 7, blockEnd: 9, bbox },
      derivation: { parser, ...CHUNK_DERIVATION_VERSIONS },
    };
    expect(chunk.id).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(chunk.locator.bbox).toEqual(bbox);
  });

  it("uses length-safe canonical fields rather than ambiguous concatenation or object order", () => {
    const base = {
      locator: { pageStart: 1 },
      derivation: {
        parser: { id: "parser", version: "1" },
        chunkerVersion: 2,
        embeddingInputVersion: 1,
      },
    } as const;
    const left = createEvidenceChunkId({ ...base, text: "ab", headings: ["c"] });
    const right = createEvidenceChunkId({ ...base, text: "a", headings: ["bc"] });
    expect(left).not.toBe(right);
    expect(createEvidenceChunkId({
      text: "ab",
      headings: ["c"],
      locator: { pageStart: 1 },
      derivation: {
        embeddingInputVersion: 1,
        chunkerVersion: 2,
        parser: { version: "1", id: "parser" },
      },
    })).toBe(left);
  });

  it("derives a host-neutral stable id without paper identity or path", () => {
    const input = {
      text: "Same canonical evidence",
      headings: ["Results"],
      locator: { pageStart: 5, blockStart: 11, blockEnd: 11 },
      derivation: {
        parser: { id: "parser", version: "1" },
        chunkerVersion: 2,
        embeddingInputVersion: 1,
      },
    } as const;
    expect(createEvidenceChunkId(input)).toBe(createEvidenceChunkId({ ...input }));
    expect(createEvidenceChunkId({ ...input, text: "Different evidence" })).not.toBe(createEvidenceChunkId(input));
  });
});
