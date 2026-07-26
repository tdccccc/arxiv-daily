import { describe, expect, it } from "vitest";
import {
  formatPaperKey,
  paperKeyFromArxivId,
  paperPathStem,
  parsePaperKey,
  resolvePaperLookupKey,
  tryParsePaperKey,
  PaperKeyError,
} from "../src/services/paper-key";

describe("paper-key", () => {
  it("formats source:externalId with lowercase source", () => {
    expect(formatPaperKey("ArXiv", "2606.12345")).toBe("arxiv:2606.12345");
    expect(formatPaperKey("arxiv", "2606.12345")).toBe("arxiv:2606.12345");
  });

  it("rejects invalid sources and externalIds", () => {
    expect(() => formatPaperKey("Ar-Xiv", "2606.12345")).toThrow(PaperKeyError);
    expect(() => formatPaperKey("arxiv", "")).toThrow(PaperKeyError);
    expect(() => formatPaperKey("arxiv", "a:b")).toThrow(PaperKeyError);
  });

  it("parses paperKey and rejects uppercase source in stored form", () => {
    expect(parsePaperKey("arxiv:2606.12345")).toEqual({
      source: "arxiv",
      externalId: "2606.12345",
    });
    expect(tryParsePaperKey("Arxiv:2606.12345")).toBeNull();
    expect(tryParsePaperKey("2606.12345")).toBeNull();
  });

  it("builds arXiv paperKey from bare id, paperKey, or URL", () => {
    expect(paperKeyFromArxivId("2606.12345")).toBe("arxiv:2606.12345");
    expect(paperKeyFromArxivId("2606.12345v2")).toBe("arxiv:2606.12345");
    expect(paperKeyFromArxivId("https://arxiv.org/abs/2606.12345")).toBe(
      "arxiv:2606.12345",
    );
    expect(() => paperKeyFromArxivId("not-an-id")).toThrow(PaperKeyError);
  });

  it("resolves lookup keys from paperKey or bare arXiv id", () => {
    expect(resolvePaperLookupKey("2606.12345")).toBe("arxiv:2606.12345");
    expect(resolvePaperLookupKey("arxiv:2606.12345")).toBe("arxiv:2606.12345");
    expect(resolvePaperLookupKey("arxiv:2606.12345v3")).toBe("arxiv:2606.12345");
    expect(() => resolvePaperLookupKey("")).toThrow(PaperKeyError);
    expect(() => resolvePaperLookupKey("s2:abc")).not.toThrow();
    expect(resolvePaperLookupKey("s2:abc")).toBe("s2:abc");
  });

  it("path stem never includes paperKey colon", () => {
    expect(paperPathStem("arxiv", "2606.12345")).toBe("2606.12345");
    expect(paperPathStem("arxiv", "2606.12345v2")).toBe("2606.12345");
    expect(paperPathStem("s2", "abc123")).toBe("abc123");
    expect(() => paperPathStem("s2", "a/b")).toThrow(PaperKeyError);
  });
});
