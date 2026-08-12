import { describe, expect, it } from "vitest";
import { lexicalTitleSimilarity, normalizeTitleText } from "../src/library/fulltext/title-similarity";

describe("normalizeTitleText", () => {
  it("lowercases, strips punctuation, and collapses whitespace", () => {
    expect(normalizeTitleText("Attention Is All You Need")).toBe("attention is all you need");
    expect(normalizeTitleText("  BERT: Pre-training  of Deep  Bidirectional  Transformers ")).toBe(
      "bert pre training of deep bidirectional transformers",
    );
    expect(normalizeTitleText("")).toBe("");
    expect(normalizeTitleText("   ")).toBe("");
    expect(normalizeTitleText("!!!")).toBe("");
  });
});

describe("lexicalTitleSimilarity", () => {
  it("scores 1 for identical normalized titles (case and punctuation insensitive)", () => {
    expect(lexicalTitleSimilarity("Attention is all you need", "Attention Is All You Need")).toBe(1);
    expect(lexicalTitleSimilarity(
      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      "BERT Pre-training of Deep Bidirectional Transformers for Language Understanding",
    )).toBe(1);
  });

  it("scores 0.95 when the query is a token-prefix of the title", () => {
    expect(lexicalTitleSimilarity("BERT", "BERT: Pre-training of Deep Bidirectional Transformers")).toBe(0.95);
    expect(lexicalTitleSimilarity("attention is all", "Attention Is All You Need")).toBe(0.95);
  });

  it("does not treat a mid-title token as a prefix match", () => {
    // "learning" appears in the ResNet title but not at its start.
    expect(lexicalTitleSimilarity("learning", "Deep Residual Learning for Image Recognition")).toBe(0);
  });

  it("returns the Jaccard value for high token overlap", () => {
    // Prefix wins over Jaccard: typing the exact title matches its extension.
    expect(lexicalTitleSimilarity(
      "Deep Residual Learning for Image Recognition",
      "Deep Residual Learning for Image Recognition, with Applications",
    )).toBe(0.95);
    // Word-order variant has the same token set and scores 1.
    expect(lexicalTitleSimilarity(
      "Residual Learning Deep for Recognition Image",
      "Deep Residual Learning for Image Recognition",
    )).toBe(1);
  });

  it("scores 0 for low-overlap or unrelated queries", () => {
    expect(lexicalTitleSimilarity("Maxout Networks", "Attention Is All You Need")).toBe(0);
    expect(lexicalTitleSimilarity("how does dropout regularization work", "Improving neural networks by preventing co-adaptation")).toBe(0);
  });

  it("scores 0 when either side is empty", () => {
    expect(lexicalTitleSimilarity("", "Attention Is All You Need")).toBe(0);
    expect(lexicalTitleSimilarity("attention", "")).toBe(0);
    expect(lexicalTitleSimilarity("", "")).toBe(0);
  });
});
