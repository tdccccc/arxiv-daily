import { describe, expect, it } from "vitest";
import { classifyPaperNote } from "../src/dashboard/paper-note-classifier";

const id = "2606.12345";

function detailMarkdown(arxivId = id): string {
  return [
    "---",
    `arxiv_id: "${arxivId}"`,
    "---",
    "# Verified detail",
    "",
    "## Research question",
    "A".repeat(150),
    "## Method",
    "B".repeat(150),
    "## Evidence",
    "C".repeat(150),
    "## Limitations",
    "D".repeat(150),
  ].join("\n");
}

function emptyStub(arxivId = id): string {
  return [
    "---",
    `arxiv_id: "${arxivId}"`,
    "---",
    `# Paper title`,
    "",
    `- **arXiv**: [${arxivId}](https://arxiv.org/abs/${arxivId})`,
    `- **PDF**: [PDF](https://arxiv.org/pdf/${arxivId})`,
    "",
    "## Notes",
    "",
  ].join("\n");
}

describe("classifyPaperNote", () => {
  it("recognizes only identity-matched generated detail summaries", () => {
    expect(classifyPaperNote(detailMarkdown(), id).kind).toBe("verified_detail");
    expect(classifyPaperNote(detailMarkdown("2606.54321"), id)).toMatchObject({
      kind: "conflict",
      reason: "identity_mismatch",
    });
  });

  it("allows empty, frontmatter-only, and the exact generated empty stub", () => {
    expect(classifyPaperNote(" \n", id)).toMatchObject({ kind: "replaceable", form: "empty" });
    expect(classifyPaperNote(`---\narxiv_id: "${id}"\n---\n`, id)).toMatchObject({
      kind: "replaceable",
      form: "frontmatter_only",
    });
    expect(classifyPaperNote(emptyStub(), id)).toMatchObject({
      kind: "replaceable",
      form: "generated_empty_stub",
    });
  });

  it("protects edited stubs, handwritten notes, ambiguous YAML, and ID mismatches", () => {
    expect(classifyPaperNote(`${emptyStub()}user text`, id)).toMatchObject({
      kind: "conflict",
      reason: "user_content",
    });
    expect(classifyPaperNote("# Handwritten", id)).toMatchObject({
      kind: "conflict",
      reason: "user_content",
    });
    expect(classifyPaperNote(`---\narxiv_id: [${id}]\n---\n`, id)).toMatchObject({
      kind: "conflict",
      reason: "identity_invalid",
    });
    expect(classifyPaperNote(`---\narxiv_id: "2606.54321"\n---\n`, id)).toMatchObject({
      kind: "conflict",
      reason: "identity_mismatch",
    });
  });

  it("never treats URL-like or nested YAML values as trusted identity", () => {
    expect(classifyPaperNote(`---\narxiv_id: "https://arxiv.org/abs/${id}"\n---\n# text`, id)).toMatchObject({
      kind: "conflict",
      reason: "identity_invalid",
    });
    expect(classifyPaperNote(`---\nmeta:\n  arxiv_id: "${id}"\n---\n# text`, id)).toMatchObject({
      kind: "conflict",
      reason: "user_content",
    });
  });
});
