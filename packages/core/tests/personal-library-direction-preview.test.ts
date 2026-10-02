import { describe, expect, it } from "vitest";
import { previewPersonalLibraryDirection } from "../src/library/personal-library-direction-preview";

const papers = [
  { paperKey: "arxiv:2608.00001", title: "Research agents", abstract: "Methods for reliable agents", categories: ["cs.LG"] },
  { paperKey: "file:sha256:" + "a".repeat(64), title: "Galaxy observations", abstract: "Galactic surveys", categories: [] },
];

describe("direction matching preview", () => {
  it("uses the current direction in the daily filter contract and reports category gaps without mutating input", async () => {
    const before = JSON.stringify(papers);
    const result = await previewPersonalLibraryDirection({
      text: "Reliable research agents", papers, categories: ["astro-ph"],
      llm: { call: async (messages) => {
        expect(messages[0]!.content).toContain("Reliable research agents");
        expect(messages[1]!.content).toContain("Methods for reliable agents");
        return JSON.stringify({ papers: [
          { id: "sample-1", category: "preview", directions: ["preview#1"], relevanceScore: 92 },
          { id: "sample-2", category: "skip", directions: [], relevanceScore: 0 },
        ] });
      } },
    });
    expect(result.papers[0]).toMatchObject({ paperKey: papers[0]!.paperKey, matched: true, directionText: "Reliable research agents", categoryCoverage: "outside" });
    expect(result.papers[1]).toMatchObject({ matched: false, categoryCoverage: "unknown" });
    expect(result.missingCategories).toEqual(["cs.LG"]);
    expect(JSON.stringify(papers)).toBe(before);
  });

  it("rejects invented response identities", async () => {
    await expect(previewPersonalLibraryDirection({ text: "Research agents", papers, categories: ["cs.LG"],
      llm: { call: async () => JSON.stringify({ papers: [{ id: "invented", category: "preview", directions: ["preview#1"], relevanceScore: 90 }] }) },
    })).rejects.toThrow(/invalid/i);
  });

  it("treats an umbrella category as covering its subcategory", async () => {
    const result = await previewPersonalLibraryDirection({ text: "Galaxies", categories: ["astro-ph"],
      papers: [{ ...papers[0]!, categories: ["astro-ph.GA"] }],
      llm: { call: async () => JSON.stringify({ papers: [{ id: "sample-1", category: "preview", directions: ["preview#1"], relevanceScore: 90 }] }) },
    });
    expect(result.papers[0]!.categoryCoverage).toBe("inside");
    expect(result.missingCategories).toEqual([]);
  });
});
