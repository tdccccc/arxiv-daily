import { describe, it, expect } from "vitest";
import { migrateArxivSettings } from "../src/settings/migration";

describe("migrateArxivSettings", () => {
  it("returns the same topics when already in new shape", () => {
    const input = {
      category: "cs.CL",
      topics: [
        { id: "u1", name: "LLM", tag: "llm", description: "x", detail: true },
      ],
      timezone: "UTC",
    };
    const out = migrateArxivSettings(input);
    expect(out.topics).toEqual(input.topics);
    expect(out.category).toBe("cs.CL");
    expect(out.timezone).toBe("UTC");
  });

  it("builds topics from legacy detailCategories + displayMap", () => {
    const legacy = {
      category: "astro-ph",
      researchInterests: "ignored",
      detailCriteria: "ignored",
      detailCategories: ["photo-z", "galaxy-cluster"],
      categoryTagMap: { "photo-z": "photo-z", "galaxy-cluster": "galaxy-cluster" },
      categoryDisplayMap: {
        "photo-z": "Photo-z 相关",
        "galaxy-cluster": "Galaxy Cluster 相关",
        "other": "其他",
      },
      timezone: "Asia/Shanghai",
    };
    const out = migrateArxivSettings(legacy);
    expect(out.topics).toHaveLength(2);
    expect(out.topics[0]).toMatchObject({
      tag: "photo-z",
      name: "Photo-z 相关",
      description: "",
      detail: true,
    });
    expect(out.topics[0].id).toMatch(/^[0-9a-f-]{36}$/i);
    expect(out.topics[1]).toMatchObject({
      tag: "galaxy-cluster",
      name: "Galaxy Cluster 相关",
      description: "",
      detail: true,
    });
  });

  it("falls back to title-case when display map lacks an entry", () => {
    const legacy = {
      category: "astro-ph",
      detailCategories: ["photo-z"],
      categoryDisplayMap: {},
      timezone: "UTC",
    };
    const out = migrateArxivSettings(legacy);
    expect(out.topics[0].name).toBe("Photo Z");
  });

  it("uses defaults when neither topics nor legacy detailCategories are present", () => {
    const out = migrateArxivSettings({ category: "astro-ph", timezone: "UTC" });
    expect(out.topics.length).toBeGreaterThan(0);
    for (const t of out.topics) {
      expect(t.id).toMatch(/^[0-9a-f-]{36}$/i);
    }
  });

  it("never carries legacy fields through to the returned shape", () => {
    const legacy = {
      category: "astro-ph",
      researchInterests: "ABC",
      detailCriteria: "XYZ",
      detailCategories: ["photo-z"],
      categoryDisplayMap: {},
      timezone: "UTC",
    };
    const out = migrateArxivSettings(legacy) as unknown as Record<string, unknown>;
    expect(out.researchInterests).toBeUndefined();
    expect(out.detailCriteria).toBeUndefined();
    expect(out.detailCategories).toBeUndefined();
    expect(out.categoryTagMap).toBeUndefined();
    expect(out.categoryDisplayMap).toBeUndefined();
  });

  it("handles null / undefined raw input", () => {
    const out = migrateArxivSettings(undefined);
    expect(out.topics.length).toBeGreaterThan(0);
    expect(out.category.length).toBeGreaterThan(0);
    expect(out.timezone.length).toBeGreaterThan(0);
  });
});
