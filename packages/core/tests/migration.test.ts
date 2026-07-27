import { describe, it, expect } from "vitest";
import { migrateArxivSettings, migrateEmailSettings } from "../src/settings/migration";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

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
    expect(out.categories).toEqual(["cs.CL"]);
    expect(out.timezone).toBe("UTC");
  });

  it("keeps normalized categories when already configured", () => {
    const out = migrateArxivSettings({
      category: "astro-ph",
      categories: ["astro-ph", "cs.LG", "astro-ph", " "],
      topics: [
        { id: "u1", name: "ML", tag: "ml", description: "x", detail: false },
      ],
      timezone: "UTC",
    });
    expect(out.category).toBe("astro-ph");
    expect(out.categories).toEqual(["astro-ph", "cs.LG"]);
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

  it("yields empty topics when neither topics nor legacy detailCategories are present", () => {
    const out = migrateArxivSettings({ category: "astro-ph", timezone: "UTC" });
    expect(out.topics).toEqual([]);
    expect(out.categories).toEqual(["astro-ph"]);
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

  it("handles null / undefined raw input by returning empty topics", () => {
    const out = migrateArxivSettings(undefined);
    expect(out.topics).toEqual([]);
    expect(out.category.length).toBeGreaterThan(0);
    expect(out.categories.length).toBeGreaterThan(0);
    expect(out.timezone.length).toBeGreaterThan(0);
  });

  // ── Corner cases ──

  it("handles null raw input (not just undefined)", () => {
    const out = migrateArxivSettings(null);
    expect(out.topics).toEqual([]);
    expect(out.category).toBe(DEFAULT_SETTINGS.arxiv.category);
    expect(out.timezone).toBe(DEFAULT_SETTINGS.arxiv.timezone);
  });

  it("falls back to default timezone when timezone is null or empty", () => {
    const out1 = migrateArxivSettings({ category: "cs.CL", topics: [], timezone: null });
    expect(out1.timezone).toBe(DEFAULT_SETTINGS.arxiv.timezone);

    const out2 = migrateArxivSettings({ category: "cs.CL", topics: [], timezone: "" });
    expect(out2.timezone).toBe(DEFAULT_SETTINGS.arxiv.timezone);
  });

  it("falls back to default category when category is null or empty", () => {
    const out1 = migrateArxivSettings({ category: "", timezone: "UTC" });
    expect(out1.category).toBe(DEFAULT_SETTINGS.arxiv.category);

    const out2 = migrateArxivSettings({ category: null, timezone: "UTC" });
    expect(out2.category).toBe(DEFAULT_SETTINGS.arxiv.category);
  });

  it("preserves an explicitly empty categories array", () => {
    const out = migrateArxivSettings({ category: "cs.LG", categories: [], timezone: "UTC" });
    expect(out.categories).toEqual([]);
    expect(out.category).toBe("cs.LG");
  });

  it("handles categories as non-array (old format single string)", () => {
    const out = migrateArxivSettings({ category: "astro-ph", categories: "astro-ph", timezone: "UTC" });
    expect(out.categories).toEqual(["astro-ph"]);
  });

  it("treats non-array topics as missing and falls back to legacy or fresh defaults", () => {
    const out = migrateArxivSettings({ category: "math", topics: "should-not-crash", timezone: "UTC" });
    expect(Array.isArray(out.topics)).toBe(true);
  });

  it("handles topics with some entries missing optional fields", () => {
    const out = migrateArxivSettings({
      category: "cs.CL",
      topics: [
        { id: "u1", name: "LLM", tag: "llm" }, // no description, no detail
      ],
      timezone: "UTC",
    });
    expect(out.topics).toHaveLength(1);
    expect(out.topics[0].name).toBe("LLM");
  });

  it("handles legacy detailCategories that is not an array", () => {
    const out = migrateArxivSettings({
      category: "astro-ph",
      detailCategories: "photo-z",
      categoryDisplayMap: {},
      timezone: "UTC",
    });
    // non-Array detailCategories should be treated as missing → fresh defaults (empty)
    expect(out.topics).toEqual([]);
  });

  it("filters out non-string entries from legacy detailCategories", () => {
    const out = migrateArxivSettings({
      category: "astro-ph",
      detailCategories: ["good", 123, null, undefined, true],
      categoryDisplayMap: {},
      timezone: "UTC",
    }) as unknown as Record<string, unknown>;
    // Non-string entries are filtered out; only "good" survives
    expect(Array.isArray(out.topics)).toBe(true);
    expect((out.topics as Array<{ tag: string }>)).toHaveLength(1);
    expect((out.topics as Array<{ tag: string }>)[0].tag).toBe("good");
  });

  it("handles categoryDisplayMap that is not an object", () => {
    const out = migrateArxivSettings({
      category: "astro-ph",
      detailCategories: ["photo-z"],
      categoryDisplayMap: "not-a-map",
      timezone: "UTC",
    });
    // Falls back to default {} which means title-case fallback
    expect(out.topics[0].name).toBe("Photo Z");
  });

  it("handles categoryDisplayMap that is an array (unexpected type)", () => {
    const out = migrateArxivSettings({
      category: "astro-ph",
      detailCategories: ["photo-z"],
      categoryDisplayMap: ["a", "b"],
      timezone: "UTC",
    });
    // Array casts to object but with no string keys → title-case fallback
    expect(out.topics[0].name).toBe("Photo Z");
  });

  it("prefers new-format topics over legacy detailCategories when both are present", () => {
    const out = migrateArxivSettings({
      category: "cs.CL",
      topics: [
        { id: "u1", name: "LLM", tag: "llm", description: "", detail: true },
      ],
      detailCategories: ["photo-z"],
      categoryDisplayMap: {},
      timezone: "UTC",
    });
    // Should use topics, not detailCategories
    expect(out.topics).toHaveLength(1);
    expect(out.topics[0].tag).toBe("llm");
  });

  it("ignores topics array that is empty and falls back to legacy detailCategories", () => {
    const out = migrateArxivSettings({
      category: "astro-ph",
      topics: [],
      detailCategories: ["photo-z"],
      categoryDisplayMap: {},
      timezone: "UTC",
    });
    // topics is an empty array → not truthy → falls through to detailCategories
    expect(out.topics).toHaveLength(1);
    expect(out.topics[0].tag).toBe("photo-z");
  });

  it("survives extremely long category names", () => {
    const long = "x".repeat(500);
    const out = migrateArxivSettings({ category: long, timezone: "UTC" });
    expect(out.category).toBe(long);
    expect(out.categories).toEqual([long]);
  });
});

describe("migrateEmailSettings", () => {
  it("returns defaults when raw is missing", () => {
    expect(migrateEmailSettings(undefined)).toEqual(DEFAULT_SETTINGS.email);
    expect(migrateEmailSettings(null)).toEqual(DEFAULT_SETTINGS.email);
  });

  it("merges partial email settings", () => {
    const out = migrateEmailSettings({
      enabled: true,
      to: "you@example.com",
      fromEmail: "from@example.com",
    });
    expect(out.enabled).toBe(true);
    expect(out.mode).toBe("self");
    expect(out.to).toBe("you@example.com");
    expect(out.fromEmail).toBe("from@example.com");
    expect(out.fromName).toBe(DEFAULT_SETTINGS.email.fromName);
    expect(out.apiKey).toBe(DEFAULT_SETTINGS.email.apiKey);
    expect(out.hostedToken).toBe(DEFAULT_SETTINGS.email.hostedToken);
  });
});
