import { describe, it, expect } from "vitest";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { migrateArxivSettings } from "../src/settings/migration";
import { validateFilterConfig } from "../src/settings/validation";
import { buildPaperFilterRequest } from "../src/pipeline/paper-filter-contract";
import type { ArxivSettings, PluginSettings } from "../src/settings/types";

/**
 * The executable form of P1's "the migration is reversible" criterion.
 *
 * `description` stays in data.json as a shadow of the first direction, so a
 * build from before directions existed reads the same file and behaves exactly
 * as it did. That older build knows only four fields per topic, so this suite
 * hands it exactly those four and checks it still works — and that the filter
 * prompt is unchanged, which is what keeps cached classifications valid.
 */

const LEGACY_ARXIV = {
  category: "astro-ph.GA",
  categories: ["astro-ph.GA", "astro-ph.CO"],
  timezone: "Asia/Shanghai",
  topics: [
    {
      id: "a1",
      name: "Photo-z",
      tag: "photo-z",
      description: "Photometric redshift methods, catalogs, comparisons.",
      detail: false,
    },
    {
      id: "a2",
      name: "Galaxy Cluster",
      tag: "galaxy-cluster",
      description: "Cluster surveys, mass calibration, SZ/X-ray/optical.",
      detail: true,
    },
  ],
};

/** What a build from before directions existed sees in the same file. */
function asOlderBuildReadsIt(arxiv: ArxivSettings): ArxivSettings {
  return {
    ...arxiv,
    topics: arxiv.topics.map((topic) => ({
      id: topic.id,
      name: topic.name,
      tag: topic.tag,
      description: topic.description,
      detail: topic.detail,
    })) as ArxivSettings["topics"],
  };
}

function settingsWith(arxiv: ArxivSettings): PluginSettings {
  return {
    ...structuredClone(DEFAULT_SETTINGS),
    llm: {
      ...structuredClone(DEFAULT_SETTINGS.llm),
      apiKey: "sk-test",
      baseUrl: "https://api.example.com/v1",
      model: "test-model",
    },
    arxiv,
  };
}

describe("rollback to a build without directions", () => {
  it("leaves the filter prompt byte-identical after migration", () => {
    const migrated = migrateArxivSettings(LEGACY_ARXIV);

    const before = buildPaperFilterRequest([], LEGACY_ARXIV as ArxivSettings);
    const after = buildPaperFilterRequest([], migrated);

    // Nothing the classifier sees changed, so cached results stay valid.
    expect(JSON.stringify(after)).toBe(JSON.stringify(before));
  });

  it("still builds the same prompt from only the fields an older build reads", () => {
    const migrated = migrateArxivSettings(LEGACY_ARXIV);
    const downgraded = asOlderBuildReadsIt(migrated);

    expect(JSON.stringify(buildPaperFilterRequest([], downgraded))).toBe(
      JSON.stringify(buildPaperFilterRequest([], LEGACY_ARXIV as ArxivSettings)),
    );
  });

  it("passes the older build's config check without a missing description", () => {
    const migrated = migrateArxivSettings(LEGACY_ARXIV);
    const downgraded = asOlderBuildReadsIt(migrated);

    const result = validateFilterConfig(settingsWith(downgraded));

    expect(result.reasons.filter((r) => /description/i.test(r))).toEqual([]);
    expect(result.ok).toBe(true);
  });

  it("keeps the shadow equal to the first direction after migration", () => {
    const migrated = migrateArxivSettings(LEGACY_ARXIV);

    for (const topic of migrated.topics) {
      expect(topic.description).toBe(topic.directions[0]?.text ?? "");
    }
  });

  it("round-trips: migrating what an older build wrote back is stable", () => {
    const once = migrateArxivSettings(LEGACY_ARXIV);
    // The user downgrades, the old build rewrites data.json from the four
    // fields it knows, then they upgrade again.
    const twice = migrateArxivSettings(asOlderBuildReadsIt(once));

    expect(twice.topics.map((t) => t.description)).toEqual(
      once.topics.map((t) => t.description),
    );
    expect(twice.topics.map((t) => t.directions[0].text)).toEqual(
      once.topics.map((t) => t.directions[0].text),
    );
  });
});
