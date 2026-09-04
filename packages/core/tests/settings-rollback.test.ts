import { describe, it, expect } from "vitest";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { migrateArxivSettings } from "../src/settings/migration";
import { validateFilterConfig } from "../src/settings/validation";
import {
  DAILY_FILTER_PROMPT_CONTRACT_VERSION,
  DAILY_FILTER_RESULT_CONTRACT_VERSION,
  buildPaperFilterRequest,
} from "../src/pipeline/paper-filter-contract";
import type { ArxivSettings, PluginSettings } from "../src/settings/types";

/**
 * The executable form of P1's "the migration is reversible" criterion.
 *
 * `description` stays in data.json as a shadow of the first direction, so a
 * build from before directions existed reads the same file and behaves exactly
 * as it did. That older build knows only four fields per topic, so this suite
 * hands it exactly those four and checks it still works.
 *
 * P3 changed what rollback means here. This suite used to also assert that the
 * filter prompt was byte-identical after migration — the guard for P1's
 * promise not to touch classification. P3 makes the filter judge by direction,
 * so that assertion no longer describes a promise anyone is keeping: what
 * survives is **data** reversibility, and the prompt change is declared by the
 * bumped contract versions rather than hidden. The cost — cached daily filter
 * results are invalidated once — is the one ADR 0012 line 45 accepts.
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
  it("classifies by the migrated directions, not by the description shadow", () => {
    const migrated = migrateArxivSettings(LEGACY_ARXIV);

    const system = buildPaperFilterRequest([], migrated).messages[0]!.content;

    expect(system).toContain("- photo-z:\n  - photo-z#1: Photometric redshift methods, catalogs, comparisons.");
    expect(buildPaperFilterRequest([], migrated).identity.directions).toEqual(
      migrated.topics.map((topic) => ({
        ref: `${topic.tag}#1`,
        tag: topic.tag,
        id: topic.directions[0]!.id,
        text: topic.description,
      })),
    );
  });

  it("declares the prompt change with both contract versions instead of hiding it", () => {
    // The old one-line-per-topic prompt is gone, and cached results keyed to it
    // must not be reused. Both versions moved past the 1 they shipped with.
    expect(buildPaperFilterRequest([], migrateArxivSettings(LEGACY_ARXIV)).messages[0]!.content)
      .not.toContain("- photo-z: Photometric redshift methods, catalogs, comparisons.");
    expect(DAILY_FILTER_PROMPT_CONTRACT_VERSION).toBeGreaterThan(1);
    expect(DAILY_FILTER_RESULT_CONTRACT_VERSION).toBeGreaterThan(1);
  });

  it("passes the config check again after a downgrade-and-upgrade round trip", () => {
    const migrated = migrateArxivSettings(LEGACY_ARXIV);
    // What the user actually does: downgrade, let the old build rewrite
    // data.json from the four fields it knows, then upgrade. Every build
    // migrates on load, so the four-field shape is never what gets validated —
    // P3's check asks for directions, and migration is what restores them.
    const upgradedAgain = migrateArxivSettings(asOlderBuildReadsIt(migrated));

    const result = validateFilterConfig(settingsWith(upgradedAgain));

    expect(result.reasons.filter((r) => /direction|description/i.test(r))).toEqual([]);
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
