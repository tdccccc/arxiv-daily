import { describe, expect, it } from "vitest";
import { acceptProposedTopics } from "../src/settings/accept-proposed-topics";
import { normalizeTopic, deriveTopicDescription } from "../src/settings/topics";
import { validateFilterConfig } from "../src/settings/validation";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

/**
 * Accepting a library proposal writes product settings (ADR 0014 §1), so the
 * bar is not "produces topics" but "produces settings the product will run
 * on": unique tags, the rollback shadow intact, and `validateFilterConfig`
 * with nothing to say about them.
 */

function proposed(suggestedName: string, texts: string[]) {
  return {
    suggestedName,
    directions: texts.map((text, index) => ({
      id: `candidate.${index}`,
      text,
      discoveryCues: ["cue"],
      representatives: [],
      representativeSetFingerprint: `sha256:${"0".repeat(64)}`,
      lineage: { candidateIds: [`candidate.${index}`] },
    })),
  };
}

function settingsWith(topics: ReturnType<typeof acceptProposedTopics>) {
  return {
    ...DEFAULT_SETTINGS,
    llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
    arxiv: { ...DEFAULT_SETTINGS.arxiv, topics },
  };
}

describe("acceptProposedTopics", () => {
  it("writes settings the product will actually run on", () => {
    const topics = acceptProposedTopics({
      topics: [proposed("Photometric redshifts", [
        "Neural network photometric redshift estimation for wide surveys",
        "Calibrating photo-z uncertainty against spectroscopic subsamples",
      ])],
      existingTopics: [],
    });
    expect(topics).toHaveLength(1);
    expect(topics[0]).toMatchObject({ name: "Photometric redshifts", tag: "photometric-redshifts" });
    expect(topics[0]!.directions.map(({ origin }) => origin)).toEqual(["library", "library"]);
    // ADR 0012 §1: the shadow has exactly one writer, and acceptance is not it.
    expect(topics[0]!.description).toBe(deriveTopicDescription(topics[0]!.directions));
    const validation = validateFilterConfig(settingsWith(topics), {});
    expect(validation.ok).toBe(true);
    expect(validation.reasons).toEqual([]);
  });

  it("never emits a tag that collides with settings or with a sibling", () => {
    const topics = acceptProposedTopics({
      topics: [
        proposed("Galaxy clusters", ["Optical cluster catalogues from wide imaging"]),
        proposed("Galaxy Clusters", ["Sunyaev-Zeldovich selected cluster samples"]),
        proposed("galaxy   clusters!", ["Cluster mass calibration from weak lensing"]),
      ],
      existingTopics: [{ tag: "galaxy-clusters" }],
    });
    const tags = topics.map(({ tag }) => tag);
    expect(tags).toEqual(["galaxy-clusters-2", "galaxy-clusters-3", "galaxy-clusters-4"]);
    // Duplicate tags are a hard validation error, so this is the property that
    // decides whether an accepted proposal can run at all.
    const validation = validateFilterConfig(
      settingsWith([normalizeTopic({ name: "Existing", tag: "galaxy-clusters", directions: [
        { text: "An existing hand-written direction", origin: "manual" },
      ] }), ...topics]),
      {},
    );
    expect(validation.ok).toBe(true);
  });

  it("falls back to an ordinal tag when a name has no tag characters at all", () => {
    const topics = acceptProposedTopics({
      topics: [proposed("星系团", ["星系团星表与质量标定"]), proposed("测光红移", ["神经网络测光红移"])],
      existingTopics: [],
    });
    expect(topics.map(({ tag }) => tag)).toEqual(["topic-1", "topic-2"]);
    expect(topics.map(({ name }) => name)).toEqual(["星系团", "测光红移"]);
    expect(validateFilterConfig(settingsWith(topics), {}).ok).toBe(true);
  });

  it("is deterministic: accepting the same proposal twice yields the same tags", () => {
    const input = {
      topics: [proposed("Photo z", ["A direction"]), proposed("Photo-Z", ["Another direction"])],
      existingTopics: [],
    };
    // Identities are freshly minted each time, deliberately; everything the
    // researcher sees and everything the filter keys on must not be.
    const strip = (topics: ReturnType<typeof acceptProposedTopics>) => topics.map((topic) => ({
      name: topic.name, tag: topic.tag, description: topic.description, detail: topic.detail,
      directions: topic.directions.map(({ text, origin }) => ({ text, origin })),
    }));
    expect(strip(acceptProposedTopics(input))).toEqual(strip(acceptProposedTopics(input)));
    expect(acceptProposedTopics(input).map(({ tag }) => tag)).toEqual(["photo-z", "photo-z-2"]);
  });
});
