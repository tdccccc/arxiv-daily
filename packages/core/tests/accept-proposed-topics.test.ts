import { describe, expect, it } from "vitest";
import { acceptProposedTopics, topicNameKey } from "../src/settings/accept-proposed-topics";
import { normalizeTopic } from "../src/settings/topics";
import { validateFilterConfig } from "../src/settings/validation";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const scopeFingerprint = `sha256:${"a".repeat(64)}`;
const identity = { proposalId: "proposal-current", scopeFingerprint };

function proposed(id: string, suggestedName: string, texts: string[], targetTopicId?: string) {
  return {
    id, suggestedName, ...(targetTopicId ? { targetTopicId } : {}),
    directions: texts.map((text, index) => ({
      id: `${id}-direction-${index}`, text, discoveryCues: ["evidence cue"], representatives: [],
      representativeSetFingerprint: `sha256:${"0".repeat(64)}`,
      lineage: { candidateIds: [`${id}-direction-${index}`] },
    })),
  };
}

function manualTopic(name = "Photo-z") {
  return normalizeTopic({
    id: "existing-topic", name, tag: "photo-z", detail: true,
    directions: [{ id: "manual-a", text: "Redshift estimation", origin: "manual" }],
  });
}

describe("acceptProposedTopics", () => {
  it("returns complete runnable settings and a receipt for the selected directions", () => {
    const result = acceptProposedTopics({
      ...identity, topics: [proposed("new-topic", "Galaxy clusters", ["Cluster mass calibration"])],
      existingTopics: [manualTopic()],
    });
    expect(result).toMatchObject({ addedTopicCount: 1, addedDirectionCount: 1 });
    expect(result.topics.map(({ name }) => name)).toEqual(["Photo-z", "Galaxy clusters"]);
    expect(result.topics[1]).toMatchObject({
      description: "Cluster mass calibration", detail: false,
      directions: [{ id: "new-topic-direction-0", text: "Cluster mass calibration", origin: "library" }],
    });
    expect(result.acceptance).toMatchObject({
      ...identity, processedCandidateIds: ["new-topic-direction-0"],
    });
    expect(validateFilterConfig({
      ...DEFAULT_SETTINGS, llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
      arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: result.topics },
    }, {}).ok).toBe(true);
  });

  it("keeps derived tags unique across existing settings and the same batch", () => {
    const existing = normalizeTopic({ ...manualTopic(), name: "Existing", tag: "galaxy-clusters" });
    const result = acceptProposedTopics({ ...identity, existingTopics: [existing], topics: [
      proposed("a", "Galaxy clusters", ["Optical cluster catalogues"]),
      proposed("b", "Galaxy_Clusters", ["Sunyaev-Zeldovich selected samples"]),
      proposed("c", "galaxy   clusters!", ["Weak lensing calibration"]),
    ] });
    expect(result).toMatchObject({ addedTopicCount: 3 });
    expect(result.topics.map(({ tag }) => tag)).toEqual([
      "galaxy-clusters", "galaxy-clusters-2", "galaxy-clusters-3", "galaxy-clusters-4",
    ]);
  });

  it("gives non-Latin topic names distinct usable tags", () => {
    const result = acceptProposedTopics({ ...identity, existingTopics: [], topics: [
      proposed("a", "星系团", ["星系团质量标定"]), proposed("b", "测光红移", ["神经网络测光红移"]),
    ] });
    expect(result).toMatchObject({ addedTopicCount: 2 });
    expect(result.topics.map(({ tag }) => tag)).toEqual(["topic-1", "topic-2"]);
  });

  it("can accept the second direction of the same proposal later", () => {
    const topic = proposed("new-topic", "Photo-z", ["Redshift estimation", "Uncertainty calibration"]);
    const first = acceptProposedTopics({ ...identity, existingTopics: [], topics: [
      { ...topic, directions: topic.directions.slice(0, 1) },
    ] });
    expect(first).toMatchObject({ addedTopicCount: 1, addedDirectionCount: 1 });
    const second = acceptProposedTopics({
      ...identity, existingTopics: first.topics, acceptance: first.acceptance,
      topics: [{ ...topic, directions: topic.directions.slice(1) }],
    });
    expect(second).toMatchObject({ addedTopicCount: 0, addedDirectionCount: 1 });
    expect(second.topics[0]!.directions.map(({ text }) => text))
      .toEqual(["Redshift estimation", "Uncertainty calibration"]);
    expect(second.acceptance.processedCandidateIds)
      .toEqual(["new-topic-direction-0", "new-topic-direction-1"]);
  });

  it("appends to an existing destination without changing its manual direction or detail policy", () => {
    const existing = manualTopic();
    const result = acceptProposedTopics({ ...identity, existingTopics: [existing], topics: [
      proposed("extension", "Photo-z", ["Uncertainty calibration"], existing.id),
    ] });
    expect(result).toMatchObject({ addedTopicCount: 0, addedDirectionCount: 1 });
    expect(result.topics[0]).toMatchObject({ id: existing.id, detail: true, description: "Redshift estimation" });
    expect(result.topics[0]!.directions[0]).toEqual(existing.directions[0]);
    expect(existing.directions).toHaveLength(1);
  });

  it("resolves a same-name topic as a destination without treating its missing direction as accepted", () => {
    const result = acceptProposedTopics({ ...identity, existingTopics: [manualTopic()], topics: [
      proposed("extension", "  PHOTO-Z  ", ["Uncertainty calibration"]),
    ] });
    expect(result).toMatchObject({ addedTopicCount: 0, addedDirectionCount: 1 });
    expect(result.topics[0]!.directions.map(({ text }) => text))
      .toEqual(["Redshift estimation", "Uncertainty calibration"]);
  });

  it("preserves a renamed destination when the rest of its original proposal is accepted", () => {
    const topic = proposed("new-topic", "Photo-z", ["Redshift estimation", "Uncertainty calibration"]);
    const first = acceptProposedTopics({ ...identity, existingTopics: [], topics: [
      { ...topic, directions: topic.directions.slice(0, 1) },
    ] });
    expect(first).toMatchObject({ addedTopicCount: 1 });
    const existingTopics = first.topics.map((item) => ({ ...item, name: "Photometric redshifts", tag: "photometric-redshifts" }));
    const result = acceptProposedTopics({ ...identity, existingTopics, acceptance: first.acceptance, topics: [topic] });
    expect(result).toMatchObject({ addedTopicCount: 0, addedDirectionCount: 1 });
    expect(result.topics.map(({ name }) => name)).toEqual(["Photometric redshifts"]);
  });

  it.each(["edited", "deleted"])("does not restore a previously accepted direction after it is %s", (change) => {
    const topic = proposed("extension", "Photo-z", ["Uncertainty calibration"], "existing-topic");
    const first = acceptProposedTopics({ ...identity, existingTopics: [manualTopic()], topics: [topic] });
    expect(first).toMatchObject({ addedDirectionCount: 1 });
    const existingTopics = first.topics.map((item) => ({
      ...item,
      directions: change === "deleted" ? item.directions.slice(0, 1) : item.directions.map((direction) =>
        direction.id === "extension-direction-0" ? { ...direction, text: "User's narrower question" } : direction),
    }));
    const result = acceptProposedTopics({ ...identity, existingTopics, acceptance: first.acceptance, topics: [topic] });
    expect(result).toMatchObject({ addedTopicCount: 0, addedDirectionCount: 0 });
    expect(result.topics).toEqual(existingTopics);
  });

  it("does not recreate a deleted destination for pending directions, even if its old name was reused", () => {
    const topic = proposed("new-topic", "Photo-z", ["Redshift estimation", "Uncertainty calibration"]);
    const acceptance = { ...identity, topicTargets: { "new-topic": "deleted-topic" }, processedCandidateIds: ["new-topic-direction-0"] };
    expect(() => acceptProposedTopics({
      ...identity, topics: [topic], acceptance, existingTopics: [manualTopic()],
    })).toThrow(/destination|target/i);
  });

  it("requires a new choice when an explicitly selected target no longer exists", () => {
    expect(() => acceptProposedTopics({ ...identity, existingTopics: [], topics: [
      proposed("extension", "Photo-z", ["Uncertainty calibration"], "deleted-topic"),
    ] })).toThrow(/destination|target/i);
  });

  it("records text already present as processed without modifying the manual direction", () => {
    const existing = manualTopic();
    const result = acceptProposedTopics({ ...identity, existingTopics: [existing], topics: [
      proposed("extension", "Photo-z", ["  REDSHIFT   estimation  "], existing.id),
    ] });
    expect(result).toMatchObject({ addedTopicCount: 0, addedDirectionCount: 0 });
    expect(result.topics).toEqual([existing]);
    expect(result.acceptance.processedCandidateIds).toEqual(["extension-direction-0"]);
  });

  it("does not infer acceptance in a different proposal from reused candidate identifiers", () => {
    const result = acceptProposedTopics({ ...identity, existingTopics: [manualTopic()],
      acceptance: { ...identity, proposalId: "another-proposal", topicTargets: {}, processedCandidateIds: ["extension-direction-0"] },
      topics: [proposed("extension", "Photo-z", ["Uncertainty calibration"], "existing-topic")],
    });
    expect(result).toMatchObject({ addedDirectionCount: 1 });
  });

  it("combines same-name proposals without dropping their different directions", () => {
    const result = acceptProposedTopics({ ...identity, existingTopics: [], topics: [
      proposed("a", "Photo-z", ["Redshift estimation"]),
      proposed("b", "photo-z", ["Uncertainty calibration"]),
    ] });
    expect(result).toMatchObject({ addedTopicCount: 1, addedDirectionCount: 2 });
    expect(result.topics[0]!.directions.map(({ text }) => text))
      .toEqual(["Redshift estimation", "Uncertainty calibration"]);
  });

  it("allows an explicit reassignment of the still-pending directions", () => {
    const existing = manualTopic("Other topic");
    const result = acceptProposedTopics({ ...identity, existingTopics: [existing],
      acceptance: { ...identity, topicTargets: { extension: "deleted-topic" }, processedCandidateIds: [] },
      topics: [proposed("extension", "Other topic", ["Uncertainty calibration"], existing.id)],
    });
    expect(result).toMatchObject({ addedTopicCount: 0, addedDirectionCount: 1 });
    expect(result.acceptance.topicTargets.extension).toBe("existing-topic");
  });

  it("exposes consistent name normalization for the review surface", () => {
    expect(topicNameKey("  Photometric REDSHIFTS  ")).toBe(topicNameKey("photometric redshifts"));
  });
});
