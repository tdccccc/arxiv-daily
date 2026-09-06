import { describe, expect, it } from "vitest";
import {
  decodeOrganizedTopics,
  PERSONAL_LIBRARY_ORGANIZATION_MAX_DIRECTIONS_PER_TOPIC,
  PERSONAL_LIBRARY_ORGANIZATION_MAX_TOPICS,
  PERSONAL_LIBRARY_ORGANIZATION_MIN_TOPICS,
  type OrganizedDirection,
  type OrganizedTopic,
  type OrganizedTopicsResult,
  type OrganizationValidationReason,
  type PersonalLibraryOrganizationGroup,
} from "../src/library/personal-library-topic-organization";
import {
  PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS,
  PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUES,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH,
  PERSONAL_LIBRARY_MAX_NAME_LENGTH,
  PERSONAL_LIBRARY_MAX_REPRESENTATIVES,
} from "../src/library/personal-library-interest-profile";

function paperKey(index: number): string {
  return `arxiv:2608.${String(index).padStart(5, "0")}`;
}

function group(id: string, indexes: number[]): PersonalLibraryOrganizationGroup {
  return { id, papers: indexes.map((index) => ({ paperKey: paperKey(index) })) };
}

const groups = [
  group("group-b", [4, 3]),
  group("group-c", [5]),
  group("group-a", [2, 1]),
];

function direction(groupIds: string[], representatives: number[]): OrganizedDirection {
  return {
    text: "Galaxy formation through observations",
    discoveryCues: ["survey data", "galaxy evolution"],
    groupIds,
    representativePaperKeys: representatives.map(paperKey),
  };
}

function organization(): OrganizedTopicsResult {
  return {
    topics: [
      {
        suggestedName: "Galaxies",
        directions: [direction(["group-b", "group-a"], [4, 1])],
      },
      {
        suggestedName: "Methods",
        directions: [direction(["group-c"], [5])],
      },
    ],
  };
}

function decode(value: unknown, inputGroups = groups) {
  return decodeOrganizedTopics(JSON.stringify(value), inputGroups);
}

function expectRejection(
  value: unknown,
  reason: OrganizationValidationReason,
  inputGroups = groups,
): void {
  expect(decode(value, inputGroups)).toEqual({ ok: false, reason });
}

describe("decodeOrganizedTopics", () => {
  it("exports the agreed small topic and direction bounds", () => {
    expect(PERSONAL_LIBRARY_ORGANIZATION_MIN_TOPICS).toBe(2);
    expect(PERSONAL_LIBRARY_ORGANIZATION_MAX_TOPICS).toBe(4);
    expect(PERSONAL_LIBRARY_ORGANIZATION_MAX_DIRECTIONS_PER_TOPIC).toBe(2);
  });

  it("accepts complete multi-group directions and canonicalizes set-like fields", () => {
    const result = decode(organization());
    expect(result).toEqual({
      ok: true,
      value: {
        topics: [
          {
            suggestedName: "Galaxies",
            directions: [{
              text: "Galaxy formation through observations",
              discoveryCues: ["galaxy evolution", "survey data"],
              groupIds: ["group-a", "group-b"],
              representativePaperKeys: [paperKey(1), paperKey(4)],
            }],
          },
          {
            suggestedName: "Methods",
            directions: [{
              text: "Galaxy formation through observations",
              discoveryCues: ["galaxy evolution", "survey data"],
              groupIds: ["group-c"],
              representativePaperKeys: [paperKey(5)],
            }],
          },
        ],
      },
    });
  });

  it("is independent of input group and set-field ordering using code-unit sort", () => {
    const first = organization();
    first.topics[0]!.directions[0]!.discoveryCues = ["ä", "z", "Z", "a"];
    const second = structuredClone(first);
    for (const topic of second.topics) {
      for (const item of topic.directions) {
        item.discoveryCues.reverse();
        item.groupIds.reverse();
        item.representativePaperKeys.reverse();
      }
    }
    const result = decode(first);
    expect(result).toEqual(decode(second, [...groups].reverse()));
    expect(result.ok && result.value.topics[0]!.directions[0]!.discoveryCues)
      .toEqual(["Z", "a", "z", "ä"]);
  });

  it("preserves the model's topic and direction order", () => {
    const value = organization();
    value.topics[0]!.directions = [
      { ...direction(["group-b"], [3]), text: "Z direction" },
      { ...direction(["group-a"], [1]), text: "A direction" },
    ];
    value.topics.reverse();
    const result = decode(value);
    expect(result.ok && result.value.topics.map((topic: OrganizedTopic) => topic.suggestedName))
      .toEqual(["Methods", "Galaxies"]);
    expect(result.ok && result.value.topics[1]!.directions.map((item: OrganizedDirection) => item.text))
      .toEqual(["Z direction", "A direction"]);
  });

  it("allows one topic only when there is a single evidence group", () => {
    const value = { topics: [{ suggestedName: "Galaxies", directions: [direction(["group-a"], [1])] }] };
    expect(decode(value, [group("group-a", [1, 2])]).ok).toBe(true);
    value.topics[0]!.directions[0]!.groupIds.push("group-b");
    expectRejection(value, "topic-count", [group("group-a", [1, 2]), group("group-b", [3])]);
  });

  it("accepts four topics and rejects a fifth", () => {
    const inputGroups = Array.from({ length: 5 }, (_, index) => group(`g${index}`, [index + 1]));
    const topics = inputGroups.map((item, index) => ({
      suggestedName: `Topic ${index}`,
      directions: [direction([item.id], [index + 1])],
    }));
    expect(decode({ topics: topics.slice(0, 4) }, inputGroups.slice(0, 4)).ok).toBe(true);
    expectRejection({ topics }, "topic-count", inputGroups);
  });

  it("rejects no topics, no evidence groups, or more topics than groups", () => {
    expectRejection({ topics: [] }, "topic-count");
    expectRejection({ topics: [] }, "topic-count", []);
    expectRejection(organization(), "topic-count", [group("group-a", [1])]);
  });

  it.each([0, 3])("rejects a topic with %i directions", (count) => {
    const value = organization();
    value.topics[0]!.directions = Array.from({ length: count }, () => direction(["group-a"], [1]));
    expectRejection(value, "direction-count");
  });

  it.each(["plain text", "```json\n{}\n```", "{\"topics\":"])("rejects non-JSON output %s", (raw) => {
    expect(decodeOrganizedTopics(raw, groups)).toEqual({ ok: false, reason: "not-json" });
  });

  it.each([null, [], {}, { topics: null }, { topics: [], extra: true }].map((value) => ({ value })))(
    "rejects a malformed or extended root $value",
    ({ value }) => expectRejection(value, "wrong-shape"),
  );

  it.each([
    null,
    [],
    { suggestedName: "Galaxies" },
    { suggestedName: "Galaxies", directions: null },
    { suggestedName: "Galaxies", directions: [], id: "invented" },
  ].map((topic) => ({ topic })))("rejects a malformed or extended topic $topic", ({ topic }) => {
    expectRejection({ topics: [topic, organization().topics[1]] }, "wrong-shape");
  });

  it("rejects missing or extra direction fields", () => {
    const valid = direction(["group-a", "group-b"], [1]);
    const variants: unknown[] = [null, [], { ...valid, id: "invented" }];
    for (const key of Object.keys(valid)) {
      const incomplete: Record<string, unknown> = { ...valid };
      delete incomplete[key];
      variants.push(incomplete);
    }
    for (const item of variants) {
      expectRejection({ topics: [
        { suggestedName: "Galaxies", directions: [item] },
        organization().topics[1],
      ] }, "wrong-shape");
    }
  });

  it.each([
    "", " ", " leading", "trailing ", "line\nline", "line\rline", "line\u2028line", "line\u2029line",
  ])("rejects non-canonical or multiline names and text %j", (text) => {
    const badName = organization();
    badName.topics[0]!.suggestedName = text;
    expectRejection(badName, "text-bounds");
    const badText = organization();
    badText.topics[0]!.directions[0]!.text = text;
    expectRejection(badText, "text-bounds");
  });

  it("enforces shared name and direction length bounds inclusively", () => {
    const value = organization();
    value.topics[0]!.suggestedName = "n".repeat(PERSONAL_LIBRARY_MAX_NAME_LENGTH);
    value.topics[0]!.directions[0]!.text = "t".repeat(PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH);
    expect(decode(value).ok).toBe(true);
    value.topics[0]!.suggestedName += "n";
    expectRejection(value, "text-bounds");
    value.topics[0]!.suggestedName = "Galaxies";
    value.topics[0]!.directions[0]!.text += "t";
    expectRejection(value, "text-bounds");
  });

  it.each([
    null, [], [1], [""], [" cue"], ["cue", "cue"],
    ["c".repeat(PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH + 1)],
    Array.from({ length: PERSONAL_LIBRARY_MAX_DISCOVERY_CUES + 1 }, (_, index) => `cue ${index}`),
  ].map((cues) => ({ cues })))("rejects invalid discovery cues $cues", ({ cues }) => {
    const value = organization();
    expectRejection({ topics: [
      { ...value.topics[0], directions: [{ ...value.topics[0]!.directions[0], discoveryCues: cues }] },
      value.topics[1],
    ] }, "cues-invalid");
  });

  it("accepts the shared discovery cue count and length bounds", () => {
    const value = organization();
    value.topics[0]!.directions[0]!.discoveryCues = [
      "c".repeat(PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH),
      ...Array.from({ length: PERSONAL_LIBRARY_MAX_DISCOVERY_CUES - 1 }, (_, index) => `cue ${index}`),
    ];
    expect(decode(value).ok).toBe(true);
  });

  it.each([null, [], [7], ["missing-group"], ["group-a", "group-a"], "group-a"].map((groupIds) => ({ groupIds })))(
    "rejects invalid, empty, unknown or duplicate group references $groupIds",
    ({ groupIds }) => {
      const value = organization();
      expectRejection({ topics: [
        { ...value.topics[0], directions: [{ ...value.topics[0]!.directions[0], groupIds }] },
        value.topics[1],
      ] }, "group-assignment");
    },
  );

  it("rejects a group assigned to two directions within a topic", () => {
    const value = organization();
    value.topics[0]!.directions.push(direction(["group-a"], [1]));
    expectRejection(value, "group-assignment");
  });

  it("rejects a group assigned to two topics", () => {
    const value = organization();
    value.topics[1]!.directions[0]!.groupIds.push("group-b");
    expectRejection(value, "group-assignment");
  });

  it("rejects unassigned groups even when all listed references are legal", () => {
    const value = organization();
    value.topics[0]!.directions[0] = direction(["group-a"], [1]);
    expectRejection(value, "group-assignment");
  });

  it("rejects ambiguous source group IDs or an assigned source group with no papers", () => {
    expectRejection(organization(), "group-assignment", [...groups, group("group-a", [6])]);
    expectRejection(organization(), "group-assignment", groups.map((item) =>
      item.id === "group-a" ? group("group-a", []) : item));
  });

  it.each([
    null, [], [1], [paperKey(1), paperKey(1)],
    Array.from({ length: PERSONAL_LIBRARY_MAX_REPRESENTATIVES + 1 }, (_, index) => paperKey(index + 1)),
  ].map((representativePaperKeys) => ({ representativePaperKeys })))(
    "rejects malformed, empty, duplicate or excessive representatives $representativePaperKeys",
    ({ representativePaperKeys }) => {
      const value = organization();
      expectRejection({ topics: [
        { ...value.topics[0], directions: [{ ...value.topics[0]!.directions[0], representativePaperKeys }] },
        value.topics[1],
      ] }, "representatives-invalid");
    },
  );

  it.each([5, 99])("rejects a representative outside the direction's groups: %i", (index) => {
    const value = organization();
    value.topics[0]!.directions[0]!.representativePaperKeys = [paperKey(index)];
    expectRejection(value, "reference-out-of-scope");
  });

  it("accepts the maximum representatives when they belong to assigned groups", () => {
    const indexes = Array.from({ length: PERSONAL_LIBRARY_MAX_REPRESENTATIVES }, (_, index) => index + 1);
    const value = { topics: [{ suggestedName: "Galaxies", directions: [direction(["group-a"], indexes)] }] };
    expect(decode(value, [group("group-a", indexes)]).ok).toBe(true);
  });

  it("accepts indexed file identities without requiring an arXiv identity", () => {
    const key = `file:sha256:${"a".repeat(64)}`;
    const item = direction(["files"], []);
    item.representativePaperKeys = [key];
    expect(decode({ topics: [{ suggestedName: "Methods", directions: [item] }] }, [
      { id: "files", papers: [{ paperKey: key }] },
    ]).ok).toBe(true);
  });

  it("bounds the full direction member union, not each group or representatives alone", () => {
    const indexes = Array.from({ length: PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS }, (_, index) => index + 1);
    const inputGroups = [group("group-a", indexes.slice(0, 256)), group("group-b", indexes.slice(256)), group("group-c", [9000])];
    const value = organization();
    value.topics[0]!.directions[0]!.representativePaperKeys = [paperKey(1)];
    value.topics[1]!.directions[0]!.representativePaperKeys = [paperKey(9000)];
    expect(decode(value, inputGroups).ok).toBe(true);
    const oversizedGroups = [inputGroups[0]!, group("group-b", [...indexes.slice(256), 8000]), inputGroups[2]!];
    expectRejection(value, "member-count", oversizedGroups);
  });
});
