import { describe, it, expect } from "vitest";
import { normalizeTopic, topicFromSeed } from "../src/settings/topics";
import { TOPIC_TEMPLATES } from "../src/settings/topic-templates";

/**
 * The shadow invariant (ADR 0012, P1): `directions` is the authority and
 * `description` is derived from it, so an older build reading the same
 * data.json keeps working. Every write path must funnel through this one
 * function — two truth sources syncing themselves is the failure mode this
 * design is guarding against.
 */
describe("normalizeTopic — shadow invariant", () => {
  const base = { id: "t1", name: "Photo-z", tag: "photo-z", detail: true };

  it("derives description from the first direction", () => {
    const out = normalizeTopic({
      ...base,
      directions: [
        { id: "d1", text: "Photometric redshift methods.", origin: "manual" },
        { id: "d2", text: "Catalog cross-matching.", origin: "library" },
      ],
    });
    expect(out.description).toBe("Photometric redshift methods.");
  });

  it("empties description once the last direction is gone", () => {
    // The settings page deletes a direction by handing back the remaining
    // list. An explicitly empty list must not be resurrected from the stale
    // shadow, or the last direction could never be deleted.
    const out = normalizeTopic({ ...base, description: "stale text", directions: [] });
    expect(out.directions).toEqual([]);
    expect(out.description).toBe("");
  });

  it("migrates description only when the directions key is absent entirely", () => {
    const legacy = normalizeTopic({ ...base, description: "Written before directions existed." });
    expect(legacy.directions).toHaveLength(1);

    const current = normalizeTopic({ ...base, description: "Derived shadow.", directions: [] });
    expect(current.directions).toEqual([]);
  });

  it("lets directions win over a conflicting stored description", () => {
    const out = normalizeTopic({
      ...base,
      description: "what an older build last wrote",
      directions: [{ id: "d1", text: "what the user actually means", origin: "manual" }],
    });
    expect(out.description).toBe("what the user actually means");
    expect(out.directions).toHaveLength(1);
  });

  it("drops blank direction text instead of keeping an empty line", () => {
    const out = normalizeTopic({
      ...base,
      directions: [
        { id: "d1", text: "   ", origin: "manual" },
        { id: "d2", text: "Real direction.", origin: "manual" },
      ],
    });
    expect(out.directions).toHaveLength(1);
    expect(out.directions[0].text).toBe("Real direction.");
    expect(out.description).toBe("Real direction.");
  });

  it("keeps a direction id that already exists", () => {
    const out = normalizeTopic({
      ...base,
      directions: [{ id: "d-keep", text: "Stable.", origin: "library" }],
    });
    expect(out.directions[0].id).toBe("d-keep");
    expect(out.directions[0].origin).toBe("library");
  });

  it("assigns an id to a direction that lacks one", () => {
    const out = normalizeTopic({
      ...base,
      directions: [{ text: "No id here.", origin: "manual" }],
    });
    expect(out.directions[0].id.length).toBeGreaterThan(0);
  });

  it("treats a direction with an unknown origin as manual", () => {
    const out = normalizeTopic({
      ...base,
      directions: [{ id: "d1", text: "Typed by hand.", origin: "nonsense" }],
    });
    expect(out.directions[0].origin).toBe("manual");
  });

  it("still migrates a legacy topic that only has a description", () => {
    const out = normalizeTopic({ ...base, description: "Legacy interest." });
    expect(out.directions).toHaveLength(1);
    expect(out.directions[0].text).toBe("Legacy interest.");
    expect(out.directions[0].origin).toBe("migrated");
    expect(out.description).toBe("Legacy interest.");
  });

  it("yields a usable blank topic from nothing at all", () => {
    const out = normalizeTopic({});
    expect(out.directions).toEqual([]);
    expect(out.description).toBe("");
    expect(out.id.length).toBeGreaterThan(0);
  });

  it("ignores a directions value that is not an array", () => {
    const out = normalizeTopic({ ...base, description: "Legacy.", directions: "nope" });
    expect(out.directions).toHaveLength(1);
    expect(out.directions[0].text).toBe("Legacy.");
  });
});

describe("topicFromSeed — quick-start templates", () => {
  it("authors a template topic's directions rather than migrating them", () => {
    const template = TOPIC_TEMPLATES.find((t) => t.id === "astro-ml")!;
    const topics = template.topics.map(topicFromSeed);

    expect(topics).toHaveLength(3);
    for (const topic of topics) {
      // "migrated" would be a lie: nothing was carried over from an older
      // format, the user picked a template.
      expect(topic.directions.every((d) => d.origin === "manual")).toBe(true);
      expect(topic.description).toBe(topic.directions[0].text);
    }
  });

  it("keeps every template topic to a single direction for now", () => {
    // Until filtering reads the whole list (P3), only the first direction
    // reaches the classifier through the description shadow. More than one
    // here would silently narrow what a new user matches.
    for (const template of TOPIC_TEMPLATES) {
      for (const seed of template.topics) {
        expect(topicFromSeed(seed).directions).toHaveLength(1);
      }
    }
  });

  it("gives every template topic a name, a tag and a direction", () => {
    for (const template of TOPIC_TEMPLATES) {
      for (const seed of template.topics) {
        const topic = topicFromSeed(seed);
        expect(topic.name.trim().length).toBeGreaterThan(0);
        expect(topic.tag.trim().length).toBeGreaterThan(0);
        expect(topic.directions[0].text.trim().length).toBeGreaterThan(0);
      }
    }
  });

  it("gives each applied template topic its own identity", () => {
    const template = TOPIC_TEMPLATES.find((t) => t.id === "nlp")!;
    const topicIds = template.topics.map((seed) => topicFromSeed(seed).id);
    const directionIds = template.topics.flatMap(
      (seed) => topicFromSeed(seed).directions.map((d) => d.id),
    );
    expect(new Set(topicIds).size).toBe(topicIds.length);
    expect(new Set(directionIds).size).toBe(directionIds.length);
  });
});
