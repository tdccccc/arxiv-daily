import { describe, it, expect } from "vitest";
import { normalizeTopic } from "../src/settings/topics";

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
