import { describe, it, expect } from "vitest";
import { TOPIC_TEMPLATES } from "../src/settings/topic-templates";
import { slugify } from "../src/utils/slugify";

describe("TOPIC_TEMPLATES", () => {
  it("includes a Blank template with no topics", () => {
    const blank = TOPIC_TEMPLATES.find((t) => t.id === "blank");
    expect(blank).toBeDefined();
    expect(blank!.topics).toEqual([]);
  });

  it("every template has a unique id", () => {
    const ids = TOPIC_TEMPLATES.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("every template has a non-empty arXiv category", () => {
    for (const t of TOPIC_TEMPLATES) {
      expect(t.category.length).toBeGreaterThan(0);
    }
  });

  it("every non-blank template has at least one topic", () => {
    for (const t of TOPIC_TEMPLATES) {
      if (t.id === "blank") continue;
      expect(t.topics.length).toBeGreaterThan(0);
    }
  });

  it("topic tags within a template are unique and slug-shaped", () => {
    for (const t of TOPIC_TEMPLATES) {
      const tags = t.topics.map((x) => x.tag);
      expect(new Set(tags).size).toBe(tags.length);
      for (const tag of tags) {
        expect(tag).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
        expect(slugify(tag)).toBe(tag);
      }
    }
  });

  it("every topic has a non-empty name and description", () => {
    for (const t of TOPIC_TEMPLATES) {
      for (const topic of t.topics) {
        expect(topic.name.length).toBeGreaterThan(0);
        expect(topic.description.length).toBeGreaterThan(0);
      }
    }
  });
});
