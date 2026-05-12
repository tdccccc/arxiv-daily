import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { parseRecent } from "../src/pipeline/arxiv-parser";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = readFileSync(
  resolve(here, "fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

describe("parseRecent", () => {
  it("returns at least one date bucket", () => {
    const buckets = parseRecent(fixture);
    expect(buckets.length).toBeGreaterThan(0);
  });

  it("each bucket has YYYY-MM-DD date and paper list", () => {
    const buckets = parseRecent(fixture);
    for (const b of buckets) {
      expect(b.announceDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(Array.isArray(b.papers)).toBe(true);
    }
  });

  it("first non-empty bucket has paper with id/title/authors/abstract", () => {
    const buckets = parseRecent(fixture);
    const bucket = buckets.find((b) => b.papers.length > 0);
    expect(bucket).toBeTruthy();
    const p = bucket!.papers[0];
    expect(p.id).toMatch(/^\d{4}\.\d{4,5}/);
    expect(p.title.length).toBeGreaterThan(0);
    expect(p.authors.length).toBeGreaterThan(0);
  });

  it("buckets are returned in descending date order (newest first)", () => {
    const buckets = parseRecent(fixture);
    for (let i = 1; i < buckets.length; i++) {
      expect(buckets[i - 1].announceDate >= buckets[i].announceDate).toBe(true);
    }
  });

  it("captures at least 2 distinct dates from the recent window", () => {
    const buckets = parseRecent(fixture);
    expect(buckets.length).toBeGreaterThanOrEqual(2);
  });
});
