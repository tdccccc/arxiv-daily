import { markupParser } from "./markup-parser";
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { parseRecent, parseRecentListing } from "../src/pipeline/arxiv-parser";
import { listingIds, recentListingWorld } from "./recent-listing-world";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = readFileSync(
  resolve(here, "fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

describe("parseRecent", () => {
  it("returns at least one date bucket", () => {
    const buckets = parseRecent(fixture, markupParser);
    expect(buckets.length).toBeGreaterThan(0);
  });

  it("each bucket has YYYY-MM-DD date and paper list", () => {
    const buckets = parseRecent(fixture, markupParser);
    for (const b of buckets) {
      expect(b.announceDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(Array.isArray(b.papers)).toBe(true);
    }
  });

  it("first non-empty bucket has paper with id/title/authors/abstract", () => {
    const buckets = parseRecent(fixture, markupParser);
    const bucket = buckets.find((b) => b.papers.length > 0);
    expect(bucket).toBeTruthy();
    const p = bucket!.papers[0];
    expect(p.id).toMatch(/^\d{4}\.\d{4,5}/);
    expect(p.title.length).toBeGreaterThan(0);
    expect(p.authors.length).toBeGreaterThan(0);
  });

  it("buckets are returned in descending date order (newest first)", () => {
    const buckets = parseRecent(fixture, markupParser);
    for (let i = 1; i < buckets.length; i++) {
      expect(buckets[i - 1].announceDate >= buckets[i].announceDate).toBe(true);
    }
  });

  it("captures at least 2 distinct dates from the recent window", () => {
    const buckets = parseRecent(fixture, markupParser);
    expect(buckets.length).toBeGreaterThanOrEqual(2);
  });

  it("parses abbreviated month names from arXiv headers", () => {
    const html = `
      <dl id="articles">
        <h3>Wed, 10 Jun 2026 (showing 1 of 1 entries )</h3>
        <dt><a title="Abstract" href="/abs/2606.11165">arXiv:2606.11165</a></dt>
        <dd>
          <div class="list-title">Title: Example Paper</div>
          <div class="list-authors"><a>Jane Doe</a></div>
        </dd>
      </dl>
    `;
    const buckets = parseRecent(html, markupParser);
    expect(buckets).toHaveLength(1);
    expect(buckets[0].announceDate).toBe("2026-06-10");
    expect(buckets[0].papers[0].id).toBe("2606.11165");
  });

  it("rejects malformed arXiv ids from listing HTML before exposing papers", () => {
    const paper = (id: string) => `
      <dt><a title="Abstract" href="/abs/${id}">arXiv:${id}</a></dt>
      <dd>
        <div class="list-title">Title: ${id}</div>
        <div class="list-authors"><a>Jane Doe</a></div>
      </dd>
    `;
    const html = `
      <dl id="articles">
        <h3>Wed, 10 Jun 2026 (showing 5 of 5 entries )</h3>
        ${paper("../../../etc/passwd")}
        ${paper("foo bar")}
        ${paper("")}
        ${paper("2606.12345")}
        ${paper("2606.12345v2")}
      </dl>
    `;

    const buckets = parseRecent(html, markupParser);

    expect(buckets).toHaveLength(1);
    expect(buckets[0].papers.map((p) => p.id)).toEqual([
      "2606.12345",
      "2606.12345v2",
    ]);
  });

  it("keeps script and style text out of parsed fields whatever their end tags look like", () => {
    const html = `
      <script>should not run</script >
      <style>.x { color: red } </style  >
      <dl id="articles">
        <h3>Wed, 10 Jun 2026 (showing 1 of 1 entries )</h3>
        <dt><a title="Abstract" href="/abs/2606.11165">arXiv:2606.11165</a></dt>
        <dd>
          <div class="list-title">Title: Example <script>leak()</script\t\n bar>Paper</div>
          <div class="list-authors">Authors: Jane Doe<style>.y{}</style x><link rel="stylesheet" href="a.css"></div>
        </dd>
      </dl>
    `;
    const buckets = parseRecent(html, markupParser);
    expect(buckets).toHaveLength(1);
    expect(buckets[0].papers[0].id).toBe("2606.11165");
    expect(buckets[0].papers[0].title).toBe("Example Paper");
    expect(buckets[0].papers[0].authors).toBe("Jane Doe");
  });

  it("drops an unterminated script instead of reading it as text", () => {
    const html = `
      <dl id="articles">
        <h3>Wed, 10 Jun 2026 (showing 1 of 1 entries )</h3>
        <dt><a title="Abstract" href="/abs/2606.11165">arXiv:2606.11165</a></dt>
        <dd>
          <div class="list-title">Title: Example Paper</div>
          <div class="list-authors">Authors: Jane Doe</div>
        </dd>
      </dl>
      <script>never closed
    `;
    const buckets = parseRecent(html, markupParser);
    expect(buckets[0].papers[0].title).toBe("Example Paper");
    expect(JSON.stringify(buckets)).not.toContain("never closed");
  });
});

describe("parseRecentListing", () => {
  it("reads the total, the day links with their offsets and what each day declares", () => {
    const listing = parseRecentListing(fixture, markupParser);

    expect(listing.totalEntries).toBe(403);
    expect(listing.days).toEqual([
      { announceDate: "2026-05-11", skip: 0 },
      { announceDate: "2026-05-08", skip: 59 },
      { announceDate: "2026-05-07", skip: 144 },
      { announceDate: "2026-05-06", skip: 221 },
      { announceDate: "2026-05-05", skip: 296 },
    ]);
    expect(listing.buckets.map((b) => [b.announceDate, b.declaredEntries, b.listedEntries])).toEqual([
      ["2026-05-11", 59, 59],
      ["2026-05-08", 85, 85],
      ["2026-05-07", 77, 77],
      ["2026-05-06", 75, 75],
      ["2026-05-05", 107, 107],
    ]);
  });

  it("is what parseRecent returns the buckets of", () => {
    expect(parseRecent(fixture, markupParser)).toEqual(parseRecentListing(fixture, markupParser).buckets);
  });

  it("shows a day cut by the page as listing fewer entries than it declares", () => {
    const world = recentListingWorld([
      { date: "2026-05-11", ids: listingIds("2605", 10000, 5) },
      { date: "2026-05-08", ids: listingIds("2605", 20000, 4) },
    ]);
    const listing = parseRecentListing(world.page(0, 7), markupParser);

    const cut = listing.buckets.find((b) => b.announceDate === "2026-05-08")!;
    expect(cut.listedEntries).toBe(2);
    expect(cut.declaredEntries).toBe(4);
    expect(cut.papers.map((paper) => paper.id)).toEqual(["2605.20000", "2605.20001"]);
    expect(listing.days.map((day) => day.skip)).toEqual([0, 5]);
  });

  it("counts a listed entry the parser cannot use, so a day is not mistaken for cut", () => {
    const html = `
      <dl id="articles">
        <h3>Wed, 10 Jun 2026 (showing 2 of 2 entries )</h3>
        <dt><a title="Abstract" href="/abs/2606.11165">arXiv:2606.11165</a></dt>
        <dd><div class="list-title">Title: Example</div><div class="list-authors"><a>Jane Doe</a></div></dd>
        <dt><a title="Abstract" href="/abs/bogus">arXiv:bogus</a></dt>
        <dd><div class="list-title">Title: Odd one</div></dd>
      </dl>`;
    const [bucket] = parseRecentListing(html, markupParser).buckets;

    expect(bucket!.papers).toHaveLength(1);
    expect(bucket!.listedEntries).toBe(2);
    expect(bucket!.declaredEntries).toBe(2);
  });

  it("works on a minimal page that has no total, no day links and no counts", () => {
    const html = `<dl id="articles"><h3>Wed, 10 Jun 2026</h3>
      <dt><a title="Abstract" href="/abs/2606.11165">arXiv:2606.11165</a></dt>
      <dd><div class="list-title">Title: Example</div></dd></dl>`;
    const listing = parseRecentListing(html, markupParser);

    expect(listing.totalEntries).toBeUndefined();
    expect(listing.days).toEqual([]);
    expect(listing.buckets[0]!.declaredEntries).toBeUndefined();
    expect(listing.buckets[0]!.papers).toHaveLength(1);
  });

  it("ignores links that are not day links", () => {
    const html = `<ul><li><a href="/list/cs.LG/recent?skip=0&amp;show=1000">fewer</a></li>
      <li><a href="/list/cs.LG/new">Mon, 11 May 2026</a></li>
      <li><a href="/list/cs.LG/recent?skip=12&amp;show=2000">Fri, 8 May 2026</a></li></ul>`;

    expect(parseRecentListing(html, markupParser).days).toEqual([{ announceDate: "2026-05-08", skip: 12 }]);
  });
});
