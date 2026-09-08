import { describe, expect, it } from "vitest";
import {
  parseDailyReportTopicDirections,
  parseTopicDirectionMarker,
  renderTopicDirectionMarker,
} from "../src/pipeline/topic-direction-marker";
import { parseDailyReportDiscoveryProvenance } from "../src/pipeline/discovery-provenance-marker";
import { normalizeTopicDirectionHits } from "../src/pipeline/topic-direction-hits";

const reportDate = "2026-09-04";
const arxivId = "2609.00001";
const hits = [
  { tag: "photo-z", id: "d1", text: "photo-z methods" },
  { tag: "photo-z", id: "d2", text: "photo-z catalog comparisons" },
];

function block(markers: string[], id = arxivId): string {
  return [
    `### A paper`,
    ...markers,
    `> 信息来源： Abstract`,
    `- **作者**: A. Author`,
    `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
  ].join("\n");
}

describe("topic direction marker", () => {
  it("round-trips the hits it was given, in order", () => {
    const marker = renderTopicDirectionMarker(hits, arxivId, reportDate);

    expect(parseTopicDirectionMarker(marker)).toEqual({
      v: 1, d: reportDate, id: arxivId, h: hits,
    });
  });

  it("refuses to render hits it would refuse to read back", () => {
    expect(() => renderTopicDirectionMarker([], arxivId, reportDate)).toThrow(/malformed/);
    expect(() => renderTopicDirectionMarker(
      [hits[0]!, { tag: "other", id: "d3", text: "x" }], arxivId, reportDate,
    )).toThrow(/malformed/);
    expect(normalizeTopicDirectionHits([{ tag: "t", id: "d", text: " " }])).not.toBeNull();
    expect(normalizeTopicDirectionHits([{ tag: "t", id: "d", text: "" }])).toBeNull();
  });

  it("reads the hits back out of a rendered report block", () => {
    const markdown = block([renderTopicDirectionMarker(hits, arxivId, reportDate)]);

    expect(parseDailyReportTopicDirections(markdown, reportDate)).toEqual({
      kind: "valid",
      occurrences: [{ arxivId, hits }],
    });
  });

  it.each(["[!info]- 命中方向与信息来源", "[!info]+ Matched directions and sources"])(
    "reads a marker inside the canonical metadata callout %s",
    (heading) => {
      const markdown = block([
        `> ${heading}`,
        `> ${renderTopicDirectionMarker(hits, arxivId, reportDate)}`,
        "> - photo-z methods",
        "> - photo-z catalog comparisons",
        ">",
      ]);

      expect(parseDailyReportTopicDirections(markdown, reportDate)).toEqual({
        kind: "valid", occurrences: [{ arxivId, hits }],
      });
    },
  );

  it.each([
    ["an ordinary quote", "> Metadata"],
    ["a callout later in the paper", "> Information\n> [!info]- Sources"],
  ])("rejects a quoted marker inside %s", (_name, heading) => {
    const markdown = block([
      heading,
      `> ${renderTopicDirectionMarker(hits, arxivId, reportDate)}`,
    ]);

    expect(parseDailyReportTopicDirections(markdown, reportDate))
      .toEqual({ kind: "invalid", reason: "topic direction marker placement is invalid" });
  });

  it("rejects duplicate markers across old and folded placements", () => {
    const marker = renderTopicDirectionMarker(hits, arxivId, reportDate);
    const markdown = block([marker, "> [!info]- Sources", `> ${marker}`]);

    expect(parseDailyReportTopicDirections(markdown, reportDate))
      .toEqual({ kind: "invalid", reason: "topic direction marker count is invalid" });
  });

  it("still validates the identity of a folded marker", () => {
    const other = renderTopicDirectionMarker(hits, "2609.00002", reportDate);
    const markdown = block(["> [!info]- Sources", `> ${other}`]);

    expect(parseDailyReportTopicDirections(markdown, reportDate))
      .toEqual({ kind: "invalid", reason: "topic direction marker identity does not match its report occurrence" });
  });

  it("rejects a marker whose paper or date is not the one it sits with", () => {
    const other = renderTopicDirectionMarker(hits, "2609.00002", reportDate);

    expect(parseDailyReportTopicDirections(block([other]), reportDate))
      .toMatchObject({ kind: "invalid" });
    expect(parseDailyReportTopicDirections(
      block([renderTopicDirectionMarker(hits, arxivId, "2026-09-03")]), reportDate,
    )).toMatchObject({ kind: "invalid" });
  });

  /**
   * Inside a block but not in its canonical slot. Without this, untrusted
   * summary text that happens to render a marker line further down the block
   * would be read as authoritative — the reason the other two marker families
   * pin a slot too.
   */
  it("rejects a marker sitting anywhere but its canonical slot in the block", () => {
    const marker = renderTopicDirectionMarker(hits, arxivId, reportDate);
    const trailing = [
      `### A paper`,
      `> 信息来源： Abstract`,
      `- **作者**: A. Author`,
      `- **arXiv**: [${arxivId}](https://arxiv.org/abs/${arxivId})`,
      marker,
    ].join("\n");

    expect(parseDailyReportTopicDirections(trailing, reportDate))
      .toEqual({ kind: "invalid", reason: "topic direction marker placement is invalid" });
  });

  it("rejects a marker parked outside a paper block", () => {
    const stray = `${renderTopicDirectionMarker(hits, arxivId, reportDate)}\n\n${block([])}`;

    expect(parseDailyReportTopicDirections(stray, reportDate)).toMatchObject({ kind: "invalid" });
  });

  /**
   * The marker family is appended after discovery provenance and personal
   * novelty precisely so their canonical-slot rules keep holding. A report
   * written before topic directions existed carries neither of the new lines
   * and must still parse exactly as it did.
   */
  it("leaves a report that predates it parsing unchanged", () => {
    const legacy = block([]);

    expect(parseDailyReportTopicDirections(legacy, reportDate))
      .toEqual({ kind: "valid", occurrences: [] });
    expect(parseDailyReportDiscoveryProvenance(legacy, reportDate))
      .toEqual({ kind: "valid", occurrences: [] });
  });
});
