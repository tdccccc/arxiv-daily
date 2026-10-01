import { describe, expect, it } from "vitest";
import {
  normalizePaperDiscoveryProvenance,
  parseDailyReportDiscoveryProvenance,
  parseDiscoveryProvenanceMarker,
  renderDiscoveryProvenanceMarker,
  type PaperDiscoveryProvenance,
} from "../src/pipeline/discovery-provenance-marker";

/**
 * Discovery provenance is a read-only legacy format: the library-profile
 * classifier that wrote these markers retired with the profile document
 * (ADR 0012 / ADR 0014). Reports already committed to vaults still carry them
 * and `paper-index` and the Dashboard still read those reports, so the parse
 * path is pinned here on its own — the daily pipeline can no longer produce a
 * marker for another test to round-trip through.
 */

const reportDate = "2026-07-22";
const arxivId = "2607.00020";

const provenance: PaperDiscoveryProvenance = {
  manualTopicTags: ["methods"],
  directions: [{
    id: "direction.001",
    name: "Direction one",
    representatives: [{
      paperKey: "arxiv:2501.00001",
      title: "Representative one",
      evidenceDepth: "metadata-and-abstract",
    }],
  }],
};

function encodeBase64Url(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function markerFor(payload: unknown): string {
  return `<!-- arxiv-daily-discovery-provenance:v1:${encodeBase64Url(JSON.stringify(payload))} -->`;
}

function legacyReport(marker: string): string {
  // The shape the retired path used to commit: a library-only section whose
  // heading no parser ever read, and a paper block whose marker sits in the
  // canonical slot directly under the heading.
  return [
    "# Daily report",
    "",
    "## Library-guided discoveries",
    "",
    `### Some Paper`,
    marker,
    "",
    `- **arXiv**: [${arxivId}](https://arxiv.org/abs/${arxivId})`,
    "",
  ].join("\n");
}

describe("discovery provenance marker (legacy read path)", () => {
  it("reads back a marker committed by the retired path", () => {
    const marker = renderDiscoveryProvenanceMarker(provenance, arxivId, reportDate);
    expect(parseDiscoveryProvenanceMarker(marker)).toEqual({
      v: 1, d: reportDate, id: arxivId, p: provenance,
    });
    expect(parseDailyReportDiscoveryProvenance(legacyReport(marker), reportDate)).toEqual({
      kind: "valid",
      occurrences: [{ arxivId, provenance }],
    });
  });

  it("rejects a payload that normalizes to the same provenance but was not written canonically", () => {
    // Same information, different serialization: key order inside the payload
    // is flipped and a direction carries its keys out of order. The bounds
    // checks all pass and `normalizePaperDiscoveryProvenance` rebuilds an
    // identical value, so only re-rendering the accepted payload and comparing
    // it to the line can tell the two apart.
    const forged = markerFor({
      p: {
        directions: [{
          representatives: [{
            evidenceDepth: "metadata-and-abstract",
            title: "Representative one",
            paperKey: "arxiv:2501.00001",
          }],
          name: "Direction one",
          id: "direction.001",
        }],
        manualTopicTags: ["methods"],
      },
      id: arxivId,
      d: reportDate,
      v: 1,
    });
    expect(normalizePaperDiscoveryProvenance(JSON.parse(
      atob(forged.slice("<!-- arxiv-daily-discovery-provenance:v1:".length, -" -->".length)
        .replace(/-/g, "+").replace(/_/g, "/")
        + "=".repeat((4 - (forged.slice("<!-- arxiv-daily-discovery-provenance:v1:".length, -" -->".length).length % 4)) % 4)),
    ).p)).toEqual(provenance);
    expect(parseDiscoveryProvenanceMarker(forged)).toBeNull();
    expect(parseDailyReportDiscoveryProvenance(legacyReport(forged), reportDate)).toEqual({
      kind: "invalid",
      reason: "provenance marker is malformed",
    });
  });

  it("rejects a marker whose identity does not match its report occurrence", () => {
    const otherPaper = renderDiscoveryProvenanceMarker(provenance, "2607.00099", reportDate);
    expect(parseDailyReportDiscoveryProvenance(legacyReport(otherPaper), reportDate)).toEqual({
      kind: "invalid",
      reason: "provenance marker identity does not match its report occurrence",
    });
  });

  it("reads a report with no markers at all as valid and empty", () => {
    expect(parseDailyReportDiscoveryProvenance("# Daily report\n\n## Methods\n", reportDate))
      .toEqual({ kind: "valid", occurrences: [] });
  });
});
