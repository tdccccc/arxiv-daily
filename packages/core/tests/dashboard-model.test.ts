import { describe, expect, it } from "vitest";
import {
  projectDashboardOccurrenceProvenance,
  queryDashboard,
  type PaperDiscoveryProvenance,
  type PaperIndexEntry,
} from "../src/index";

function provenance(
  manualTopicTags: string[] = [],
  directionNames: string[] = [],
): PaperDiscoveryProvenance {
  return {
    manualTopicTags,
    directions: directionNames.map((name, index) => ({
      id: `direction-${index + 1}`,
      name,
      representatives: [{
        paperKey: `arxiv:2501.0000${index + 1}`,
        title: `Prior paper ${index + 1}`,
        evidenceDepth: "metadata-and-abstract",
      }],
    })),
  };
}

function paper(overrides: Partial<PaperIndexEntry> = {}): PaperIndexEntry {
  return {
    paperKey: "arxiv:2608.00001",
    source: "arxiv",
    externalId: "2608.00001",
    arxivId: "2608.00001",
    title: "Dashboard paper",
    authors: ["A. Author"],
    published: "2026-08-01",
    updated: "2026-08-01",
    category: "cs.AI",
    categories: ["cs.AI"],
    abstract: "An abstract with transformers.",
    topics: ["rag"],
    primaryTopic: "rag",
    detail: false,
    status: "inbox",
    priority: "normal",
    seenDates: ["2026-08-01"],
    dailyReports: [],
    discoveryProvenanceByReport: {},
    paperPath: null,
    arxivUrl: "https://arxiv.org/abs/2608.00001",
    pdfUrl: "https://arxiv.org/pdf/2608.00001",
    pdfPath: "",
    zoteroKey: "",
    zoteroUri: "",
    citationKey: "",
    projects: [],
    ...overrides,
  };
}

describe("Dashboard occurrence provenance", () => {
  it("projects fresh manual, library, and both-source occurrences with resolvable topic names", () => {
    const reports = ["arxiv-daily/daily/2026-08-01.md"];
    const options = { topics: [{ tag: "rag", name: "Retrieval augmented generation" }] };
    const manual = queryDashboard([paper({
      dailyReports: reports,
      discoveryProvenanceByReport: { [reports[0]!]: provenance(["rag"]) },
    })], { tab: "all" }, options).rows[0]!.occurrenceProvenance;
    const library = queryDashboard([paper({
      dailyReports: reports,
      discoveryProvenanceByReport: { [reports[0]!]: provenance([], ["Efficient retrieval"]) },
    })], { tab: "all" }, options).rows[0]!.occurrenceProvenance;
    const both = queryDashboard([paper({
      dailyReports: reports,
      discoveryProvenanceByReport: {
        [reports[0]!]: provenance(["rag"], ["Efficient retrieval", "Robust evaluation"]),
      },
    })], { tab: "all" }, options).rows[0]!.occurrenceProvenance;

    expect(manual).toMatchObject({
      source: "manual",
      manualTopics: [{ tag: "rag", name: "Retrieval augmented generation" }],
      directions: [],
    });
    expect(library).toMatchObject({
      source: "library",
      evidenceDepth: "metadata-and-abstract",
      directions: [{ name: "Efficient retrieval" }],
    });
    expect(both).toMatchObject({
      source: "both",
      manualTopics: [{ tag: "rag", name: "Retrieval augmented generation" }],
      evidenceDepth: "metadata-and-abstract",
    });
    expect(both?.directions.map(({ name }) => name)).toEqual([
      "Efficient retrieval",
      "Robust evaluation",
    ]);
    expect(both?.directions[1]?.representatives[0]).toEqual({
      paperKey: "arxiv:2501.00002",
      title: "Prior paper 2",
      evidenceDepth: "metadata-and-abstract",
    });
  });

  it("chooses the latest committed dated occurrence independent of object insertion order", () => {
    const oldReport = "arxiv-daily/daily/2026-08-01.md";
    const latestA = "archive/a/2026-08-03.md";
    const latestZ = "archive/z/2026-08-03.md";
    const uncommitted = "arxiv-daily/daily/2026-08-04.md";
    const entry = paper({
      dailyReports: [latestA, oldReport, latestZ],
      discoveryProvenanceByReport: {
        [uncommitted]: provenance([], ["Uncommitted"]),
        [latestA]: provenance([], ["Repeated A"]),
        [oldReport]: provenance(["rag"]),
        [latestZ]: provenance([], ["Repeated Z"]),
      },
    });

    expect(projectDashboardOccurrenceProvenance(entry)).toMatchObject({
      occurrenceProvenance: {
        reportPath: latestZ,
        reportDate: "2026-08-03",
        directions: [{ name: "Repeated Z" }],
      },
    });
    const reversed = paper({
      dailyReports: [...entry.dailyReports].reverse(),
      discoveryProvenanceByReport: Object.fromEntries(
        Object.entries(entry.discoveryProvenanceByReport).reverse(),
      ),
    });
    expect(projectDashboardOccurrenceProvenance(reversed)).toEqual(
      projectDashboardOccurrenceProvenance(entry),
    );
  });

  it("omits metadata for legacy entries, malformed report dates, and provenance not tied to a committed report", () => {
    expect(queryDashboard([paper()], { tab: "all" }).rows[0]!.occurrenceProvenance)
      .toBeUndefined();
    expect(projectDashboardOccurrenceProvenance(paper({
      dailyReports: ["arxiv-daily/daily/latest.md"],
      discoveryProvenanceByReport: {
        "arxiv-daily/daily/latest.md": provenance(["rag"]),
        "arxiv-daily/daily/2026-08-01.md": provenance([], ["Not committed"]),
      },
    }))).toEqual({});
  });

  it("keeps durable occurrence provenance separate from query-time match reasons", () => {
    const report = "arxiv-daily/daily/2026-08-01.md";
    const entry = paper({
      dailyReports: [report],
      discoveryProvenanceByReport: { [report]: provenance([], ["Transformer systems"]) },
    });
    const withoutSearch = queryDashboard([entry], { tab: "all" }).rows[0]!;
    const withSearch = queryDashboard([entry], { tab: "all", search: "transformers" }).rows[0]!;

    expect(withoutSearch.occurrenceProvenance?.directions[0]?.name).toBe("Transformer systems");
    expect(withoutSearch.matchReasons).toBeUndefined();
    expect(withSearch.occurrenceProvenance).toEqual(withoutSearch.occurrenceProvenance);
    expect(withSearch.matchReasons?.length).toBeGreaterThan(0);
  });
});
