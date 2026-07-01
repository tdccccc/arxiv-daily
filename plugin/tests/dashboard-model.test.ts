import { describe, expect, it } from "vitest";
import {
  planDashboardAction,
  queryDashboard,
} from "../src/dashboard/model";
import type { PaperIndexEntry } from "../src/services/paper-index";

function paper(
  arxivId: string,
  overrides: Partial<PaperIndexEntry> = {},
): PaperIndexEntry {
  return {
    arxivId,
    source: "arxiv",
    title: `Paper ${arxivId}`,
    authors: ["A. Author"],
    published: "2026-06-10",
    updated: "2026-06-10",
    category: "astro-ph",
    categories: ["astro-ph"],
    topics: ["photo-z"],
    primaryTopic: "photo-z",
    detail: false,
    status: "inbox",
    priority: "normal",
    seenDates: ["2026-06-10"],
    dailyReports: ["arxiv-daily/daily/2026-06-10.md"],
    paperPath: null,
    arxivUrl: `https://arxiv.org/abs/${arxivId}`,
    pdfUrl: `https://arxiv.org/pdf/${arxivId}`,
    pdfPath: "",
    zoteroKey: "",
    zoteroUri: "",
    citationKey: "",
    projects: [],
    ...overrides,
  };
}

function fixtures(): PaperIndexEntry[] {
  return [
    paper("2606.00001", {
      title: "Alpha Lensing Calibration",
      authors: ["A. Alpha", "B. Beta"],
      status: "to_read",
      priority: "normal",
      seenDates: ["2026-06-10"],
      summary: {
        coreProblem: "Weak lensing calibration for photometric redshifts.",
      },
    }),
    paper("2606.00002", {
      title: "Beta Transformer Photo-z",
      authors: ["C. Gamma"],
      published: "2026-06-09",
      status: "to_read",
      priority: "high",
      seenDates: ["2026-06-11"],
      summary: {
        keyMethod: "Contrastive transformer model.",
      },
    }),
    paper("2606.00003", {
      title: "Cluster Catalog",
      status: "saved",
      priority: "normal",
      topics: ["cluster"],
      primaryTopic: "cluster",
      seenDates: ["2026-06-12"],
      paperPath: "arxiv-daily/papers/2606.00003.md",
    }),
    paper("2606.00004", {
      title: "Ignored High Priority",
      status: "ignored",
      priority: "high",
      seenDates: ["2026-06-08"],
    }),
    paper("2606.00005", {
      title: "Reading Survey",
      authors: ["D. Delta"],
      status: "reading",
      priority: "low",
      topics: ["survey"],
      primaryTopic: "survey",
      seenDates: ["2026-06-13"],
    }),
  ];
}

describe("dashboard model", () => {
  it("uses the starred tab by default and counts simplified tabs", () => {
    const result = queryDashboard(fixtures());

    expect(result.rows.map((row) => row.arxivId)).toEqual(["2606.00002"]);
    expect(result.tabCounts).toEqual({
      starred: 1,
      all: 4,
    });
  });

  it("searches title, authors, topics, and structured summaries", () => {
    const bySummary = queryDashboard(fixtures(), {
      tab: "all",
      search: "calibration redshifts",
    });
    const byTitleAndPriority = queryDashboard(fixtures(), {
      tab: "all",
      search: "transformer",
      topics: ["photo-z"],
      priorities: ["high"],
    });

    expect(bySummary.rows.map((row) => row.arxivId)).toEqual(["2606.00001"]);
    expect(byTitleAndPriority.rows.map((row) => row.arxivId)).toEqual([
      "2606.00002",
    ]);
  });

  it("filters by date range and detail summary existence", () => {
    const seenRange = queryDashboard(fixtures(), {
      tab: "all",
      dateFrom: "2026-06-12",
      dateTo: "2026-06-13",
    });
    const withDetailSummary = queryDashboard(
      fixtures(),
      {
        tab: "all",
        detailSummary: true,
      },
      { detailSummaryIds: new Set(["2606.00003"]) },
    );

    expect(seenRange.rows.map((row) => row.arxivId)).toEqual([
      "2606.00003",
      "2606.00005",
    ]);
    expect(withDetailSummary.rows.map((row) => row.arxivId)).toEqual([
      "2606.00003",
    ]);
    expect(withDetailSummary.rows[0].hasDetailSummary).toBe(true);
  });

  it("builds filtered summary stats", () => {
    const result = queryDashboard(
      fixtures(),
      { tab: "all" },
      { now: new Date("2026-06-13T12:00:00") },
    );

    expect(result.stats.total).toBe(4);
    expect(result.stats.topicCounts).toEqual({
      "photo-z": 2,
      cluster: 1,
      survey: 1,
    });
    expect(result.stats.statusCounts.saved).toBe(1);
    expect(result.stats.priorityCounts.high).toBe(1);
    expect(result.stats.weekAdded).toBe(4);
    expect(result.stats.starred).toBe(1);
  });

  it("sorts with explicit keys and directions", () => {
    const byPublishedAsc = queryDashboard(fixtures(), {
      tab: "all",
      sort: { key: "published", direction: "asc" },
    });
    const byPublishedDesc = queryDashboard(fixtures(), {
      tab: "all",
      sort: { key: "published", direction: "desc" },
    });

    expect(byPublishedAsc.rows[0].arxivId).toBe("2606.00002");
    expect(byPublishedDesc.rows[0].arxivId).not.toBe("2606.00002");
  });

  it("sorts by published, topic, and title", () => {
    const byPublishedDesc = queryDashboard(fixtures(), {
      tab: "all",
      sort: { key: "published", direction: "desc" },
    });
    const byTopicAsc = queryDashboard(fixtures(), {
      tab: "all",
      sort: { key: "topic", direction: "asc" },
    });
    const byTitleAsc = queryDashboard(fixtures(), {
      tab: "all",
      sort: { key: "title", direction: "asc" },
    });

    expect(byPublishedDesc.rows.map((row) => row.arxivId)).toEqual([
      "2606.00005",
      "2606.00003",
      "2606.00002",
      "2606.00001",
    ]);
    expect(byTopicAsc.rows.map((row) => row.arxivId)).toEqual([
      "2606.00003",
      "2606.00002",
      "2606.00001",
      "2606.00005",
    ]);
    expect(byTitleAsc.rows.map((row) => row.arxivId)).toEqual([
      "2606.00001",
      "2606.00002",
      "2606.00003",
      "2606.00005",
    ]);
  });

  it("plans host-neutral dashboard actions without side effects", () => {
    const entries = fixtures();
    const statusPlan = planDashboardAction(entries, {
      type: "set_status",
      arxivIds: ["2606.00001", "2606.00003", "missing", "2606.00001"],
      status: "saved",
    });
    const priorityPlan = planDashboardAction(entries, {
      type: "set_priority",
      arxivIds: ["2606.00001", "2606.00002"],
      priority: "high",
    });
    const markPlan = planDashboardAction(entries, {
      type: "set_mark",
      arxivIds: ["2606.00001", "2606.00003"],
      status: "to_read",
      priority: "high",
    });

    expect(statusPlan).toEqual({
      patches: [
        {
          arxivId: "2606.00001",
          status: "saved",
        },
      ],
      missingIds: ["missing"],
      requiresConfirmation: false,
    });
    expect(priorityPlan.patches).toEqual([
      { arxivId: "2606.00001", priority: "high" },
    ]);
    expect(markPlan.patches).toEqual([
      {
        arxivId: "2606.00001",
        status: "to_read",
        priority: "high",
      },
      {
        arxivId: "2606.00003",
        status: "to_read",
        priority: "high",
      },
    ]);
  });
});
