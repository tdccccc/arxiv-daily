import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  applyStarButtonState,
  collectIndexedDetailSummaryRefs,
  executeObsidianCommand,
  filterDashboardMarkdownFiles,
  formatLogEntries,
  openDashboardView,
  openMarkdownFileOnce,
  paginateDashboardRows,
  shouldSkipDashboardHistorySync,
} from "../src/dashboard/view";
import type { PaperIndexEntry } from "../src/services/paper-index";

const dashboardViewSource = readFileSync(
  resolve(process.cwd(), "src/dashboard/view.ts"),
  "utf-8",
);

describe("openDashboardView", () => {
  it("reveals an existing dashboard leaf", async () => {
    const leaf = { setViewState: vi.fn() };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([leaf]),
      getLeaf: vi.fn(),
      revealLeaf: vi.fn().mockResolvedValue(undefined),
    };

    await openDashboardView({ app: { workspace } } as any);

    expect(workspace.getLeavesOfType).toHaveBeenCalledWith(
      ARXIV_DAILY_DASHBOARD_VIEW,
    );
    expect(workspace.revealLeaf).toHaveBeenCalledWith(leaf);
    expect(workspace.getLeaf).not.toHaveBeenCalled();
    expect(leaf.setViewState).not.toHaveBeenCalled();
  });

  it("creates a dashboard leaf when none exists", async () => {
    const leaf = { setViewState: vi.fn().mockResolvedValue(undefined) };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([]),
      getLeaf: vi.fn().mockReturnValue(leaf),
      revealLeaf: vi.fn().mockResolvedValue(undefined),
    };

    await openDashboardView({ app: { workspace } } as any);

    expect(workspace.getLeaf).toHaveBeenCalledWith(true);
    expect(leaf.setViewState).toHaveBeenCalledWith({
      type: ARXIV_DAILY_DASHBOARD_VIEW,
      active: true,
    });
    expect(workspace.revealLeaf).toHaveBeenCalledWith(leaf);
  });
});

describe("openMarkdownFileOnce", () => {
  it("reveals an already open markdown file", async () => {
    const leaf = {
      getViewState: vi.fn().mockReturnValue({
        state: { file: "arxiv-daily/papers/2606.12345.md" },
      }),
    };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([leaf]),
      revealLeaf: vi.fn().mockResolvedValue(undefined),
      openLinkText: vi.fn().mockResolvedValue(undefined),
    };

    await openMarkdownFileOnce(
      { workspace },
      "arxiv-daily/papers/2606.12345.md",
    );

    expect(workspace.getLeavesOfType).toHaveBeenCalledWith("markdown");
    expect(workspace.revealLeaf).toHaveBeenCalledWith(leaf);
    expect(workspace.openLinkText).not.toHaveBeenCalled();
  });

  it("opens the markdown file when no existing leaf matches", async () => {
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([
        { view: { file: { path: "arxiv-daily/papers/2606.54321.md" } } },
      ]),
      revealLeaf: vi.fn().mockResolvedValue(undefined),
      openLinkText: vi.fn().mockResolvedValue(undefined),
    };

    await openMarkdownFileOnce(
      { workspace },
      "arxiv-daily/papers/2606.12345.md",
    );

    expect(workspace.revealLeaf).not.toHaveBeenCalled();
    expect(workspace.openLinkText).toHaveBeenCalledWith(
      "arxiv-daily/papers/2606.12345.md",
      "",
      false,
    );
  });
});

describe("executeObsidianCommand", () => {
  it("uses executeCommandById when available", async () => {
    const executeCommandById = vi.fn().mockReturnValue(true);

    const executed = await executeObsidianCommand(
      { commands: { executeCommandById } },
      "arxiv-daily-run-for-date",
    );

    expect(executed).toBe(true);
    expect(executeCommandById).toHaveBeenCalledWith(
      "arxiv-daily-run-for-date",
    );
  });

  it("tries the Obsidian plugin-prefixed command id first", async () => {
    const executeCommandById = vi
      .fn()
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);

    const executed = await executeObsidianCommand(
      { commands: { executeCommandById } },
      "arxiv-daily-run-for-date",
      "arxiv-daily",
    );

    expect(executed).toBe(true);
    expect(executeCommandById).toHaveBeenNthCalledWith(
      1,
      "arxiv-daily:arxiv-daily-run-for-date",
    );
    expect(executeCommandById).toHaveBeenNthCalledWith(
      2,
      "arxiv-daily-run-for-date",
    );
  });

  it("uses the registered command id when the registry is available", async () => {
    const executeCommandById = vi.fn().mockReturnValue(true);

    const executed = await executeObsidianCommand(
      {
        commands: {
          executeCommandById,
          commands: {
            "arxiv-daily-run-for-date": {},
          },
        },
      },
      "arxiv-daily-run-for-date",
      "arxiv-daily",
    );

    expect(executed).toBe(true);
    expect(executeCommandById).toHaveBeenCalledTimes(1);
    expect(executeCommandById).toHaveBeenCalledWith(
      "arxiv-daily-run-for-date",
    );
  });

  it("falls back to command callbacks when executeCommandById is unavailable", async () => {
    const callback = vi.fn();

    const executed = await executeObsidianCommand(
      {
        commands: {
          commands: {
            "arxiv-daily-run-for-date": { callback },
          },
        },
      },
      "arxiv-daily-run-for-date",
    );

    expect(executed).toBe(true);
    expect(callback).toHaveBeenCalledTimes(1);
  });

  it("finds plugin-prefixed callbacks in the command registry", async () => {
    const callback = vi.fn();

    const executed = await executeObsidianCommand(
      {
        commands: {
          commands: {
            "arxiv-daily:arxiv-daily-run-for-date": { callback },
          },
        },
      },
      "arxiv-daily-run-for-date",
      "arxiv-daily",
    );

    expect(executed).toBe(true);
    expect(callback).toHaveBeenCalledTimes(1);
  });

  it("returns false for missing commands", async () => {
    await expect(
      executeObsidianCommand({ commands: { commands: {} } }, "missing"),
    ).resolves.toBe(false);
  });
});

describe("dashboard star controls", () => {
  it("updates the star button state in place", () => {
    const button = document.createElement("button");

    applyStarButtonState(button, true);

    expect(button.classList.contains("is-starred")).toBe(true);
    expect(button.getAttribute("aria-pressed")).toBe("true");
    expect(button.getAttribute("aria-label")).toBe("Unstar paper");

    applyStarButtonState(button, false);

    expect(button.classList.contains("is-starred")).toBe(false);
    expect(button.getAttribute("aria-pressed")).toBe("false");
    expect(button.getAttribute("aria-label")).toBe("Star paper");
  });

  it("does not rerender the result table when updating a star", () => {
    const updateStarBody = dashboardViewSource.match(
      /private async updateStar\([\s\S]*?\n  private async openDetailSummary/,
    )?.[0];

    expect(updateStarBody).toBeDefined();
    expect(updateStarBody).toContain("applyStarButtonState");
    expect(updateStarBody).not.toContain("renderCurrentResults");
  });
});

describe("HubModal tabs", () => {
  it("links each tabpanel to its tab button", () => {
    expect(dashboardViewSource).toContain('role: "tab"');
    expect(dashboardViewSource).toContain('"aria-selected": "false"');
    expect(dashboardViewSource).toContain('content.setAttribute("role", "tabpanel")');
    expect(dashboardViewSource).toContain('content.setAttribute("aria-labelledby", tabId)');
    expect(dashboardViewSource).toContain("button.id = tabId");
    expect(dashboardViewSource).toContain("content.id = panelId");
  });
});

describe("collectIndexedDetailSummaryRefs", () => {
  it("uses synced index detail fields without reading markdown files again", () => {
    const refs = collectIndexedDetailSummaryRefs([
      indexedPaper("2606.00001", {
        detail: true,
        paperPath: "arxiv/papers/2606.00001.md",
      }),
      indexedPaper("2606.00002", {
        detail: false,
        paperPath: "arxiv/papers/2606.00002.md",
      }),
      indexedPaper("2606.00003", {
        detail: true,
        paperPath: null,
      }),
    ]);

    expect([...refs.ids]).toEqual(["2606.00001"]);
    expect(refs.paths.get("2606.00001")).toBe("arxiv/papers/2606.00001.md");
    expect(refs.paths.has("2606.00002")).toBe(false);
    expect(refs.paths.has("2606.00003")).toBe(false);
  });
});

describe("dashboard reload helpers", () => {
  it("filters markdown files to configured daily and papers directories", () => {
    const files = [
      { path: "arxiv/daily/2026-06-30.md" },
      { path: "arxiv/daily/nested/ignore.md" },
      { path: "arxiv/papers/2606.00001.md" },
      { path: "arxiv-paper-notes/2606.00002.md" },
      { path: "notes/random.md" },
    ];

    expect(
      filterDashboardMarkdownFiles(files, "/arxiv/daily/", "arxiv/papers").map(
        (file) => file.path,
      ),
    ).toEqual([
      "arxiv/daily/2026-06-30.md",
      "arxiv/daily/nested/ignore.md",
      "arxiv/papers/2606.00001.md",
    ]);
  });

  it("skips dashboard history sync only when daily paths are unchanged and entries exist", () => {
    expect(
      shouldSkipDashboardHistorySync(null, new Set(["arxiv/daily/2026-06-30.md"]), 1),
    ).toBe(false);
    expect(
      shouldSkipDashboardHistorySync(
        new Set(["arxiv/daily/2026-06-30.md", "arxiv/daily/2026-07-01.md"]),
        new Set(["arxiv/daily/2026-07-01.md", "arxiv/daily/2026-06-30.md"]),
        12,
      ),
    ).toBe(true);
    expect(
      shouldSkipDashboardHistorySync(
        new Set(["arxiv/daily/2026-06-30.md"]),
        new Set(["arxiv/daily/2026-06-30.md"]),
        0,
      ),
    ).toBe(false);
    expect(
      shouldSkipDashboardHistorySync(
        new Set(["arxiv/daily/2026-06-30.md"]),
        new Set(["arxiv/daily/2026-07-01.md"]),
        12,
      ),
    ).toBe(false);
  });
});

describe("paginateDashboardRows", () => {
  it("returns a 20-row page summary with one-based visible bounds", () => {
    const rows = Array.from({ length: 45 }, (_, index) => `row-${index + 1}`);

    const page = paginateDashboardRows(rows, 2, 20);

    expect(page.currentPage).toBe(2);
    expect(page.totalPages).toBe(3);
    expect(page.start).toBe(41);
    expect(page.end).toBe(45);
    expect(page.rows).toEqual([
      "row-41",
      "row-42",
      "row-43",
      "row-44",
      "row-45",
    ]);
  });

  it("clamps out-of-range pages and handles empty rows", () => {
    const rows = Array.from({ length: 21 }, (_, index) => index);

    expect(paginateDashboardRows(rows, 99, 20)).toMatchObject({
      currentPage: 1,
      totalPages: 2,
      start: 21,
      end: 21,
      rows: [20],
    });
    expect(paginateDashboardRows([], 3, 20)).toMatchObject({
      currentPage: 0,
      totalPages: 1,
      start: 0,
      end: 0,
      rows: [],
    });
  });
});

function indexedPaper(
  id: string,
  overrides: Partial<PaperIndexEntry> = {},
): PaperIndexEntry {
  return {
    arxivId: id,
    source: "arxiv",
    title: `Paper ${id}`,
    authors: ["A. Author"],
    published: "2026-06-10",
    updated: "2026-06-10",
    category: "photo-z",
    categories: ["photo-z"],
    summary: undefined,
    topics: ["photo-z"],
    primaryTopic: "photo-z",
    detail: false,
    status: "inbox",
    priority: "normal",
    seenDates: ["2026-06-10"],
    dailyReports: [],
    paperPath: null,
    arxivUrl: `https://arxiv.org/abs/${id}`,
    pdfUrl: `https://arxiv.org/pdf/${id}`,
    pdfPath: "",
    zoteroKey: "",
    zoteroUri: "",
    citationKey: "",
    projects: [],
    ...overrides,
  };
}

describe("formatLogEntries", () => {
  it("renders newest-first by default", () => {
    const buf = [
      "2026-07-03 09:00:00.000 [INFO] first",
      "2026-07-03 09:00:01.000 [INFO] second",
      "2026-07-03 09:00:02.000 [ERROR] boom",
    ];
    const out = formatLogEntries(buf);
    const lines = out.split("\n");
    expect(lines[0]).toContain("boom");
    expect(lines[2]).toContain("first");
  });

  it("filters out levels not in the enabled set", () => {
    const buf = [
      "2026-07-03 09:00:00.000 [DEBUG] d",
      "2026-07-03 09:00:01.000 [INFO] i",
      "2026-07-03 09:00:02.000 [WARN] w",
      "2026-07-03 09:00:03.000 [ERROR] e",
    ];
    const out = formatLogEntries(buf, { levels: new Set(["info", "warn", "error"]) });
    const lines = out.split("\n");
    expect(lines).toHaveLength(3);
    expect(out).not.toContain("[DEBUG]");
    expect(lines[0]).toContain("[ERROR] e"); // newest first
  });

  it("returns a placeholder when buffer is empty", () => {
    expect(formatLogEntries([])).toBe("(no log entries)");
  });

  it("keeps lines without a parseable level tag when filter is active (kept, not hidden)", () => {
    const buf = ["weird line without level", "2026-07-03 09:00:00.000 [INFO] ok"];
    const out = formatLogEntries(buf, { levels: new Set(["info"]) });
    const lines = out.split("\n");
    // Output is reversed newest-first, so the tagged (later) line is first,
    // the untagged (earlier) line is last — both must survive the filter.
    expect(lines).toHaveLength(2);
    expect(lines[0]).toBe("2026-07-03 09:00:00.000 [INFO] ok");
    expect(lines[1]).toBe("weird line without level");
  });
});
