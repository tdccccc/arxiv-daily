import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  applyStarButtonState,
  collectIndexedDetailSummaryRefs,
  dashboardHistoryPathSet,
  dashboardOccurrenceProvenanceLines,
  dashboardPersonalNoveltyLines,
  deferDashboardAction,
  executeObsidianCommand,
  expectedDetailSummaryPath,
  filterDashboardMarkdownFiles,
  formatLogEntries,
  isExpectedGeneratedDetailSummary,
  openDashboardView,
  openMarkdownFileOnce,
  paginateDashboardRows,
  refreshOpenDashboardViews,
  shouldForceDashboardHistorySyncAfterDetailDeletion,
  shouldSkipDashboardHistorySync,
  trashFileWithUserPreference,
} from "../src/dashboard/view";
import type { PaperIndexEntry } from "@arxiv-daily/core";

const dashboardViewSource = readFileSync(
  resolve(process.cwd(), "src/dashboard/view.ts"),
  "utf-8",
);
const detailRefsSource = readFileSync(
  resolve(process.cwd(), "src/dashboard/detail-refs.ts"),
  "utf-8",
);
const hubModalSource = readFileSync(
  resolve(process.cwd(), "src/dashboard/hub-modal.ts"),
  "utf-8",
);
const pluginStyles = readFileSync(resolve(process.cwd(), "styles.css"), "utf-8");

describe("openDashboardView", () => {
  it("activates an existing dashboard leaf with the Obsidian 1.4 API", async () => {
    const leaf = { setViewState: vi.fn() };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([leaf]),
      getLeaf: vi.fn(),
      setActiveLeaf: vi.fn(),
    };

    await openDashboardView({ app: { workspace } } as any);

    expect(workspace.getLeavesOfType).toHaveBeenCalledWith(
      ARXIV_DAILY_DASHBOARD_VIEW,
    );
    expect(workspace.setActiveLeaf).toHaveBeenCalledWith(leaf, { focus: true });
    expect(workspace.getLeaf).not.toHaveBeenCalled();
    expect(leaf.setViewState).not.toHaveBeenCalled();
  });

  it("creates a dashboard leaf when none exists", async () => {
    const leaf = { setViewState: vi.fn().mockResolvedValue(undefined) };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([]),
      getLeaf: vi.fn().mockReturnValue(leaf),
      setActiveLeaf: vi.fn(),
    };

    await openDashboardView({ app: { workspace } } as any);

    expect(workspace.getLeaf).toHaveBeenCalledWith(true);
    expect(leaf.setViewState).toHaveBeenCalledWith({
      type: ARXIV_DAILY_DASHBOARD_VIEW,
      active: true,
    });
    expect(workspace.setActiveLeaf).toHaveBeenCalledWith(leaf, { focus: true });
  });
});

describe("openMarkdownFileOnce", () => {
  it("activates an already open markdown file with the Obsidian 1.4 API", async () => {
    const leaf = {
      getViewState: vi.fn().mockReturnValue({
        state: { file: "arxiv-daily/papers/2606.12345.md" },
      }),
    };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([leaf]),
      setActiveLeaf: vi.fn(),
      openLinkText: vi.fn().mockResolvedValue(undefined),
    };

    await openMarkdownFileOnce(
      { workspace },
      "arxiv-daily/papers/2606.12345.md",
    );

    expect(workspace.getLeavesOfType).toHaveBeenCalledWith("markdown");
    expect(workspace.setActiveLeaf).toHaveBeenCalledWith(leaf, { focus: true });
    expect(workspace.openLinkText).not.toHaveBeenCalled();
  });

  it("opens the markdown file when no existing leaf matches", async () => {
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([
        { view: { file: { path: "arxiv-daily/papers/2606.54321.md" } } },
      ]),
      setActiveLeaf: vi.fn(),
      openLinkText: vi.fn().mockResolvedValue(undefined),
    };

    await openMarkdownFileOnce(
      { workspace },
      "arxiv-daily/papers/2606.12345.md",
    );

    expect(workspace.setActiveLeaf).not.toHaveBeenCalled();
    expect(workspace.openLinkText).toHaveBeenCalledWith(
      "arxiv-daily/papers/2606.12345.md",
      "",
      false,
    );
  });
});

describe("deferDashboardAction", () => {
  it("runs modal-opening menu actions after the current event loop", () => {
    vi.useFakeTimers();
    const action = vi.fn();

    deferDashboardAction(action);

    expect(action).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(action).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
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

describe("dashboard manual summaries", () => {
  it("refreshes every open dashboard view from the vault", async () => {
    const refreshA = vi.fn().mockResolvedValue(undefined);
    const refreshB = vi.fn().mockResolvedValue(undefined);
    const plugin = {
      app: {
        workspace: {
          getLeavesOfType: vi.fn(() => [
            { view: { refreshFromVault: refreshA } },
            { view: { refreshFromVault: refreshB } },
          ]),
        },
      },
    };

    await refreshOpenDashboardViews(plugin as any);

    expect(refreshA).toHaveBeenCalledTimes(1);
    expect(refreshB).toHaveBeenCalledTimes(1);
  });

  it("opens only successful results and forces history refresh before opening", () => {
    const body = dashboardViewSource.match(
      /private async summarizeDetailById\([\s\S]*?\n  private async openDailyReport/,
    )?.[0];
    expect(body).toContain('result.kind !== "done" && result.kind !== "already_exists"');
    expect(body).toContain("this.lastSyncedHistoryPaths = null");
    expect(body).toContain("await this.reloadIndex()");
    expect(body).toContain("openMarkdownFileOnce");
  });
});

describe("dashboard date-run controls", () => {
  it("runs the selected scheduler path and always refreshes the dashboard", () => {
    const body = dashboardViewSource.match(
      /private async runSelectedDate\([\s\S]*?\n  private async runToday/,
    )?.[0];

    expect(body).toBeDefined();
    expect(body).toContain("forceRunForDate(date)");
    expect(body).toContain("runForDateNow(date)");
    expect(body).toMatch(/finally \{\s*await this\.reloadIndex\(\);/);
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

  it("rerenders query-dependent results and restores sensible focus", () => {
    const updateStarBody = dashboardViewSource.match(
      /private async updateStar\([\s\S]*?\n  private async openDetailSummary/,
    )?.[0];

    expect(updateStarBody).toBeDefined();
    expect(updateStarBody).toContain("this.dailyReports = this.loadDailyReports");
    expect(updateStarBody).toContain("refreshCalendarDailyReports");
    expect(updateStarBody).toContain("this.render()");
    expect(updateStarBody).toContain("nextButton.focus()");
    expect(updateStarBody).toContain(".arxiv-daily-dashboard__tab.is-active");
  });

  it("updates the detail-summary filter button state after toggling", () => {
    const filterBody = dashboardViewSource.match(
      /private renderToolbarFilter\([\s\S]*?\n  private countToolbarFilter/,
    )?.[0];

    expect(filterBody).toBeDefined();
    expect(filterBody).toContain('button.toggleClass("is-active", isActive)');
    expect(filterBody).toContain(
      'button.setAttribute("aria-pressed", String(isActive))',
    );
  });

  it("fully resets controls and cancels pending search debounce", () => {
    const resetBody = dashboardViewSource.match(
      /private resetFilters\(\)[\s\S]*?\n  private openSettings/,
    )?.[0];

    expect(resetBody).toBeDefined();
    expect(resetBody).toContain("clearSearchDebounce");
    expect(resetBody).toContain("this.render()");
    expect(resetBody).not.toContain("renderCurrentResults");
  });
});

describe("dashboard render regressions", () => {
  it("points missing configuration to the Settings onboarding steps", () => {
    expect(dashboardViewSource).toContain('text: "Finish setup in Settings"');
    expect(dashboardViewSource).toContain(
      'text: "Connect AI, choose paper sources, and describe your research interests, then generate your first report."',
    );
    expect(dashboardViewSource).toContain('"Open Settings"');
    expect(dashboardViewSource).not.toContain(
      "Add your LLM settings and at least one research topic",
    );
  });

  it("guards overlapping calendar month refreshes with a sequence token", () => {
    expect(dashboardViewSource).toContain("calendarRefreshSeq");
    expect(dashboardViewSource).toContain("refreshCalendarMonth");
    expect(dashboardViewSource).toContain("token !== this.calendarRefreshSeq");
  });

  it("refreshes toolbar filter counts after tab switches", () => {
    const renderToolbarBody = dashboardViewSource.match(
      /private renderToolbar\([\s\S]*?\n  private updateTabButtonState/,
    )?.[0];

    expect(renderToolbarBody).toBeDefined();
    expect(renderToolbarBody).toContain("updateToolbarFilterCounts");
  });
});

describe("HubModal tabs", () => {
  it("links each tabpanel to its tab button", () => {
    expect(hubModalSource).toContain('role: "tab"');
    expect(hubModalSource).toContain('"aria-selected": "false"');
    expect(hubModalSource).toContain('content.setAttribute("role", "tabpanel")');
    expect(hubModalSource).toContain('content.setAttribute("aria-labelledby", tabId)');
    expect(hubModalSource).toContain("button.id = tabId");
    expect(hubModalSource).toContain("content.id = panelId");
  });

  it("keeps Clear logs actionable only on the Logs tab", () => {
    expect(hubModalSource).toContain('text: "Clear logs"');
    expect(hubModalSource).toContain('const visible = this.activeTab === "logs"');
    expect(hubModalSource).toContain("this.clearButton.hidden = !visible");
    expect(hubModalSource).toContain("this.clearButton.disabled = !visible");
  });

  it("uses separate flex sizing classes for short viewports", () => {
    expect(hubModalSource).toContain('contentEl.addClass("arxiv-daily-hub-modal__content")');
    expect(pluginStyles).toContain(".arxiv-daily-hub-modal__content");
    expect(pluginStyles).toContain("min-height: 0");
    expect(pluginStyles).toContain("max-height: min(82vh, 740px)");
  });
});

describe("dashboard occurrence provenance", () => {
  it("formats manual, library, and both-source metadata with every direction and representative", () => {
    const row = {
      entry: indexedPaper("2606.10001"),
      arxivId: "2606.10001",
      title: "New paper",
      authors: "A. Author",
      topic: "rag",
      firstSeen: "2026-06-10",
      hasDetailSummary: false,
      occurrenceProvenance: {
        reportPath: "arxiv-daily/daily/2026-06-10.md",
        reportDate: "2026-06-10",
        source: "both" as const,
        manualTopics: [{ tag: "rag", name: "RAG" }],
        directions: [{
          id: "direction-1",
          name: "Reliable retrieval",
          representatives: [{
            paperKey: "arxiv:2501.00001",
            title: "Prior one",
            evidenceDepth: "metadata-and-abstract" as const,
          }],
        }, {
          id: "direction-2",
          name: "Evaluation",
          representatives: [{
            paperKey: "arxiv:2501.00002",
            title: "Prior two",
            evidenceDepth: "metadata-and-abstract" as const,
          }],
        }],
        evidenceDepth: "metadata-and-abstract" as const,
      },
    };

    expect(dashboardOccurrenceProvenanceLines(row)).toEqual([
      "Discovery source: Manual + library",
      "Manual topics: RAG (rag)",
      "Library directions: Reliable retrieval: Prior one (arxiv:2501.00001); Evaluation: Prior two (arxiv:2501.00002)",
      "Evidence depth: metadata and abstract",
    ]);
    expect(dashboardOccurrenceProvenanceLines({
      ...row,
      occurrenceProvenance: {
        ...row.occurrenceProvenance,
        source: "manual",
        directions: [],
        evidenceDepth: undefined,
      },
    })).toEqual(["Discovery source: Manual", "Manual topics: RAG (rag)"]);
    expect(dashboardOccurrenceProvenanceLines({
      ...row,
      occurrenceProvenance: {
        ...row.occurrenceProvenance,
        source: "library",
        manualTopics: [],
      },
    })[0]).toBe("Discovery source: Library");
  });

  it("keeps hostile metadata literal and renders provenance independently from search reasons", () => {
    const lines = dashboardOccurrenceProvenanceLines({
      entry: indexedPaper("2606.10002"),
      arxivId: "2606.10002",
      title: "Hostile paper",
      authors: "A. Author",
      topic: "hostile",
      firstSeen: "2026-06-10",
      hasDetailSummary: false,
      matchReasons: [{ field: "title", text: "title matched hostile" }],
      occurrenceProvenance: {
        reportPath: "arxiv-daily/daily/2026-06-10.md",
        reportDate: "2026-06-10",
        source: "library",
        manualTopics: [],
        directions: [{
          id: "hostile",
          name: "<img src=x onerror=alert(1)> **direction**",
          representatives: [{
            paperKey: "arxiv:2501.00001",
            title: "[prior](javascript:alert(1)) <script>bad()</script>",
            evidenceDepth: "metadata-and-abstract",
          }],
        }],
        evidenceDepth: "metadata-and-abstract",
      },
    });

    expect(lines.join("\n")).toContain("<img src=x onerror=alert(1)> **direction**");
    expect(lines.join("\n")).toContain("[prior](javascript:alert(1)) <script>bad()</script>");
    expect(lines.join("\n")).not.toContain("title matched hostile");
    expect(dashboardViewSource).toContain("dashboardOccurrenceProvenanceLines(row)");
    expect(dashboardViewSource).toContain('text: line');
    expect(dashboardViewSource).not.toContain("attr: { title: line }");
    expect(dashboardViewSource).toContain("this.isActiveRelevanceSearch() && row.matchReasons?.length");
    const provenanceStyles = pluginStyles.match(
      /\.arxiv-daily-dashboard__provenance\s*\{[\s\S]*?\}/,
    )?.[0];
    expect(provenanceStyles).toBeDefined();
    expect(provenanceStyles).toContain("overflow-wrap: anywhere");
    expect(provenanceStyles).toContain("white-space: normal");
    expect(provenanceStyles).not.toContain("text-overflow: ellipsis");
    expect(provenanceStyles).not.toContain("overflow: hidden");
  });
});

describe("dashboard occurrence personal novelty", () => {
  const noveltyRow = {
    entry: indexedPaper("2606.10001"),
    arxivId: "2606.10001",
    title: "New paper",
    authors: "A. Author",
    topic: "rag",
    firstSeen: "2026-06-10",
    hasDetailSummary: false,
    personalNovelty: {
      reportPath: "arxiv-daily/daily/2026-06-10.md",
      reportDate: "2026-06-10",
      differenceType: "new-method",
      comparisonBasis: [
        { paperKey: "arxiv:2501.00001" },
        { paperKey: "arxiv:2501.00002" },
      ],
      evidenceDepth: "metadata-and-abstract",
      explanation: "Introduces a method absent from the representative abstracts.",
    },
  };

  it("formats a clearly labeled novelty block with difference type, paperKey-only basis, depth, and explanation", () => {
    expect(dashboardPersonalNoveltyLines(noveltyRow)).toEqual([
      "Personal novelty: new method vs. prior papers: arxiv:2501.00001, arxiv:2501.00002",
      "Evidence depth: metadata and abstract",
      "Introduces a method absent from the representative abstracts.",
    ]);
    expect(dashboardPersonalNoveltyLines({
      ...noveltyRow,
      personalNovelty: {
        ...noveltyRow.personalNovelty,
        differenceType: "counter-evidence",
      },
    })[0]).toBe(
      "Personal novelty: counter-evidence vs. prior papers: arxiv:2501.00001, arxiv:2501.00002",
    );
  });

  it("keeps hostile novelty text literal, collapses whitespace, and stays separate from provenance and match reasons", () => {
    const row = {
      ...noveltyRow,
      matchReasons: [{ field: "title", text: "title matched hostile" }],
      occurrenceProvenance: {
        reportPath: "arxiv-daily/daily/2026-06-10.md",
        reportDate: "2026-06-10",
        source: "library",
        manualTopics: [],
        directions: [{
          id: "direction-1",
          name: "Reliable retrieval",
          representatives: [{
            paperKey: "arxiv:2501.00001",
            title: "Prior one",
            evidenceDepth: "metadata-and-abstract",
          }],
        }],
        evidenceDepth: "metadata-and-abstract",
      },
      personalNovelty: {
        ...noveltyRow.personalNovelty,
        explanation: "<img src=x onerror=alert(1)> [link](javascript:alert(1))\n  hostile   text ",
      },
    };

    const lines = dashboardPersonalNoveltyLines(row);
    const joined = lines.join("\n");
    expect(joined).toContain("<img src=x onerror=alert(1)> [link](javascript:alert(1))");
    // Whitespace runs collapse to single spaces like the daily-report line.
    expect(joined).toContain("hostile text");
    expect(joined).not.toContain("hostile   text");
    expect(joined).not.toContain("title matched hostile");
    expect(joined).not.toContain("Reliable retrieval");
    expect(joined).not.toContain("Prior one");
    expect(dashboardOccurrenceProvenanceLines(row).join("\n"))
      .not.toContain("Personal novelty:");
    expect(dashboardViewSource).toContain("dashboardPersonalNoveltyLines(row)");
    expect(dashboardViewSource).toContain('text: line');
    expect(dashboardViewSource).toContain('"arxiv-daily-dashboard__novelty"');
    expect(dashboardViewSource).not.toContain("attr: { title: line }");
    expect(dashboardViewSource).not.toContain("innerHTML");
    const noveltyStyles = pluginStyles.match(
      /\.arxiv-daily-dashboard__novelty\s*\{[\s\S]*?\}/,
    )?.[0];
    expect(noveltyStyles).toBeDefined();
    expect(noveltyStyles).toContain("overflow-wrap: anywhere");
    expect(noveltyStyles).toContain("white-space: normal");
    expect(noveltyStyles).not.toContain("text-overflow: ellipsis");
    expect(noveltyStyles).not.toContain("overflow: hidden");
    expect(pluginStyles).toContain(".arxiv-daily-dashboard__provenance");
    expect(pluginStyles).toContain(".arxiv-daily-dashboard__match-reason");
    // The novelty block is visually distinct from the provenance block and
    // from query-time match-reason truncation styles.
    expect(noveltyStyles).not.toEqual(
      pluginStyles.match(/\.arxiv-daily-dashboard__provenance\s*\{[\s\S]*?\}/)?.[0],
    );
  });

  it("returns no novelty lines for legacy rows without novelty metadata", () => {
    expect(dashboardPersonalNoveltyLines({
      entry: indexedPaper("2606.10002"),
      arxivId: "2606.10002",
      title: "Legacy paper",
      authors: "A. Author",
      topic: "rag",
      firstSeen: "2026-06-10",
      hasDetailSummary: false,
    })).toEqual([]);
  });
});

describe("dashboard pane responsiveness", () => {
  it("styles compact search explanations and narrow similar-paper content", () => {
    expect(pluginStyles).toContain(".arxiv-daily-dashboard__match-reason");
    expect(pluginStyles).toContain(".arxiv-daily-similar-modal__actions");
    expect(pluginStyles).toContain("@media (max-width: 520px)");
  });

  it("uses dashboard container queries for overview, filters, and calendar", () => {
    expect(pluginStyles).toContain("container-name: arxiv-daily-dashboard");
    expect(pluginStyles).toContain("container-type: inline-size");
    expect(pluginStyles).toContain("@container arxiv-daily-dashboard (max-width: 920px)");
    expect(pluginStyles).toContain("@container arxiv-daily-dashboard (max-width: 520px)");
  });
});

describe("detail-summary deletion boundaries", () => {
  const generated = [
    "---",
    'arxiv_id: "2606.12345v2"',
    "---",
    "# Verified paper",
    "",
    "- **arXiv**: [2606.12345v2](https://arxiv.org/abs/2606.12345v2)",
    "",
    "## Research question",
    "A".repeat(150),
    "## Method",
    "B".repeat(150),
    "## Evidence",
    "C".repeat(150),
    "## Limitations",
    "D".repeat(150),
  ].join("\n");

  it("derives only the canonical configured papers path", () => {
    expect(expectedDetailSummaryPath("arxiv/papers", "2606.12345v7")).toBe(
      "arxiv/papers/2606.12345.md",
    );
    expect(expectedDetailSummaryPath("/arxiv/papers", "2606.12345")).toBeNull();
    expect(expectedDetailSummaryPath("arxiv/papers", "../../notes")).toBeNull();
  });

  it("requires generated detail content with exact matching frontmatter arxiv ID", () => {
    expect(isExpectedGeneratedDetailSummary(generated, "2606.12345")).toBe(true);
    expect(isExpectedGeneratedDetailSummary(generated, "2606.54321")).toBe(false);
    const spoofedBodyUrl = generated
      .replace(/^---[\s\S]*?---\n/, "")
      .replace("# Verified paper", "# https://arxiv.org/abs/2606.12345");
    expect(isExpectedGeneratedDetailSummary(spoofedBodyUrl, "2606.12345")).toBe(false);
    expect(
      isExpectedGeneratedDetailSummary(
        generated.replace(
          'arxiv_id: "2606.12345v2"',
          'arxiv_id: "https://arxiv.org/abs/2606.12345"',
        ),
        "2606.12345",
      ),
    ).toBe(false);
    expect(
      isExpectedGeneratedDetailSummary(
        "---\narxiv_id: '2606.12345'\n---\n# Paper\n\nToo short",
        "2606.12345",
      ),
    ).toBe(false);
  });

  it("accepts matching legacy arxiv frontmatter", () => {
    const legacy = generated.replace("arxiv_id:", "arxiv:");

    expect(isExpectedGeneratedDetailSummary(legacy, "2606.12345")).toBe(true);
    expect(isExpectedGeneratedDetailSummary(legacy, "2606.54321")).toBe(false);
  });

  it("rejects conflicting or invalid frontmatter arxiv IDs", () => {
    const conflicting = generated.replace(
      'arxiv_id: "2606.12345v2"',
      'arxiv_id: "2606.12345v2"\narxiv: "2606.54321"',
    );
    const invalidLegacy = generated.replace(
      'arxiv_id: "2606.12345v2"',
      'arxiv: "https://arxiv.org/abs/2606.12345"',
    );

    expect(isExpectedGeneratedDetailSummary(conflicting, "2606.12345")).toBe(false);
    expect(isExpectedGeneratedDetailSummary(invalidLegacy, "2606.12345")).toBe(false);
  });

  it("reuses the core classifier for deletion validation", () => {
    const validatorBody = detailRefsSource.match(
      /export function isExpectedGeneratedDetailSummary\([\s\S]*?\n}/,
    )?.[0];
    expect(validatorBody).toContain("classifyPaperNote(markdown, canonicalArxivId)");
  });

  it("requires the indexed path to equal the expected configured path", () => {
    const deletionBody = dashboardViewSource.match(
      /private async runBatchDeleteSummary\(\)[\s\S]*?\n  private async runBatchAction/,
    )?.[0];
    expect(deletionBody).toContain("this.hasDeletableDetailSummary(entry)");
    expect(deletionBody).toContain("removePaperDetailsAtPath(");
    expect(deletionBody).not.toContain("this.detailSummaryPaths.get");
  });

  it("uses FileManager.trashFile when the API is available", async () => {
    const file = { path: "arxiv/papers/2606.12345.md" };
    const trashFile = vi.fn().mockResolvedValue(undefined);
    const vaultTrash = vi.fn();
    const adapterRemove = vi.fn();

    await trashFileWithUserPreference(
      { fileManager: { trashFile }, vault: { trash: vaultTrash } } as any,
      file as any,
    );

    expect(trashFile).toHaveBeenCalledWith(file);
    expect(vaultTrash).not.toHaveBeenCalled();
    expect(adapterRemove).not.toHaveBeenCalled();

    const deletionBody = dashboardViewSource.match(
      /private async runBatchDeleteSummary\(\)[\s\S]*?\n  private async runBatchAction/,
    )?.[0];
    expect(deletionBody).toBeDefined();
    expect(deletionBody).toContain("vault.read(abstractFile)");
    expect(deletionBody).toContain("trashFileWithUserPreference(");
    expect(deletionBody).toContain("trashed but index update failed");
    expect(deletionBody).toContain("this.lastSyncedHistoryPaths = null");
    expect(deletionBody).not.toContain("vault.trash");
    expect(deletionBody).not.toContain("adapter.remove");
  });

  it("uses non-destructive Vault trash on Obsidian before 1.6.6", async () => {
    const file = { path: "arxiv/papers/2606.12345.md" };
    const vaultTrash = vi.fn().mockResolvedValue(undefined);

    await trashFileWithUserPreference(
      { fileManager: {}, vault: { trash: vaultTrash } } as any,
      file as any,
    );

    expect(vaultTrash).toHaveBeenCalledWith(file, true);
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
  it("forces history sync after detail storage or index state changes", () => {
    expect(shouldForceDashboardHistorySyncAfterDetailDeletion(0, 0)).toBe(false);
    expect(shouldForceDashboardHistorySyncAfterDetailDeletion(1, 0)).toBe(true);
    expect(shouldForceDashboardHistorySyncAfterDetailDeletion(0, 1)).toBe(true);
    expect(shouldForceDashboardHistorySyncAfterDetailDeletion(1, 1)).toBe(true);
  });

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

  it("fingerprints daily paths and direct-child paper Markdown paths", () => {
    const files = [
      { path: "notes/random.md" },
      { path: "arxiv\\papers\\2606.00002.MD" },
      { path: "arxiv/papers/nested/2606.00003.md" },
      { path: "arxiv/daily/2026-07-01.md" },
      { path: "arxiv/papers/2606.00001.md" },
      { path: "arxiv/papers/attachment.pdf" },
      { path: "arxiv/daily/nested/managed.md" },
    ];

    expect(
      dashboardHistoryPathSet(files, "/arxiv/daily/", "/arxiv/papers/"),
    ).toEqual(
      new Set([
        "arxiv/daily/2026-07-01.md",
        "arxiv/daily/nested/managed.md",
        "arxiv/papers/2606.00001.md",
        "arxiv/papers/2606.00002.MD",
      ]),
    );
  });

  it("skips only when the complete managed history path set is unchanged", () => {
    const daily = "arxiv/daily/2026-06-30.md";
    const paper = "arxiv/papers/2606.00001.md";
    expect(
      shouldSkipDashboardHistorySync(null, new Set([daily, paper]), 1),
    ).toBe(false);
    expect(
      shouldSkipDashboardHistorySync(
        new Set([daily, paper]),
        new Set([paper, daily]),
        12,
      ),
    ).toBe(true);
    expect(
      shouldSkipDashboardHistorySync(
        new Set([daily, paper]),
        new Set([paper, daily]),
        0,
      ),
    ).toBe(false);
  });

  it.each([
    {
      change: "creates a paper",
      before: ["arxiv/daily/2026-06-30.md"],
      after: [
        "arxiv/daily/2026-06-30.md",
        "arxiv/papers/2606.00001.md",
      ],
    },
    {
      change: "deletes a paper",
      before: [
        "arxiv/daily/2026-06-30.md",
        "arxiv/papers/2606.00001.md",
      ],
      after: ["arxiv/daily/2026-06-30.md"],
    },
    {
      change: "renames a paper",
      before: [
        "arxiv/daily/2026-06-30.md",
        "arxiv/papers/2606.00001.md",
      ],
      after: [
        "arxiv/papers/2606.00002.md",
        "arxiv/daily/2026-06-30.md",
      ],
    },
  ])("does not skip when an external change $change", ({ before, after }) => {
    expect(
      shouldSkipDashboardHistorySync(new Set(before), new Set(after), 12),
    ).toBe(false);
  });

  it("logs cache hits as unchanged managed history files", () => {
    expect(dashboardViewSource).toContain(
      "unchanged managed history files",
    );
    expect(dashboardViewSource).not.toContain("unchanged daily files");
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
