import {
  ItemView,
  Menu,
  Notice,
  setIcon,
  type WorkspaceLeaf,
} from "obsidian";
import type ArxivDailyPlugin from "../../main";
import {
  planDashboardAction,
  queryDashboard,
  type DashboardAction,
  type DashboardPatch,
  type DashboardQuery,
  type DashboardRow,
  type DashboardSortDirection,
  type DashboardSortKey,
  type DashboardTab,
} from "./model";
import { looksLikeDetailSummary } from "./detail-summary";
import { syncDashboardHistory } from "./history-sync";
import {
  validateFilterConfig,
  validateLlmConfig,
} from "../settings/validation";
import { daysBefore, formatDate, isWeekendDate, todayInTz } from "../utils/time";
import { getSetupStatus } from "../onboarding";
import { chooseModal } from "../services/modal";

export const ARXIV_DAILY_DASHBOARD_VIEW = "arxiv-daily-dashboard";

const DASHBOARD_TABS: Array<{ id: DashboardTab; label: string }> = [
  { id: "all", label: "All" },
  { id: "starred", label: "Starred" },
];

const SORT_OPTIONS: Array<{
  value: string;
  label: string;
  key: DashboardSortKey;
  direction: DashboardSortDirection;
}> = [
  {
    value: "priority:asc",
    label: "Starred first",
    key: "priority",
    direction: "asc",
  },
  {
    value: "firstSeen:desc",
    label: "Recently seen",
    key: "firstSeen",
    direction: "desc",
  },
  {
    value: "published:desc",
    label: "Published newest",
    key: "published",
    direction: "desc",
  },
  {
    value: "published:asc",
    label: "Published oldest",
    key: "published",
    direction: "asc",
  },
  {
    value: "topic:asc",
    label: "Topic A-Z",
    key: "topic",
    direction: "asc",
  },
  {
    value: "title:asc",
    label: "Title A-Z",
    key: "title",
    direction: "asc",
  },
];

const DISPLAY_OPTIONS = [
  { value: "20", label: "20" },
  { value: "50", label: "50" },
  { value: "100", label: "100" },
  { value: "all", label: "All" },
] as const;

type DashboardDisplayLimit = (typeof DISPLAY_OPTIONS)[number]["value"];

interface DailyReportDay {
  date: string;
  path: string;
  papers: number;
  starred: number;
}

export type CalendarCellState =
  | "empty"        // No date or outside lookback
  | "runnable"     // Can generate report
  | "has-report";  // Report exists

export interface CalendarCell {
  date: string | null;
  state: CalendarCellState;
  report?: DailyReportDay;
}

export function registerDashboardView(plugin: ArxivDailyPlugin): void {
  plugin.registerView(
    ARXIV_DAILY_DASHBOARD_VIEW,
    (leaf) => new ArxivDailyDashboardView(leaf, plugin),
  );
}

export async function openDashboardView(
  plugin: ArxivDailyPlugin,
): Promise<void> {
  const workspace = plugin.app.workspace;
  const existing = workspace.getLeavesOfType(ARXIV_DAILY_DASHBOARD_VIEW)[0];
  if (existing) {
    await workspace.revealLeaf(existing);
    return;
  }

  const leaf = workspace.getLeaf(true);
  if (!leaf) {
    new Notice("arXiv Daily: no workspace leaf available");
    return;
  }
  await leaf.setViewState({
    type: ARXIV_DAILY_DASHBOARD_VIEW,
    active: true,
  });
  await workspace.revealLeaf(leaf);
}

export async function openMarkdownFileOnce(
  app: {
    workspace: {
      getLeavesOfType?(type: string): unknown[];
      revealLeaf?(leaf: unknown): Promise<void>;
      openLinkText(path: string, sourcePath: string, newLeaf?: boolean): Promise<void>;
    };
  },
  path: string,
): Promise<void> {
  const target = normalizeVaultPath(path);
  const leaves = app.workspace.getLeavesOfType?.("markdown") ?? [];
  for (const leaf of leaves) {
    const leafPath = markdownPathFromLeaf(leaf);
    if (leafPath && normalizeVaultPath(leafPath) === target) {
      if (app.workspace.revealLeaf) {
        await app.workspace.revealLeaf(leaf);
      } else {
        await app.workspace.openLinkText(path, "", false);
      }
      return;
    }
  }
  await app.workspace.openLinkText(path, "", false);
}

export function appendSettingsButton(
  parent: HTMLElement,
  onClick: () => void,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "arxiv-daily-dashboard__settings-btn";
  button.type = "button";
  button.setAttribute("aria-label", "Open arXiv Daily settings");
  setIcon(button, "settings");
  const label = document.createElement("span");
  label.textContent = "Settings";
  button.appendChild(label);
  button.addEventListener("click", onClick);
  parent.appendChild(button);
  return button;
}

class ArxivDailyDashboardView extends ItemView {
  private entries: DashboardRow["entry"][] = [];
  private dailyReports: DailyReportDay[] = [];
  private detailSummaryIds = new Set<string>();
  private detailSummaryPaths = new Map<string, string>();
  private calendarMonth: string | null = null;
  private query: DashboardQuery = { tab: "starred" };
  private displayLimit: DashboardDisplayLimit = "20";
  private error: string | null = null;
  private selectedIds = new Set<string>();
  private batchEl: HTMLElement | null = null;
  private statsEl: HTMLElement | null = null;
  private resultsEl: HTMLElement | null = null;

  constructor(
    leaf: WorkspaceLeaf,
    private plugin: ArxivDailyPlugin,
  ) {
    super(leaf);
  }

  getViewType(): string {
    return ARXIV_DAILY_DASHBOARD_VIEW;
  }

  getDisplayText(): string {
    return "arXiv Daily Dashboard";
  }

  getIcon(): string {
    return "book-open-check";
  }

  async onOpen(): Promise<void> {
    await this.reloadIndex();
  }

  async onClose(): Promise<void> {
    this.contentEl.empty();
  }

  private async reloadIndex(): Promise<void> {
    this.renderLoading();
    try {
      await this.clearRunStateForMissingDailyReports();
      const store = this.plugin.buildPaperIndex();
      const index = await syncDashboardHistory({
        vault: this.plugin.app.vault,
        store,
        output: this.plugin.settings.output,
        topics: this.plugin.settings.arxiv.topics,
        logger: this.plugin.logger,
      });
      this.entries = Object.values(index.papers);
      await this.loadDetailSummaries(store, this.entries);
      this.dailyReports = this.loadDailyReports(this.entries);
      this.calendarMonth ??= latestReportMonth(this.dailyReports);
      this.error = null;
    } catch (e) {
      this.entries = [];
      this.dailyReports = [];
      this.detailSummaryIds = new Set();
      this.detailSummaryPaths = new Map();
      this.error = (e as Error).message;
    }
    this.render();
  }

  private async clearRunStateForMissingDailyReports(): Promise<void> {
    const writer = this.plugin.buildMarkdownWriter();
    const snapshot = this.plugin.stateStore.snapshot();
    let cleared = 0;
    for (const [date, entry] of Object.entries(snapshot)) {
      // Only completed runs imply a daily file should exist; failed/skipped
      // states are scheduling decisions and should survive file cleanup.
      if (entry.status !== "completed") continue;
      const path = normalizeVaultPath(writer.dailyPath(date));
      if (await this.plugin.app.vault.adapter.exists(path)) continue;
      await this.plugin.stateStore.clearDate(date);
      cleared += 1;
    }
    if (cleared > 0) {
      this.plugin.logger.info(
        `dashboard: cleared ${cleared} completed run-state entries for missing daily reports`,
      );
    }
  }

  private async loadDetailSummaries(
    store: ReturnType<ArxivDailyPlugin["buildPaperIndex"]>,
    entries: DashboardRow["entry"][],
  ): Promise<void> {
    const ids = new Set<string>();
    const paths = new Map<string, string>();
    const writer = this.plugin.buildMarkdownWriter();

    for (const entry of entries) {
      const path = await this.findDetailSummaryPath(
        entry,
        writer.paperDetailPath(entry.arxivId),
      );
      if (!path) continue;
      ids.add(entry.arxivId);
      paths.set(entry.arxivId, path);
      if (
        normalizeVaultPath(entry.paperPath ?? "") !== normalizeVaultPath(path)
      ) {
        try {
          const updated = await store.setPaperPath(entry.arxivId, path);
          entry.paperPath = updated?.paperPath ?? path;
        } catch (e) {
          this.plugin.logger.warn(
            `dashboard: failed to sync paperPath for ${entry.arxivId}`,
            e,
          );
        }
      }
    }

    this.detailSummaryIds = ids;
    this.detailSummaryPaths = paths;
  }

  private async findDetailSummaryPath(
    entry: DashboardRow["entry"],
    defaultPath: string,
  ): Promise<string | null> {
    const candidates = uniquePaths([entry.paperPath, defaultPath]);
    for (const path of candidates) {
      try {
        const normalized = normalizeVaultPath(path);
        if (!(await this.plugin.app.vault.adapter.exists(normalized))) continue;
        const markdown = await this.plugin.app.vault.adapter.read(normalized);
        if (looksLikeDetailSummary(markdown)) return normalized;
      } catch (e) {
        this.plugin.logger.warn(
          `dashboard: failed to inspect detail summary for ${entry.arxivId}`,
          e,
        );
      }
    }
    return null;
  }

  private loadDailyReports(entries: DashboardRow["entry"][]): DailyReportDay[] {
    const dailyDir = normalizeVaultPath(this.plugin.settings.output.dailyDir);
    const byDate = new Map<string, DailyReportDay>();

    for (const file of this.plugin.app.vault.getMarkdownFiles()) {
      const path = normalizeVaultPath(file.path);
      const date = dailyDateFromPath(path, dailyDir);
      if (!date) continue;
      byDate.set(date, {
        date,
        path,
        papers: 0,
        starred: 0,
      });
    }

    const counted = new Set<string>();
    for (const entry of entries) {
      for (const reportPath of entry.dailyReports) {
        const path = normalizeVaultPath(reportPath);
        const date = dailyDateFromPath(path, dailyDir);
        if (!date) continue;
        const report = byDate.get(date);
        if (!report) continue;
        const countKey = `${date}:${entry.arxivId}`;
        if (counted.has(countKey)) continue;
        counted.add(countKey);
        report.papers += 1;
        if (isStarredEntry(entry)) report.starred += 1;
      }
    }

    return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
  }

  private renderLoading(): void {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("arxiv-daily-dashboard");
    this.renderHeader(contentEl);
    contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__state",
      text: "Loading...",
    });
  }

  private render(): void {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("arxiv-daily-dashboard");
    this.renderHeader(contentEl);

    if (this.error) {
      contentEl.createEl("div", {
        cls: "arxiv-daily-dashboard__state arxiv-daily-dashboard__state--error",
        text: `Failed to load paper index: ${this.error}`,
      });
      return;
    }

    const result = queryDashboard(this.entries, this.query, {
      detailSummaryIds: this.detailSummaryIds,
    });
    this.renderToolbar(contentEl, result);

    const overview = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__overview",
    });
    const filterPanel = overview.createEl("div", {
      cls: "arxiv-daily-dashboard__overview-main",
    });
    const calendarPanel = overview.createEl("div", {
      cls: "arxiv-daily-dashboard__overview-calendar",
    });
    this.renderFilters(filterPanel);
    this.statsEl = filterPanel.createEl("div");
    this.renderDailyCalendar(calendarPanel);

    this.batchEl = contentEl.createEl("div");
    this.resultsEl = contentEl.createEl("div");
    this.renderCurrentResults();
  }

  private renderCurrentResults(): void {
    if (!this.statsEl || !this.batchEl || !this.resultsEl) return;
    const result = queryDashboard(this.entries, this.query, {
      detailSummaryIds: this.detailSummaryIds,
    });
    this.statsEl.empty();
    this.batchEl.empty();
    this.resultsEl.empty();
    const visibleRows = this.displayedRows(result.rows);
    this.renderStats(this.statsEl, result);
    this.renderBatchControls(this.batchEl, visibleRows, result.rows.length);
    if (result.rows.length === 0) {
      this.renderEmptyState(this.resultsEl, result);
      return;
    }
    this.renderTable(this.resultsEl, visibleRows);
  }

  private renderEmptyState(
    contentEl: HTMLElement,
    result: ReturnType<typeof queryDashboard>,
  ): void {
    const setup = getSetupStatus(this.plugin.settings);
    const state = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__state arxiv-daily-dashboard__empty",
    });

    if (!setup.readyToRun && this.entries.length === 0) {
      state.createEl("div", {
        cls: "arxiv-daily-dashboard__empty-title",
        text: "Finish setup before running arXiv Daily",
      });
      state.createEl("div", {
        cls: "arxiv-daily-dashboard__empty-desc",
        text: "Add your LLM settings and at least one research topic, then run today from the Dashboard.",
      });
      if (setup.reasons.length > 0) {
        const reasons = state.createEl("ul", {
          cls: "arxiv-daily-dashboard__empty-list",
        });
        for (const reason of setup.reasons) {
          reasons.createEl("li", { text: reason });
        }
      }
      const actions = this.createEmptyActions(state);
      this.createEmptyActionButton(actions, "settings", "Open Settings", (button) => {
        button.disabled = true;
        this.openSettings();
        button.disabled = false;
      });
      return;
    }

    if (this.entries.length === 0) {
      state.createEl("div", {
        cls: "arxiv-daily-dashboard__empty-title",
        text: "No papers indexed yet",
      });
      state.createEl("div", {
        cls: "arxiv-daily-dashboard__empty-desc",
        text: "Run today or run pending dates to create daily reports and populate this Dashboard.",
      });
      const actions = this.createEmptyActions(state);
      this.createEmptyActionButton(actions, "play", "Run Today", (button) => {
        void this.runControlAction(button, () => this.runToday());
      });
      this.createEmptyActionButton(actions, "layers", "Run Pending", (button) => {
        void this.runControlAction(button, () => this.runAllPending());
      });
      return;
    }

    if (this.hasActiveFilters()) {
      state.createEl("div", {
        cls: "arxiv-daily-dashboard__empty-title",
        text: "No papers match these filters",
      });
      state.createEl("div", {
        cls: "arxiv-daily-dashboard__empty-desc",
        text: "Reset filters to return to the current reading list.",
      });
      const actions = this.createEmptyActions(state);
      this.createEmptyActionButton(actions, "rotate-ccw", "Reset Filters", () => {
        this.resetFilters();
      });
      return;
    }

    if ((this.query.tab ?? "starred") === "starred" && result.tabCounts.all > 0) {
      state.createEl("div", {
        cls: "arxiv-daily-dashboard__empty-title",
        text: "No starred papers yet",
      });
      state.createEl("div", {
        cls: "arxiv-daily-dashboard__empty-desc",
        text: "Star the papers worth returning to. Use All to browse everything already indexed.",
      });
      const actions = this.createEmptyActions(state);
      this.createEmptyActionButton(actions, "list", "Show All", () => {
        this.query = { ...this.query, tab: "all" };
        this.render();
      });
      return;
    }

    state.createEl("div", {
      cls: "arxiv-daily-dashboard__empty-title",
      text: "No papers in this view",
    });
    state.createEl("div", {
      cls: "arxiv-daily-dashboard__empty-desc",
      text: "Run arXiv Daily again or adjust your topic settings if this looks unexpected.",
    });
  }

  private createEmptyActions(parent: HTMLElement): HTMLElement {
    return parent.createEl("div", {
      cls: "arxiv-daily-dashboard__empty-actions",
    });
  }

  private createEmptyActionButton(
    parent: HTMLElement,
    icon: string,
    label: string,
    onClick: (button: HTMLButtonElement) => void,
  ): HTMLButtonElement {
    const button = parent.createEl("button", {
      cls: "arxiv-daily-dashboard__empty-action",
      attr: { type: "button" },
    }) as HTMLButtonElement;
    setIcon(button, icon);
    button.createSpan({ text: label });
    button.addEventListener("click", () => onClick(button));
    return button;
  }

  private hasActiveFilters(): boolean {
    return Boolean(
      this.query.search ||
        (this.query.topics?.length ?? 0) > 0 ||
        (this.query.statuses?.length ?? 0) > 0 ||
        (this.query.priorities?.length ?? 0) > 0 ||
        this.query.dateFrom ||
        this.query.dateTo ||
        this.query.detailSummary != null,
    );
  }

  private resetFilters(): void {
    this.query = {
      tab: this.query.tab ?? "starred",
      ...(this.query.sort ? { sort: this.query.sort } : {}),
    };
    this.render();
  }

  private openSettings(): void {
    const settings = (this.plugin.app as any).setting;
    if (settings?.open && settings?.openTabById) {
      settings.open();
      settings.openTabById(this.plugin.manifest.id);
      return;
    }
    new Notice("Open Settings -> Community plugins -> arXiv Daily.");
  }

  private createSettingsButton(parent: HTMLElement): void {
    appendSettingsButton(parent, () => this.openSettings());
  }

  private renderHeader(contentEl: HTMLElement): void {
    const header = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__header",
    });
    header.createEl("h2", { text: "arXiv Daily Dashboard" });
  }

  private renderToolbar(
    contentEl: HTMLElement,
    result: ReturnType<typeof queryDashboard>,
  ): void {
    const toolbar = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__toolbar",
    });
    const tabs = toolbar.createEl("div", {
      cls: "arxiv-daily-dashboard__tabs",
    });
    const active = this.query.tab ?? "starred";
    for (const tab of DASHBOARD_TABS) {
      const button = tabs.createEl("button", {
        cls: "arxiv-daily-dashboard__tab",
        attr: {
          type: "button",
          "aria-pressed": String(tab.id === active),
        },
      });
      if (tab.id === active) button.addClass("is-active");
      button.createSpan({ text: tab.label });
      button.createSpan({
        cls: "arxiv-daily-dashboard__tab-count",
        text: String(result.tabCounts[tab.id]),
      });
      button.addEventListener("click", () => {
        this.query = { ...this.query, tab: tab.id };
        this.render();
      });
    }
    this.renderToolbarFilter(
      tabs,
      "Detail summary",
      this.query.detailSummary === true,
      this.countToolbarFilter((entry) =>
        this.detailSummaryIds.has(entry.arxivId),
      ),
      () => {
        this.query = {
          ...this.query,
          detailSummary:
            this.query.detailSummary === true ? undefined : true,
        };
        this.render();
      },
    );

    const actions = toolbar.createEl("div", {
      cls: "arxiv-daily-dashboard__toolbar-actions",
    });
    this.createToolbarButton(
      actions,
      "refresh-cw",
      "Refresh",
      "Refresh dashboard",
      (button) => {
        void this.runControlAction(button, () => this.reloadIndex());
      },
    );
    this.createToolbarButton(
      actions,
      "play",
      "Run Today",
      "Run today",
      (button) => {
        void this.runControlAction(button, () => this.runToday());
      },
    );
    this.createToolbarButton(
      actions,
      "file-text",
      "Summarize by ID",
      "Summarize paper by arXiv ID",
      (_button, evt) => {
        void this.runDashboardCommand("arxiv-daily-summarize-by-id", false);
      },
    );
    this.createToolbarButton(
      actions,
      "more-horizontal",
      "More",
      "More arXiv Daily actions",
      (_button, evt) => this.showMoreMenu(evt),
    );
    this.createSettingsButton(actions);
  }

  private renderDailyCalendar(contentEl: HTMLElement): void {
    const section = contentEl.createEl("section", {
      cls: "arxiv-daily-dashboard__calendar",
    });
    const header = section.createEl("div", {
      cls: "arxiv-daily-dashboard__calendar-header",
    });
    header.createEl("h3", { text: "Daily reports" });

    const controls = header.createEl("div", {
      cls: "arxiv-daily-dashboard__calendar-controls",
    });
    const today = this.todayDate();
    const todayMonth = today.slice(0, 7);
    const month =
      this.calendarMonth ?? latestReportMonth(this.dailyReports) ?? todayMonth;
    const todayButton = controls.createEl("button", {
      cls: "arxiv-daily-dashboard__calendar-today",
      text: "Today",
      attr: {
        type: "button",
        "aria-label": "Go to current month",
      },
    }) as HTMLButtonElement;
    const prev = controls.createEl("button", {
      cls: "clickable-icon",
      attr: { type: "button", "aria-label": "Previous month" },
    }) as HTMLButtonElement;
    setIcon(prev, "chevron-left");
    controls.createEl("span", {
      cls: "arxiv-daily-dashboard__calendar-month",
      text: month || "No reports",
    });
    const next = controls.createEl("button", {
      cls: "clickable-icon",
      attr: { type: "button", "aria-label": "Next month" },
    }) as HTMLButtonElement;
    setIcon(next, "chevron-right");

    if (!month) {
      todayButton.disabled = true;
      prev.disabled = true;
      next.disabled = true;
      section.createEl("div", {
        cls: "arxiv-daily-dashboard__state",
        text: "No daily reports found.",
      });
      return;
    }

    todayButton.disabled = month === todayMonth;
    todayButton.addEventListener("click", () => {
      this.calendarMonth = todayMonth;
      this.render();
    });
    prev.addEventListener("click", () => {
      this.calendarMonth = shiftMonth(month, -1);
      this.render();
    });
    next.addEventListener("click", () => {
      this.calendarMonth = shiftMonth(month, 1);
      this.render();
    });

    const weekdays = section.createEl("div", {
      cls: "arxiv-daily-dashboard__calendar-weekdays",
    });
    for (const label of ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]) {
      weekdays.createSpan({ text: label });
    }

    const grid = section.createEl("div", {
      cls: "arxiv-daily-dashboard__calendar-grid",
    });

    // Use the new buildCalendarCells method
    for (const cell of this.buildCalendarCells(month)) {
      const button = grid.createEl("button", {
        cls: this.getCalendarCellClasses(cell),
        attr: {
          type: "button",
          "aria-label": this.getCalendarCellAriaLabel(cell),
        },
      }) as HTMLButtonElement;

      if (!cell.date) {
        button.disabled = true;
        button.addClass("is-empty");
        continue;
      }

      // Date number
      button.createSpan({
        cls: "arxiv-daily-dashboard__calendar-day-number",
        text: String(Number(cell.date.slice(-2))),
      });

      // Today indicator
      if (cell.date === today) button.addClass("is-today");

      // State-specific rendering
      switch (cell.state) {
        case "has-report":
          this.renderReportCell(button, cell);
          break;
        case "runnable":
          this.renderRunnableCell(button, cell);
          break;
      }
    }
  }

  private renderToolbarFilter(
    parent: HTMLElement,
    label: string,
    active: boolean,
    count: number,
    onClick: () => void,
  ): void {
    const button = parent.createEl("button", {
      cls: "arxiv-daily-dashboard__tab arxiv-daily-dashboard__tab--filter",
      attr: {
        type: "button",
        "aria-pressed": String(active),
      },
    });
    if (active) button.addClass("is-active");
    button.createSpan({ text: label });
    button.createSpan({
      cls: "arxiv-daily-dashboard__tab-count",
      text: String(count),
    });
    button.addEventListener("click", onClick);
  }

  private countToolbarFilter(
    predicate: (entry: DashboardRow["entry"]) => boolean,
  ): number {
    const active = this.query.tab ?? "starred";
    return this.entries.filter((entry) => {
      if (active === "starred" && !isStarredEntry(entry)) return false;
      if (active === "all" && entry.status === "ignored") return false;
      return predicate(entry);
    }).length;
  }

  private todayDate(): string {
    return formatDate(
      todayInTz(new Date(), this.plugin.settings.arxiv.timezone),
    );
  }

  private getLookbackDates(): Set<string> {
    const dates = new Set<string>();
    const today = todayInTz(new Date(), this.plugin.settings.arxiv.timezone);
    const lookbackDays = 5; // LOOKBACK_DAYS from scheduler.ts

    for (let i = 0; i < lookbackDays; i++) {
      const date = daysBefore(today, i);
      if (!isWeekendDate(date)) {
        dates.add(formatDate(date));
      }
    }

    return dates;
  }

  private buildCalendarCells(month: string): CalendarCell[] {
    const cells: CalendarCell[] = [];
    const byDate = new Map(this.dailyReports.map(r => [r.date, r]));
    const lookbackDates = this.getLookbackDates();

    for (const cellDate of calendarCells(month)) {
      if (!cellDate.date) {
        cells.push({ date: null, state: "empty" });
        continue;
      }

      const report = byDate.get(cellDate.date);

      let state: CalendarCellState;
      if (report) {
        state = "has-report";
      } else if (lookbackDates.has(cellDate.date)) {
        state = "runnable";
      } else {
        state = "empty";
      }

      cells.push({
        date: cellDate.date,
        state,
        report,
      });
    }

    return cells;
  }

  private getCalendarCellClasses(cell: CalendarCell): string {
    const classes = ["arxiv-daily-dashboard__calendar-day"];

    if (!cell.date) {
      classes.push("is-empty");
    } else if (cell.state === "has-report") {
      classes.push("has-report");
    } else if (cell.state === "runnable") {
      classes.push("is-runnable");
    }

    return classes.join(" ");
  }

  private getCalendarCellAriaLabel(cell: CalendarCell): string {
    if (!cell.date) {
      return "Empty calendar cell";
    }

    if (cell.state === "has-report" && cell.report) {
      return `Open daily report ${cell.report.date}: ${cell.report.papers} indexed papers${cell.report.starred ? `, ${cell.report.starred} starred` : ""}`;
    }

    if (cell.state === "runnable") {
      return `Click to run for ${cell.date}`;
    }

    return `No daily report ${cell.date}`;
  }

  private renderRunnableCell(button: HTMLButtonElement, cell: CalendarCell): void {
    button.addClass("is-runnable");

    // Set tooltip for CSS ::after pseudo-element
    button.setAttribute("data-tooltip", `Click to run for ${cell.date}`);

    // Play icon
    const icon = button.createSpan({
      cls: "arxiv-daily-dashboard__calendar-day-icon",
    });
    setIcon(icon, "play");

    // Click handler to run
    button.addEventListener("click", () => {
      void this.runDateFromCalendar(cell.date!);
    });
  }

  private renderReportCell(button: HTMLButtonElement, cell: CalendarCell): void {
    if (!cell.report) return;

    button.addClass("has-report");
    button.createSpan({
      cls: "arxiv-daily-dashboard__calendar-day-count",
      text: String(cell.report.papers),
    });
    button.addEventListener("click", () => {
      void openMarkdownFileOnce(this.plugin.app, cell.report!.path);
    });
  }

  private async runDateFromCalendar(date: string): Promise<void> {
    const setup = getSetupStatus(this.plugin.settings);
    if (!setup.readyToRun) {
      new Notice("arXiv Daily: Please complete setup first");
      this.openSettings();
      return;
    }

    new Notice(`arXiv Daily: running for ${date}…`);
    const result = await this.plugin.scheduler.runForDateNow(date);
    new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);

    // Refresh dashboard
    await this.reloadIndex();
  }

  private renderStats(
    contentEl: HTMLElement,
    result: ReturnType<typeof queryDashboard>,
  ): void {
    const stats = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__stats",
    });
    const items = [
      ["Shown", result.stats.total],
      ["This week", result.stats.weekAdded],
      ["Starred", result.stats.starred],
      ["Details", result.rows.filter((row) => row.hasDetailSummary).length],
    ] as const;

    for (const [label, value] of items) {
      const item = stats.createEl("div", {
        cls: "arxiv-daily-dashboard__stat",
      });
      item.createEl("span", {
        cls: "arxiv-daily-dashboard__stat-value",
        text: String(value),
      });
      item.createEl("span", {
        cls: "arxiv-daily-dashboard__stat-label",
        text: label,
      });
    }
  }

  private renderFilters(contentEl: HTMLElement): void {
    const filters = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__filters",
    });

    const search = this.createFilterField(
      filters,
      "Search",
      "arxiv-daily-dashboard__filter--search",
    ).createEl("input", {
      attr: {
        type: "search",
        placeholder: "ID, title, author, topic, summary",
      },
    }) as HTMLInputElement;
    search.value = this.query.search ?? "";
    search.addEventListener("input", () => {
      this.query = { ...this.query, search: search.value.trim() || undefined };
      this.renderCurrentResults();
    });

    const topic = this.createSelect(
      this.createFilterField(
        filters,
        "Topic",
        "arxiv-daily-dashboard__filter--topic",
      ),
      [
        { value: "", label: "Any topic" },
        ...topicOptions(this.entries).map((value) => ({ value, label: value })),
      ],
      this.query.topics?.[0] ?? "",
    );
    topic.addEventListener("change", () => {
      this.query = {
        ...this.query,
        topics: topic.value ? [topic.value] : undefined,
      };
      this.renderCurrentResults();
    });

    const dateField = this.createFilterField(
      filters,
      "Date",
      "arxiv-daily-dashboard__filter--date-range",
    );
    const dateInputs = dateField.createEl("div", {
      cls: "arxiv-daily-dashboard__date-range",
    });
    dateInputs.createSpan({
      cls: "arxiv-daily-dashboard__date-range-prefix",
      text: "from",
    });
    const dateFrom = dateInputs.createEl("input", {
      attr: { type: "date", "aria-label": "Date from" },
    }) as HTMLInputElement;
    dateFrom.value = this.query.dateFrom ?? "";
    dateFrom.addEventListener("change", () => {
      this.query = {
        ...this.query,
        dateFrom: dateFrom.value || undefined,
      };
      this.renderCurrentResults();
    });
    dateInputs.createSpan({
      cls: "arxiv-daily-dashboard__date-range-separator",
      text: "to",
    });

    const dateTo = dateInputs.createEl("input", {
      attr: { type: "date", "aria-label": "Date to" },
    }) as HTMLInputElement;
    dateTo.value = this.query.dateTo ?? "";
    dateTo.addEventListener("change", () => {
      this.query = {
        ...this.query,
        dateTo: dateTo.value || undefined,
      };
      this.renderCurrentResults();
    });

    const resetWrap = filters.createEl("div", {
      cls: "arxiv-daily-dashboard__filter-reset-wrap",
    });
    const reset = resetWrap.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__filter-reset",
      attr: {
        type: "button",
        "aria-label": "Reset filters",
      },
    });
    setIcon(reset, "rotate-ccw");
    reset.addEventListener("click", () => {
      this.resetFilters();
    });
  }

  private createFilterField(
    parent: HTMLElement,
    label: string,
    cls?: string,
  ): HTMLElement {
    const field = parent.createEl("label", {
      cls: cls
        ? `arxiv-daily-dashboard__filter ${cls}`
        : "arxiv-daily-dashboard__filter",
    });
    field.createEl("span", {
      cls: "arxiv-daily-dashboard__filter-label",
      text: label,
    });
    return field;
  }

  private createSelect(
    parent: HTMLElement,
    options: Array<{ value: string; label: string }>,
    selected: string,
  ): HTMLSelectElement {
    const select = parent.createEl("select") as HTMLSelectElement;
    for (const option of options) {
      const el = select.createEl("option", { text: option.label });
      el.value = option.value;
    }
    select.value = selected;
    return select;
  }

  private renderTable(contentEl: HTMLElement, rows: DashboardRow[]): void {
    const scroller = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__table-wrap",
    });
    const table = scroller.createEl("table", {
      cls: "arxiv-daily-dashboard__table",
    });
    const thead = table.createEl("thead");
    const headRow = thead.createEl("tr");
    const selectAllCell = headRow.createEl("th");
    const selectAll = selectAllCell.createEl("input", {
      attr: {
        type: "checkbox",
        "aria-label": "Select visible papers",
      },
    }) as HTMLInputElement;
    const visibleIds = rows.map((row) => row.arxivId);
    const selectedVisible = visibleIds.filter((id) => this.selectedIds.has(id));
    selectAll.checked =
      visibleIds.length > 0 && selectedVisible.length === visibleIds.length;
    selectAll.indeterminate =
      selectedVisible.length > 0 && selectedVisible.length < visibleIds.length;
    selectAll.addEventListener("change", () => {
      for (const id of visibleIds) {
        if (selectAll.checked) this.selectedIds.add(id);
        else this.selectedIds.delete(id);
      }
      this.renderCurrentResults();
    });

    for (const label of [
      "Star",
      "Title",
      "Topic",
      "Published",
      "Actions",
    ]) {
      headRow.createEl("th", { text: label });
    }

    const tbody = table.createEl("tbody");
    for (const row of rows) {
      const tr = tbody.createEl("tr");
      const selectCell = tr.createEl("td");
      const checkbox = selectCell.createEl("input", {
        attr: {
          type: "checkbox",
          "aria-label": `Select ${row.arxivId}`,
        },
      }) as HTMLInputElement;
      checkbox.checked = this.selectedIds.has(row.arxivId);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) this.selectedIds.add(row.arxivId);
        else this.selectedIds.delete(row.arxivId);
        this.renderCurrentResults();
      });

      const markCell = tr.createEl("td");
      this.createStarToggle(markCell, row.entry);

      const titleCell = tr.createEl("td", {
        cls: "arxiv-daily-dashboard__title-cell",
      });
      titleCell.createEl("div", {
        cls: "arxiv-daily-dashboard__title",
        text: row.title,
      });
      titleCell.createEl("div", {
        cls: "arxiv-daily-dashboard__meta",
        text: `${row.arxivId} · ${row.authors || "Unknown authors"}`,
      });

      tr.createEl("td", { text: row.topic });
      tr.createEl("td", { text: row.entry.published || "-" });
      const actionCell = tr.createEl("td", {
        cls: "arxiv-daily-dashboard__actions",
      });
      this.createIconButton(
        actionCell,
        row.hasDetailSummary ? "file-text" : "sparkles",
        row.hasDetailSummary
          ? "Open detail summary"
          : "Summarize by arXiv ID",
        (button) => {
          void this.runControlAction(button, () => {
            return row.hasDetailSummary
              ? this.openDetailSummary(row.entry)
              : this.summarizeDetailById(row.entry);
          });
        },
      );
      this.createIconButton(actionCell, "calendar", "Open daily report", (button) => {
        void this.runControlAction(button, () =>
          this.openDailyReport(row.entry),
        );
      });
      this.createIconButton(actionCell, "external-link", "Open arXiv", (button) => {
        void this.runControlAction(button, async () => {
          openUrl(row.entry.arxivUrl, "arXiv");
        });
      });
      this.createIconButton(actionCell, "file-down", "Open PDF", (button) => {
        void this.runControlAction(button, () => this.openPdf(row.entry));
      });
      this.createIconButton(actionCell, "download", "Download PDF", (button) => {
        void this.runControlAction(button, () =>
          this.downloadPdf(row.entry),
        );
      });
    }
  }

  private renderBatchControls(
    contentEl: HTMLElement,
    visibleRows: DashboardRow[],
    totalRows: number,
  ): void {
    const selectedCount = this.selectedIds.size;
    const toolbar = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__batch",
    });
    const actions = toolbar.createEl("div", {
      cls: "arxiv-daily-dashboard__batch-actions",
    });
    actions.createEl("span", {
      cls: "arxiv-daily-dashboard__batch-count",
      text: `${selectedCount} selected`,
    });

    this.createBatchButton(
      actions,
      "star",
      "Star",
      selectedCount,
      () =>
        this.runBatchStar(true),
    );
    this.createBatchButton(
      actions,
      "star-off",
      "Unstar",
      selectedCount,
      () =>
        this.runBatchStar(false),
    );
    this.createBatchButton(
      actions,
      "trash-2",
      "Delete summary",
      this.selectedDetailSummaryCount(),
      () =>
        this.runBatchDeleteSummary(),
      { warning: true },
    );

    const clear = actions.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__batch-icon",
      attr: {
        type: "button",
        "aria-label": "Clear selection",
      },
    }) as HTMLButtonElement;
    setIcon(clear, "x");
    clear.disabled = selectedCount === 0;
    clear.addEventListener("click", () => {
      this.selectedIds.clear();
      this.renderCurrentResults();
    });

    const controls = toolbar.createEl("div", {
      cls: "arxiv-daily-dashboard__batch-controls",
    });
    controls.createEl("span", {
      cls: "arxiv-daily-dashboard__batch-showing",
      text: `Showing ${visibleRows.length} of ${totalRows}`,
    });
    this.renderDisplayControl(controls);
    this.renderSortControl(controls);

    if (visibleRows.length === 0) toolbar.addClass("is-empty");
  }

  private renderDisplayControl(parent: HTMLElement): void {
    const field = parent.createEl("label", {
      cls: "arxiv-daily-dashboard__batch-display",
    });
    field.createSpan({
      cls: "arxiv-daily-dashboard__batch-display-label",
      text: "Display",
    });
    const display = this.createSelect(
      field,
      DISPLAY_OPTIONS.map((option) => ({
        value: option.value,
        label: option.label,
      })),
      this.displayLimit,
    );
    display.addEventListener("change", () => {
      if (isDashboardDisplayLimit(display.value)) {
        this.displayLimit = display.value;
        this.renderCurrentResults();
      }
    });
  }

  private renderSortControl(parent: HTMLElement): void {
    const field = parent.createEl("label", {
      cls: "arxiv-daily-dashboard__batch-sort",
    });
    field.createSpan({
      cls: "arxiv-daily-dashboard__batch-sort-label",
      text: "Sort",
    });
    const sort = this.createSelect(
      field,
      SORT_OPTIONS.map((option) => ({
        value: option.value,
        label: option.label,
      })),
      sortValue(this.query.sort),
    );
    sort.addEventListener("change", () => {
      this.query = {
        ...this.query,
        sort: sortQuery(sort.value),
      };
      this.renderCurrentResults();
    });
  }

  private displayedRows(rows: DashboardRow[]): DashboardRow[] {
    if (this.displayLimit === "all") return rows;
    return rows.slice(0, Number(this.displayLimit));
  }

  private createStarToggle(
    parent: HTMLElement,
    entry: DashboardRow["entry"],
  ): void {
    const starred = isStarredEntry(entry);
    const label = starred ? "Unstar paper" : "Star paper";
    const button = parent.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__star",
      attr: {
        type: "button",
        "aria-label": label,
        "aria-pressed": String(starred),
      },
    }) as HTMLButtonElement;
    if (starred) button.addClass("is-starred");
    setIcon(button, "star");
    button.addEventListener("click", () => {
      void this.runControlAction(button, () =>
        this.updateStar(entry, !starred),
      );
    });
  }

  private createToolbarButton(
    parent: HTMLElement,
    icon: string,
    label: string,
    title: string,
    onClick: (button: HTMLButtonElement, evt: MouseEvent) => void,
  ): void {
    const button = parent.createEl("button", {
      cls: "arxiv-daily-dashboard__toolbar-button",
      attr: {
        type: "button",
        "aria-label": title,
      },
    }) as HTMLButtonElement;
    setIcon(button, icon);
    button.createSpan({ text: label });
    button.addEventListener("click", (evt) => onClick(button, evt));
  }

  private showMoreMenu(evt: MouseEvent): void {
    const menu = new Menu();
    const enabled = this.plugin.settings.schedule.enabled;
    const activeRuns = this.plugin.scheduler.activeRuns();

    menu.addItem((item) =>
      item
        .setTitle(`Scheduler: ${enabled ? "Enabled" : "Disabled"}`)
        .setIcon(enabled ? "circle-check" : "circle-slash")
        .setDisabled(true),
    );
    menu.addItem((item) =>
      item
        .setTitle(enabled ? "Disable scheduler" : "Enable scheduler")
        .setIcon(enabled ? "pause" : "play")
        .onClick(async () => {
          const applied = await this.plugin.setScheduleEnabled(!enabled);
          if (applied) {
            new Notice(`arXiv Daily: ${!enabled ? "enabled" : "disabled"}`);
          }
        }),
    );

    menu.addSeparator();
    this.addCommandMenuItem(menu, "Run for date...", "calendar", "arxiv-daily-run-for-date");
    this.addCommandMenuItem(
      menu,
      "Force run for date...",
      "rotate-cw",
      "arxiv-daily-force-run-for-date",
    );
    menu.addItem((item) =>
      item
        .setTitle("Retry failed dates")
        .setIcon("refresh-cw")
        .onClick(() => {
          void this.retryFailedInLookback();
        }),
    );
    this.addCommandMenuItem(
      menu,
      "Cancel current run",
      "circle-stop",
      "arxiv-daily-cancel-current-run",
      true,
      activeRuns.length === 0,
    );

    menu.addSeparator();
    menu.addItem((item) =>
      item
        .setTitle("Run Pending")
        .setIcon("layers")
        .onClick(() => {
          void this.runAllPending();
        }),
    );

    menu.addSeparator();
    this.addCommandMenuItem(
      menu,
      "Show recent run state",
      "list",
      "arxiv-daily-show-state",
    );
    this.addCommandMenuItem(
      menu,
      "Show diagnostics",
      "clipboard-list",
      "arxiv-daily-show-diagnostics",
    );
    this.addCommandMenuItem(
      menu,
      "Clear run state...",
      "trash-2",
      "arxiv-daily-clear-run-state",
    );

    menu.showAtMouseEvent(evt);
  }

  private addCommandMenuItem(
    menu: Menu,
    label: string,
    icon: string,
    commandId: string,
    refreshAfter = false,
    disabled = false,
  ): void {
    menu.addItem((item) =>
      item
        .setTitle(label)
        .setIcon(icon)
        .setDisabled(disabled)
        .onClick(() => {
          void this.runDashboardCommand(commandId, refreshAfter).catch((e) => {
            new Notice(`arXiv Daily: ${(e as Error).message}`, 10_000);
          });
        }),
    );
  }

  private createIconButton(
    parent: HTMLElement,
    icon: string,
    label: string,
    onClick: (button: HTMLButtonElement) => void,
    disabled = false,
  ): void {
    const button = parent.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__action",
      attr: {
        type: "button",
        "aria-label": label,
      },
    }) as HTMLButtonElement;
    setIcon(button, icon);
    button.disabled = disabled;
    button.addEventListener("click", () => onClick(button));
  }

  private createBatchButton(
    parent: HTMLElement,
    icon: string,
    label: string,
    selectedCount: number,
    action: () => Promise<void>,
    options: { warning?: boolean } = {},
  ): void {
    const button = parent.createEl("button", {
      cls: "arxiv-daily-dashboard__batch-button",
      attr: {
        type: "button",
        "aria-label": label,
      },
    }) as HTMLButtonElement;
    setIcon(button, icon);
    button.createSpan({ text: label });
    if (options.warning) button.addClass("mod-warning");
    button.disabled = selectedCount === 0;
    button.addEventListener("click", () => {
      void this.runControlAction(button, action);
    });
  }

  private async runControlAction(
    control: HTMLButtonElement,
    action: () => Promise<void>,
  ): Promise<void> {
    control.disabled = true;
    try {
      await action();
    } catch (e) {
      new Notice(`arXiv Daily: ${(e as Error).message}`, 10_000);
    } finally {
      control.disabled = false;
    }
  }

  private gateFilter(): boolean {
    const validation = validateFilterConfig(this.plugin.settings);
    if (!validation.ok) {
      new Notice(
        `arXiv Daily - cannot run:\n${validation.reasons.map((reason) => `- ${reason}`).join("\n")}`,
        10_000,
      );
      return false;
    }
    return true;
  }

  private gateLlm(): boolean {
    const validation = validateLlmConfig(this.plugin.settings);
    if (!validation.ok) {
      new Notice(
        `arXiv Daily - cannot summarize:\n${validation.reasons.map((reason) => `- ${reason}`).join("\n")}`,
        10_000,
      );
      return false;
    }
    return true;
  }

  private async runToday(): Promise<void> {
    if (!this.gateFilter()) return;
    const date = this.todayDate();
    new Notice(`arXiv Daily: running for ${date}...`);
    const result = await this.plugin.scheduler.runForDateNow(date);
    new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
    await this.reloadIndex();
  }

  private async runAllPending(): Promise<void> {
    if (!this.gateFilter()) return;
    new Notice("arXiv Daily: running all pending in lookback...");
    const results = await this.plugin.scheduler.runAllPending();
    if (results.length === 0) {
      new Notice("arXiv Daily: nothing pending in lookback window");
      return;
    }
    new Notice(`arXiv Daily (lookback):\n${describeRunResults(results)}`, 10_000);
    await this.reloadIndex();
  }

  private async retryFailedInLookback(): Promise<void> {
    if (!this.gateFilter()) return;
    new Notice("arXiv Daily: retrying failed dates in lookback...");
    const results = await this.plugin.scheduler.retryFailedInLookback();
    if (results.length === 0) {
      new Notice("arXiv Daily: no failed dates in lookback window");
      return;
    }
    new Notice(`arXiv Daily retry:\n${describeRunResults(results)}`, 10_000);
    await this.reloadIndex();
  }

  private async runDashboardCommand(
    commandId: string,
    refreshAfter: boolean,
  ): Promise<void> {
    const executed = await executeObsidianCommand(
      this.plugin.app,
      commandId,
      this.plugin.manifest.id,
    );
    if (!executed) {
      throw new Error(`command not found: ${commandId}`);
    }
    if (refreshAfter) await this.reloadIndex();
  }

  private async updateStar(
    entry: DashboardRow["entry"],
    starred: boolean,
  ): Promise<void> {
    const store = this.plugin.buildPaperIndex();
    const updated = await store.setPriority(
      entry.arxivId,
      starred ? "high" : "normal",
    );
    if (!updated) throw new Error(`${entry.arxivId} is not in papers.json`);
    new Notice(
      `arXiv Daily: ${entry.arxivId} ${starred ? "starred" : "unstarred"}`,
    );
    await this.reloadIndex();
  }

  private async openDetailSummary(entry: DashboardRow["entry"]): Promise<void> {
    const path = this.detailSummaryPaths.get(entry.arxivId);
    if (!path) {
      new Notice(`arXiv Daily: ${entry.arxivId} has no detail summary`);
      return;
    }
    await openMarkdownFileOnce(this.plugin.app, path);
  }

  private async summarizeDetailById(
    entry: DashboardRow["entry"],
  ): Promise<void> {
    if (!this.gateLlm()) return;
    new Notice(`arXiv Daily: summarizing ${entry.arxivId}...`);
    const result = await this.plugin.manualFetch.fetchAndSummarize(
      entry.arxivId,
      this.todayDate(),
    );
    new Notice(`arXiv Daily: ${describeManualResult(result)}`, 10_000);
    if (result.kind !== "done" && result.kind !== "already_exists") return;
    await this.reloadIndex();
    await openMarkdownFileOnce(this.plugin.app, result.path);
  }

  private async openDailyReport(entry: DashboardRow["entry"]): Promise<void> {
    const path = entry.dailyReports[0];
    if (!path) {
      new Notice(`arXiv Daily: ${entry.arxivId} has no daily report`);
      return;
    }
    await openMarkdownFileOnce(this.plugin.app, path);
  }

  private async openPdf(entry: DashboardRow["entry"]): Promise<void> {
    if (entry.pdfPath.trim()) {
      await this.plugin.app.workspace.openLinkText(entry.pdfPath, "", false);
      return;
    }
    openUrl(entry.pdfUrl, "PDF");
  }

  private async downloadPdf(entry: DashboardRow["entry"]): Promise<void> {
    const result = await this.plugin.buildPdfService().downloadForEntry(entry);
    if (result.kind !== "done") {
      new Notice(`arXiv Daily: PDF download failed - ${result.reason}`, 10_000);
      return;
    }
    new Notice(
      `arXiv Daily: downloaded PDF for ${result.arxivId} -> ${result.path}`,
      10_000,
    );
    await this.reloadIndex();
  }

  private selectedArxivIds(): string[] {
    return [...this.selectedIds];
  }

  private selectedDetailSummaryCount(): number {
    return this.selectedArxivIds().filter((id) =>
      this.detailSummaryIds.has(id),
    ).length;
  }

  private async runBatchStar(starred: boolean): Promise<void> {
    await this.runBatchAction({
      type: "set_priority",
      arxivIds: this.selectedArxivIds(),
      priority: starred ? "high" : "normal",
    });
  }

  private async runBatchDeleteSummary(): Promise<void> {
    const entries = this.selectedArxivIds()
      .map((id) => this.entries.find((entry) => entry.arxivId === id))
      .filter((entry): entry is DashboardRow["entry"] =>
        Boolean(entry && this.detailSummaryPaths.has(entry.arxivId)),
      );
    if (entries.length === 0) {
      new Notice("arXiv Daily: no selected papers have detail summaries");
      return;
    }

    const choice = await chooseModal(
      this.plugin.app,
      "Delete detail summaries",
      `Delete ${entries.length} detail summary file${entries.length === 1 ? "" : "s"} from the vault? Daily report entries will stay in the Dashboard.`,
      [
        { label: "Cancel", value: "cancel" },
        { label: "Delete", value: "delete", warning: true },
      ],
    );
    if (choice !== "delete") return;

    const store = this.plugin.buildPaperIndex();
    let deletedFiles = 0;
    for (const path of uniquePaths(
      entries.map((entry) => this.detailSummaryPaths.get(entry.arxivId)),
    )) {
      try {
        if (!(await this.plugin.app.vault.adapter.exists(path))) continue;
        await this.plugin.app.vault.adapter.remove(path);
        deletedFiles += 1;
      } catch (e) {
        this.plugin.logger.warn(
          `dashboard: failed to delete detail summary ${path}`,
          e,
        );
        throw new Error(`failed to delete ${path}: ${(e as Error).message}`);
      }
    }

    const clearIds = entries
      .filter((entry) => entry.dailyReports.length > 0)
      .map((entry) => entry.arxivId);
    const removeIds = entries
      .filter((entry) => entry.dailyReports.length === 0)
      .map((entry) => entry.arxivId);
    const cleared = await store.clearPaperDetails(clearIds);
    const removed = await store.removePapers(removeIds);

    this.selectedIds.clear();
    new Notice(
      `arXiv Daily: deleted ${deletedFiles} summaries, cleared ${cleared}, removed ${removed} orphan entries`,
      10_000,
    );
    await this.reloadIndex();
  }

  private async runBatchAction(action: DashboardAction): Promise<void> {
    const plan = planDashboardAction(this.entries, action);
    if (plan.patches.length === 0) {
      new Notice("arXiv Daily: no selected papers need changes");
      return;
    }

    const store = this.plugin.buildPaperIndex();
    let changed = 0;
    for (const patch of plan.patches) {
      const entry = await this.applyBatchPatch(store, patch);
      if (!entry) continue;
      changed += 1;
    }
    this.selectedIds.clear();
    new Notice(`arXiv Daily: updated ${changed} papers`);
    await this.reloadIndex();
  }

  private async applyBatchPatch(
    store: ReturnType<ArxivDailyPlugin["buildPaperIndex"]>,
    patch: DashboardPatch,
  ): Promise<DashboardRow["entry"] | null> {
    let entry = await store.get(patch.arxivId);
    if (!entry) return null;
    if (patch.status) {
      entry = await store.setStatus(patch.arxivId, patch.status);
      if (!entry) return null;
    }
    if (patch.priority) {
      entry = await store.setPriority(patch.arxivId, patch.priority);
      if (!entry) return null;
    }
    return entry;
  }
}

function topicOptions(entries: DashboardRow["entry"][]): string[] {
  const topics = new Set<string>();
  for (const entry of entries) {
    for (const topic of [entry.primaryTopic, ...entry.topics]) {
      const trimmed = topic.trim();
      if (trimmed) topics.add(trimmed);
    }
  }
  return [...topics].sort((a, b) => a.localeCompare(b));
}

function sortValue(sort: DashboardQuery["sort"]): string {
  const key = sort?.key ?? "priority";
  const direction = sort?.direction ?? "asc";
  return `${key}:${direction}`;
}

function sortQuery(value: string): DashboardQuery["sort"] {
  const option =
    SORT_OPTIONS.find((candidate) => candidate.value === value) ??
    SORT_OPTIONS[0];
  if (option.value === SORT_OPTIONS[0].value) return undefined;
  return {
    key: option.key,
    direction: option.direction,
  };
}

function isDashboardDisplayLimit(value: string): value is DashboardDisplayLimit {
  return DISPLAY_OPTIONS.some((option) => option.value === value);
}

function openUrl(url: string, label: string): void {
  if (!url.trim()) {
    new Notice(`arXiv Daily: no ${label} URL`);
    return;
  }
  window.open(url, "_blank", "noopener");
}

function isStarredEntry(entry: DashboardRow["entry"]): boolean {
  return entry.status !== "ignored" && entry.priority === "high";
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return Boolean(
    value &&
      (typeof value === "object" || typeof value === "function") &&
      typeof (value as { then?: unknown }).then === "function",
  );
}

export async function executeObsidianCommand(
  app: unknown,
  commandId: string,
  pluginId?: string,
): Promise<boolean> {
  const commands = (app as {
    commands?: {
      executeCommandById?: (id: string) => unknown;
      commands?: Record<
        string,
        {
          callback?: () => unknown;
          checkCallback?: (checking: boolean) => unknown;
        }
      >;
    };
  })?.commands;
  if (!commands) return false;
  const ids = commandId.includes(":")
    ? [commandId]
    : uniqueCommandIds([
        pluginId ? `${pluginId}:${commandId}` : "",
        commandId,
      ]);
  const registeredIds = ids.filter((id) => commands.commands?.[id]);
  const executableIds = registeredIds.length ? registeredIds : ids;

  if (typeof commands.executeCommandById === "function") {
    for (const id of executableIds) {
      const result = commands.executeCommandById(id);
      if (isPromiseLike(result)) await result;
      if (result !== false) return true;
    }
    return false;
  }

  const id = registeredIds[0];
  const command = id ? commands.commands?.[id] : undefined;
  if (!command) return false;
  const callback = command.callback;
  if (typeof callback === "function") {
    const result = callback();
    if (isPromiseLike(result)) await result;
    return result !== false;
  }
  const checkCallback = command.checkCallback;
  if (typeof checkCallback === "function") {
    const result = checkCallback(false);
    if (isPromiseLike(result)) await result;
    return result !== false;
  }
  return false;
}

function uniqueCommandIds(ids: string[]): string[] {
  const out: string[] = [];
  for (const id of ids) {
    if (!id || out.includes(id)) continue;
    out.push(id);
  }
  return out;
}

function describeResult(result: any): string {
  if (!result) return "no result";
  if (result.kind === "completed") return `done (${result.papersWritten} papers)`;
  if (result.kind === "failed_transient") return `transient: ${result.reason}`;
  if (result.kind === "failed_permanent") return `permanent: ${result.reason}`;
  if (result.kind === "skipped") return `skipped: ${result.reason}`;
  return JSON.stringify(result);
}

function describeManualResult(result: any): string {
  if (!result) return "no result";
  if (result.kind === "done") return `done -> ${result.path}`;
  if (result.kind === "already_exists") return `already exists at ${result.path}`;
  if (result.kind === "not_found") return `not found: ${result.reason}`;
  if (result.kind === "no_html") return `no full text: ${result.reason}`;
  if (result.kind === "error") return `error: ${result.reason}`;
  return JSON.stringify(result);
}

function describeRunResults(results: Array<{ date: string; result: any }>): string {
  return results
    .map((entry) => `${entry.date}: ${describeResult(entry.result)}`)
    .join("\n");
}

function normalizeVaultPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
}

function markdownPathFromLeaf(leaf: unknown): string | null {
  const candidate = leaf as {
    getViewState?: () => { state?: { file?: unknown } };
    view?: { file?: { path?: unknown } };
  };
  const stateFile = candidate.getViewState?.().state?.file;
  if (typeof stateFile === "string") return stateFile;
  const viewPath = candidate.view?.file?.path;
  return typeof viewPath === "string" ? viewPath : null;
}

function uniquePaths(paths: Array<string | null | undefined>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const path of paths) {
    const normalized = normalizeVaultPath(path ?? "");
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function dailyDateFromPath(path: string, dailyDir: string): string | null {
  const normalized = normalizeVaultPath(path);
  const prefix = `${dailyDir}/`;
  if (!normalized.startsWith(prefix)) return null;
  const rest = normalized.slice(prefix.length);
  const match = /^(\d{4}-\d{2}-\d{2})\.md$/i.exec(rest);
  return match?.[1] ?? null;
}

function latestReportMonth(reports: DailyReportDay[]): string | null {
  const latest = reports[reports.length - 1]?.date;
  return latest ? latest.slice(0, 7) : null;
}

function shiftMonth(month: string, delta: number): string {
  const [year, monthIndex] = month.split("-").map(Number);
  const date = new Date(Date.UTC(year, monthIndex - 1 + delta, 1));
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
}

function calendarCells(month: string): Array<{ date: string | null }> {
  const [year, monthIndex] = month.split("-").map(Number);
  const first = new Date(Date.UTC(year, monthIndex - 1, 1));
  const daysInMonth = new Date(Date.UTC(year, monthIndex, 0)).getUTCDate();
  const mondayOffset = (first.getUTCDay() + 6) % 7;
  const cells: Array<{ date: string | null }> = Array.from(
    { length: mondayOffset },
    () => ({ date: null }),
  );
  for (let day = 1; day <= daysInMonth; day++) {
    cells.push({
      date: `${month}-${String(day).padStart(2, "0")}`,
    });
  }
  while (cells.length % 7 !== 0) cells.push({ date: null });
  return cells;
}
