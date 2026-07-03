import {
  ItemView,
  Menu,
  Modal,
  Notice,
  setIcon,
  type App,
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
import { syncDashboardHistory, type DashboardMarkdownFile } from "./history-sync";
import {
  validateFilterConfig,
  validateLlmConfig,
} from "../settings/validation";
import type { RunStateEntry } from "../settings/types";
import { daysBefore, formatDate, isTimeWithinLocalWindow, isWeekendDate, todayInTz } from "../utils/time";
import { getSetupStatus, logSetupStatus } from "../onboarding";
import { chooseModal } from "../services/modal";
import { buildDiagnosticsReport } from "../services/diagnostics";
import { formatRunHistoryRecords } from "../services/run-history";

export const ARXIV_DAILY_DASHBOARD_VIEW = "arxiv-daily-dashboard";
const RECENT_DATES_FOREGROUND_TIMEOUT_MS = 3000;

const DASHBOARD_TABS: Array<{ id: DashboardTab; label: string }> = [
  { id: "all", label: "All" },
  { id: "starred", label: "Starred" },
];

const SORT_LABELS: Record<DashboardSortKey, string> = {
  priority: "Starred first",
  published: "Published",
  topic: "Topic",
  title: "Title",
};

const DEFAULT_SORT_KEY: DashboardSortKey = "priority";

export interface DashboardPage<T> {
  rows: T[];
  total: number;
  totalPages: number;
  currentPage: number;
  start: number;
  end: number;
  pageSize: number;
}

export interface DailyReportDay {
  date: string;
  path: string;
  papers: number;
  starred: number;
}

export type CalendarCellState =
  | "empty"        // No date or outside lookback
  | "runnable"     // Can generate report
  | "has-report"   // Report exists
  | "no-relevant-papers"; // LLM filtered to 0 relevant papers

export type CalendarEmptyReason =
  | "blank"
  | "arxiv-not-updated"
  | "future"
  | "before-tracking"
  | "report-missing";

export interface CalendarCell {
  date: string | null;
  state: CalendarCellState;
  report?: DailyReportDay;
  emptyReason?: CalendarEmptyReason;
}

export interface CalendarCellResolution {
  state: CalendarCellState;
  emptyReason?: CalendarEmptyReason;
}

export interface CalendarRunWhitelistInput {
  date: string;
  today: string;
  now: Date;
  timezone: string;
  runAtLocal: string;
  runUntilLocal: string;
  inLookback: boolean;
  isWeekend: boolean;
  hasDailyReport: boolean;
  recentDates: ReadonlySet<string>;
  runState?: RunStateEntry;
}

export function resolveCalendarCellState({
  report,
  runnable,
  runState,
  emptyReason,
}: {
  report?: { papers: number };
  runnable: boolean;
  runState?: RunStateEntry;
  emptyReason?: CalendarEmptyReason;
}): CalendarCellResolution {
  if (report) {
    return {
      state: report.papers === 0 ? "no-relevant-papers" : "has-report",
    };
  }

  if (isArxivNotUpdatedRunState(runState)) {
    return { state: "empty", emptyReason: "arxiv-not-updated" };
  }

  if (isCompletedRunState(runState)) {
    return { state: "empty", emptyReason: "report-missing" };
  }

  if (runnable) return { state: "runnable" };

  return emptyReason
    ? { state: "empty", emptyReason }
    : { state: "empty" };
}

export function isCalendarRunWhitelisted(
  input: CalendarRunWhitelistInput,
): boolean {
  if (!input.inLookback) return false;
  if (input.isWeekend) return false;
  if (input.hasDailyReport) return false;
  if (isRunStateBlockedForCalendarRun(input.runState)) return false;

  if (input.date === input.today) {
    return isTimeWithinLocalWindow(
      input.now,
      input.timezone,
      input.runAtLocal,
      input.runUntilLocal,
    );
  }

  return input.recentDates.has(input.date);
}

export interface CalendarEmptyReasonInput {
  date: string;
  today: string;
  trackingStartDate: string;
  recentDates: ReadonlySet<string>;
}

export interface CalendarDailyReportMapInput {
  month: string;
  scannedReports: DailyReportDay[];
  runState: Record<string, RunStateEntry | undefined>;
  dailyPath: (date: string) => string;
  exists: (path: string) => Promise<boolean>;
  normalizePath: (path: string) => string;
}

export function resolveCalendarEmptyReason(
  input: CalendarEmptyReasonInput,
): CalendarEmptyReason {
  if (input.date > input.today) return "future";
  if (
    input.date < input.trackingStartDate &&
    !input.recentDates.has(input.date)
  ) {
    return "before-tracking";
  }
  return "arxiv-not-updated";
}

export function calendarCellAriaLabel(cell: CalendarCell): string | undefined {
  if (!cell.date) {
    return undefined;
  }

  if (cell.state === "has-report" && cell.report) {
    return `${cell.report.papers} indexed papers${cell.report.starred ? `, ${cell.report.starred} starred` : ""}`;
  }

  if (cell.state === "no-relevant-papers") {
    return "No relevant papers";
  }

  if (cell.state === "runnable") {
    return "Run daily report";
  }

  if (cell.emptyReason === "arxiv-not-updated") {
    return "arXiv not updated";
  }

  if (cell.emptyReason === "report-missing") {
    return "Daily report missing";
  }

  if (cell.emptyReason === "future") {
    return "Future date";
  }

  return undefined;
}

export function collectIndexedDetailSummaryRefs(
  entries: DashboardRow["entry"][],
): { ids: Set<string>; paths: Map<string, string> } {
  const ids = new Set<string>();
  const paths = new Map<string, string>();
  for (const entry of entries) {
    const path = normalizeVaultPath(entry.paperPath ?? "");
    if (!entry.detail || !path) continue;
    ids.add(entry.arxivId);
    paths.set(entry.arxivId, path);
  }
  return { ids, paths };
}

export async function buildCalendarDailyReportMap(
  input: CalendarDailyReportMapInput,
): Promise<Map<string, DailyReportDay>> {
  const scannedByDate = new Map(
    input.scannedReports.map((report) => [report.date, report]),
  );
  const out = new Map<string, DailyReportDay>();

  for (const cell of calendarCells(input.month)) {
    if (!cell.date) continue;
    const path = input.normalizePath(input.dailyPath(cell.date));
    if (!(await input.exists(path))) continue;
    const scanned = scannedByDate.get(cell.date);
    out.set(
      cell.date,
      scanned
        ? { ...scanned, path: input.normalizePath(scanned.path) }
        : {
            date: cell.date,
            path,
            papers: completedPapersWritten(input.runState[cell.date]) ?? 0,
            starred: 0,
          },
    );
  }

  return out;
}

function completedPapersWritten(runState?: RunStateEntry): number | undefined {
  if (runState?.status !== "completed") return undefined;
  return runState.papersWritten;
}

function isArxivNotUpdatedRunState(runState?: RunStateEntry): boolean {
  return (
    runState?.status === "skipped" ||
    runState?.status === "failed_permanent" ||
    (runState?.status === "completed" && runState.papersWritten === 0)
  );
}

function isRunStateBlockedForCalendarRun(runState?: RunStateEntry): boolean {
  return (
    runState?.status === "running" ||
    runState?.status === "skipped" ||
    runState?.status === "failed_permanent" ||
    runState?.status === "completed"
  );
}

function isCompletedRunState(runState?: RunStateEntry): boolean {
  return runState?.status === "completed";
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
    plugin.logger.info("ArXiv Daily: no workspace leaf available");
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

export const DEFAULT_LOG_LEVELS: ReadonlySet<string> = new Set(["debug", "info", "warn", "error"]);

const LOG_LEVEL_TAG = /\[(DEBUG|INFO|WARN|ERROR)\]/;

function parseLogLevelTag(line: string): string | null {
  const m = line.match(LOG_LEVEL_TAG);
  return m ? m[1].toLowerCase() : null;
}

export interface FormatLogEntriesOptions {
  /** Levels to keep. Defaults to all four levels. */
  levels?: Set<string>;
}

export function formatLogEntries(
  buffer: string[],
  opts: FormatLogEntriesOptions = {},
): string {
  if (buffer.length === 0) return "(no log entries)";
  const levels = opts.levels ?? DEFAULT_LOG_LEVELS;
  const kept: string[] = [];
  // Iterate oldest→newest, push allowed; untagged lines are kept.
  for (const line of buffer) {
    const lvl = parseLogLevelTag(line);
    if (lvl === null || levels.has(lvl)) kept.push(line);
  }
  if (kept.length === 0) return "(no log entries at this level)";
  // Reverse so newest is on top.
  return kept.reverse().join("\n");
}

class ArxivDailyDashboardView extends ItemView {
  private entries: DashboardRow["entry"][] = [];
  private dailyReports: DailyReportDay[] = [];
  private calendarDailyReports = new Map<string, DailyReportDay>();
  private detailSummaryIds = new Set<string>();
  private detailSummaryPaths = new Map<string, string>();
  private calendarMonth: string | null = null;
  private query: DashboardQuery = { tab: "starred" };
  private pageSize = 20;
  private currentPage = 0;
  private error: string | null = null;
  private selectedIds = new Set<string>();
  private batchEl: HTMLElement | null = null;
  private statsEl: HTMLElement | null = null;
  private resultsEl: HTMLElement | null = null;
  private recentDatesNotice: string | null = null;
  private recentDatesRefresh: Promise<unknown> | null = null;
  private lastSyncedDailyPaths: Set<string> | null = null;
  private isOpen = false;

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
    this.isOpen = true;
    await this.reloadIndex();
  }

  async onClose(): Promise<void> {
    this.isOpen = false;
    this.contentEl.empty();
  }

  private async reloadIndex(): Promise<void> {
    this.renderLoading();
    try {
      this.refreshRecentDatesForForeground();
      const allFiles = this.plugin.app.vault.getMarkdownFiles();
      const dailyDir = normalizeVaultPath(this.plugin.settings.output.dailyDir);
      const papersDir = normalizeVaultPath(this.plugin.settings.output.papersDir);
      const markdownFiles = filterDashboardMarkdownFiles(
        allFiles,
        dailyDir,
        papersDir,
      );
      this.plugin.logger.info(
        `dashboard: scanning ${markdownFiles.length}/${allFiles.length} files (${dailyDir}, ${papersDir})`,
      );
      const dailyPaths = dailyFilePathSet(markdownFiles, dailyDir);
      if (
        shouldSkipDashboardHistorySync(
          this.lastSyncedDailyPaths,
          dailyPaths,
          this.entries.length,
        )
      ) {
        this.error = null;
        this.plugin.logger.info(
          `dashboard: skipped history sync for ${dailyPaths.size} unchanged daily files`,
        );
        this.render();
        return;
      }
      const store = this.plugin.buildPaperIndex();
      const index = await syncDashboardHistory({
        vault: this.plugin.app.vault,
        store,
        output: this.plugin.settings.output,
        topics: this.plugin.settings.arxiv.topics,
        markdownFiles,
        logger: this.plugin.logger,
      });
      this.lastSyncedDailyPaths = dailyPaths;
      this.entries = Object.values(index.papers);
      this.loadDetailSummaries(this.entries);
      this.dailyReports = this.loadDailyReports(this.entries, markdownFiles);
      this.calendarMonth ??= this.todayDate().slice(0, 7);
      await this.refreshCalendarDailyReports(
        this.calendarMonth ?? this.todayDate().slice(0, 7),
      );
      this.error = null;
    } catch (e) {
      this.entries = [];
      this.dailyReports = [];
      this.calendarDailyReports = new Map();
      this.detailSummaryIds = new Set();
      this.detailSummaryPaths = new Map();
      this.error = (e as Error).message;
    }
    this.render();
  }

  private async refreshRecentDatesForForeground(): Promise<void> {
    const result = await this.plugin.recentDates.refreshWithin(
      RECENT_DATES_FOREGROUND_TIMEOUT_MS,
    );
    this.recentDatesNotice = result.timedOut
      ? recentDatesFallbackNotice(result.snapshot.refreshedAt)
      : result.snapshot.error
        ? `arXiv recent dates partially refreshed: ${result.snapshot.error}`
        : null;
    if (result.completed) return;

    const refresh = result.refresh
      .then((snapshot) => {
        this.recentDatesNotice = snapshot.error
          ? `arXiv recent dates partially refreshed: ${snapshot.error}`
          : null;
        if (this.isOpen) this.render();
      })
      .catch((e) => {
        this.recentDatesNotice = `arXiv recent dates refresh failed: ${(e as Error).message}`;
        if (this.isOpen) this.render();
      })
      .finally(() => {
        if (this.recentDatesRefresh === refresh) {
          this.recentDatesRefresh = null;
        }
      });
    this.recentDatesRefresh = refresh;
  }

  private loadDetailSummaries(entries: DashboardRow["entry"][]): void {
    const refs = collectIndexedDetailSummaryRefs(entries);
    this.detailSummaryIds = refs.ids;
    this.detailSummaryPaths = refs.paths;
  }

  private loadDailyReports(
    entries: DashboardRow["entry"][],
    markdownFiles: DashboardMarkdownFile[] = this.plugin.app.vault.getMarkdownFiles(),
  ): DailyReportDay[] {
    const dailyDir = normalizeVaultPath(this.plugin.settings.output.dailyDir);
    const byDate = new Map<string, DailyReportDay>();

    for (const file of markdownFiles) {
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

  private async refreshCalendarDailyReports(month: string): Promise<void> {
    const writer = this.plugin.buildMarkdownWriter();
    this.calendarDailyReports = await buildCalendarDailyReportMap({
      month,
      scannedReports: this.dailyReports,
      runState: this.plugin.stateStore.snapshot(),
      dailyPath: (date) => writer.dailyPath(date),
      exists: (path) => this.plugin.app.vault.adapter.exists(path),
      normalizePath: normalizeVaultPath,
    });
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
    this.renderRecentDatesNotice(contentEl);

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

  private renderRecentDatesNotice(contentEl: HTMLElement): void {
    if (!this.recentDatesNotice) return;
    contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__notice",
      text: this.recentDatesNotice,
    });
  }

  private renderCurrentResults(): void {
    if (!this.statsEl || !this.batchEl || !this.resultsEl) return;
    const result = queryDashboard(this.entries, this.query, {
      detailSummaryIds: this.detailSummaryIds,
    });
    this.statsEl.empty();
    this.batchEl.empty();
    this.resultsEl.empty();
    const page = paginateDashboardRows(
      result.rows,
      this.currentPage,
      this.pageSize,
    );
    this.currentPage = page.currentPage;
    this.renderStats(this.statsEl, result);
    this.renderBatchControls(this.batchEl, page);
    if (result.rows.length === 0) {
      this.renderEmptyState(this.resultsEl, result);
      return;
    }
    this.renderTable(this.resultsEl, page.rows);
    this.renderPaginationControls(this.resultsEl, page);
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
    this.currentPage = 0;
    this.render();
  }

  private openSettings(): void {
    this.plugin.logger.info("dashboard: open settings requested");
    const settings = (this.plugin.app as any).setting;
    if (settings?.open && settings?.openTabById) {
      settings.open();
      settings.openTabById(this.plugin.manifest.id);
      return;
    }
    this.notice("Open Settings -> Community plugins -> arXiv Daily.");
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
        this.currentPage = 0;
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
        this.currentPage = 0;
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
      void this.refreshCalendarDailyReports(todayMonth).then(() => this.render());
    });
    prev.addEventListener("click", () => {
      const nextMonth = shiftMonth(month, -1);
      this.calendarMonth = nextMonth;
      void this.refreshCalendarDailyReports(nextMonth).then(() => this.render());
    });
    next.addEventListener("click", () => {
      const nextMonth = shiftMonth(month, 1);
      this.calendarMonth = nextMonth;
      void this.refreshCalendarDailyReports(nextMonth).then(() => this.render());
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
        },
      }) as HTMLButtonElement;
      const ariaLabel = this.getCalendarCellAriaLabel(cell);
      if (ariaLabel) button.setAttribute("aria-label", ariaLabel);

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
        case "no-relevant-papers":
          this.renderNoRelevantPapersCell(button, cell);
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

  private isRunnable(
    date: string,
    runState: RunStateEntry | undefined,
    hasDailyReport: boolean,
  ): boolean {
    const today = this.todayDate();
    const parsed = parseCalendarDate(date);
    const settings = this.plugin.settings;

    return isCalendarRunWhitelisted({
      date,
      today,
      now: new Date(),
      timezone: settings.arxiv.timezone,
      runAtLocal: settings.schedule.runAtLocal,
      runUntilLocal: settings.schedule.runUntilLocal,
      inLookback: this.getLookbackDates().has(date),
      isWeekend: parsed ? isWeekendDate(parsed) : false,
      hasDailyReport,
      recentDates: this.plugin.recentDates.snapshot().dates,
      runState,
    });
  }

  private getCalendarEmptyReason(date: string): CalendarEmptyReason {
    return resolveCalendarEmptyReason({
      date,
      today: this.todayDate(),
      trackingStartDate: this.getTrackingStartDate(),
      recentDates: this.plugin.recentDates.snapshot().dates,
    });
  }

  private getTrackingStartDate(): string {
    const candidates = [
      ...this.dailyReports.map((report) => report.date),
      ...Object.keys(this.plugin.stateStore.snapshot()),
    ].sort();
    return candidates[0] ?? this.todayDate();
  }

  private buildCalendarCells(month: string): CalendarCell[] {
    const cells: CalendarCell[] = [];
    const byDate = this.calendarDailyReports;
    const runState = this.plugin.stateStore.snapshot();

    for (const cellDate of calendarCells(month)) {
      if (!cellDate.date) {
        cells.push({ date: null, state: "empty", emptyReason: "blank" });
        continue;
      }

      const report = byDate.get(cellDate.date);
      const dateRunState = runState[cellDate.date];
      const resolution = resolveCalendarCellState({
        report,
        runnable: this.isRunnable(cellDate.date, dateRunState, Boolean(report)),
        runState: dateRunState,
        emptyReason: this.getCalendarEmptyReason(cellDate.date),
      });

      cells.push({
        date: cellDate.date,
        state: resolution.state,
        emptyReason: resolution.emptyReason,
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
    } else if (cell.state === "no-relevant-papers") {
      classes.push("has-report");
      classes.push("no-relevant-papers");
    } else if (cell.state === "runnable") {
      classes.push("is-runnable");
    }

    return classes.join(" ");
  }

  private getCalendarCellAriaLabel(cell: CalendarCell): string | undefined {
    return calendarCellAriaLabel(cell);
  }

  private renderRunnableCell(button: HTMLButtonElement, cell: CalendarCell): void {
    button.addClass("is-runnable");

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

  private renderNoRelevantPapersCell(button: HTMLButtonElement, cell: CalendarCell): void {
    button.addClass("has-report");
    button.addClass("no-relevant-papers");

    // Show "0" as the count
    button.createSpan({
      cls: "arxiv-daily-dashboard__calendar-day-count",
      text: "0",
    });

    // Click handler to open the report
    if (cell.report) {
      button.addEventListener("click", () => {
        void openMarkdownFileOnce(this.plugin.app, cell.report!.path);
      });
    }
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
      logSetupStatus(this.plugin.logger, "dashboard calendar run blocked", setup);
      this.notice("arXiv Daily: Please complete setup first");
      this.openSettings();
      return;
    }

    await this.plugin.recentDates.refresh();
    if (date !== this.todayDate() && !this.plugin.recentDates.hasDate(date)) {
      this.notice(`arXiv Daily ${date}: arXiv not updated`);
      await this.reloadIndex();
      return;
    }

    this.plugin.logger.info(`dashboard: manual calendar run requested for ${date}`);
    this.notice(`arXiv Daily: running for ${date}…`);
    const result = await this.plugin.scheduler.runForDateNow(date, {
      trigger: "calendar",
    });
    this.notice(`arXiv Daily ${date}: ${describeResult(result)}`);

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
      this.currentPage = 0;
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
      this.currentPage = 0;
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
      this.currentPage = 0;
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
      this.currentPage = 0;
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
          openUrl(row.entry.arxivUrl, "arXiv", this.plugin.logger);
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
    page: DashboardPage<DashboardRow>,
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
      text:
        page.total === 0
          ? "Showing 0 of 0 papers"
          : `Showing ${page.start}-${page.end} of ${page.total} papers`,
    });
    this.renderSortControl(controls);

    if (page.rows.length === 0) toolbar.addClass("is-empty");
  }

  private renderSortControl(parent: HTMLElement): void {
    const field = parent.createEl("label", {
      cls: "arxiv-daily-dashboard__batch-sort",
    });
    field.createSpan({
      cls: "arxiv-daily-dashboard__batch-sort-label",
      text: "Sort",
    });
    const currentKey = this.query.sort?.key ?? DEFAULT_SORT_KEY;
    const currentDir = this.query.sort?.direction ?? "asc";

    const keySelect = this.createSelect(
      field,
      Object.entries(SORT_LABELS).map(([key, label]) => ({
        value: key,
        label,
      })),
      currentKey,
    );
    keySelect.addEventListener("change", () => {
      const newKey = keySelect.value as DashboardSortKey;
      if (newKey === currentKey) return;
      this.query = {
        ...this.query,
        sort: { key: newKey, direction: currentDir },
      };
      this.currentPage = 0;
      this.renderCurrentResults();
    });

    const dirIcon = currentDir === "asc" ? "arrow-up" : "arrow-down";
    const dirButton = field.createEl("button", {
      cls: "clickable-icon",
      attr: { type: "button", "aria-label": "Toggle sort direction" },
    }) as HTMLButtonElement;
    setIcon(dirButton, dirIcon);
    dirButton.addEventListener("click", () => {
      const newDir: DashboardSortDirection = currentDir === "asc" ? "desc" : "asc";
      this.query = {
        ...this.query,
        sort: { key: currentKey, direction: newDir },
      };
      this.currentPage = 0;
      this.renderCurrentResults();
    });
  }

  private renderPaginationControls(
    contentEl: HTMLElement,
    page: DashboardPage<DashboardRow>,
  ): void {
    const controls = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__pagination",
    });
    const prev = controls.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__pagination-button",
      attr: {
        type: "button",
        "aria-label": "Previous page",
      },
    }) as HTMLButtonElement;
    setIcon(prev, "chevron-left");
    prev.disabled = page.currentPage === 0;
    prev.addEventListener("click", () => {
      this.setPage(page.currentPage - 1);
    });

    controls.createEl("span", {
      cls: "arxiv-daily-dashboard__pagination-label",
      text: `Page ${page.currentPage + 1} / ${page.totalPages}`,
    });

    const next = controls.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__pagination-button",
      attr: {
        type: "button",
        "aria-label": "Next page",
      },
    }) as HTMLButtonElement;
    setIcon(next, "chevron-right");
    next.disabled = page.currentPage >= page.totalPages - 1;
    next.addEventListener("click", () => {
      this.setPage(page.currentPage + 1);
    });

    controls.createEl("span", {
      cls: "arxiv-daily-dashboard__pagination-size",
      text: `Show ${page.pageSize} per page`,
    });
  }

  private setPage(n: number): void {
    this.currentPage = n;
    this.renderCurrentResults();
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
            this.notice(`arXiv Daily: ${!enabled ? "enabled" : "disabled"}`);
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
    menu.addItem((item) =>
      item
        .setTitle("Show logs & history")
        .setIcon("scroll-text")
        .onClick(() => {
          new HubModal(this.plugin.app, this.plugin).open();
        }),
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
            this.plugin.logger.warn(`dashboard command failed: ${commandId}`, e);
            this.notice(`arXiv Daily: ${(e as Error).message}`, 10_000);
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
      this.plugin.logger.warn("dashboard control action failed", e);
      this.notice(`arXiv Daily: ${(e as Error).message}`, 10_000);
    } finally {
      control.disabled = false;
    }
  }

  private notice(message: string, timeoutMs?: number): void {
    this.plugin.logger.info(message);
    new Notice(message, timeoutMs);
  }

  private gateFilter(): boolean {
    const validation = validateFilterConfig(this.plugin.settings);
    if (!validation.ok) {
      this.plugin.logger.info(
        `dashboard: filter validation failed (${validation.reasons.join("; ")})`,
      );
      this.notice(
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
      this.plugin.logger.info(
        `dashboard: LLM validation failed (${validation.reasons.join("; ")})`,
      );
      this.notice(
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
    this.plugin.logger.info(`dashboard: manual run today requested for ${date}`);
    this.notice(`arXiv Daily: running for ${date}...`);
    await this.plugin.recentDates.refresh();
    const result = await this.plugin.scheduler.runForDateNow(date);
    this.notice(`arXiv Daily ${date}: ${describeResult(result)}`);
    await this.reloadIndex();
  }

  private async runAllPending(): Promise<void> {
    if (!this.gateFilter()) return;
    this.plugin.logger.info("dashboard: run all pending requested");
    this.notice("arXiv Daily: running all pending in lookback...");
    await this.plugin.recentDates.refresh();
    const results = await this.plugin.scheduler.runAllPending();
    if (results.length === 0) {
      this.notice("arXiv Daily: nothing pending in lookback window");
      return;
    }
    this.notice(`arXiv Daily (lookback):\n${describeRunResults(results)}`, 10_000);
    await this.reloadIndex();
  }

  private async retryFailedInLookback(): Promise<void> {
    if (!this.gateFilter()) return;
    this.plugin.logger.info("dashboard: retry failed dates requested");
    this.notice("arXiv Daily: retrying failed dates in lookback...");
    const results = await this.plugin.scheduler.retryFailedInLookback();
    if (results.length === 0) {
      this.notice("arXiv Daily: no failed dates in lookback window");
      return;
    }
    this.notice(`arXiv Daily retry:\n${describeRunResults(results)}`, 10_000);
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
    this.notice(
      `arXiv Daily: ${entry.arxivId} ${starred ? "starred" : "unstarred"}`,
    );
    await this.reloadIndex();
  }

  private async openDetailSummary(entry: DashboardRow["entry"]): Promise<void> {
    const path = this.detailSummaryPaths.get(entry.arxivId);
    if (!path) {
      this.notice(`arXiv Daily: ${entry.arxivId} has no detail summary`);
      return;
    }
    await openMarkdownFileOnce(this.plugin.app, path);
  }

  private async summarizeDetailById(
    entry: DashboardRow["entry"],
  ): Promise<void> {
    if (!this.gateLlm()) return;
    this.plugin.logger.info(`dashboard: summarize detail requested for ${entry.arxivId}`);
    this.notice(`arXiv Daily: summarizing ${entry.arxivId}...`);
    const result = await this.plugin.manualFetch.fetchAndSummarize(
      entry.arxivId,
      this.todayDate(),
    );
    this.notice(`arXiv Daily: ${describeManualResult(result)}`, 10_000);
    if (result.kind !== "done" && result.kind !== "already_exists") return;
    await this.reloadIndex();
    await openMarkdownFileOnce(this.plugin.app, result.path);
  }

  private async openDailyReport(entry: DashboardRow["entry"]): Promise<void> {
    const path = entry.dailyReports[0];
    if (!path) {
      this.notice(`arXiv Daily: ${entry.arxivId} has no daily report`);
      return;
    }
    await openMarkdownFileOnce(this.plugin.app, path);
  }

  private async openPdf(entry: DashboardRow["entry"]): Promise<void> {
    if (entry.pdfPath.trim()) {
      await this.plugin.app.workspace.openLinkText(entry.pdfPath, "", false);
      return;
    }
    openUrl(entry.pdfUrl, "PDF", this.plugin.logger);
  }

  private async downloadPdf(entry: DashboardRow["entry"]): Promise<void> {
    this.plugin.logger.info(`dashboard: PDF download requested for ${entry.arxivId}`);
    const result = await this.plugin.buildPdfService().downloadForEntry(entry);
    if (result.kind !== "done") {
      this.plugin.logger.warn(
        `dashboard: PDF download failed for ${entry.arxivId}: ${result.reason}`,
      );
      this.notice(`arXiv Daily: PDF download failed - ${result.reason}`, 10_000);
      return;
    }
    this.notice(
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
      this.notice("arXiv Daily: no selected papers have detail summaries");
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

    this.plugin.logger.info(
      `dashboard: deleting detail summaries for ${entries.length} selected papers`,
    );
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
    this.notice(
      `arXiv Daily: deleted ${deletedFiles} summaries, cleared ${cleared}, removed ${removed} orphan entries`,
      10_000,
    );
    await this.reloadIndex();
  }

  private async runBatchAction(action: DashboardAction): Promise<void> {
    this.plugin.logger.info(`dashboard: batch action requested (${describeDashboardAction(action)})`);
    const plan = planDashboardAction(this.entries, action);
    if (plan.patches.length === 0) {
      this.notice("arXiv Daily: no selected papers need changes");
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
    this.notice(`arXiv Daily: updated ${changed} papers`);
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

function openUrl(
  url: string,
  label: string,
  logger: ArxivDailyPlugin["logger"],
): void {
  if (!url.trim()) {
    logger.info(`arXiv Daily: no ${label} URL`);
    new Notice(`arXiv Daily: no ${label} URL`);
    return;
  }
  logger.info(`dashboard: opening ${label} URL ${url}`);
  window.open(url, "_blank", "noopener");
}

function describeDashboardAction(action: DashboardAction): string {
  const count = action.arxivIds.length;
  if (action.type === "set_priority") {
    return `set_priority:${action.priority}:${count}`;
  }
  if (action.type === "set_status") {
    return `set_status:${action.status}:${count}`;
  }
  return `${action.type}:${count}`;
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

function recentDatesFallbackNotice(refreshedAt: number): string {
  if (!refreshedAt) {
    return "arXiv recent dates are still refreshing in the background.";
  }
  const refreshed = new Date(refreshedAt).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  return `arXiv recent dates are still refreshing in the background; using cached data from ${refreshed}.`;
}

function normalizeVaultPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
}

export function filterDashboardMarkdownFiles<T extends DashboardMarkdownFile>(
  files: T[],
  dailyDir: string,
  papersDir: string,
): T[] {
  const normalizedDailyDir = normalizeVaultPath(dailyDir);
  const normalizedPapersDir = normalizeVaultPath(papersDir);
  return files.filter((file) => {
    const path = normalizeVaultPath(file.path);
    return (
      path.startsWith(`${normalizedDailyDir}/`) ||
      path.startsWith(`${normalizedPapersDir}/`)
    );
  });
}

function dailyFilePathSet(
  files: DashboardMarkdownFile[],
  dailyDir: string,
): Set<string> {
  const normalizedDailyDir = normalizeVaultPath(dailyDir);
  return new Set(
    files
      .map((file) => normalizeVaultPath(file.path))
      .filter((path) => path.startsWith(`${normalizedDailyDir}/`)),
  );
}

export function shouldSkipDashboardHistorySync(
  previousDailyPaths: ReadonlySet<string> | null,
  currentDailyPaths: ReadonlySet<string>,
  currentEntryCount: number,
): boolean {
  if (!previousDailyPaths || currentEntryCount === 0) return false;
  if (previousDailyPaths.size !== currentDailyPaths.size) return false;
  for (const path of currentDailyPaths) {
    if (!previousDailyPaths.has(path)) return false;
  }
  return true;
}

export function paginateDashboardRows<T>(
  rows: T[],
  currentPage: number,
  pageSize: number,
): DashboardPage<T> {
  const safePageSize = Math.max(1, Math.floor(pageSize));
  const total = rows.length;
  const totalPages = Math.ceil(total / safePageSize) || 1;
  const clampedPage = Math.max(
    0,
    Math.min(Math.floor(currentPage), totalPages - 1),
  );
  const offset = clampedPage * safePageSize;
  const pageRows = rows.slice(offset, offset + safePageSize);
  return {
    rows: pageRows,
    total,
    totalPages,
    currentPage: clampedPage,
    start: pageRows.length === 0 ? 0 : offset + 1,
    end: offset + pageRows.length,
    pageSize: safePageSize,
  };
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

function parseCalendarDate(date: string): { y: number; m: number; d: number } | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date);
  if (!match) return null;
  return {
    y: Number(match[1]),
    m: Number(match[2]),
    d: Number(match[3]),
  };
}

type HubModalTab = "logs" | "history" | "state" | "diagnostics";

interface HubPanel {
  button: HTMLButtonElement;
  content: HTMLPreElement;
  text: string;
}

class HubModal extends Modal {
  private activeTab: HubModalTab = "logs";
  private panels = new Map<HubModalTab, HubPanel>();
  private logLevels: Set<string> = new Set(DEFAULT_LOG_LEVELS);
  private levelRow: HTMLDivElement | null = null;

  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app);
  }

  onOpen() {
    const { contentEl, modalEl } = this;
    modalEl.style.width = "min(90vw, 900px)";
    contentEl.addClass("arxiv-daily-hub-modal");
    contentEl.createEl("h2", { text: "arXiv Daily — Logs & History" });

    const tabs = contentEl.createDiv({ cls: "arxiv-daily-hub-modal__tabs" });
    const body = contentEl.createDiv({ cls: "arxiv-daily-hub-modal__body" });

    this.createPanel(tabs, body, "logs", "Logs");
    this.createPanel(tabs, body, "history", "Run History");
    this.createPanel(tabs, body, "state", "Run State");
    this.createPanel(tabs, body, "diagnostics", "Diagnostics");

    const levelRow = body.createDiv({ cls: "arxiv-daily-hub-modal__level-filter" });
    this.levelRow = levelRow;
    this.renderLevelChips(levelRow);
    levelRow.style.display = "none"; // shown only when logs tab active

    this.activateTab("logs");
    this.refreshActiveTab();

    const footer = contentEl.createDiv({ cls: "arxiv-daily-hub-modal__footer" });
    footer.createEl("button", {
      text: "Refresh",
      attr: { type: "button" },
    }).onclick = () => {
      this.refreshActiveTab();
    };

    footer.createEl("button", {
      text: "Clear",
      attr: { type: "button" },
    }).onclick = () => {
      if (this.activeTab === "logs") {
        this.plugin.logger.clearBuffer();
      }
      this.refreshActiveTab();
    };

    footer.createEl("button", {
      text: "Copy",
      attr: { type: "button" },
    }).onclick = () => {
      void this.copyActiveTab();
    };

    footer.createEl("button", {
      text: "Close",
      attr: { type: "button" },
    }).onclick = () => {
      this.close();
    };
  }

  onClose() {
    this.panels.clear();
    this.contentEl.empty();
  }

  private createPanel(
    tabs: HTMLElement,
    body: HTMLElement,
    tab: HubModalTab,
    label: string,
  ): void {
    const button = tabs.createEl("button", {
      cls: "arxiv-daily-hub-modal__tab",
      text: label,
      attr: {
        type: "button",
        "aria-pressed": "false",
      },
    }) as HTMLButtonElement;
    button.addEventListener("click", () => {
      this.activateTab(tab);
      this.refreshActiveTab();
    });

    const content = body.createEl("pre", {
      cls: "arxiv-daily-hub-modal__panel",
    }) as HTMLPreElement;
    content.style.userSelect = "text";
    content.style.cursor = "text";
    this.panels.set(tab, { button, content, text: "" });
  }

  private activateTab(tab: HubModalTab): void {
    this.activeTab = tab;
    for (const [id, panel] of this.panels) {
      const active = id === tab;
      panel.button.toggleClass("hub-modal-tab-active", active);
      panel.button.setAttribute("aria-pressed", String(active));
      panel.content.toggleClass("is-active", active);
    }
    this.setLevelRowVisibility();
  }

  private renderLevelChips(container: HTMLElement): void {
    container.empty();
    const order: Array<{ key: string; label: string }> = [
      { key: "debug", label: "Debug" },
      { key: "info", label: "Info" },
      { key: "warn", label: "Warn" },
      { key: "error", label: "Error" },
    ];
    for (const { key, label } of order) {
      const active = this.logLevels.has(key);
      const chip = container.createEl("button", {
        cls: `arxiv-daily-hub-modal__level-chip${active ? " is-active" : ""}`,
        text: label,
        attr: { type: "button", "aria-pressed": String(active) },
      });
      chip.onclick = () => {
        if (this.logLevels.has(key)) this.logLevels.delete(key);
        else this.logLevels.add(key);
        this.renderLevelChips(container);
        this.refreshActiveTab();
      };
    }
    const all = container.createEl("button", {
      cls: "arxiv-daily-hub-modal__level-chip",
      text: "All",
      attr: { type: "button" },
    });
    all.onclick = () => {
      this.logLevels = new Set(DEFAULT_LOG_LEVELS);
      this.renderLevelChips(container);
      this.refreshActiveTab();
    };
  }

  private setLevelRowVisibility(): void {
    if (this.levelRow) {
      this.levelRow.style.display = this.activeTab === "logs" ? "" : "none";
    }
  }

  private refreshActiveTab(): void {
    const tab = this.activeTab;
    if (tab === "logs") {
      this.setPanelText(
        tab,
        formatLogEntries(this.plugin.logger.getBuffer(), { levels: this.logLevels }),
      );
      return;
    }
    this.setPanelText(tab, tab === "history" ? "Loading run history..." : tab === "state" ? "Loading run state..." : "Loading diagnostics...");
    if (tab === "history") {
      void this.loadRunHistory();
    } else if (tab === "state") {
      this.loadRunState();
    } else {
      this.loadDiagnostics();
    }
  }

  private async loadRunHistory(): Promise<void> {
    try {
      const records = await this.plugin.runHistoryStore.readLatest(100);
      this.setPanelText("history", formatRunHistoryRecords(records));
    } catch (e) {
      this.plugin.logger.warn("run history load failed", e);
      this.setPanelText(
        "history",
        `Failed to load run history: ${(e as Error).message}`,
      );
    }
  }

  private loadRunState(): void {
    try {
      const snap = this.plugin.stateStore.snapshot();
      const entries = Object.entries(snap).sort((a, b) => (a[0] < b[0] ? 1 : -1));
      if (entries.length === 0) {
        this.setPanelText("state", "No runs yet.");
        return;
      }
      const lines = entries.slice(0, 50).map(([date, e]) => {
        let line = `${date}: ${e.status} (attempts=${e.attempts}`;
        if (e.papersWritten != null) line += `, papers=${e.papersWritten}`;
        if (e.error) line += `, err=${e.error.slice(0, 120)}`;
        line += ")";
        return line;
      });
      this.setPanelText("state", lines.join("\n"));
    } catch (e) {
      this.plugin.logger.warn("run state load failed", e);
      this.setPanelText("state", `Failed to load run state: ${(e as Error).message}`);
    }
  }

  private loadDiagnostics(): void {
    try {
      this.setPanelText(
        "diagnostics",
        buildDiagnosticsReport({
          settings: this.plugin.settings,
          runState: this.plugin.stateStore.snapshot(),
          version: this.plugin.manifest?.version,
        }),
      );
    } catch (e) {
      this.plugin.logger.warn("diagnostics load failed", e);
      this.setPanelText(
        "diagnostics",
        `Failed to build diagnostics: ${(e as Error).message}`,
      );
    }
  }

  private setPanelText(tab: HubModalTab, text: string): void {
    const panel = this.panels.get(tab);
    if (!panel) return;
    panel.text = text;
    panel.content.setText(text);
    panel.content.scrollTop = panel.content.scrollHeight;
  }

  private async copyActiveTab(): Promise<void> {
    const panel = this.panels.get(this.activeTab);
    const text = panel?.text ?? "";
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else if (panel) {
        const range = document.createRange();
        range.selectNodeContents(panel.content);
        window.getSelection()?.removeAllRanges();
        window.getSelection()?.addRange(range);
        document.execCommand("copy");
        window.getSelection()?.removeAllRanges();
      }
      new Notice("arXiv Daily: copied");
    } catch (e) {
      this.plugin.logger.warn("Could not copy hub modal text", e);
      if (panel) {
        const range = document.createRange();
        range.selectNodeContents(panel.content);
        window.getSelection()?.removeAllRanges();
        window.getSelection()?.addRange(range);
      }
      new Notice("Could not copy; text is selectable");
    }
  }
}
