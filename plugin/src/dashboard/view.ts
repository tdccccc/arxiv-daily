import {
  ItemView,
  Menu,
  Modal,
  Notice,
  setIcon,
  TFile,
  type App,
  type WorkspaceLeaf,
} from "obsidian";
import type ArxivDailyPlugin from "../../main";
import {
  PaperSearchIndex,
  looksLikeDetailSummary,
  modernArxivResources,
  normalizeArxivId,
  planDashboardAction,
  queryDashboard,
  validateVaultRelativeDirectory,
  type DashboardAction,
  type DashboardPatch,
  type DashboardQuery,
  type DashboardRow,
  type DashboardSortDirection,
  type DashboardSortKey,
  type DashboardTab,
} from "@arxiv-daily/core";
import { syncDashboardHistory, type DashboardMarkdownFile } from "@arxiv-daily/core";
import {
  validateFilterConfig,
  validateLlmConfig,
} from "@arxiv-daily/core";
import type { RunStateEntry } from "@arxiv-daily/core";
import { daysBefore, formatDate, isTimeWithinLocalWindow, isWeekendDate, todayInTz } from "@arxiv-daily/core";
import { getSetupStatus, logSetupStatus } from "../onboarding";
import { chooseModal } from "../services/modal";
import { SimilarPapersModal } from "./similar-papers-modal";
import { buildDiagnosticsReport, redactText } from "@arxiv-daily/core";
import { formatRunHistoryRecords } from "@arxiv-daily/core";
import { LOOKBACK_DAYS } from "@arxiv-daily/core";
import {
  describeManualResult,
  describeResult,
  describeRunResults,
} from "@arxiv-daily/core";

export const ARXIV_DAILY_DASHBOARD_VIEW = "arxiv-daily-dashboard";
const RECENT_DATES_FOREGROUND_TIMEOUT_MS = 3000;
const DASHBOARD_SEARCH_DEBOUNCE_MS = 250;

const DASHBOARD_TABS: Array<{ id: DashboardTab; label: string }> = [
  { id: "all", label: "All" },
  { id: "starred", label: "Starred" },
];

const PAGE_SIZE_OPTIONS: Array<{ value: number; label: string }> = [
  { value: 20, label: "20" },
  { value: 50, label: "50" },
  { value: 100, label: "100" },
  { value: Infinity, label: "All" },
];

const SORT_LABELS: Record<DashboardSortKey, string> = {
  relevance: "Relevance",
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
  normalizePath: (path: string) => string;
}

interface AppWithSettingsApi extends App {
  setting?: {
    open?: () => void;
    openTabById?: (id: string) => void;
  };
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
  if (!cell.date) return undefined;
  const date = cell.date;

  if (cell.state === "has-report" && cell.report) {
    return `${date}: open daily report, ${cell.report.papers} indexed papers${cell.report.starred ? `, ${cell.report.starred} starred` : ""}`;
  }

  if (cell.state === "no-relevant-papers") {
    return `${date}: open daily report, no relevant papers`;
  }

  if (cell.state === "runnable") {
    return `${date}: run daily report`;
  }

  if (cell.emptyReason === "arxiv-not-updated") {
    return `${date}: arXiv not updated`;
  }

  if (cell.emptyReason === "report-missing") {
    return `${date}: daily report missing`;
  }

  if (cell.emptyReason === "future") {
    return `${date}: future date`;
  }

  return date;
}

export function applyEmptyCalendarCellA11y(button: HTMLButtonElement): void {
  button.disabled = true;
  button.setAttribute("tabindex", "-1");
  button.setAttribute("aria-hidden", "true");
}

export function dashboardHeaderStatusText(input: {
  isRunning: boolean;
  lastCompletedDate?: string;
}): string {
  if (input.isRunning) return "Running…";
  return `Last run: ${input.lastCompletedDate ?? "never"}`;
}

export function latestCompletedRunDate(
  runState: Record<string, RunStateEntry | undefined>,
): string | undefined {
  const completed = Object.entries(runState)
    .filter(([, entry]) => entry?.status === "completed")
    .map(([date]) => date)
    .sort();
  return completed[completed.length - 1];
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

export function expectedDetailSummaryPath(
  papersDir: string,
  rawArxivId: string,
): string | null {
  const canonicalId = normalizeArxivId(rawArxivId);
  const directory = validateVaultRelativeDirectory(papersDir);
  if (!canonicalId || !directory.ok || !directory.value) return null;
  return `${directory.value}/${canonicalId}.md`;
}

export function isExpectedGeneratedDetailSummary(
  markdown: string,
  canonicalArxivId: string,
): boolean {
  const expectedId = normalizeArxivId(canonicalArxivId);
  if (!expectedId || !looksLikeDetailSummary(markdown)) return false;
  const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---(?:\s|$)/.exec(markdown)?.[1];
  if (frontmatter == null) return false;
  const frontmatterIds: string[] = [];
  for (const line of frontmatter.split(/\r?\n/)) {
    const item = /^(?:arxiv_id|arxiv):\s*(.*?)\s*$/.exec(line);
    if (!item) continue;
    const raw = (item[1] ?? "").replace(/^(["'])(.*)\1$/, "$2").trim();
    if (!/^\d{4}\.\d{4,5}(?:v\d+)?$/.test(raw)) return false;
    const normalized = normalizeArxivId(raw);
    if (!normalized) return false;
    frontmatterIds.push(normalized);
  }
  return (
    frontmatterIds.length > 0 &&
    frontmatterIds.every((arxivId) => arxivId === expectedId)
  );
}

export function shouldForceDashboardHistorySyncAfterDetailDeletion(
  trashedFiles: number,
  updatedEntries: number,
): boolean {
  return trashedFiles > 0 || updatedEntries > 0;
}

export async function buildCalendarDailyReportMap(
  input: CalendarDailyReportMapInput,
): Promise<Map<string, DailyReportDay>> {
  const out = new Map<string, DailyReportDay>();
  const monthPrefix = `${input.month}-`;

  for (const report of input.scannedReports) {
    if (!report.date.startsWith(monthPrefix)) continue;
    out.set(report.date, {
      ...report,
      path: input.normalizePath(report.path),
    });
  }

  return out;
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

export function applyStarButtonState(
  button: HTMLButtonElement,
  starred: boolean,
): void {
  button.classList.toggle("is-starred", starred);
  button.setAttribute("aria-pressed", String(starred));
  button.setAttribute(
    "aria-label",
    starred ? "Unstar paper" : "Star paper",
  );
  button.replaceChildren();
  setIcon(button, "star");
}

export const DEFAULT_LOG_LEVELS: ReadonlySet<string> = new Set(["debug", "info", "warn", "error"]);

const LOG_LEVEL_TAG = /\[(DEBUG|INFO|WARN|ERROR)\]/;

function parseLogLevelTag(line: string): string | null {
  const m = line.match(LOG_LEVEL_TAG);
  return m?.[1] ? m[1].toLowerCase() : null;
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
  private searchIndex: PaperSearchIndex | null = null;
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
  private searchDebounceTimer: ReturnType<typeof setTimeout> | null = null;
  private lastSyncedHistoryPaths: Set<string> | null = null;
  private calendarRefreshSeq = 0;
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
    this.clearSearchDebounce();
    this.contentEl.empty();
  }

  private async reloadIndex(): Promise<void> {
    this.renderLoading();
    try {
      void this.refreshRecentDatesForForeground().catch(() => {});
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
      const historyPaths = dashboardHistoryPathSet(
        markdownFiles,
        dailyDir,
        papersDir,
      );
      if (
        shouldSkipDashboardHistorySync(
          this.lastSyncedHistoryPaths,
          historyPaths,
          this.entries.length,
        )
      ) {
        this.error = null;
        this.plugin.logger.info(
          `dashboard: skipped history sync for ${historyPaths.size} unchanged managed history files`,
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
      this.lastSyncedHistoryPaths = historyPaths;
      this.entries = Object.values(index.papers);
      try {
        this.searchIndex = new PaperSearchIndex(this.entries);
      } catch (e) {
        this.searchIndex = null;
        this.plugin.logger.warn("dashboard: local search index construction failed; using substring fallback", e);
      }
      this.loadDetailSummaries(this.entries);
      this.dailyReports = this.loadDailyReports(this.entries, markdownFiles);
      this.calendarMonth ??= this.todayDate().slice(0, 7);
      await this.refreshCalendarDailyReports(
        this.calendarMonth ?? this.todayDate().slice(0, 7),
      );
      this.error = null;
    } catch (e) {
      this.entries = [];
      this.searchIndex = null;
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
    this.calendarDailyReports = await this.buildCalendarDailyReports(month);
  }

  private async buildCalendarDailyReports(month: string): Promise<Map<string, DailyReportDay>> {
    return await buildCalendarDailyReportMap({
      month,
      scannedReports: this.dailyReports,
      normalizePath: normalizeVaultPath,
    });
  }

  private refreshCalendarMonth(month: string): void {
    const token = ++this.calendarRefreshSeq;
    this.calendarMonth = month;
    void this.buildCalendarDailyReports(month)
      .then((reports) => {
        if (token !== this.calendarRefreshSeq || this.calendarMonth !== month) return;
        this.calendarDailyReports = reports;
        this.render();
      })
      .catch((e) => {
        if (token !== this.calendarRefreshSeq || this.calendarMonth !== month) return;
        this.plugin.logger.warn("dashboard: calendar month refresh failed", e);
      });
  }

  private renderLoading(): void {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("arxiv-daily-dashboard");
    this.renderHeader(contentEl);
    contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__state",
      text: "Loading…",
    });
  }

  private render(): void {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("arxiv-daily-dashboard");
    this.renderHeader(contentEl);

    if (this.error) {
      this.renderErrorToolbar(contentEl);
      contentEl.createEl("div", {
        cls: "arxiv-daily-dashboard__state arxiv-daily-dashboard__state--error",
        text: `Failed to load paper index: ${this.error}`,
      });
      return;
    }

    const result = queryDashboard(this.entries, this.query, {
      detailSummaryIds: this.detailSummaryIds,
      searchIndex: this.searchIndex,
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
    this.renderCurrentResults(result);
  }

  private renderRecentDatesNotice(contentEl: HTMLElement): void {
    if (!this.recentDatesNotice) return;
    contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__notice",
      text: this.recentDatesNotice,
    });
  }

  private renderCurrentResults(
    precomputed?: ReturnType<typeof queryDashboard>,
  ): void {
    if (!this.statsEl || !this.batchEl || !this.resultsEl) return;
    const result =
      precomputed ??
      queryDashboard(this.entries, this.query, {
        detailSummaryIds: this.detailSummaryIds,
        searchIndex: this.searchIndex,
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
      this.createEmptyActionButton(actions, "play", "Run today", (button) => {
        void this.runControlAction(button, () => this.runToday());
      });
      this.createEmptyActionButton(actions, "layers", "Run pending", (button) => {
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
        this.renderCurrentResults();
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
    this.clearSearchDebounce();
    this.query = {
      tab: this.query.tab ?? "starred",
      ...(this.query.sort ? { sort: this.query.sort } : {}),
    };
    this.currentPage = 0;
    this.render();
  }

  private openSettings(): void {
    this.plugin.logger.info("dashboard: open settings requested");
    const settings = (this.plugin.app as AppWithSettingsApi).setting;
    if (settings?.open && settings?.openTabById) {
      settings.open();
      settings.openTabById(this.plugin.manifest.id);
      return;
    }
    this.notice("Open Settings → Community plugins → arXiv Daily.");
  }

  private createSettingsButton(parent: HTMLElement): void {
    appendSettingsButton(parent, () => this.openSettings());
  }

  private renderHeader(contentEl: HTMLElement): void {
    const header = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__header",
    });
    const titleGroup = header.createEl("div", {
      cls: "arxiv-daily-dashboard__header-main",
    });
    titleGroup.createEl("h2", { text: "arXiv Daily Dashboard" });
    titleGroup.createEl("div", {
      cls: "arxiv-daily-dashboard__status-line",
      text: dashboardHeaderStatusText({
        isRunning: this.plugin.operations.snapshot().length > 0,
        lastCompletedDate: latestCompletedRunDate(
          this.plugin.stateStore.snapshot(),
        ),
      }),
    });
  }

  private renderErrorToolbar(contentEl: HTMLElement): void {
    const toolbar = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__toolbar arxiv-daily-dashboard__toolbar--error",
    });
    const actions = toolbar.createEl("div", {
      cls: "arxiv-daily-dashboard__toolbar-actions",
    });
    this.createToolbarButton(
      actions,
      "refresh-cw",
      "Retry",
      "Retry loading dashboard",
      (button) => {
        void this.runControlAction(button, () => this.reloadIndex());
      },
    );
    this.createSettingsButton(actions);
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
          "data-tab": tab.id,
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
        this.updateTabButtonState(tabs);
        this.updateToolbarFilterCounts(tabs);
        this.renderCurrentResults();
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
        this.renderCurrentResults();
        return this.query.detailSummary === true;
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
      "Run today",
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
        void this.runDashboardCommand("summarize-by-id", false);
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

  private updateTabButtonState(tabs: HTMLElement): void {
    const active = this.query.tab ?? "starred";
    const buttons = Array.from(
      tabs.querySelectorAll<HTMLButtonElement>(
        ".arxiv-daily-dashboard__tab",
      ),
    );
    for (const button of buttons) {
      const isActive = button.getAttribute("data-tab") === active;
      button.toggleClass("is-active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    }
  }

  private updateToolbarFilterCounts(tabs: HTMLElement): void {
    const countEl = tabs.querySelector<HTMLElement>(
      ".arxiv-daily-dashboard__tab--filter .arxiv-daily-dashboard__tab-count",
    );
    if (!countEl) return;
    countEl.textContent = String(
      this.countToolbarFilter((entry) =>
        this.detailSummaryIds.has(entry.arxivId),
      ),
    );
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
      this.refreshCalendarMonth(todayMonth);
    });
    prev.addEventListener("click", () => {
      const nextMonth = shiftMonth(month, -1);
      this.refreshCalendarMonth(nextMonth);
    });
    next.addEventListener("click", () => {
      const nextMonth = shiftMonth(month, 1);
      this.refreshCalendarMonth(nextMonth);
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
      const interactive =
        cell.state === "runnable" ||
        ((cell.state === "has-report" || cell.state === "no-relevant-papers") &&
          Boolean(cell.report));
      const day = grid.createEl(interactive ? "button" : "span", {
        cls: this.getCalendarCellClasses(cell),
      });
      if (interactive) day.setAttribute("type", "button");
      const ariaLabel = this.getCalendarCellAriaLabel(cell);
      if (ariaLabel) day.setAttribute("aria-label", ariaLabel);

      if (!cell.date) {
        day.setAttribute("aria-hidden", "true");
        day.addClass("is-empty");
        continue;
      }

      day.createSpan({
        cls: "arxiv-daily-dashboard__calendar-day-number",
        text: String(Number(cell.date.slice(-2))),
      });

      if (cell.date === today) day.addClass("is-today");

      if (!(day instanceof HTMLButtonElement)) continue;
      switch (cell.state) {
        case "has-report":
          this.renderReportCell(day, cell);
          break;
        case "no-relevant-papers":
          this.renderNoRelevantPapersCell(day, cell);
          break;
        case "runnable":
          this.renderRunnableCell(day, cell);
          break;
      }
    }
  }

  private renderToolbarFilter(
    parent: HTMLElement,
    label: string,
    active: boolean,
    count: number,
    onClick: () => boolean,
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
    button.addEventListener("click", () => {
      const isActive = onClick();
      button.toggleClass("is-active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });
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
    const timezone = this.plugin.settings.arxiv.timezone;
    const today = todayInTz(new Date(), timezone);

    for (let i = 0; i < LOOKBACK_DAYS; i++) {
      const date = daysBefore(today, i, timezone);
      if (!isWeekendDate(date, timezone)) {
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
      isWeekend: parsed ? isWeekendDate(parsed, settings.arxiv.timezone) : false,
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
      void this.runControlAction(button, () =>
        this.runDateFromCalendar(cell.date!),
      );
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

    this.notice(`arXiv Daily: running for ${date}…`);
    await this.plugin.recentDates.refresh();
    if (date !== this.todayDate() && !this.plugin.recentDates.hasDate(date)) {
      this.notice(`arXiv Daily ${date}: arXiv not updated`);
      await this.reloadIndex();
      return;
    }

    this.plugin.logger.info(`dashboard: manual calendar run requested for ${date}`);
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
    this.clearSearchDebounce();
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
      this.clearSearchDebounce();
      this.searchDebounceTimer = setTimeout(() => {
        this.searchDebounceTimer = null;
        this.query = { ...this.query, search: search.value.trim() || undefined };
        this.currentPage = 0;
        this.renderCurrentResults();
      }, DASHBOARD_SEARCH_DEBOUNCE_MS);
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
    const selectAllCell = headRow.createEl("th", {
      attr: { scope: "col" },
    });
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
      for (const checkbox of Array.from(
        tbody.querySelectorAll<HTMLInputElement>("input[type='checkbox']"),
      )) {
        checkbox.checked = selectAll.checked;
      }
      this.updateVisibleSelectionControls(selectAll, visibleIds);
      this.refreshBatchControlsForCurrentPage();
    });

    for (const label of [
      "Star",
      "Title",
      "Topic",
      "Published",
      "Actions",
    ]) {
      headRow.createEl("th", { text: label, attr: { scope: "col" } });
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
        this.updateVisibleSelectionControls(selectAll, visibleIds);
        this.refreshBatchControlsForCurrentPage();
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
      if (this.isActiveRelevanceSearch() && row.matchReasons?.length) {
        titleCell.createEl("div", {
          cls: "arxiv-daily-dashboard__match-reason",
          text: row.matchReasons.slice(0, 2).map((reason) => reason.text).join(" · "),
        });
      }

      tr.createEl("td", { text: row.topic });
      tr.createEl("td", { text: row.entry.published || "-" });
      const actionCell = tr.createEl("td", {
        cls: "arxiv-daily-dashboard__actions",
      });
      this.createIconButton(actionCell, "scan-search", "Find similar papers", () => {
        this.openSimilarPapers(row.entry);
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
          openArxivResource(row.entry.arxivId, "abs", this.plugin.logger);
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

  private updateVisibleSelectionControls(
    selectAll: HTMLInputElement,
    visibleIds: string[],
  ): void {
    const selectedVisible = visibleIds.filter((id) => this.selectedIds.has(id));
    selectAll.checked =
      visibleIds.length > 0 && selectedVisible.length === visibleIds.length;
    selectAll.indeterminate =
      selectedVisible.length > 0 && selectedVisible.length < visibleIds.length;
  }

  private refreshBatchControlsForCurrentPage(): void {
    if (!this.batchEl) return;
    const result = queryDashboard(this.entries, this.query, {
      detailSummaryIds: this.detailSummaryIds,
      searchIndex: this.searchIndex,
    });
    const page = paginateDashboardRows(
      result.rows,
      this.currentPage,
      this.pageSize,
    );
    this.currentPage = page.currentPage;
    this.batchEl.empty();
    this.renderBatchControls(this.batchEl, page);
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
      text: showingText(page),
    });
    this.renderSortControl(controls);
    this.renderPageSizeControl(controls);

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
    const currentKey = this.query.sort?.key ?? ((this.query.search?.trim() ? "relevance" : DEFAULT_SORT_KEY));
    const currentDir = this.query.sort?.direction ?? (currentKey === "relevance" ? "desc" : "asc");

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

  private renderPageSizeControl(parent: HTMLElement): void {
    const field = parent.createEl("label", {
      cls: "arxiv-daily-dashboard__batch-sort",
    });
    field.createSpan({
      cls: "arxiv-daily-dashboard__batch-sort-label",
      text: "Per page",
    });
    const currentSize = this.pageSize;
    const select = this.createSelect(
      field,
      PAGE_SIZE_OPTIONS.map((opt) => ({
        value: String(opt.value),
        label: opt.label,
      })),
      String(currentSize),
    );
    select.addEventListener("change", () => {
      const size = Number(select.value);
      if (size === this.pageSize) return;
      this.pageSize = size;
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
      text: isFinite(page.pageSize)
        ? `Show ${page.pageSize} per page`
        : "",
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
    const button = parent.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__star",
      attr: {
        type: "button",
        "data-arxiv-id": entry.arxivId,
      },
    }) as HTMLButtonElement;
    applyStarButtonState(button, starred);
    button.addEventListener("click", () => {
      const nextStarred = !isStarredEntry(entry);
      void this.runControlAction(button, () =>
        this.updateStar(entry, nextStarred, button),
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
    const activeRuns = this.plugin.operations.snapshot();

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
    this.addCommandMenuItem(menu, "Run for date…", "calendar", "run-for-date");
    this.addCommandMenuItem(
      menu,
      "Force run for date…",
      "rotate-cw",
      "force-run-for-date",
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
      "Cancel active tasks",
      "circle-stop",
      "cancel-current-run",
      true,
      activeRuns.length === 0,
    );

    menu.addSeparator();
    menu.addItem((item) =>
      item
        .setTitle("Run pending")
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
      "Clear run state…",
      "trash-2",
      "clear-run-state",
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

  private clearSearchDebounce(): void {
    if (!this.searchDebounceTimer) return;
    clearTimeout(this.searchDebounceTimer);
    this.searchDebounceTimer = null;
  }

  private gateFilter(): boolean {
    const validation = validateFilterConfig(this.plugin.settings);
    if (!validation.ok) {
      this.plugin.logger.info(
        `dashboard: filter validation failed (${validation.reasons.join("; ")})`,
      );
      this.notice(
        `arXiv Daily — cannot run:\n${validation.reasons.map((reason) => `• ${reason}`).join("\n")}`,
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
        `arXiv Daily — cannot summarize:\n${validation.reasons.map((reason) => `• ${reason}`).join("\n")}`,
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
    this.notice(`arXiv Daily: running for ${date}…`);
    const result = await this.plugin.scheduler.runForDateNow(date);
    this.notice(`arXiv Daily ${date}: ${describeResult(result)}`);
    await this.reloadIndex();
  }

  private async runAllPending(): Promise<void> {
    if (!this.gateFilter()) return;
    this.plugin.logger.info("dashboard: run all pending requested");
    this.notice("arXiv Daily: running all pending in lookback…");
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
    this.notice("arXiv Daily: retrying failed dates in lookback…");
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
    button: HTMLButtonElement,
  ): Promise<void> {
    const previousPriority = entry.priority;
    const previousStarred = isStarredEntry(entry);
    entry.priority = starred ? "high" : "normal";
    applyStarButtonState(button, starred);
    const store = this.plugin.buildPaperIndex();
    let updated: DashboardRow["entry"] | null = null;
    try {
      updated = await store.setPriority(
        entry.arxivId,
        starred ? "high" : "normal",
      );
    } catch (e) {
      entry.priority = previousPriority;
      applyStarButtonState(button, previousStarred);
      throw e;
    }
    if (!updated) {
      entry.priority = previousPriority;
      applyStarButtonState(button, previousStarred);
      throw new Error(`${entry.arxivId} is not in papers.json`);
    }
    entry.priority = updated.priority;
    this.dailyReports = this.loadDailyReports(this.entries);
    await this.refreshCalendarDailyReports(
      this.calendarMonth ?? this.todayDate().slice(0, 7),
    );
    this.render();
    const nextButton = this.contentEl.querySelector<HTMLButtonElement>(
      `.arxiv-daily-dashboard__star[data-arxiv-id="${entry.arxivId}"]`,
    );
    if (nextButton) {
      nextButton.focus();
    } else {
      this.contentEl
        .querySelector<HTMLButtonElement>(".arxiv-daily-dashboard__tab.is-active")
        ?.focus();
    }
    this.notice(
      `arXiv Daily: ${entry.arxivId} ${starred ? "starred" : "unstarred"}`,
    );
  }

  private isActiveRelevanceSearch(): boolean {
    return Boolean(this.query.search?.trim()) && (!this.query.sort || this.query.sort.key === "relevance");
  }

  private openSimilarPapers(entry: DashboardRow["entry"]): void {
    let index = this.searchIndex;
    if (!index) {
      try {
        index = new PaperSearchIndex(this.entries);
      } catch (e) {
        this.plugin.logger.warn("dashboard: similar-paper index construction failed", e);
        this.notice("arXiv Daily: local similarity search is unavailable");
        return;
      }
    }
    const results = index.similar(entry, { limit: 10 });
    new SimilarPapersModal(this.plugin.app, {
      source: entry,
      results,
      openDetail: (candidate) => this.openDetailSummary(candidate),
      openDaily: (candidate) => this.openDailyReport(candidate),
      openArxiv: (candidate) =>
        openArxivResource(candidate.arxivId, "abs", this.plugin.logger),
      openPdf: (candidate) => this.openPdf(candidate),
      onActionError: (error, action, candidate) => {
        this.plugin.logger.error(
          `dashboard: similar papers ${action.toLowerCase()} failed for ${candidate.arxivId}`,
          error,
        );
        this.notice(`arXiv Daily: ${action} failed`);
      },
    }).open();
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
    this.notice(`arXiv Daily: summarizing ${entry.arxivId}…`);
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
    openArxivResource(entry.arxivId, "pdf", this.plugin.logger);
  }

  private async downloadPdf(entry: DashboardRow["entry"]): Promise<void> {
    this.plugin.logger.info(`dashboard: PDF download requested for ${entry.arxivId}`);
    const result = await this.plugin.downloadPdf(entry);
    if (result.kind !== "done") {
      this.plugin.logger.warn(
        `dashboard: PDF download failed for ${entry.arxivId}: ${result.reason}`,
      );
      this.notice(`arXiv Daily: PDF download failed - ${result.reason}`, 10_000);
      return;
    }
    this.notice(
      `arXiv Daily: downloaded PDF for ${result.arxivId} → ${result.path}`,
      10_000,
    );
    await this.reloadIndex();
  }

  private selectedArxivIds(): string[] {
    return [...this.selectedIds];
  }

  private selectedDetailSummaryCount(): number {
    return this.selectedArxivIds().filter((id) => {
      const entry = this.entries.find((candidate) => candidate.arxivId === id);
      return Boolean(entry && this.hasDeletableDetailSummary(entry));
    }).length;
  }

  private hasDeletableDetailSummary(entry: DashboardRow["entry"]): boolean {
    const expectedPath = expectedDetailSummaryPath(
      this.plugin.settings.output.papersDir,
      entry.arxivId,
    );
    const indexedPath = normalizeVaultPath(entry.paperPath ?? "");
    return Boolean(
      entry.detail &&
        expectedPath &&
        indexedPath === normalizeVaultPath(expectedPath),
    );
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
        Boolean(entry && this.hasDeletableDetailSummary(entry)),
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
    let trashedFiles = 0;
    let updatedEntries = 0;
    let indexFailures = 0;
    let refused = 0;
    for (const entry of entries) {
      const canonicalId = normalizeArxivId(entry.arxivId);
      const path = expectedDetailSummaryPath(
        this.plugin.settings.output.papersDir,
        entry.arxivId,
      );
      try {
        if (!canonicalId || !path) {
          throw new Error(`invalid arXiv ID: ${entry.arxivId}`);
        }
        const result = await store.removePaperDetailsAtPath(
          canonicalId,
          path,
          async () => {
            const abstractFile = this.plugin.app.vault.getAbstractFileByPath(path);
            if (!(abstractFile instanceof TFile)) {
              throw new Error(`expected detail summary does not exist: ${path}`);
            }
            const markdown = await this.plugin.app.vault.read(abstractFile);
            if (!isExpectedGeneratedDetailSummary(markdown, canonicalId)) {
              throw new Error(`file is not a generated detail summary for ${canonicalId}`);
            }
            await this.plugin.app.vault.trash(abstractFile, true);
            trashedFiles += 1;
          },
        );
        if (result.kind === "path_mismatch") {
          throw new Error(
            `indexed detail path does not match expected path: ${String(result.actualPath)}`,
          );
        }
        if (result.kind === "missing") {
          throw new Error(`paper index entry no longer exists: ${canonicalId}`);
        }
        if (result.kind === "index_failed") {
          indexFailures += 1;
          this.selectedIds.delete(entry.arxivId);
          this.plugin.logger.warn(
            `dashboard: trashed detail summary but failed to update index for ${canonicalId}`,
            result.error,
          );
          continue;
        }
        updatedEntries += 1;
        this.selectedIds.delete(entry.arxivId);
      } catch (e) {
        refused += 1;
        this.plugin.logger.warn(
          `dashboard: refused or failed before trashing detail summary for ${entry.arxivId}`,
          e,
        );
      }
    }

    this.notice(
      `arXiv Daily: trashed ${trashedFiles} summaries; updated ${updatedEntries} index entries${indexFailures ? `; ${indexFailures} trashed but index update failed` : ""}${refused ? `; ${refused} refused or failed before trash` : ""}`,
      10_000,
    );
    if (
      shouldForceDashboardHistorySyncAfterDetailDeletion(
        trashedFiles,
        updatedEntries,
      )
    ) {
      // Always bypass the reload shortcut after a successful deletion so the
      // vault and index receive a full reconciliation, even if paths appear unchanged.
      this.lastSyncedHistoryPaths = null;
    }
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
      const local = this.entries.find((candidate) => candidate.arxivId === entry.arxivId);
      if (local) {
        local.status = entry.status;
        local.priority = entry.priority;
      }
      changed += 1;
    }
    this.selectedIds.clear();
    this.dailyReports = this.loadDailyReports(this.entries);
    await this.refreshCalendarDailyReports(
      this.calendarMonth ?? this.todayDate().slice(0, 7),
    );
    this.notice(`arXiv Daily: updated ${changed} papers`);
    this.render();
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

function openArxivResource(
  rawArxivId: string,
  kind: "abs" | "pdf",
  logger: ArxivDailyPlugin["logger"],
): void {
  const resources = modernArxivResources(rawArxivId);
  const label = kind === "pdf" ? "PDF" : "arXiv";
  if (!resources) {
    logger.warn(`dashboard: refused invalid arXiv ID for ${label}`);
    new Notice(`arXiv Daily: invalid arXiv ID; ${label} was not opened`);
    return;
  }
  const url = kind === "pdf" ? resources.pdfUrl : resources.absUrl;
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

export function dashboardHistoryPathSet(
  files: DashboardMarkdownFile[],
  dailyDir: string,
  papersDir: string,
): Set<string> {
  const normalizedDailyDir = normalizeVaultPath(dailyDir);
  const normalizedPapersDir = normalizeVaultPath(papersDir);
  return new Set(
    files
      .map((file) => normalizeVaultPath(file.path))
      .filter(
        (path) =>
          path.startsWith(`${normalizedDailyDir}/`) ||
          isDirectChildMarkdown(path, normalizedPapersDir),
      ),
  );
}

export function shouldSkipDashboardHistorySync(
  previousHistoryPaths: ReadonlySet<string> | null,
  currentHistoryPaths: ReadonlySet<string>,
  currentEntryCount: number,
): boolean {
  if (!previousHistoryPaths || currentEntryCount === 0) return false;
  if (previousHistoryPaths.size !== currentHistoryPaths.size) return false;
  for (const path of currentHistoryPaths) {
    if (!previousHistoryPaths.has(path)) return false;
  }
  return true;
}

function isDirectChildMarkdown(path: string, dir: string): boolean {
  const prefix = `${dir}/`;
  if (!path.startsWith(prefix) || !/\.md$/i.test(path)) return false;
  return !path.slice(prefix.length).includes("/");
}

export function showingText(page: DashboardPage<unknown>): string {
  if (page.total === 0) return "Showing 0 of 0 papers";
  if (!isFinite(page.pageSize)) return `Showing all ${page.total} papers`;
  return `Showing ${page.start}-${page.end} of ${page.total} papers`;
}

export function paginateDashboardRows<T>(
  rows: T[],
  currentPage: number,
  pageSize: number,
): DashboardPage<T> {
  const total = rows.length;
  // Infinity means "show all" — bypass arithmetic to avoid 0 * Infinity = NaN.
  if (!isFinite(pageSize)) {
    return {
      rows: rows.slice(),
      total,
      totalPages: 1,
      currentPage: 0,
      start: total === 0 ? 0 : 1,
      end: total,
      pageSize,
    };
  }
  const safePageSize = Math.max(1, Math.floor(pageSize));
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
  const [rawYear, rawMonthIndex] = month.split("-").map(Number);
  if (
    typeof rawYear !== "number" ||
    typeof rawMonthIndex !== "number" ||
    !Number.isFinite(rawYear) ||
    !Number.isFinite(rawMonthIndex)
  ) {
    return month;
  }
  const year = rawYear;
  const monthIndex = rawMonthIndex;
  const date = new Date(Date.UTC(year, monthIndex - 1 + delta, 1));
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
}

function calendarCells(month: string): Array<{ date: string | null }> {
  const [rawYear, rawMonthIndex] = month.split("-").map(Number);
  if (
    typeof rawYear !== "number" ||
    typeof rawMonthIndex !== "number" ||
    !Number.isFinite(rawYear) ||
    !Number.isFinite(rawMonthIndex)
  ) {
    return [];
  }
  const year = rawYear;
  const monthIndex = rawMonthIndex;
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
  private clearButton: HTMLButtonElement | null = null;

  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app);
  }

  onOpen() {
    const { contentEl, modalEl } = this;
    modalEl.addClass("arxiv-daily-hub-modal");
    contentEl.addClass("arxiv-daily-hub-modal__content");
    contentEl.createEl("h2", { text: "arXiv Daily — Logs & History" });

    const tabs = contentEl.createDiv({ cls: "arxiv-daily-hub-modal__tabs" });
    tabs.setAttribute("role", "tablist");
    const body = contentEl.createDiv({ cls: "arxiv-daily-hub-modal__body" });

    this.createPanel(tabs, body, "logs", "Logs");
    this.createPanel(tabs, body, "history", "Run History");
    this.createPanel(tabs, body, "state", "Run State");
    this.createPanel(tabs, body, "diagnostics", "Diagnostics");

    const levelRow = body.createDiv({ cls: "arxiv-daily-hub-modal__level-filter" });
    this.levelRow = levelRow;
    this.renderLevelChips(levelRow);
    levelRow.addClass("arxiv-daily-hub-modal__level-filter--hidden");

    this.activateTab("logs");
    this.refreshActiveTab();

    const footer = contentEl.createDiv({ cls: "arxiv-daily-hub-modal__footer" });
    footer.createEl("button", {
      text: "Refresh",
      attr: { type: "button" },
    }).onclick = () => {
      this.refreshActiveTab();
    };

    this.clearButton = footer.createEl("button", {
      text: "Clear logs",
      attr: { type: "button" },
    }) as HTMLButtonElement;
    this.clearButton.onclick = () => {
      this.plugin.logger.clearBuffer();
      this.refreshActiveTab();
    };
    this.updateClearButton();

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
    const tabId = `arxiv-daily-hub-modal-tab-${tab}`;
    const panelId = `arxiv-daily-hub-modal-panel-${tab}`;
    const button = tabs.createEl("button", {
      cls: "arxiv-daily-hub-modal__tab",
      text: label,
      attr: {
        type: "button",
        role: "tab",
        "aria-selected": "false",
        "aria-controls": panelId,
      },
    }) as HTMLButtonElement;
    button.id = tabId;
    button.addEventListener("click", () => {
      this.activateTab(tab);
      this.refreshActiveTab();
    });

    const content = body.createEl("pre", {
      cls: "arxiv-daily-hub-modal__panel",
    }) as HTMLPreElement;
    content.id = panelId;
    content.setAttribute("role", "tabpanel");
    content.setAttribute("aria-labelledby", tabId);
    this.panels.set(tab, { button, content, text: "" });
  }

  private activateTab(tab: HubModalTab): void {
    this.activeTab = tab;
    for (const [id, panel] of this.panels) {
      const active = id === tab;
      panel.button.toggleClass("hub-modal-tab-active", active);
      panel.button.setAttribute("aria-selected", String(active));
      panel.content.toggleClass("is-active", active);
    }
    this.setLevelRowVisibility();
    this.updateClearButton();
  }

  private updateClearButton(): void {
    if (!this.clearButton) return;
    const visible = this.activeTab === "logs";
    this.clearButton.hidden = !visible;
    this.clearButton.disabled = !visible;
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
      this.levelRow.toggleClass(
        "arxiv-daily-hub-modal__level-filter--hidden",
        this.activeTab !== "logs",
      );
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
    this.setPanelText(tab, tab === "history" ? "Loading run history…" : tab === "state" ? "Loading run state…" : "Loading diagnostics…");
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
        `Failed to build diagnostics: ${redactText(e instanceof Error ? e.message : e, {
          secrets: [this.plugin.settings.llm.apiKey],
        })}`,
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
