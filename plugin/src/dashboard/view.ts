import {
  ItemView,
  Modal,
  Notice,
  Setting,
  setIcon,
  type App,
  type WorkspaceLeaf,
} from "obsidian";
import type ArxivDailyPlugin from "../../main";
import {
  planDashboardAction,
  queryDashboard,
  type DashboardAction,
  type DashboardDateField,
  type DashboardPatch,
  type DashboardQuery,
  type DashboardRow,
  type DashboardTab,
} from "./model";
import { type PaperIndexEntry } from "../services/paper-index";
import { ensurePaperNote } from "../services/paper-note";
import { chooseModal } from "../services/modal";
import { formatDate, todayInTz } from "../utils/time";

export const ARXIV_DAILY_DASHBOARD_VIEW = "arxiv-daily-dashboard";

const DASHBOARD_TABS: Array<{ id: DashboardTab; label: string }> = [
  { id: "starred", label: "Starred" },
  { id: "all", label: "All" },
];

interface DailyReportDay {
  date: string;
  path: string;
  papers: number;
  starred: number;
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

class ArxivDailyDashboardView extends ItemView {
  private entries: DashboardRow["entry"][] = [];
  private dailyReports: DailyReportDay[] = [];
  private calendarMonth: string | null = null;
  private query: DashboardQuery = { tab: "starred" };
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
    return "arXiv Daily Reading Dashboard";
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
      const index = await this.plugin.buildPaperIndex().load();
      this.entries = Object.values(index.papers);
      this.dailyReports = this.loadDailyReports(this.entries);
      this.calendarMonth ??= latestReportMonth(this.dailyReports);
      this.error = null;
    } catch (e) {
      this.entries = [];
      this.dailyReports = [];
      this.error = (e as Error).message;
    }
    this.render();
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
        const report =
          byDate.get(date) ??
          {
            date,
            path,
            papers: 0,
            starred: 0,
          };
        byDate.set(date, report);
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

    const result = queryDashboard(this.entries, this.query);
    this.renderTabs(contentEl, result.tabCounts);

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
    const result = queryDashboard(this.entries, this.query);
    this.statsEl.empty();
    this.batchEl.empty();
    this.resultsEl.empty();
    this.renderStats(this.statsEl, result);
    this.renderBatchControls(this.batchEl, result.rows);
    if (result.rows.length === 0) {
      this.resultsEl.createEl("div", {
        cls: "arxiv-daily-dashboard__state",
        text: "No papers in this view.",
      });
      return;
    }
    this.renderTable(this.resultsEl, result.rows);
  }

  private renderHeader(contentEl: HTMLElement): void {
    const header = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__header",
    });
    header.createEl("h2", { text: "Reading Dashboard" });
    const refresh = header.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__refresh",
      attr: {
        type: "button",
        "aria-label": "Refresh dashboard",
        title: "Refresh dashboard",
      },
    });
    setIcon(refresh, "refresh-cw");
    refresh.addEventListener("click", () => {
      void this.reloadIndex();
    });
  }

  private renderTabs(
    contentEl: HTMLElement,
    counts: Record<DashboardTab, number>,
  ): void {
    const tabs = contentEl.createEl("div", {
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
        text: String(counts[tab.id]),
      });
      button.addEventListener("click", () => {
        this.query = { ...this.query, tab: tab.id };
        this.render();
      });
    }
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
        title: "Go to current month",
      },
    }) as HTMLButtonElement;
    const prev = controls.createEl("button", {
      cls: "clickable-icon",
      attr: { type: "button", "aria-label": "Previous month", title: "Previous month" },
    }) as HTMLButtonElement;
    setIcon(prev, "chevron-left");
    controls.createEl("span", {
      cls: "arxiv-daily-dashboard__calendar-month",
      text: month || "No reports",
    });
    const next = controls.createEl("button", {
      cls: "clickable-icon",
      attr: { type: "button", "aria-label": "Next month", title: "Next month" },
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

    const byDate = new Map(this.dailyReports.map((report) => [report.date, report]));
    const weekdays = section.createEl("div", {
      cls: "arxiv-daily-dashboard__calendar-weekdays",
    });
    for (const label of ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]) {
      weekdays.createSpan({ text: label });
    }

    const grid = section.createEl("div", {
      cls: "arxiv-daily-dashboard__calendar-grid",
    });
    for (const cell of calendarCells(month)) {
      const report = cell.date ? byDate.get(cell.date) : undefined;
      const button = grid.createEl("button", {
        cls: "arxiv-daily-dashboard__calendar-day",
        attr: {
          type: "button",
          "aria-label": report
            ? `Open daily report ${report.date}`
            : cell.date
              ? `No daily report ${cell.date}`
              : "Empty calendar cell",
          title: report
            ? `${report.date}: ${report.papers} indexed papers${report.starred ? `, ${report.starred} starred` : ""}`
            : cell.date ?? "",
        },
      }) as HTMLButtonElement;
      if (!cell.date) {
        button.disabled = true;
        button.addClass("is-empty");
        continue;
      }
      button.createSpan({
        cls: "arxiv-daily-dashboard__calendar-day-number",
        text: String(Number(cell.date.slice(-2))),
      });
      if (cell.date === today) button.addClass("is-today");
      if (report) {
        button.addClass("has-report");
        button.createSpan({
          cls: "arxiv-daily-dashboard__calendar-day-count",
          text: String(report.papers),
        });
        button.addEventListener("click", () => {
          void this.plugin.app.workspace.openLinkText(report.path, "", false);
        });
      } else {
        button.disabled = true;
      }
    }
  }

  private todayDate(): string {
    return formatDate(
      todayInTz(new Date(), this.plugin.settings.arxiv.timezone),
    );
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
      ["Notes", result.rows.filter((row) => row.hasNote).length],
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

    const search = this.createFilterField(filters, "Search").createEl("input", {
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
      this.createFilterField(filters, "Topic"),
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

    const dateField = this.createSelect(
      this.createFilterField(filters, "Date field"),
      [
        { value: "seen", label: "First seen" },
        { value: "published", label: "Published" },
      ],
      this.query.dateField ?? "seen",
    );
    dateField.addEventListener("change", () => {
      this.query = {
        ...this.query,
        dateField: dateField.value as DashboardDateField,
      };
      this.renderCurrentResults();
    });

    const dateFrom = this.createFilterField(filters, "From").createEl("input", {
      attr: { type: "date" },
    }) as HTMLInputElement;
    dateFrom.value = this.query.dateFrom ?? "";
    dateFrom.addEventListener("change", () => {
      this.query = {
        ...this.query,
        dateFrom: dateFrom.value || undefined,
      };
      this.renderCurrentResults();
    });

    const dateTo = this.createFilterField(filters, "To").createEl("input", {
      attr: { type: "date" },
    }) as HTMLInputElement;
    dateTo.value = this.query.dateTo ?? "";
    dateTo.addEventListener("change", () => {
      this.query = {
        ...this.query,
        dateTo: dateTo.value || undefined,
      };
      this.renderCurrentResults();
    });

    const note = this.createSelect(
      this.createFilterField(filters, "Note"),
      [
        { value: "", label: "Any note" },
        { value: "yes", label: "Has note" },
        { value: "no", label: "No note" },
      ],
      boolSelectValue(this.query.hasNote),
    );
    note.addEventListener("change", () => {
      this.query = { ...this.query, hasNote: boolSelectQuery(note.value) };
      this.renderCurrentResults();
    });

    const detail = this.createSelect(
      this.createFilterField(filters, "Detail"),
      [
        { value: "", label: "Any detail" },
        { value: "yes", label: "Detail" },
        { value: "no", label: "No detail" },
      ],
      boolSelectValue(this.query.detail),
    );
    detail.addEventListener("change", () => {
      this.query = { ...this.query, detail: boolSelectQuery(detail.value) };
      this.renderCurrentResults();
    });

    const reset = filters.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__filter-reset",
      attr: {
        type: "button",
        "aria-label": "Reset filters",
        title: "Reset filters",
      },
    });
    setIcon(reset, "rotate-ccw");
    reset.addEventListener("click", () => {
      this.query = { tab: this.query.tab ?? "starred" };
      this.render();
    });
  }

  private createFilterField(parent: HTMLElement, label: string): HTMLElement {
    const field = parent.createEl("label", {
      cls: "arxiv-daily-dashboard__filter",
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
      const summary = summaryLine(row.entry);
      if (summary) {
        titleCell.createEl("div", {
          cls: "arxiv-daily-dashboard__summary",
          text: summary,
        });
      }

      tr.createEl("td", { text: row.topic });
      tr.createEl("td", { text: row.entry.published || "-" });
      const actionCell = tr.createEl("td", {
        cls: "arxiv-daily-dashboard__actions",
      });
      this.createIconButton(
        actionCell,
        row.hasNote ? "file-text" : "file-plus",
        row.hasNote ? "Open note" : "Create note",
        (button) => {
          void this.runControlAction(button, () =>
            this.openOrCreateNote(row.entry),
          );
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
      this.createIconButton(actionCell, "folder-plus", "Add to project", (button) => {
        void this.runControlAction(button, () =>
          this.openProjectNoteModal(row.entry),
        );
      });
    }
  }

  private renderBatchControls(
    contentEl: HTMLElement,
    visibleRows: DashboardRow[],
  ): void {
    const selectedCount = this.selectedIds.size;
    const toolbar = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__batch",
    });
    toolbar.createEl("span", {
      cls: "arxiv-daily-dashboard__batch-count",
      text: `${selectedCount} selected`,
    });

    this.createBatchButton(
      toolbar,
      "star",
      "Star",
      selectedCount,
      () =>
        this.runBatchStar(true),
    );
    this.createBatchButton(
      toolbar,
      "star-off",
      "Unstar",
      selectedCount,
      () =>
        this.runBatchStar(false),
    );
    this.createBatchButton(
      toolbar,
      "file-plus",
      "Notes",
      selectedCount,
      () =>
        this.runBatchAction({
          type: "create_notes",
          arxivIds: this.selectedArxivIds(),
        }),
    );

    const clear = toolbar.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__batch-icon",
      attr: {
        type: "button",
        "aria-label": "Clear selection",
        title: "Clear selection",
      },
    }) as HTMLButtonElement;
    setIcon(clear, "x");
    clear.disabled = selectedCount === 0;
    clear.addEventListener("click", () => {
      this.selectedIds.clear();
      this.renderCurrentResults();
    });

    if (visibleRows.length === 0) toolbar.addClass("is-empty");
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
        title: label,
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

  private createIconButton(
    parent: HTMLElement,
    icon: string,
    label: string,
    onClick: (button: HTMLButtonElement) => void,
  ): void {
    const button = parent.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__action",
      attr: {
        type: "button",
        "aria-label": label,
        title: label,
      },
    }) as HTMLButtonElement;
    setIcon(button, icon);
    button.addEventListener("click", () => onClick(button));
  }

  private createBatchButton(
    parent: HTMLElement,
    icon: string,
    label: string,
    selectedCount: number,
    action: () => Promise<void>,
  ): void {
    const button = parent.createEl("button", {
      cls: "arxiv-daily-dashboard__batch-button",
      attr: {
        type: "button",
        title: label,
      },
    }) as HTMLButtonElement;
    setIcon(button, icon);
    button.createSpan({ text: label });
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

  private async openOrCreateNote(entry: DashboardRow["entry"]): Promise<void> {
    const store = this.plugin.buildPaperIndex();
    const latest = (await store.get(entry.arxivId)) ?? entry;
    const path = await ensurePaperNote(this.plugin, store, latest);
    await this.plugin.app.workspace.openLinkText(path, "", false);
    await this.reloadIndex();
  }

  private async openDailyReport(entry: DashboardRow["entry"]): Promise<void> {
    const path = entry.dailyReports[0];
    if (!path) {
      new Notice(`arXiv Daily: ${entry.arxivId} has no daily report`);
      return;
    }
    await this.plugin.app.workspace.openLinkText(path, "", false);
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
      new Notice(`arXiv Daily: PDF download failed — ${result.reason}`, 10_000);
      return;
    }
    new Notice(
      `arXiv Daily: downloaded PDF for ${result.arxivId} → ${result.path}`,
      10_000,
    );
    await this.reloadIndex();
  }

  private async openProjectNoteModal(
    entry: DashboardRow["entry"],
  ): Promise<void> {
    new ProjectNoteModal(this.plugin.app, entry, async (projectPath) => {
      const result = await this.plugin
        .buildProjectNotesService()
        .addPaperToProject(entry, projectPath);
      if (result.kind !== "done") {
        throw new Error(result.reason);
      }
      new Notice(
        `arXiv Daily: ${result.appended ? "added" : "already listed"} ${result.arxivId} in ${result.projectPath}`,
        10_000,
      );
      await this.reloadIndex();
    }).open();
  }

  private selectedArxivIds(): string[] {
    return [...this.selectedIds];
  }

  private async runBatchStar(starred: boolean): Promise<void> {
    await this.runBatchAction({
      type: "set_priority",
      arxivIds: this.selectedArxivIds(),
      priority: starred ? "high" : "normal",
    });
  }

  private async runBatchAction(action: DashboardAction): Promise<void> {
    const plan = planDashboardAction(this.entries, action);
    if (plan.patches.length === 0) {
      new Notice("arXiv Daily: no selected papers need changes");
      return;
    }
    if (plan.requiresConfirmation) {
      const noteCount = plan.patches.filter((patch) => patch.ensureNote).length;
      const choice = await chooseModal(
        this.plugin.app,
        "Create paper notes",
        `Create ${noteCount} paper notes?`,
        [
          { label: "Cancel", value: "cancel" },
          { label: "Create notes", value: "create", cta: true },
        ],
      );
      if (choice !== "create") return;
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
    if (patch.ensureNote) {
      await ensurePaperNote(this.plugin, store, entry);
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

function boolSelectValue(value: boolean | undefined): string {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "";
}

function boolSelectQuery(value: string): boolean | undefined {
  if (value === "yes") return true;
  if (value === "no") return false;
  return undefined;
}

function openUrl(url: string, label: string): void {
  if (!url.trim()) {
    new Notice(`arXiv Daily: no ${label} URL`);
    return;
  }
  window.open(url, "_blank", "noopener");
}

function summaryLine(entry: DashboardRow["entry"]): string {
  const summary = entry.summary;
  if (!summary) return "";
  return (
    summary.coreProblem ||
    summary.whyRelevant ||
    summary.keyMethod ||
    summary.mainResult ||
    summary.limitations ||
    ""
  );
}

function isStarredEntry(entry: DashboardRow["entry"]): boolean {
  return entry.status !== "ignored" && entry.priority === "high";
}

function normalizeVaultPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
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

class ProjectNoteModal extends Modal {
  private projectPath = "";

  constructor(
    app: App,
    private entry: PaperIndexEntry,
    private onSubmit: (projectPath: string) => Promise<void>,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Add paper to project" });
    contentEl.createEl("p", {
      text: `${this.entry.arxivId} · ${this.entry.title}`,
    });
    new Setting(contentEl)
      .setName("Project note")
      .setDesc("Vault path, for example Projects/photo-z.md. .md is added when omitted.")
      .addText((text) =>
        text.setPlaceholder("Projects/photo-z.md").onChange((value) => {
          this.projectPath = value.trim();
        }),
      );
    new Setting(contentEl).addButton((button) =>
      button
        .setButtonText("Add")
        .setCta()
        .onClick(() => {
          if (!this.projectPath) {
            new Notice("arXiv Daily: project note path is required");
            return;
          }
          this.close();
          this.onSubmit(this.projectPath).catch((e) => {
            new Notice(`arXiv Daily: ${(e as Error).message}`, 10_000);
          });
        }),
    );
  }

  onClose(): void {
    this.contentEl.empty();
  }
}
