import { ItemView, Notice, setIcon, type WorkspaceLeaf } from "obsidian";
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
import type { PaperPriority, PaperStatus } from "../services/paper-index";
import { ensurePaperNote } from "../services/paper-note";
import { chooseModal } from "../services/modal";

export const ARXIV_DAILY_DASHBOARD_VIEW = "arxiv-daily-dashboard";

const DASHBOARD_TABS: Array<{ id: DashboardTab; label: string }> = [
  { id: "watch", label: "Watch" },
  { id: "highlight", label: "Highlight" },
  { id: "reading", label: "Reading" },
  { id: "saved", label: "Saved" },
  { id: "read", label: "Read" },
  { id: "all", label: "All" },
  { id: "ignored", label: "Ignored" },
];

const STATUS_OPTIONS: Array<{ value: "" | PaperStatus; label: string }> = [
  { value: "", label: "Any status" },
  { value: "inbox", label: "Inbox" },
  { value: "to_read", label: "To read" },
  { value: "reading", label: "Reading" },
  { value: "read", label: "Read" },
  { value: "saved", label: "Saved" },
  { value: "ignored", label: "Ignored" },
];

const PRIORITY_OPTIONS: Array<{ value: "" | PaperPriority; label: string }> = [
  { value: "", label: "Any priority" },
  { value: "high", label: "High" },
  { value: "normal", label: "Normal" },
  { value: "low", label: "Low" },
];
const ROW_STATUS_OPTIONS = STATUS_OPTIONS.filter((option) => option.value) as Array<{
  value: PaperStatus;
  label: string;
}>;
const ROW_PRIORITY_OPTIONS = PRIORITY_OPTIONS.filter((option) => option.value) as Array<{
  value: PaperPriority;
  label: string;
}>;

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
  private query: DashboardQuery = { tab: "watch" };
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
      this.error = null;
    } catch (e) {
      this.entries = [];
      this.error = (e as Error).message;
    }
    this.render();
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
    this.renderFilters(contentEl);

    this.statsEl = contentEl.createEl("div");
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
    const active = this.query.tab ?? "watch";
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
      ["Watch", result.stats.watch],
      ["Highlight", result.stats.highlight],
      ["Saved", result.stats.saved],
      ["Saved missing citation", result.stats.savedMissingCitationKey],
      ["Saved missing Zotero", result.stats.savedMissingZoteroKey],
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

    const status = this.createSelect(
      this.createFilterField(filters, "Status"),
      STATUS_OPTIONS,
      this.query.statuses?.[0] ?? "",
    );
    status.addEventListener("change", () => {
      this.query = {
        ...this.query,
        statuses: status.value ? [status.value as PaperStatus] : undefined,
      };
      this.renderCurrentResults();
    });

    const priority = this.createSelect(
      this.createFilterField(filters, "Priority"),
      PRIORITY_OPTIONS,
      this.query.priorities?.[0] ?? "",
    );
    priority.addEventListener("change", () => {
      this.query = {
        ...this.query,
        priorities: priority.value
          ? [priority.value as PaperPriority]
          : undefined,
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

    this.createCheckboxFilter(
      filters,
      "Missing citation",
      Boolean(this.query.missingCitationKey),
      (checked) => {
        this.query = {
          ...this.query,
          missingCitationKey: checked || undefined,
        };
        this.renderCurrentResults();
      },
    );

    this.createCheckboxFilter(
      filters,
      "Missing Zotero",
      Boolean(this.query.missingZoteroKey),
      (checked) => {
        this.query = {
          ...this.query,
          missingZoteroKey: checked || undefined,
        };
        this.renderCurrentResults();
      },
    );

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
      this.query = { tab: this.query.tab ?? "watch" };
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

  private createCheckboxFilter(
    parent: HTMLElement,
    label: string,
    checked: boolean,
    onChange: (checked: boolean) => void,
  ): void {
    const field = parent.createEl("label", {
      cls: "arxiv-daily-dashboard__filter arxiv-daily-dashboard__filter--checkbox",
    });
    const input = field.createEl("input", {
      attr: { type: "checkbox" },
    }) as HTMLInputElement;
    input.checked = checked;
    field.createSpan({ text: label });
    input.addEventListener("change", () => onChange(input.checked));
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
      "Priority",
      "Status",
      "Title",
      "Topic",
      "Published",
      "First seen",
      "Actions",
      "Citation",
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

      const priorityCell = tr.createEl("td");
      this.createInlineSelect(
        priorityCell,
        ROW_PRIORITY_OPTIONS,
        row.entry.priority,
        (value, control) => {
          void this.runControlAction(control, () =>
            this.updatePriority(row.entry, value as PaperPriority),
          );
        },
      );

      const statusCell = tr.createEl("td");
      this.createInlineSelect(
        statusCell,
        ROW_STATUS_OPTIONS,
        row.entry.status,
        (value, control) => {
          void this.runControlAction(control, () =>
            this.updateStatus(row.entry, value as PaperStatus),
          );
        },
      );

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
      tr.createEl("td", { text: row.firstSeen || "-" });
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
        void this.runControlAction(button, async () => {
          openUrl(row.entry.pdfUrl, "PDF");
        });
      });
      tr.createEl("td", {
        text: row.missingCitationKey ? "missing" : row.entry.citationKey,
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
      "archive-x",
      "Ignore",
      selectedCount,
      () =>
        this.runBatchAction({
          type: "set_status",
          arxivIds: this.selectedArxivIds(),
          status: "ignored",
        }),
    );
    this.createBatchButton(
      toolbar,
      "check-check",
      "Read",
      selectedCount,
      () =>
        this.runBatchAction({
          type: "set_status",
          arxivIds: this.selectedArxivIds(),
          status: "read",
        }),
    );
    this.createBatchButton(
      toolbar,
      "bookmark",
      "Saved",
      selectedCount,
      () =>
        this.runBatchAction({
          type: "set_status",
          arxivIds: this.selectedArxivIds(),
          status: "saved",
        }),
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

    const priority = this.createSelect(
      toolbar.createEl("label", {
        cls: "arxiv-daily-dashboard__batch-priority",
      }),
      ROW_PRIORITY_OPTIONS,
      "normal",
    );
    priority.setAttribute("aria-label", "Batch priority");
    priority.disabled = selectedCount === 0;
    const applyPriority = toolbar.createEl("button", {
      cls: "clickable-icon arxiv-daily-dashboard__batch-icon",
      attr: {
        type: "button",
        "aria-label": "Apply priority",
        title: "Apply priority",
      },
    }) as HTMLButtonElement;
    setIcon(applyPriority, "check");
    applyPriority.disabled = selectedCount === 0;
    applyPriority.addEventListener("click", () => {
      void this.runBatchAction({
        type: "set_priority",
        arxivIds: this.selectedArxivIds(),
        priority: priority.value as PaperPriority,
      });
    });

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

  private createInlineSelect(
    parent: HTMLElement,
    options: Array<{ value: string; label: string }>,
    selected: string,
    onChange: (value: string, control: HTMLSelectElement) => void,
  ): void {
    const select = parent.createEl("select", {
      cls: "arxiv-daily-dashboard__inline-select",
    }) as HTMLSelectElement;
    for (const option of options) {
      const el = select.createEl("option", { text: option.label });
      el.value = option.value;
    }
    select.value = selected;
    select.addEventListener("change", () => onChange(select.value, select));
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
    control: HTMLButtonElement | HTMLSelectElement,
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

  private async updateStatus(
    entry: DashboardRow["entry"],
    status: PaperStatus,
  ): Promise<void> {
    const store = this.plugin.buildPaperIndex();
    const updated = await store.setStatus(entry.arxivId, status);
    if (!updated) throw new Error(`${entry.arxivId} is not in papers.json`);
    if (status === "saved") {
      await ensurePaperNote(this.plugin, store, updated);
    }
    new Notice(`arXiv Daily: ${entry.arxivId} marked ${status}`);
    await this.reloadIndex();
  }

  private async updatePriority(
    entry: DashboardRow["entry"],
    priority: PaperPriority,
  ): Promise<void> {
    const store = this.plugin.buildPaperIndex();
    const updated = await store.setPriority(entry.arxivId, priority);
    if (!updated) throw new Error(`${entry.arxivId} is not in papers.json`);
    new Notice(`arXiv Daily: ${entry.arxivId} priority set to ${priority}`);
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

  private selectedArxivIds(): string[] {
    return [...this.selectedIds];
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
