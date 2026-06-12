import { ItemView, Notice, setIcon, type WorkspaceLeaf } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import {
  queryDashboard,
  type DashboardDateField,
  type DashboardQuery,
  type DashboardRow,
  type DashboardTab,
} from "./model";
import type { PaperPriority, PaperStatus } from "../services/paper-index";

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
    this.resultsEl = contentEl.createEl("div");
    this.renderCurrentResults();
  }

  private renderCurrentResults(): void {
    if (!this.statsEl || !this.resultsEl) return;
    const result = queryDashboard(this.entries, this.query);
    this.statsEl.empty();
    this.resultsEl.empty();
    this.renderStats(this.statsEl, result);
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
    for (const label of [
      "Priority",
      "Status",
      "Title",
      "Topic",
      "Published",
      "First seen",
      "Note",
      "Citation",
    ]) {
      headRow.createEl("th", { text: label });
    }

    const tbody = table.createEl("tbody");
    for (const row of rows) {
      const tr = tbody.createEl("tr");
      tr.createEl("td", { text: row.entry.priority });
      tr.createEl("td", { text: row.entry.status });

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
      tr.createEl("td", { text: row.hasNote ? "yes" : "-" });
      tr.createEl("td", {
        text: row.missingCitationKey ? "missing" : row.entry.citationKey,
      });
    }
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
