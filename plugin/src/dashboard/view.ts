import { ItemView, Notice, setIcon, type WorkspaceLeaf } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import {
  queryDashboard,
  type DashboardQuery,
  type DashboardRow,
  type DashboardTab,
} from "./model";

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
    this.renderStats(contentEl, result);

    if (result.rows.length === 0) {
      contentEl.createEl("div", {
        cls: "arxiv-daily-dashboard__state",
        text: "No papers in this view.",
      });
      return;
    }

    this.renderTable(contentEl, result.rows);
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
