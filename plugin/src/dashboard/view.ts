import { ItemView, Notice, type WorkspaceLeaf } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { queryDashboard } from "./model";

export const ARXIV_DAILY_DASHBOARD_VIEW = "arxiv-daily-dashboard";

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
    await this.render();
  }

  async onClose(): Promise<void> {
    this.contentEl.empty();
  }

  private async render(): Promise<void> {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("arxiv-daily-dashboard");

    const header = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__header",
    });
    header.createEl("h2", { text: "Reading Dashboard" });

    const status = contentEl.createEl("div", {
      cls: "arxiv-daily-dashboard__status",
      text: "Loading...",
    });

    try {
      const index = await this.plugin.buildPaperIndex().load();
      const entries = Object.values(index.papers);
      const result = queryDashboard(entries, { tab: "all" });
      status.setText(
        `${result.stats.total} papers indexed; ` +
          `${result.tabCounts.watch} watch, ` +
          `${result.tabCounts.highlight} highlight, ` +
          `${result.tabCounts.saved} saved`,
      );
    } catch (e) {
      status.setText(`Failed to load paper index: ${(e as Error).message}`);
    }
  }
}
