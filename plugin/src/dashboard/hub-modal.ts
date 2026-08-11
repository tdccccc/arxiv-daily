import { Modal, Notice, type App } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { DEFAULT_LOG_LEVELS, formatLogEntries } from "./log-format";
import { errorMessage } from "./actions";
import { formatRunHistoryRecords } from "@arxiv-daily/core";
import { buildSafePluginDiagnosticsReport } from "../services/paper-index-diagnostics";

export type HubModalTab = "logs" | "history" | "state" | "diagnostics";

interface HubPanel {
  button: HTMLButtonElement;
  content: HTMLPreElement;
  text: string;
}

export class HubModal extends Modal {
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
    });
    this.clearButton.onclick = () => {
      this.plugin.logger.clearBuffer();
      this.refreshActiveTab();
    };
    this.updateClearButton();

    footer.createEl("button", {
      text: "Copy",
      attr: { type: "button" },
    }).onclick = () => {
      void this.copyActiveTab().catch((error: unknown) => {
        this.plugin.logger.error("dashboard: failed to copy hub text", error);
        new Notice(
          `arXiv Daily: failed to copy hub text: ${errorMessage(error)}`,
          10_000,
        );
      });
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
    });
    button.id = tabId;
    button.addEventListener("click", () => {
      this.activateTab(tab);
      this.refreshActiveTab();
    });

    const content = body.createEl("pre", {
      cls: "arxiv-daily-hub-modal__panel",
    });
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
      void this.loadDiagnostics();
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

  private async loadDiagnostics(): Promise<void> {
    try {
      this.setPanelText(
        "diagnostics",
        await buildSafePluginDiagnosticsReport(this.plugin),
      );
    } catch (error) {
      this.plugin.logger.warn("diagnostics render failed", error);
      this.setPanelText(
        "diagnostics",
        `Failed to build diagnostics: ${errorMessage(error)}`,
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
    const ownerDocument = this.contentEl.ownerDocument;
    const ownerWindow = ownerDocument.defaultView;
    try {
      const clipboard = ownerWindow?.navigator.clipboard;
      if (clipboard?.writeText) {
        await clipboard.writeText(text);
      } else if (panel) {
        const range = ownerDocument.createRange();
        range.selectNodeContents(panel.content);
        const selection = ownerWindow?.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
        ownerDocument.execCommand("copy");
        selection?.removeAllRanges();
      }
      new Notice("arXiv Daily: copied");
    } catch (e) {
      this.plugin.logger.warn("Could not copy hub modal text", e);
      if (panel) {
        const range = ownerDocument.createRange();
        range.selectNodeContents(panel.content);
        const selection = ownerWindow?.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
      }
      new Notice("Could not copy; text is selectable");
    }
  }
}
