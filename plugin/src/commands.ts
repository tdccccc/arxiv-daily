import { App, Menu, Modal, Notice, Setting } from "obsidian";
import type ArxivDailyPlugin from "../main";
import { todayInTz, formatDate } from "./utils/time";

export function registerCommands(plugin: ArxivDailyPlugin): void {
  const tz = () => plugin.settings.arxiv.timezone;
  const today = () => formatDate(todayInTz(new Date(), tz()));

  async function runToday() {
    const date = today();
    new Notice(`arXiv Daily: running for ${date}…`);
    const result = await plugin.scheduler.runForDateNow(date);
    new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
  }

  async function runAllPending() {
    new Notice(`arXiv Daily: running all pending in lookback…`);
    const results = await plugin.scheduler.runAllPending();
    if (results.length === 0) {
      new Notice("arXiv Daily: nothing pending in lookback window");
      return;
    }
    const summary = results
      .map((r) => `${r.date}: ${describeResult(r.result)}`)
      .join("\n");
    new Notice(`arXiv Daily (lookback):\n${summary}`, 10_000);
  }

  function openDatePicker() {
    new DatePickerModal(plugin.app, async (date) => {
      if (!date) return;
      new Notice(`arXiv Daily: running for ${date}…`);
      const result = await plugin.scheduler.runForDateNow(date);
      new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
    }).open();
  }

  function openArxivIdPicker() {
    new ArxivIdModal(plugin.app, async (raw) => {
      if (!raw) return;
      new Notice(`arXiv Daily: summarizing ${raw}…`);
      const today = formatDate(todayInTz(new Date(), tz()));
      const result = await plugin.manualFetch.fetchAndSummarize(raw, today);
      new Notice(`arXiv Daily: ${describeManualResult(result)}`, 10_000);
    }).open();
  }

  plugin.addCommand({
    id: "arxiv-daily-run-now",
    name: "Run now (today)",
    callback: runToday,
  });

  plugin.addCommand({
    id: "arxiv-daily-run-for-date",
    name: "Run for date…",
    callback: openDatePicker,
  });

  plugin.addCommand({
    id: "arxiv-daily-run-all-pending",
    name: "Run all pending in lookback window",
    callback: runAllPending,
  });

  plugin.addCommand({
    id: "arxiv-daily-summarize-by-id",
    name: "Summarize by arXiv ID…",
    callback: openArxivIdPicker,
  });

  plugin.addCommand({
    id: "arxiv-daily-open-today",
    name: "Open today's daily report",
    callback: async () => {
      const path = `${plugin.settings.output.dailyDir}/${today()}.md`;
      const file = plugin.app.vault.getAbstractFileByPath(path);
      if (file) {
        await plugin.app.workspace.openLinkText(path, "", false);
      } else {
        new Notice(`No daily report at ${path}`);
      }
    },
  });

  plugin.addCommand({
    id: "arxiv-daily-show-state",
    name: "Show recent run state",
    callback: () => new StateModal(plugin.app, plugin).open(),
  });

  plugin.addRibbonIcon("calendar-clock", "arXiv Daily", (evt: MouseEvent) => {
    const menu = new Menu();
    menu.addItem((item) =>
      item
        .setTitle("Run for today")
        .setIcon("play")
        .onClick(runToday),
    );
    menu.addSeparator();
    menu.addItem((item) =>
      item
        .setTitle("Run all pending (lookback)")
        .setIcon("layers")
        .onClick(runAllPending),
    );
    menu.addItem((item) =>
      item
        .setTitle("Run for specific date…")
        .setIcon("calendar")
        .onClick(openDatePicker),
    );
    menu.addItem((item) =>
      item
        .setTitle("Summarize by arXiv ID…")
        .setIcon("file-text")
        .onClick(openArxivIdPicker),
    );
    menu.showAtMouseEvent(evt);
  });
}

function describeResult(r: any): string {
  if (!r) return "no result";
  if (r.kind === "completed") return `done (${r.papersWritten} papers)`;
  if (r.kind === "failed_transient") return `transient: ${r.reason}`;
  if (r.kind === "failed_permanent") return `permanent: ${r.reason}`;
  if (r.kind === "skipped") return `skipped: ${r.reason}`;
  return JSON.stringify(r);
}

function describeManualResult(r: any): string {
  if (!r) return "no result";
  if (r.kind === "done") return `done → ${r.path}`;
  if (r.kind === "already_exists") return `already exists at ${r.path}`;
  if (r.kind === "not_found") return `not found: ${r.reason}`;
  if (r.kind === "no_html") return `no full HTML: ${r.reason}`;
  if (r.kind === "error") return `error: ${r.reason}`;
  return JSON.stringify(r);
}

class DatePickerModal extends Modal {
  private value = "";
  constructor(app: App, private onSubmit: (date: string | null) => void) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Run arXiv Daily for date" });
    new Setting(contentEl)
      .setName("Date")
      .setDesc("YYYY-MM-DD (within the past 5 days for arXiv /recent)")
      .addText((t) =>
        t.setPlaceholder("2026-05-10").onChange((v) => {
          this.value = v.trim();
        }),
      );
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText("Run")
        .setCta()
        .onClick(() => {
          if (!/^\d{4}-\d{2}-\d{2}$/.test(this.value)) {
            new Notice("Invalid date format");
            return;
          }
          this.close();
          this.onSubmit(this.value);
        }),
    );
  }
  onClose() {
    this.contentEl.empty();
  }
}

class ArxivIdModal extends Modal {
  private value = "";
  constructor(app: App, private onSubmit: (raw: string | null) => void) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Summarize paper by arXiv ID" });
    new Setting(contentEl)
      .setName("arXiv ID or URL")
      .setDesc("e.g. 2605.08080, arXiv:2605.08080v1, https://arxiv.org/abs/2605.08080")
      .addText((t) =>
        t.setPlaceholder("2605.08080").onChange((v) => {
          this.value = v.trim();
        }),
      );
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText("Summarize")
        .setCta()
        .onClick(() => {
          if (!this.value) {
            new Notice("Please enter an arXiv ID");
            return;
          }
          this.close();
          this.onSubmit(this.value);
        }),
    );
  }
  onClose() {
    this.contentEl.empty();
  }
}

class StateModal extends Modal {
  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "arXiv Daily — Recent state" });
    const snap = this.plugin.stateStore.snapshot();
    const entries = Object.entries(snap).sort((a, b) => (a[0] < b[0] ? 1 : -1));
    if (entries.length === 0) {
      contentEl.createEl("p", { text: "No runs yet." });
      return;
    }
    const ul = contentEl.createEl("ul");
    for (const [date, e] of entries.slice(0, 20)) {
      const li = ul.createEl("li");
      li.setText(
        `${date}: ${e.status} (attempts=${e.attempts}` +
          (e.papersWritten != null ? `, papers=${e.papersWritten}` : "") +
          (e.error ? `, err=${e.error.slice(0, 80)}` : "") +
          `)`,
      );
    }
  }
  onClose() {
    this.contentEl.empty();
  }
}
