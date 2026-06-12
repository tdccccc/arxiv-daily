import { App, Menu, Modal, Notice, Setting } from "obsidian";
import type ArxivDailyPlugin from "../main";
import { todayInTz, formatDate } from "./utils/time";
import { validateFilterConfig, validateLlmConfig } from "./settings/validation";
import { chooseModal } from "./services/modal";
import {
  buildDiagnosticsReport,
  type PaperIndexDiagnostics,
} from "./services/diagnostics";
import { normalizeArxivId } from "./services/manual-fetch";
import {
  isPaperStatus,
  type PaperIndexEntry,
  type PaperStatus,
} from "./services/paper-index";
import { extractArxivIdFromMarkdown } from "./services/bibtex";
import { openDashboardView } from "./dashboard/view";
import { ensurePaperNote } from "./services/paper-note";

export function registerCommands(plugin: ArxivDailyPlugin): void {
  const tz = () => plugin.settings.arxiv.timezone;
  const today = () => formatDate(todayInTz(new Date(), tz()));

  function gateFilter(): boolean {
    const v = validateFilterConfig(plugin.settings);
    if (!v.ok) {
      new Notice(`arXiv Daily — cannot run:\n${v.reasons.map((r) => "• " + r).join("\n")}`, 10_000);
      return false;
    }
    return true;
  }

  function gateLlm(): boolean {
    const v = validateLlmConfig(plugin.settings);
    if (!v.ok) {
      new Notice(`arXiv Daily — cannot run:\n${v.reasons.map((r) => "• " + r).join("\n")}`, 10_000);
      return false;
    }
    return true;
  }

  async function runToday() {
    if (!gateFilter()) return;
    const date = today();
    new Notice(`arXiv Daily: running for ${date}…`);
    const result = await plugin.scheduler.runForDateNow(date);
    new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
  }

  async function runAllPending() {
    if (!gateFilter()) return;
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

  async function retryFailedInLookback() {
    if (!gateFilter()) return;
    new Notice(`arXiv Daily: retrying failed dates in lookback…`);
    const results = await plugin.scheduler.retryFailedInLookback();
    if (results.length === 0) {
      new Notice("arXiv Daily: no failed dates in lookback window");
      return;
    }
    new Notice(`arXiv Daily retry:\n${describeRunResults(results)}`, 10_000);
  }

  function openDatePicker() {
    if (!gateFilter()) return;
    new DatePickerModal(plugin.app, async (date) => {
      if (!date) return;
      new Notice(`arXiv Daily: running for ${date}…`);
      const result = await plugin.scheduler.runForDateNow(date);
      new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
    }).open();
  }

  function openForceDatePicker() {
    if (!gateFilter()) return;
    new DatePickerModal(
      plugin.app,
      async (date) => {
        if (!date) return;
        new Notice(`arXiv Daily: force running for ${date}…`);
        const result = await plugin.scheduler.forceRunForDate(date);
        new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
      },
      {
        title: "Force run arXiv Daily for date",
        desc: "YYYY-MM-DD. Clears stored run state for this date before running; existing daily files are still not overwritten.",
        buttonText: "Force run",
      },
    ).open();
  }

  async function clearRunState() {
    const choice = await chooseModal(
      plugin.app,
      "Clear arXiv Daily run state",
      "This clears stored completed/failed/skipped statuses. Existing markdown files are not changed.",
      [
        { label: "Cancel", value: "cancel" },
        { label: "Clear state", value: "clear", warning: true },
      ],
    );
    if (choice !== "clear") return;
    await plugin.stateStore.clearAll();
    plugin.progress.setIdle(undefined);
    new Notice("arXiv Daily: run state cleared");
  }

  function cancelCurrentRun() {
    const active = plugin.scheduler.activeRuns();
    if (active.length === 0) {
      new Notice("arXiv Daily: no active run to cancel");
      return;
    }
    const cancelled = plugin.scheduler.cancelCurrentRun();
    new Notice(`arXiv Daily: cancellation requested for ${cancelled.join(", ")}`);
  }

  function openSetPaperStatusModal() {
    new PaperStatusModal(plugin.app, async (id, status) => {
      if (!id || !status) return;
      await setPaperStatus(id, status);
    }).open();
  }

  function openCreatePaperNoteModal() {
    new PaperIdModal(
      plugin.app,
      "Create arXiv Daily paper note",
      "arXiv ID or URL",
      "Create note",
      async (raw) => {
        if (!raw) return;
        const id = normalizeArxivId(raw);
        if (!id) {
          new Notice("Invalid arXiv ID");
          return;
        }
        await createPaperNote(id);
      },
    ).open();
  }

  function openCopyBibtexModal() {
    new PaperIdModal(
      plugin.app,
      "Copy arXiv BibTeX",
      "arXiv ID or URL",
      "Copy BibTeX",
      async (raw) => {
        if (!raw) return;
        await copyBibtex(raw);
      },
    ).open();
  }

  async function setCurrentPaperStatus(status: PaperStatus) {
    const id = getCurrentPaperId(plugin);
    if (!id) {
      new Notice("arXiv Daily: current note is not an indexed arXiv paper");
      return;
    }
    await setPaperStatus(id, status);
  }

  async function setPaperStatus(rawId: string, status: PaperStatus) {
    const id = normalizeArxivId(rawId);
    if (!id) {
      new Notice("Invalid arXiv ID");
      return;
    }
    const store = plugin.buildPaperIndex();
    const entry = await store.setStatus(id, status);
    if (!entry) {
      new Notice(`arXiv Daily: ${id} is not in papers.json`);
      return;
    }
    if (status === "saved") {
      await ensurePaperNote(plugin, store, entry);
    }
    new Notice(`arXiv Daily: ${id} marked ${status}`);
  }

  async function createPaperNote(rawId: string) {
    const id = normalizeArxivId(rawId);
    if (!id) {
      new Notice("Invalid arXiv ID");
      return;
    }
    const store = plugin.buildPaperIndex();
    const entry = await store.get(id);
    if (!entry) {
      new Notice(`arXiv Daily: ${id} is not in papers.json`);
      return;
    }
    const path = await ensurePaperNote(plugin, store, entry);
    await plugin.app.workspace.openLinkText(path, "", false);
    new Notice(`arXiv Daily: paper note ready at ${path}`);
  }

  async function copyCurrentPaperBibtex() {
    const id = await getCurrentPaperIdFromActiveFile(plugin);
    if (!id) {
      new Notice("arXiv Daily: current note is not an arXiv paper");
      return;
    }
    await copyBibtex(id);
  }

  async function copyBibtex(rawId: string) {
    const result = await plugin.buildBibtexService().fetchAndStore(rawId);
    if (result.kind !== "done") {
      new Notice(`arXiv Daily: BibTeX failed — ${result.reason}`, 10_000);
      return;
    }
    try {
      await writeClipboard(result.bibtex);
      new Notice(
        `arXiv Daily: copied BibTeX for ${result.arxivId} (${result.citationKey})` +
          (result.entryUpdated ? "" : "; not in papers.json"),
      );
    } catch {
      new Notice("arXiv Daily: BibTeX fetched but clipboard copy failed", 10_000);
    }
  }

  function openArxivIdPicker() {
    if (!gateLlm()) return;
    new ArxivIdModal(plugin.app, async (raw) => {
      if (!raw) return;
      new Notice(`arXiv Daily: summarizing ${raw}…`);
      const today = formatDate(todayInTz(new Date(), tz()));
      const result = await plugin.manualFetch.fetchAndSummarize(raw, today);
      new Notice(`arXiv Daily: ${describeManualResult(result)}`, 10_000);
    }).open();
  }

  async function openTodayDaily() {
    const path = `${plugin.settings.output.dailyDir}/${today()}.md`;
    const file = plugin.app.vault.getAbstractFileByPath(path);
    if (file) {
      await plugin.app.workspace.openLinkText(path, "", false);
    } else {
      new Notice(`No daily report at ${path}`);
    }
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
    id: "arxiv-daily-retry-failed",
    name: "Retry failed dates in lookback window",
    callback: retryFailedInLookback,
  });

  plugin.addCommand({
    id: "arxiv-daily-force-run-for-date",
    name: "Force run for date…",
    callback: openForceDatePicker,
  });

  plugin.addCommand({
    id: "arxiv-daily-clear-run-state",
    name: "Clear run state…",
    callback: clearRunState,
  });

  plugin.addCommand({
    id: "arxiv-daily-cancel-current-run",
    name: "Cancel current run",
    callback: cancelCurrentRun,
  });

  plugin.addCommand({
    id: "arxiv-daily-summarize-by-id",
    name: "Summarize by arXiv ID…",
    callback: openArxivIdPicker,
  });

  plugin.addCommand({
    id: "arxiv-daily-set-paper-status",
    name: "Set paper status…",
    callback: openSetPaperStatusModal,
  });

  plugin.addCommand({
    id: "arxiv-daily-create-paper-note",
    name: "Create paper note…",
    callback: openCreatePaperNoteModal,
  });

  plugin.addCommand({
    id: "arxiv-daily-copy-current-bibtex",
    name: "Copy BibTeX for current paper",
    callback: copyCurrentPaperBibtex,
  });

  plugin.addCommand({
    id: "arxiv-daily-copy-bibtex-by-id",
    name: "Copy BibTeX by arXiv ID…",
    callback: openCopyBibtexModal,
  });

  for (const status of PAPER_STATUSES) {
    plugin.addCommand({
      id: `arxiv-daily-mark-current-${status}`,
      name: `Mark current paper as ${status}`,
      callback: () => setCurrentPaperStatus(status),
    });
  }

  plugin.addCommand({
    id: "arxiv-daily-open-today",
    name: "Open today's daily report",
    callback: openTodayDaily,
  });

  plugin.addCommand({
    id: "arxiv-daily-open-reading-dashboard",
    name: "Open reading dashboard",
    callback: () => openDashboardView(plugin),
  });

  plugin.addCommand({
    id: "arxiv-daily-show-state",
    name: "Show recent run state",
    callback: () => new StateModal(plugin.app, plugin).open(),
  });

  plugin.addCommand({
    id: "arxiv-daily-show-diagnostics",
    name: "Show diagnostics",
    callback: () => new DiagnosticsModal(plugin.app, plugin).open(),
  });

  plugin.addRibbonIcon("calendar-clock", "arXiv Daily", (evt: MouseEvent) => {
    const menu = new Menu();

    const enabled = plugin.settings.schedule.enabled;
    const activeRuns = plugin.scheduler.activeRuns();

    // Status header (non-interactive)
    menu.addItem((item) =>
      item
        .setTitle(`Status: ${enabled ? "Enabled" : "Disabled"}`)
        .setIcon(enabled ? "circle-check" : "circle-slash")
        .setDisabled(true),
    );
    // Enable/Disable toggle
    menu.addItem((item) =>
      item
        .setTitle(enabled ? "Disable" : "Enable")
        .setIcon(enabled ? "pause" : "play")
        .onClick(async () => {
          const applied = await plugin.setScheduleEnabled(!enabled);
          if (applied) {
            new Notice(`arXiv Daily: ${!enabled ? "enabled" : "disabled"}`);
          }
        }),
    );

    menu.addSeparator();
    menu.addItem((item) =>
      item.setTitle("Run for today").setIcon("play").onClick(runToday),
    );
    menu.addItem((item) =>
      item
        .setTitle("Cancel current run")
        .setIcon("circle-stop")
        .setDisabled(activeRuns.length === 0)
        .onClick(cancelCurrentRun),
    );
    menu.addItem((item) =>
      item
        .setTitle("Run all pending (lookback)")
        .setIcon("layers")
        .onClick(runAllPending),
    );
    menu.addItem((item) =>
      item
        .setTitle("Retry failed (lookback)")
        .setIcon("refresh-cw")
        .onClick(retryFailedInLookback),
    );
    menu.addItem((item) =>
      item
        .setTitle("Run for specific date…")
        .setIcon("calendar")
        .onClick(openDatePicker),
    );
    menu.addItem((item) =>
      item
        .setTitle("Force run for date…")
        .setIcon("rotate-cw")
        .onClick(openForceDatePicker),
    );

    menu.addSeparator();
    menu.addItem((item) =>
      item
        .setTitle("Open today's daily report")
        .setIcon("file-text")
        .onClick(openTodayDaily),
    );
    menu.addItem((item) =>
      item
        .setTitle("Open reading dashboard")
        .setIcon("book-open-check")
        .onClick(() => openDashboardView(plugin)),
    );
    menu.addItem((item) =>
      item
        .setTitle("Summarize by arXiv ID…")
        .setIcon("file-text")
        .onClick(openArxivIdPicker),
    );
    menu.addItem((item) =>
      item
        .setTitle("Create paper note…")
        .setIcon("file-plus")
        .onClick(openCreatePaperNoteModal),
    );
    menu.addItem((item) =>
      item
        .setTitle("Copy BibTeX by arXiv ID…")
        .setIcon("copy")
        .onClick(openCopyBibtexModal),
    );
    menu.addItem((item) =>
      item
        .setTitle("Set paper status…")
        .setIcon("list-checks")
        .onClick(openSetPaperStatusModal),
    );

    menu.addSeparator();
    menu.addItem((item) =>
      item
        .setTitle("Show recent run state")
        .setIcon("list")
        .onClick(() => new StateModal(plugin.app, plugin).open()),
    );
    menu.addItem((item) =>
      item
        .setTitle("Show diagnostics")
        .setIcon("clipboard-list")
        .onClick(() => new DiagnosticsModal(plugin.app, plugin).open()),
    );
    menu.addItem((item) =>
      item
        .setTitle("Clear run state…")
        .setIcon("trash-2")
        .onClick(clearRunState),
    );
    menu.showAtMouseEvent(evt);
  });
}

const PAPER_STATUSES: PaperStatus[] = [
  "to_read",
  "reading",
  "read",
  "saved",
  "ignored",
];

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

function describeRunResults(
  results: Array<{ date: string; result: any }>,
): string {
  return results.map((r) => `${r.date}: ${describeResult(r.result)}`).join("\n");
}

class DatePickerModal extends Modal {
  private value = "";
  constructor(
    app: App,
    private onSubmit: (date: string | null) => void,
    private opts: { title?: string; desc?: string; buttonText?: string } = {},
  ) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: this.opts.title ?? "Run arXiv Daily for date" });
    new Setting(contentEl)
      .setName("Date")
      .setDesc(this.opts.desc ?? "YYYY-MM-DD (within the past 5 days for arXiv /recent)")
      .addText((t) =>
        t.setPlaceholder("2026-05-10").onChange((v) => {
          this.value = v.trim();
        }),
      );
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText(this.opts.buttonText ?? "Run")
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

class PaperIdModal extends Modal {
  private value = "";
  constructor(
    app: App,
    private title: string,
    private fieldName: string,
    private buttonText: string,
    private onSubmit: (raw: string | null) => void,
  ) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: this.title });
    new Setting(contentEl)
      .setName(this.fieldName)
      .setDesc("e.g. 2605.08080, arXiv:2605.08080v1, https://arxiv.org/abs/2605.08080")
      .addText((t) =>
        t.setPlaceholder("2605.08080").onChange((v) => {
          this.value = v.trim();
        }),
      );
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText(this.buttonText)
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

class PaperStatusModal extends Modal {
  private value = "";
  private status: PaperStatus = "to_read";
  constructor(app: App, private onSubmit: (id: string | null, status: PaperStatus | null) => void) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Set arXiv Daily paper status" });
    new Setting(contentEl)
      .setName("arXiv ID or URL")
      .setDesc("Paper must already exist in the internal arXiv Daily paper index")
      .addText((t) =>
        t.setPlaceholder("2605.08080").onChange((v) => {
          this.value = v.trim();
        }),
      );
    new Setting(contentEl)
      .setName("Status")
      .addDropdown((d) => {
        for (const status of PAPER_STATUSES) d.addOption(status, status);
        d.setValue(this.status).onChange((v) => {
          if (isPaperStatus(v)) this.status = v;
        });
      });
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText("Set status")
        .setCta()
        .onClick(() => {
          if (!this.value) {
            new Notice("Please enter an arXiv ID");
            return;
          }
          this.close();
          this.onSubmit(this.value, this.status);
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

class DiagnosticsModal extends Modal {
  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "arXiv Daily diagnostics" });
    const textarea = contentEl.createEl("textarea");
    textarea.value = "Loading diagnostics...";
    textarea.readOnly = true;
    textarea.style.width = "100%";
    textarea.style.height = "360px";
    textarea.style.fontFamily = "var(--font-monospace)";
    textarea.style.fontSize = "var(--font-smaller)";
    textarea.style.resize = "vertical";
    let report = textarea.value;
    void collectPaperIndexDiagnostics(this.plugin)
      .then((paperIndex) => {
        report = buildDiagnosticsReport({
          settings: this.plugin.settings,
          runState: this.plugin.stateStore.snapshot(),
          version: this.plugin.manifest?.version,
          paperIndex,
        });
        textarea.value = report;
      })
      .catch((e) => {
        report = buildDiagnosticsReport({
          settings: this.plugin.settings,
          runState: this.plugin.stateStore.snapshot(),
          version: this.plugin.manifest?.version,
          paperIndex: {
            path: this.plugin.buildPaperIndex().paths.papersJsonPath,
            exists: false,
            error: (e as Error).message,
          },
        });
        textarea.value = report;
      });
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText("Copy")
        .setCta()
        .onClick(async () => {
          try {
            if (navigator.clipboard?.writeText) {
              await navigator.clipboard.writeText(report);
            } else {
              textarea.select();
              document.execCommand("copy");
            }
            new Notice("arXiv Daily: diagnostics copied");
          } catch {
            textarea.select();
            new Notice("Could not copy diagnostics; text is selectable");
          }
        }),
    );
  }
  onClose() {
    this.contentEl.empty();
  }
}

function getCurrentPaperId(plugin: ArxivDailyPlugin): string | null {
  const app = plugin.app as any;
  const file = app.workspace?.getActiveFile?.();
  const frontmatter = file
    ? app.metadataCache?.getFileCache?.(file)?.frontmatter
    : null;
  const fromFrontmatter = normalizeArxivId(
    String(frontmatter?.arxiv_id ?? frontmatter?.arxiv ?? ""),
  );
  if (fromFrontmatter) return fromFrontmatter;
  const basename =
    typeof file?.basename === "string"
      ? file.basename
      : typeof file?.name === "string"
      ? file.name.replace(/\.md$/i, "")
      : "";
  return normalizeArxivId(basename);
}

async function getCurrentPaperIdFromActiveFile(
  plugin: ArxivDailyPlugin,
): Promise<string | null> {
  const fromMetadata = getCurrentPaperId(plugin);
  if (fromMetadata) return fromMetadata;

  const app = plugin.app as any;
  const file = app.workspace?.getActiveFile?.();
  if (!file?.path) return null;
  try {
    const markdown = await plugin.app.vault.adapter.read(file.path);
    return extractArxivIdFromMarkdown(markdown);
  } catch {
    return null;
  }
}

async function writeClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    if (!document.execCommand("copy")) {
      throw new Error("execCommand copy returned false");
    }
  } finally {
    textarea.remove();
  }
}

async function collectPaperIndexDiagnostics(
  plugin: ArxivDailyPlugin,
): Promise<PaperIndexDiagnostics> {
  const store = plugin.buildPaperIndex();
  const exists =
    (await plugin.app.vault.adapter.exists(store.paths.papersJsonPath)) ||
    (await plugin.app.vault.adapter.exists(store.paths.legacyPapersJsonPath));
  const diag: PaperIndexDiagnostics = {
    path: store.paths.papersJsonPath,
    exists,
  };
  if (!exists) return diag;
  try {
    const index = await store.load();
    const entries = Object.values(index.papers);
    const statusCounts: Record<string, number> = {};
    const invalidStatuses: string[] = [];
    const missingPaperPaths: string[] = [];
    for (const entry of entries) {
      statusCounts[entry.status] = (statusCounts[entry.status] ?? 0) + 1;
      if (!isPaperStatus(entry.status)) {
        invalidStatuses.push(`${entry.arxivId}: ${entry.status}`);
      }
      if (
        entry.paperPath &&
        !(await plugin.app.vault.adapter.exists(entry.paperPath))
      ) {
        missingPaperPaths.push(`${entry.arxivId}: ${entry.paperPath}`);
      }
    }
    return {
      ...diag,
      schemaVersion: index.schemaVersion,
      total: entries.length,
      statusCounts,
      invalidStatuses,
      missingPaperPaths,
    };
  } catch (e) {
    return {
      ...diag,
      error: (e as Error).message,
    };
  }
}
