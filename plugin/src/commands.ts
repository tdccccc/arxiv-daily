import { App, Modal, Notice, Setting } from "obsidian";
import type ArxivDailyPlugin from "../main";
import { todayInTz, formatDate } from "@arxiv-daily/core";
import { validateFilterConfig, validateLlmConfig } from "@arxiv-daily/core";
import { chooseModal } from "./services/modal";
import {
  buildDiagnosticsReport,
  redactText,
  type PaperIndexDiagnostics,
} from "@arxiv-daily/core";
import { normalizeArxivId } from "@arxiv-daily/core";
import {
  PAPER_INBOX_SCHEMA_VERSION,
  isPaperPriority,
  isPaperStatus,
  type PaperIndexEntry,
  type PaperPriority,
  type PaperStatus,
} from "@arxiv-daily/core";
import { openDashboardView } from "./dashboard/view";
import { ensurePaperNote } from "./services/paper-note";
import { formatRunHistoryRecords } from "@arxiv-daily/core";
import {
  describeManualResult,
  describeResult,
  describeRunResults,
} from "@arxiv-daily/core";

export function isValidCalendarDate(value: string): boolean {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
  );
}

export function bindEnterToButton(
  input: HTMLInputElement,
  button: HTMLButtonElement,
): void {
  input.addEventListener("keydown", (evt) => {
    if (evt.key !== "Enter") return;
    evt.preventDefault();
    button.click();
  });
}

export function registerCommands(plugin: ArxivDailyPlugin): void {
  const tz = () => plugin.settings.arxiv.timezone;
  const today = () => formatDate(todayInTz(new Date(), tz()));
  const notice: CommandNotice = (message, timeoutMs) => {
    plugin.logger.info(message);
    new Notice(message, timeoutMs);
  };

  function gateFilter(): boolean {
    const v = validateFilterConfig(plugin.settings);
    if (!v.ok) {
      notice(`arXiv Daily — cannot run:\n${v.reasons.map((r) => "• " + r).join("\n")}`, 10_000);
      return false;
    }
    return true;
  }

  function gateLlm(): boolean {
    const v = validateLlmConfig(plugin.settings);
    if (!v.ok) {
      notice(`arXiv Daily — cannot run:\n${v.reasons.map((r) => "• " + r).join("\n")}`, 10_000);
      return false;
    }
    return true;
  }

  async function runToday() {
    if (!gateFilter()) return;
    const date = today();
    notice(`arXiv Daily: running for ${date}…`);
    const result = await plugin.scheduler.runForDateNow(date);
    notice(`arXiv Daily ${date}: ${describeResult(result)}`);
  }

  async function runAllPending() {
    if (!gateFilter()) return;
    notice(`arXiv Daily: running all pending in lookback…`);
    const results = await plugin.scheduler.runAllPending();
    if (results.length === 0) {
      notice("arXiv Daily: nothing pending in lookback window");
      return;
    }
    const summary = results
      .map((r) => `${r.date}: ${describeResult(r.result)}`)
      .join("\n");
    notice(`arXiv Daily (lookback):\n${summary}`, 10_000);
  }

  async function retryFailedInLookback() {
    if (!gateFilter()) return;
    notice(`arXiv Daily: retrying failed dates in lookback…`);
    const results = await plugin.scheduler.retryFailedInLookback();
    if (results.length === 0) {
      notice("arXiv Daily: no failed dates in lookback window");
      return;
    }
    notice(`arXiv Daily retry:\n${describeRunResults(results)}`, 10_000);
  }

  function openDatePicker() {
    if (!gateFilter()) return;
    new DatePickerModal(
      plugin.app,
      async (date) => {
        if (!date) return;
        notice(`arXiv Daily: running for ${date}…`);
        const result = await plugin.scheduler.runForDateNow(date);
        notice(`arXiv Daily ${date}: ${describeResult(result)}`);
      },
      {},
      notice,
    ).open();
  }

  function openForceDatePicker() {
    if (!gateFilter()) return;
    new DatePickerModal(
      plugin.app,
      async (date) => {
        if (!date) return;
        notice(`arXiv Daily: force running for ${date}…`);
        const result = await plugin.scheduler.forceRunForDate(date);
        notice(`arXiv Daily ${date}: ${describeResult(result)}`);
      },
      {
        title: "Force run arXiv Daily for date",
        desc: "YYYY-MM-DD. Clears stored run state for this date before running; existing daily files are still not overwritten.",
        buttonText: "Force run",
      },
      notice,
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
    notice("arXiv Daily: run state cleared");
  }

  function cancelCurrentRun() {
    const active = plugin.operations.snapshot();
    if (active.length === 0) {
      notice("arXiv Daily: no active tasks to cancel");
      return;
    }
    plugin.operations.cancelAll();
    notice(`arXiv Daily: cancellation requested for ${active.length} active task${active.length === 1 ? "" : "s"}`);
  }

  function openSetPaperMarkModal() {
    new PaperMarkModal(plugin.app, async (id, mark) => {
      if (!id || !mark) return;
      await setPaperMark(id, mark);
    }, notice).open();
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
          notice("Invalid arXiv ID");
          return;
        }
        await createPaperNote(id);
      },
      notice,
    ).open();
  }

  async function setCurrentPaperMark(mark: PaperMark) {
    const id = getCurrentPaperId(plugin);
    if (!id) {
      notice("arXiv Daily: current note is not an indexed arXiv paper");
      return;
    }
    await setPaperMark(id, mark);
  }

  async function setPaperMark(rawId: string, mark: PaperMark) {
    const id = normalizeArxivId(rawId);
    if (!id) {
      notice("Invalid arXiv ID");
      return;
    }
    const store = plugin.buildPaperIndex();
    const state = stateForMark(mark);
    let entry = await store.setStatus(id, state.status);
    if (!entry) {
      notice(`arXiv Daily: ${id} is not in papers.json`);
      return;
    }
    entry = await store.setPriority(id, state.priority);
    if (!entry) {
      notice(`arXiv Daily: ${id} is not in papers.json`);
      return;
    }
    if (mark === "saved") {
      await ensurePaperNote(plugin, store, entry);
    }
    notice(`arXiv Daily: ${id} marked ${labelForMark(mark)}`);
  }

  async function createPaperNote(rawId: string) {
    const id = normalizeArxivId(rawId);
    if (!id) {
      notice("Invalid arXiv ID");
      return;
    }
    const store = plugin.buildPaperIndex();
    const entry = await store.get(id);
    if (!entry) {
      notice(`arXiv Daily: ${id} is not in papers.json`);
      return;
    }
    const path = await ensurePaperNote(plugin, store, entry);
    await plugin.app.workspace.openLinkText(path, "", false);
    notice(`arXiv Daily: paper note ready at ${path}`);
  }

  function openArxivIdPicker() {
    if (!gateLlm()) return;
    new ArxivIdModal(plugin.app, async (raw) => {
      if (!raw) return;
      notice(`arXiv Daily: summarizing ${raw}…`);
      const today = formatDate(todayInTz(new Date(), tz()));
      const result = await plugin.manualFetch.fetchAndSummarize(raw, today);
      notice(`arXiv Daily: ${describeManualResult(result)}`, 10_000);
    }, notice).open();
  }

  async function openTodayDaily() {
    const path = `${plugin.settings.output.dailyDir}/${today()}.md`;
    const file = plugin.app.vault.getAbstractFileByPath(path);
    if (file) {
      await plugin.app.workspace.openLinkText(path, "", false);
    } else {
      notice(`No daily report at ${path}`);
    }
  }

  plugin.addCommand({
    id: "run-now",
    name: "Run Today",
    callback: runToday,
  });

  plugin.addCommand({
    id: "run-for-date",
    name: "Run for date…",
    callback: openDatePicker,
  });

  plugin.addCommand({
    id: "run-all-pending",
    name: "Run all pending in lookback window",
    callback: runAllPending,
  });

  plugin.addCommand({
    id: "retry-failed",
    name: "Retry failed dates in lookback window",
    callback: retryFailedInLookback,
  });

  plugin.addCommand({
    id: "force-run-for-date",
    name: "Force run for date…",
    callback: openForceDatePicker,
  });

  plugin.addCommand({
    id: "clear-run-state",
    name: "Clear run state…",
    callback: clearRunState,
  });

  plugin.addCommand({
    id: "cancel-current-run",
    name: "Cancel active tasks",
    callback: cancelCurrentRun,
  });

  plugin.addCommand({
    id: "summarize-by-id",
    name: "Summarize by arXiv ID…",
    callback: openArxivIdPicker,
  });

  plugin.addCommand({
    id: "set-paper-status",
    name: "Set paper mark…",
    callback: openSetPaperMarkModal,
  });

  plugin.addCommand({
    id: "create-paper-note",
    name: "Create paper note…",
    callback: openCreatePaperNoteModal,
  });

  for (const mark of PAPER_MARKS) {
    plugin.addCommand({
      id: `mark-current-${mark.value}`,
      name: `Mark current paper as ${mark.label}`,
      callback: () => setCurrentPaperMark(mark.value),
    });
  }

  plugin.addCommand({
    id: "open-today",
    name: "Open today's daily report",
    callback: openTodayDaily,
  });

  plugin.addCommand({
    id: "open-reading-dashboard",
    name: "Open reading dashboard",
    callback: () => openDashboardView(plugin),
  });

  plugin.addCommand({
    id: "show-state",
    name: "Show recent run state",
    callback: () => new StateModal(plugin.app, plugin).open(),
  });

  plugin.addCommand({
    id: "show-run-history",
    name: "Show run history",
    callback: () => new RunHistoryModal(plugin.app, plugin).open(),
  });

  plugin.addCommand({
    id: "show-diagnostics",
    name: "Show diagnostics",
    callback: () => new DiagnosticsModal(plugin.app, plugin).open(),
  });

  const dashboardRibbonIcon = plugin.addRibbonIcon(
    "book-open-check",
    "arXiv Daily Dashboard",
    () => {
      void openDashboardView(plugin);
    },
  );
  dashboardRibbonIcon.addClass("arxiv-daily-ribbon-dashboard");
}

type PaperMark = "inbox" | "watch" | "highlight" | "saved" | "ignored";
type CommandNotice = (message: string, timeoutMs?: number) => void;

const PAPER_MARKS: Array<{ value: PaperMark; label: string }> = [
  { value: "inbox", label: "Unmarked" },
  { value: "watch", label: "Watch" },
  { value: "highlight", label: "Highlight" },
  { value: "saved", label: "Saved" },
  { value: "ignored", label: "Ignored" },
];

function isPaperMark(value: string): value is PaperMark {
  return PAPER_MARKS.some((mark) => mark.value === value);
}

function labelForMark(mark: PaperMark): string {
  return PAPER_MARKS.find((item) => item.value === mark)?.label ?? mark;
}

function stateForMark(mark: PaperMark): {
  status: PaperStatus;
  priority: PaperPriority;
} {
  if (mark === "watch") return { status: "to_read", priority: "normal" };
  if (mark === "highlight") return { status: "to_read", priority: "high" };
  if (mark === "saved") return { status: "saved", priority: "normal" };
  if (mark === "ignored") return { status: "ignored", priority: "normal" };
  return { status: "inbox", priority: "normal" };
}

class DatePickerModal extends Modal {
  private value = "";
  constructor(
    app: App,
    private onSubmit: (date: string | null) => void,
    private opts: { title?: string; desc?: string; buttonText?: string },
    private notice: CommandNotice,
  ) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: this.opts.title ?? "Run arXiv Daily for date" });
    let inputEl: HTMLInputElement | null = null;
    let submitButton: HTMLButtonElement | null = null;
    const dateSetting = new Setting(contentEl)
      .setName("Date")
      .setDesc(this.opts.desc ?? "Choose a real calendar date within the supported arXiv window.")
      .addText((t) => {
        inputEl = t.inputEl;
        t.inputEl.type = "date";
        t.inputEl.setAttribute("aria-describedby", "arxiv-daily-date-error");
        t.onChange((v) => {
          this.value = v.trim();
          refreshValidation();
        });
      });
    const errorEl = dateSetting.descEl.createEl("div", {
      attr: { id: "arxiv-daily-date-error", "aria-live": "polite" },
    });
    const refreshValidation = () => {
      const valid = isValidCalendarDate(this.value);
      errorEl.textContent = this.value && !valid ? "Enter a valid calendar date." : "";
      if (submitButton) submitButton.disabled = !valid;
      inputEl?.setAttribute("aria-invalid", String(Boolean(this.value) && !valid));
    };
    new Setting(contentEl).addButton((b) => {
      submitButton = b.buttonEl;
      b
        .setButtonText(this.opts.buttonText ?? "Run")
        .setCta()
        .setDisabled(true)
        .onClick(() => {
          if (!isValidCalendarDate(this.value)) {
            refreshValidation();
            this.notice("Invalid calendar date");
            return;
          }
          this.close();
          this.onSubmit(this.value);
        });
    });
    if (inputEl && submitButton) bindEnterToButton(inputEl, submitButton);
  }
  onClose() {
    this.contentEl.empty();
  }
}

class ArxivIdModal extends Modal {
  private value = "";
  constructor(
    app: App,
    private onSubmit: (raw: string | null) => void,
    private notice: CommandNotice,
  ) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Summarize paper by arXiv ID" });
    let inputEl: HTMLInputElement | null = null;
    let submitButton: HTMLButtonElement | null = null;
    new Setting(contentEl)
      .setName("arXiv ID or URL")
      .setDesc("e.g. 2605.08080, arXiv:2605.08080v1, https://arxiv.org/abs/2605.08080")
      .addText((t) => {
        inputEl = t.inputEl;
        t.setPlaceholder("2605.08080").onChange((v) => {
          this.value = v.trim();
        });
      });
    new Setting(contentEl).addButton((b) => {
      submitButton = b.buttonEl;
      b
        .setButtonText("Summarize")
        .setCta()
        .onClick(() => {
          if (!this.value) {
            this.notice("Please enter an arXiv ID");
            return;
          }
          this.close();
          this.onSubmit(this.value);
        });
    });
    if (inputEl && submitButton) bindEnterToButton(inputEl, submitButton);
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
    private notice: CommandNotice,
  ) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: this.title });
    let inputEl: HTMLInputElement | null = null;
    let submitButton: HTMLButtonElement | null = null;
    new Setting(contentEl)
      .setName(this.fieldName)
      .setDesc("e.g. 2605.08080, arXiv:2605.08080v1, https://arxiv.org/abs/2605.08080")
      .addText((t) => {
        inputEl = t.inputEl;
        t.setPlaceholder("2605.08080").onChange((v) => {
          this.value = v.trim();
        });
      });
    new Setting(contentEl).addButton((b) => {
      submitButton = b.buttonEl;
      b
        .setButtonText(this.buttonText)
        .setCta()
        .onClick(() => {
          if (!this.value) {
            this.notice("Please enter an arXiv ID");
            return;
          }
          this.close();
          this.onSubmit(this.value);
        });
    });
    if (inputEl && submitButton) bindEnterToButton(inputEl, submitButton);
  }
  onClose() {
    this.contentEl.empty();
  }
}

class PaperMarkModal extends Modal {
  private value = "";
  private mark: PaperMark = "watch";
  constructor(
    app: App,
    private onSubmit: (id: string | null, mark: PaperMark | null) => void,
    private notice: CommandNotice,
  ) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Set arXiv Daily paper mark" });
    let inputEl: HTMLInputElement | null = null;
    let submitButton: HTMLButtonElement | null = null;
    new Setting(contentEl)
      .setName("arXiv ID or URL")
      .setDesc("Paper must already exist in the internal arXiv Daily paper index")
      .addText((t) => {
        inputEl = t.inputEl;
        t.setPlaceholder("2605.08080").onChange((v) => {
          this.value = v.trim();
        });
      });
    new Setting(contentEl)
      .setName("Mark")
      .addDropdown((d) => {
        for (const mark of PAPER_MARKS) d.addOption(mark.value, mark.label);
        d.setValue(this.mark).onChange((v) => {
          if (isPaperMark(v)) this.mark = v;
        });
      });
    new Setting(contentEl).addButton((b) => {
      submitButton = b.buttonEl;
      b
        .setButtonText("Set mark")
        .setCta()
        .onClick(() => {
          if (!this.value) {
            this.notice("Please enter an arXiv ID");
            return;
          }
          this.close();
          this.onSubmit(this.value, this.mark);
        });
    });
    if (inputEl && submitButton) bindEnterToButton(inputEl, submitButton);
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

class RunHistoryModal extends Modal {
  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "arXiv Daily run history" });
    const textarea = contentEl.createEl("textarea", {
      cls: "arxiv-daily-diagnostics-textarea",
    });
    textarea.value = "Loading run history…";
    textarea.readOnly = true;
    let report = textarea.value;
    void this.plugin.runHistoryStore
      .readLatest(100)
      .then((records) => {
        report = formatRunHistoryRecords(records);
        textarea.value = report;
      })
      .catch((e) => {
        this.plugin.logger.warn("run history load failed", e);
        report = `Failed to load run history: ${(e as Error).message}`;
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
            this.plugin.logger.info("arXiv Daily: run history copied");
            new Notice("arXiv Daily: run history copied");
          } catch (e) {
            this.plugin.logger.warn("Could not copy run history; text is selectable", e);
            textarea.select();
            new Notice("Could not copy run history; text is selectable");
          }
        }),
    );
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
    const textarea = contentEl.createEl("textarea", {
      cls: "arxiv-daily-diagnostics-textarea",
    });
    textarea.value = "Loading diagnostics…";
    textarea.readOnly = true;
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
        this.plugin.logger.warn("diagnostics load failed", e);
        report = buildDiagnosticsReport({
          settings: this.plugin.settings,
          runState: this.plugin.stateStore.snapshot(),
          version: this.plugin.manifest?.version,
          paperIndex: {
            path: this.plugin.buildPaperIndex().paths.papersJsonPath,
            exists: false,
            error: redactText(e instanceof Error ? e.message : e, {
              secrets: [this.plugin.settings.llm.apiKey],
            }),
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
            this.plugin.logger.info("arXiv Daily: diagnostics copied");
            new Notice("arXiv Daily: diagnostics copied");
          } catch (e) {
            this.plugin.logger.warn("Could not copy diagnostics; text is selectable", e);
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
  const file = plugin.app.workspace.getActiveFile();
  const frontmatter = file
    ? plugin.app.metadataCache.getFileCache(file)?.frontmatter
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

function recordOrEmpty(value: unknown): Record<string, unknown> {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

async function collectPaperIndexDiagnostics(
  plugin: ArxivDailyPlugin,
): Promise<PaperIndexDiagnostics> {
  const store = plugin.buildPaperIndex();
  const path = (await plugin.app.vault.adapter.exists(store.paths.papersJsonPath))
    ? store.paths.papersJsonPath
    : (await plugin.app.vault.adapter.exists(store.paths.legacyPapersJsonPath))
    ? store.paths.legacyPapersJsonPath
    : null;
  const diag: PaperIndexDiagnostics = {
    path: store.paths.papersJsonPath,
    exists: Boolean(path),
  };
  if (!path) return diag;
  try {
    const raw = JSON.parse(await plugin.app.vault.adapter.read(path));
    const obj = recordOrEmpty(raw);
    const rawSchemaVersion = obj.schemaVersion;
    const schemaVersion =
      typeof rawSchemaVersion === "number" ? rawSchemaVersion : undefined;
    const papers =
      obj.papers && typeof obj.papers === "object"
        ? (obj.papers as Record<string, unknown>)
        : {};
    const statusCounts: Record<string, number> = {};
    const invalidStatuses: string[] = [];
    const invalidPriorities: string[] = [];
    const invalidSeenDates: string[] = [];
    const missingPaperPaths: string[] = [];
    const noteArxivIdMismatches: string[] = [];

    for (const [id, value] of Object.entries(papers)) {
      const entry = recordOrEmpty(value);
      const arxivId = stringOr(entry.arxivId, id);
      const status = stringOr(entry.status, "");
      const priority = stringOr(entry.priority, "");
      if (status) statusCounts[status] = (statusCounts[status] ?? 0) + 1;
      if (!isPaperStatus(status)) {
        invalidStatuses.push(`${arxivId}: ${status || "(missing)"}`);
      }
      if (!isPaperPriority(priority)) {
        invalidPriorities.push(`${arxivId}: ${priority || "(missing)"}`);
      }
      if (!Array.isArray(entry.seenDates)) {
        invalidSeenDates.push(`${arxivId}: seenDates is not an array`);
      } else {
        for (const date of entry.seenDates) {
          if (typeof date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            invalidSeenDates.push(`${arxivId}: ${String(date)}`);
          }
        }
      }

      const paperPath = stringOr(entry.paperPath, "");
      if (paperPath && !(await plugin.app.vault.adapter.exists(paperPath))) {
        missingPaperPaths.push(`${arxivId}: ${paperPath}`);
        continue;
      }
      if (paperPath) {
        const noteArxivId = await readNoteArxivId(plugin, paperPath);
        if (noteArxivId && noteArxivId !== arxivId) {
          noteArxivIdMismatches.push(
            `${arxivId}: ${paperPath} has arxiv_id ${noteArxivId}`,
          );
        }
      }
    }
    return {
      ...diag,
      schemaVersion,
      unsupportedSchemaVersion:
        rawSchemaVersion !== 1 && rawSchemaVersion !== PAPER_INBOX_SCHEMA_VERSION
          ? String(rawSchemaVersion)
          : undefined,
      total: Object.keys(papers).length,
      statusCounts,
      invalidStatuses,
      invalidPriorities,
      invalidSeenDates,
      missingPaperPaths,
      noteArxivIdMismatches,
    };
  } catch (e) {
    return {
      ...diag,
      error: (e as Error).message,
    };
  }
}

async function readNoteArxivId(
  plugin: ArxivDailyPlugin,
  path: string,
): Promise<string | null> {
  try {
    const markdown = await plugin.app.vault.adapter.read(path);
    const frontmatter = /^---\s*\n([\s\S]*?)\n---/.exec(markdown)?.[1] ?? "";
    if (!frontmatter) return null;
    const raw = /^arxiv_id:\s*(.+)$/m.exec(frontmatter)?.[1]?.trim() ?? "";
    if (!raw) return null;
    return normalizeArxivId(raw.replace(/^["']|["']$/g, ""));
  } catch (e) {
    plugin.logger.warn(`diagnostics: failed to read arXiv ID from ${path}`, e);
    return null;
  }
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}
