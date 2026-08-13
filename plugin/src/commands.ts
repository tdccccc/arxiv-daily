import { App, Modal, Notice, Setting } from "obsidian";
import type ArxivDailyPlugin from "../main";
import { todayInTz, formatDate } from "@arxiv-daily/core";
import { validateFilterConfig, validateLlmConfig } from "@arxiv-daily/core";
import { chooseModal } from "./services/modal";
import {
  formatFullTextRuntimeDiagnostics,
  summarizeFullTextRuntimeDiagnostics,
} from "./services/fulltext-runtime-diagnostics";
import { normalizeArxivId } from "@arxiv-daily/core";
import {
  type PaperPriority,
  type PaperStatus,
} from "@arxiv-daily/core";
import { openDashboardView, refreshOpenDashboardViews } from "./dashboard/view";
import { ensurePaperNote } from "./services/paper-note";
import { bindEnterToButton, openDatePickerModal } from "./date-picker-modal";
import { formatRunHistoryRecords } from "@arxiv-daily/core";
import { buildSafePluginDiagnosticsReport } from "./services/paper-index-diagnostics";

export { bindEnterToButton, isValidCalendarDate } from "./date-picker-modal";
export {
  collectPaperIndexDiagnostics,
  isSupportedPaperIndexSchemaVersion,
} from "./services/paper-index-diagnostics";
import {
  describeManualResult,
  describeResult,
  describeRunResults,
} from "@arxiv-daily/core";

export function registerCommands(plugin: ArxivDailyPlugin): void {
  const tz = () => plugin.settings.arxiv.timezone;
  const today = () => formatDate(todayInTz(new Date(), tz()));
  const notice: CommandNotice = (message, timeoutMs) => {
    plugin.logger.info(message);
    new Notice(message, timeoutMs);
  };
  const runDetached = (promise: Promise<unknown>, action: string): void => {
    void promise.catch((error: unknown) => {
      plugin.logger.error(`commands: failed to ${action}`, error);
      notice(
        `arXiv Daily: failed to ${action}: ${errorMessage(error)}`,
        10_000,
      );
    });
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
    openDatePickerModal(
      plugin.app,
      (date) => {
        runDetached(
          (async () => {
            notice(`arXiv Daily: running for ${date}…`);
            const result = await plugin.scheduler.runForDateNow(date);
            notice(`arXiv Daily ${date}: ${describeResult(result)}`);
          })(),
          `run for ${date}`,
        );
      },
      {},
      notice,
    );
  }

  function openForceDatePicker() {
    if (!gateFilter()) return;
    openDatePickerModal(
      plugin.app,
      (date) => {
        runDetached(
          (async () => {
            notice(`arXiv Daily: force running for ${date}…`);
            const result = await plugin.scheduler.forceRunForDate(date);
            notice(`arXiv Daily ${date}: ${describeResult(result)}`);
          })(),
          `force run for ${date}`,
        );
      },
      {
        title: "Force run arXiv Daily for date",
        desc: "YYYY-MM-DD. Clears stored run state for this date before running; existing daily files are still not overwritten.",
        buttonText: "Force run",
      },
      notice,
    );
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
    new PaperMarkModal(plugin.app, (id, mark) => {
      if (!id || !mark) return;
      runDetached(setPaperMark(id, mark), `mark ${id}`);
    }, notice).open();
  }

  function openCreatePaperNoteModal() {
    new PaperIdModal(
      plugin.app,
      "Create arXiv Daily paper note",
      "arXiv ID or URL",
      "Create note",
      (raw) => {
        if (!raw) return;
        const id = normalizeArxivId(raw);
        if (!id) {
          notice("Invalid arXiv ID");
          return;
        }
        runDetached(createPaperNote(id), `create paper note for ${id}`);
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
    await plugin.withOutputOperation(
      mark === "saved" ? "paper-note" : "paper-index",
      `Set paper mark: ${id}`,
      id,
      async () => {
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
      },
    );
  }

  async function createPaperNote(rawId: string) {
    const id = normalizeArxivId(rawId);
    if (!id) {
      notice("Invalid arXiv ID");
      return;
    }
    await plugin.withOutputOperation(
      "paper-note",
      `Create paper note: ${id}`,
      id,
      async () => {
        const store = plugin.buildPaperIndex();
        const entry = await store.get(id);
        if (!entry) {
          notice(`arXiv Daily: ${id} is not in papers.json`);
          return;
        }
        const path = await ensurePaperNote(plugin, store, entry);
        await plugin.app.workspace.openLinkText(path, "", false);
        notice(`arXiv Daily: paper note ready at ${path}`);
      },
    );
  }

  function openArxivIdPicker() {
    if (!gateLlm()) return;
    new ArxivIdModal(plugin.app, (raw) => {
      if (!raw) return;
      runDetached(
        (async () => {
          notice(`arXiv Daily: summarizing ${raw}…`);
          const today = formatDate(todayInTz(new Date(), tz()));
          const result = await plugin.manualFetch.fetchAndSummarize(raw, today);
          notice(`arXiv Daily: ${describeManualResult(result)}`, 10_000);
          if (result.kind === "done" || result.kind === "already_exists") {
            await refreshOpenDashboardViews(plugin);
            await plugin.app.workspace.openLinkText(result.path, "", false);
          }
        })(),
        `summarize ${raw}`,
      );
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
    callback: () => runDetached(runToday(), "run today"),
  });

  plugin.addCommand({
    id: "run-for-date",
    name: "Run for date…",
    callback: openDatePicker,
  });

  plugin.addCommand({
    id: "run-all-pending",
    name: "Run all pending in lookback window",
    callback: () => runDetached(runAllPending(), "run pending dates"),
  });

  plugin.addCommand({
    id: "retry-failed",
    name: "Retry failed dates in lookback window",
    callback: () =>
      runDetached(retryFailedInLookback(), "retry failed dates"),
  });

  plugin.addCommand({
    id: "force-run-for-date",
    name: "Force run for date…",
    callback: openForceDatePicker,
  });

  plugin.addCommand({
    id: "clear-run-state",
    name: "Clear run state…",
    callback: () => runDetached(clearRunState(), "clear run state"),
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
      callback: () =>
        runDetached(setCurrentPaperMark(mark.value), `mark current paper ${mark.value}`),
    });
  }

  plugin.addCommand({
    id: "open-today",
    name: "Open today's daily report",
    callback: () => runDetached(openTodayDaily(), "open today's daily report"),
  });

  plugin.addCommand({
    id: "send-test-email",
    name: "Send test email",
    callback: () =>
      runDetached(
        (async () => {
          notice("arXiv Daily: sending test email…");
          const message = await plugin.sendTestEmail();
          notice(message, 8_000);
        })(),
        "send test email",
      ),
  });

  plugin.addCommand({
    id: "open-reading-dashboard",
    name: "Open reading dashboard",
    callback: () => runDetached(openDashboardView(plugin), "open dashboard"),
  });

  plugin.addCommand({
    id: "review-reading-candidates",
    name: "Review reading candidates",
    callback: () => {
      try {
        plugin.openReadingCandidatesReview();
      } catch (error) {
        plugin.logger.error("commands: failed to open reading candidates review", error);
        notice("arXiv Daily: reading candidates review could not be opened. Try again.", 10_000);
      }
    },
  });

  plugin.addCommand({
    id: "review-personal-library-directions",
    name: "Review personal library directions",
    callback: () => {
      try {
        plugin.openPersonalLibraryDirectionReview();
      } catch (error) {
        plugin.logger.error("commands: failed to open personal library direction review", error);
        notice("arXiv Daily: direction review could not be opened. Try again.", 10_000);
      }
    },
  });

  plugin.addCommand({
    id: "check-incremental-direction-updates",
    name: "Check incremental direction updates",
    callback: () =>
      runDetached(
        (async () => {
          notice("arXiv Daily: checking incremental direction updates…");
          try {
            const summary = await plugin.runIncrementalDirectionUpdate();
            const pending = summary.pendingAuthorizationBuffered > 0
              ? `, ${summary.pendingAuthorizationBuffered} awaiting model authorization`
              : "";
            const superseded = summary.superseded > 0
              ? `, ${summary.superseded} un-reviewed suggestion(s) superseded by new evidence`
              : "";
            notice(
              `arXiv Daily: incremental update — ${summary.suggestions} suggestion(s) `
              + `stored (${summary.attachments} attachment(s)), `
              + `${summary.buffered} paper(s) buffered${pending}${superseded}`,
              10_000,
            );
          } catch (error) {
            plugin.logger.error("commands: incremental direction update failed", error);
            notice(`arXiv Daily: incremental update failed: ${errorMessage(error)}`, 10_000);
          }
        })(),
        "check incremental direction updates",
      ),
  });

  plugin.addCommand({
    id: "review-incremental-suggestions",
    name: "Review incremental direction suggestions",
    callback: () => {
      try {
        plugin.openPersonalLibraryDirectionReview();
      } catch (error) {
        plugin.logger.error("commands: failed to open incremental suggestion review", error);
        notice("arXiv Daily: suggestion review could not be opened. Try again.", 10_000);
      }
    },
  });

  plugin.addCommand({
    id: "index-personal-library-fulltext",
    name: "Index personal library full text (local embeddings)",
    callback: () =>
      runDetached(
        (async () => {
          notice("arXiv Daily: indexing personal library full text…");
          try {
            const summary = await plugin.indexPersonalLibraryFullText();
            const refreshed = summary.titlesRefreshed > 0
              ? `, ${summary.titlesRefreshed} titles refreshed`
              : "";
            notice(
              `arXiv Daily: full-text index — ${summary.indexed} indexed, `
              + `${summary.reused} reused, ${summary.failed} failed, ${summary.pruned} pruned${refreshed}`,
              10_000,
            );
          } catch (error) {
            plugin.logger.error("commands: personal library full-text indexing failed", error);
            notice(`arXiv Daily: full-text indexing failed: ${errorMessage(error)}`, 10_000);
          }
        })(),
        "index personal library full text",
      ),
  });

  plugin.addCommand({
    id: "search-personal-library-fulltext",
    name: "Search personal library full text…",
    callback: () =>
      new FullTextQueryModal(plugin.app, (query) => {
        if (!query) return;
        runDetached(
          (async () => {
            try {
              const matches = await plugin.searchPersonalLibraryFullText(query);
              if (matches.length === 0) {
                notice("arXiv Daily: no similar papers found in the full-text index", 10_000);
                return;
              }
              const lines = matches.slice(0, 5).map((match) => {
                return `${match.title}\n  ${match.filePath ?? match.paperKey} · similarity ${match.score.toFixed(3)}`;
              });
              notice(`arXiv Daily: top ${Math.min(matches.length, 5)} matches\n${lines.join("\n")}`, 12_000);
            } catch (error) {
              plugin.logger.error("commands: personal library full-text search failed", error);
              notice(`arXiv Daily: full-text search failed: ${errorMessage(error)}`, 10_000);
            }
          })(),
          "search personal library full text",
        );
      }, notice).open(),
  });

  plugin.addCommand({
    id: "diagnose-fulltext-runtime",
    name: "Diagnose full-text runtime (pdf.js + embeddings)",
    callback: () =>
      runDetached(
        (async () => {
          notice("arXiv Daily: diagnosing full-text runtime…");
          try {
            const report = await plugin.diagnoseFullTextRuntime();
            const text = formatFullTextRuntimeDiagnostics(report);
            plugin.logger.info(`full-text runtime diagnostics:\n${text}`);
            notice(
              `arXiv Daily: full-text runtime — ${summarizeFullTextRuntimeDiagnostics(report)}`,
              15_000,
            );
            new FullTextRuntimeDiagnosticsModal(plugin.app, text).open();
          } catch (error) {
            plugin.logger.error("commands: full-text runtime diagnostics failed", error);
            notice(`arXiv Daily: full-text runtime diagnostics failed: ${errorMessage(error)}`, 10_000);
          }
        })(),
        "diagnose full-text runtime",
      ),
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
      runDetached(openDashboardView(plugin), "open dashboard from ribbon");
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

class FullTextQueryModal extends Modal {
  private value = "";
  constructor(
    app: App,
    private onSubmit: (query: string | null) => void,
    private notice: CommandNotice,
  ) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Search personal library full text" });
    let inputEl: HTMLInputElement | null = null;
    let submitButton: HTMLButtonElement | null = null;
    new Setting(contentEl)
      .setName("Query")
      .setDesc("A research question or description; matched against local full-text embeddings")
      .addText((t) => {
        inputEl = t.inputEl;
        t.setPlaceholder("e.g. graph neural networks for node classification")
          .onChange((v) => {
            this.value = v.trim();
          });
      });
    new Setting(contentEl).addButton((b) => {
      submitButton = b.buttonEl;
      b
        .setButtonText("Search")
        .setCta()
        .onClick(() => {
          if (!this.value) {
            this.notice("Please enter a query");
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
            const ownerDocument = textarea.ownerDocument;
            const clipboard = ownerDocument.defaultView?.navigator.clipboard;
            if (clipboard?.writeText) {
              await clipboard.writeText(report);
            } else {
              textarea.select();
              ownerDocument.execCommand("copy");
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

export class DiagnosticsModal extends Modal {
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
    void buildSafePluginDiagnosticsReport(this.plugin)
      .then((value) => {
        report = value;
        textarea.value = report;
      })
      .catch((error) => {
        this.plugin.logger.warn("diagnostics render failed", error);
        report = `Failed to build diagnostics: ${errorMessage(error)}`;
        textarea.value = report;
      });
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText("Copy")
        .setCta()
        .onClick(async () => {
          try {
            const ownerDocument = textarea.ownerDocument;
            const clipboard = ownerDocument.defaultView?.navigator.clipboard;
            if (clipboard?.writeText) {
              await clipboard.writeText(report);
            } else {
              textarea.select();
              ownerDocument.execCommand("copy");
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

class FullTextRuntimeDiagnosticsModal extends Modal {
  constructor(app: App, private readonly report: string) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "arXiv Daily full-text runtime diagnostics" });
    const textarea = contentEl.createEl("textarea", {
      cls: "arxiv-daily-diagnostics-textarea",
    });
    textarea.value = this.report;
    textarea.readOnly = true;
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText("Copy")
        .setCta()
        .onClick(async () => {
          try {
            const ownerDocument = textarea.ownerDocument;
            const clipboard = ownerDocument.defaultView?.navigator.clipboard;
            if (clipboard?.writeText) {
              await clipboard.writeText(this.report);
            } else {
              textarea.select();
              ownerDocument.execCommand("copy");
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

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error);
}
