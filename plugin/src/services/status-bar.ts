import type { ProgressReporter, ProgressStage, IdleReason } from "@arxiv-daily/core";
import type { StateStore } from "@arxiv-daily/core";

const STAGE_LABELS: Record<ProgressStage, string> = {
  "fetch-metadata": "metadata",
  "fetch-recent": "fetch /recent",
  "enrich-abstract": "abstracts",
  "filter": "filter",
  "fetch-content": "fetch",
  "summarize-daily": "summarize",
  "summarize-detail": "detail summary",
  "write-detail": "detail",
};

const STAGE_ORDER: ProgressStage[] = [
  "fetch-metadata",
  "fetch-recent",
  "enrich-abstract",
  "filter",
  "fetch-content",
  "summarize-daily",
  "summarize-detail",
  "write-detail",
];

const AUTO_HIDE_COMPLETE_MS = 4_000;
const AUTO_HIDE_IDLE_MS = 1_500;
const AUTO_HIDE_ERROR_MS = 10_000;

export interface StatusBarOpts {
  initiallyEnabled: boolean;
}

interface BatchState {
  currentDay: number;
  totalDays: number;
  date: string;
}

interface StageState {
  stage: ProgressStage;
  current?: number;
  total?: number;
}

export class StatusBarController implements ProgressReporter {
  private disabled: boolean;
  private batch: BatchState | null = null;
  private stage: StageState | null = null;
  private task: { title: string; detail?: string } | null = null;
  private panel: HTMLElement | null = null;
  private panelTitle: HTMLElement | null = null;
  private panelDetail: HTMLElement | null = null;
  private panelFill: HTMLElement | null = null;
  private panelTrack: HTMLElement | null = null;
  private panelPercent: HTMLElement | null = null;
  private panelState: "running" | "complete" | "error" | null = null;
  private hideTimer: number | null = null;
  private hideTimerView: Window | null = null;
  private lastCompletedDate: string | undefined;
  private idleReason: IdleReason | undefined;
  private disposed = false;

  constructor(
    private readonly el: HTMLElement,
    store: StateStore,
    opts: StatusBarOpts,
  ) {
    this.disabled = !opts.initiallyEnabled;
    this.lastCompletedDate = pickLastCompleted(store);
    this.render();
  }

  setTask(title: string, detail?: string): void {
    if (this.disposed) return;
    this.disabled = false;
    this.task = { title, detail };
    this.idleReason = undefined;
    this.clearHideTimer();
    this.render();
  }

  setBatch(currentDay: number, totalDays: number, date: string): void {
    if (this.disposed) return;
    this.disabled = false;
    this.idleReason = undefined;
    this.batch = { currentDay, totalDays, date };
    this.clearHideTimer();
    this.render();
  }

  setStage(stage: ProgressStage, current?: number, total?: number): void {
    if (this.disposed) return;
    this.disabled = false;
    this.idleReason = undefined;
    this.stage = { stage, current, total };
    this.clearHideTimer();
    this.render();
  }

  setComplete(message = "Complete"): void {
    if (this.disposed) return;
    this.disabled = false;
    this.stage = null;
    this.el.textContent = "Complete";
    this.renderPanel(message, "Done", 100, "complete");
    this.scheduleHide(AUTO_HIDE_COMPLETE_MS);
  }

  setError(message: string): void {
    if (this.disposed) return;
    this.disabled = false;
    this.stage = null;
    this.el.textContent = "Failed";
    this.renderPanel("Stopped", message, 100, "error");
    this.scheduleHide(AUTO_HIDE_ERROR_MS);
  }

  setIdle(lastCompletedDate?: string, reason?: IdleReason): void {
    if (this.disposed) return;
    this.disabled = false;
    this.batch = null;
    this.stage = null;
    this.task = null;
    this.idleReason = reason;
    if (lastCompletedDate) this.lastCompletedDate = lastCompletedDate;
    this.render();
    if (this.panelState !== "complete" && this.panelState !== "error") {
      this.scheduleHide(AUTO_HIDE_IDLE_MS);
    }
  }

  setDisabled(): void {
    if (this.disposed) return;
    this.disabled = true;
    this.batch = null;
    this.stage = null;
    this.task = null;
    this.idleReason = undefined;
    this.render();
    this.hidePanel();
  }

  dispose(): void {
    this.disposed = true;
    this.clearHideTimer();
    this.panel?.remove();
    this.panel = null;
    this.panelTitle = null;
    this.panelDetail = null;
    this.panelFill = null;
    this.panelTrack = null;
    this.panelPercent = null;
    this.panelState = null;
  }

  private render(): void {
    this.el.textContent = this.computeText();
    if (this.disabled || (!this.task && !this.batch && !this.stage)) return;
    const title = this.task?.title ?? "arXiv Daily";
    const detail = this.computePanelDetail();
    this.renderPanel(title, detail, this.computePercent(), "running");
  }

  private computeText(): string {
    if (this.disabled) return "arXiv: disabled";
    if (this.batch && this.stage) {
      const stagePart = formatStage(this.stage);
      if (this.batch.totalDays > 1) {
        return `arXiv: ${this.batch.date} [${this.batch.currentDay}/${this.batch.totalDays}] · ${stagePart}`;
      }
      return `arXiv: ${this.batch.date} · ${stagePart}`;
    }
    if (this.task && this.stage) {
      return `arXiv: ${this.task.title} · ${formatStage(this.stage)}`;
    }
    if (this.task) {
      return `arXiv: ${this.task.title}${this.task.detail ? ` · ${this.task.detail}` : ""}`;
    }
    if (this.batch) {
      if (this.batch.totalDays > 1) {
        return `arXiv: ${this.batch.date} [${this.batch.currentDay}/${this.batch.totalDays}]`;
      }
      return `arXiv: ${this.batch.date}`;
    }
    if (this.idleReason === "weekend") return "arXiv: idle · weekend";
    if (this.lastCompletedDate) return `arXiv: idle · last ${this.lastCompletedDate}`;
    return "arXiv: idle";
  }

  private computePanelDetail(): string {
    const parts: string[] = [];
    if (this.task?.detail) parts.push(this.task.detail);
    if (this.batch) {
      parts.push(
        this.batch.totalDays > 1
          ? `${this.batch.date} (${this.batch.currentDay}/${this.batch.totalDays})`
          : this.batch.date,
      );
    }
    if (this.stage) parts.push(formatStage(this.stage));
    return parts.join(" · ") || "Working";
  }

  private computePercent(): number {
    if (!this.stage) return this.batch ? 5 : 0;
    const stageIndex = Math.max(0, STAGE_ORDER.indexOf(this.stage.stage));
    const stageCount = STAGE_ORDER.length;
    const stageProgress =
      this.stage.current != null &&
      this.stage.total != null &&
      this.stage.total > 0
        ? clamp(this.stage.current / this.stage.total, 0, 1)
        : 0.45;
    return Math.round(((stageIndex + stageProgress) / stageCount) * 100);
  }

  private renderPanel(
    title: string,
    detail: string,
    percent: number,
    state: "running" | "complete" | "error",
  ): void {
    const panel = this.ensurePanel();
    panel.classList.remove("is-complete", "is-error");
    if (state === "complete") panel.classList.add("is-complete");
    if (state === "error") panel.classList.add("is-error");
    this.panelState = state;
    this.panelTitle!.textContent = title;
    this.panelDetail!.textContent = detail;
    const normalized = clamp(percent, 0, 100);
    this.panelFill!.style.width = `${normalized}%`;
    this.panelPercent!.textContent = `${normalized}%`;
    this.panelTrack!.setAttribute("aria-valuemin", "0");
    this.panelTrack!.setAttribute("aria-valuemax", "100");
    this.panelTrack!.setAttribute("aria-valuenow", String(normalized));
    panel.classList.remove("is-hidden");
  }

  private ensurePanel(): HTMLElement {
    if (this.panel) return this.panel;
    const panel = this.el.ownerDocument.body.createDiv({
      cls: "arxiv-daily-progress",
      attr: { "aria-live": "polite" },
    });
    const header = panel.createDiv({
      cls: "arxiv-daily-progress__header",
    });
    this.panelTitle = header.createDiv({
      cls: "arxiv-daily-progress__title",
    });
    this.panelPercent = header.createDiv({
      cls: "arxiv-daily-progress__percent",
    });
    this.panelDetail = panel.createDiv({
      cls: "arxiv-daily-progress__detail",
    });
    const track = panel.createDiv({
      cls: "arxiv-daily-progress__track",
      attr: {
        role: "progressbar",
        "aria-valuemin": "0",
        "aria-valuemax": "100",
        "aria-valuenow": "0",
      },
    });
    this.panelTrack = track;
    this.panelFill = track.createDiv({
      cls: "arxiv-daily-progress__fill",
    });
    this.panel = panel;
    return panel;
  }

  private scheduleHide(delayMs: number): void {
    this.clearHideTimer();
    const view = this.el.ownerDocument.defaultView;
    if (!view) return;
    this.hideTimerView = view;
    this.hideTimer = view.setTimeout(() => this.hidePanel(), delayMs);
  }

  private clearHideTimer(): void {
    if (this.hideTimer != null && this.hideTimerView) {
      this.hideTimerView.clearTimeout(this.hideTimer);
    }
    this.hideTimer = null;
    this.hideTimerView = null;
  }

  private hidePanel(): void {
    this.clearHideTimer();
    this.panel?.classList.add("is-hidden");
    this.panelState = null;
  }
}

function formatStage(stage: StageState): string {
  const label = STAGE_LABELS[stage.stage];
  if (stage.current != null && stage.total != null) {
    return `${label} ${stage.current}/${stage.total}`;
  }
  return label;
}

function pickLastCompleted(store: StateStore): string | undefined {
  const snap = store.snapshot();
  const completed = Object.entries(snap)
    .filter(([, v]) => v.status === "completed")
    .map(([k]) => k)
    .sort();
  return completed[completed.length - 1];
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
