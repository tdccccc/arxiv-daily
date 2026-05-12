import type { ProgressReporter, ProgressStage, IdleReason } from "./progress";
import type { StateStore } from "./state-store";

const STAGE_LABELS: Record<ProgressStage, string> = {
  "fetch-recent": "fetch /recent",
  "enrich-abstract": "abstracts",
  "filter": "filter",
  "fetch-content": "fetch",
  "summarize-daily": "summarize",
  "write-detail": "detail",
};

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
  private lastCompletedDate: string | undefined;
  private idleReason: IdleReason | undefined;

  constructor(
    private readonly el: HTMLElement,
    store: StateStore,
    opts: StatusBarOpts,
  ) {
    this.disabled = !opts.initiallyEnabled;
    this.lastCompletedDate = pickLastCompleted(store);
    this.render();
  }

  setBatch(currentDay: number, totalDays: number, date: string): void {
    this.disabled = false;
    this.idleReason = undefined;
    this.batch = { currentDay, totalDays, date };
    this.render();
  }

  setStage(stage: ProgressStage, current?: number, total?: number): void {
    this.disabled = false;
    this.idleReason = undefined;
    this.stage = { stage, current, total };
    this.render();
  }

  setIdle(lastCompletedDate?: string, reason?: IdleReason): void {
    this.disabled = false;
    this.batch = null;
    this.stage = null;
    this.idleReason = reason;
    if (lastCompletedDate) this.lastCompletedDate = lastCompletedDate;
    this.render();
  }

  setDisabled(): void {
    this.disabled = true;
    this.batch = null;
    this.stage = null;
    this.idleReason = undefined;
    this.render();
  }

  private render(): void {
    this.el.textContent = this.computeText();
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
    if (this.idleReason === "weekend") return "arXiv: idle · weekend";
    if (this.lastCompletedDate) return `arXiv: idle · last ${this.lastCompletedDate}`;
    return "arXiv: idle";
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
