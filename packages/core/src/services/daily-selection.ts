import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { daysBefore, formatDate, todayInTz } from "../utils/time";
import type { Logger } from "./logger";
import { dailySelectionMarkerRegExp } from "./daily-selection-marker";
import { LOOKBACK_DAYS } from "./scheduling/constants";
import type {
  PaperIndexEntry,
  PaperInbox,
  PaperPriority,
  PaperStatus,
  PaperIndexStore,
} from "./paper-index";
import { paperKeyFromArxivId } from "./paper-key";

export interface DailyPaperSelection {
  arxivId: string;
  watch: boolean;
  highlight: boolean;
}

export interface DailySelectionSyncResult {
  found: number;
  changed: number;
  missing: string[];
}

export interface DailySelectionStartupSyncResult
  extends DailySelectionSyncResult {
  scanned: number;
  paths: string[];
}

export function parseDailySelections(markdown: string): DailyPaperSelection[] {
  const selections = new Map<string, DailyPaperSelection>();
  const re = dailySelectionMarkerRegExp();
  let m: RegExpExecArray | null;
  while ((m = re.exec(markdown)) !== null) {
    const [, rawChecked, arxivId, rawKind] = m;
    if (!rawChecked || !arxivId || !rawKind) continue;
    const checked = rawChecked.toLowerCase() === "x";
    const kind = rawKind as "watch" | "highlight";
    const cur =
      selections.get(arxivId) ?? { arxivId, watch: false, highlight: false };
    if (kind === "watch") cur.watch = checked;
    if (kind === "highlight") cur.highlight = checked;
    selections.set(arxivId, cur);
  }
  return Array.from(selections.values());
}

export function applyDailySelections(
  store: PaperIndexStore,
  selections: DailyPaperSelection[],
): Promise<DailySelectionSyncResult> {
  return store.mutate((index) => {
    const result = applySelectionsToIndex(index, selections);
    return { result, changed: result.changed > 0 };
  });
}

export function applySelectionsToIndex(
  index: PaperInbox,
  selections: DailyPaperSelection[],
): DailySelectionSyncResult {
  let changed = 0;
  const missing: string[] = [];
  for (const selection of selections) {
    const entry = index.papers[paperKeyFromArxivId(selection.arxivId)];
    if (!entry) {
      missing.push(selection.arxivId);
      continue;
    }
    const next = stateForSelection(entry, selection);
    if (!next) continue;
    if (entry.status !== next.status || entry.priority !== next.priority) {
      entry.status = next.status;
      entry.priority = next.priority;
      changed += 1;
    }
  }
  return { found: selections.length, changed, missing };
}

function stateForSelection(
  entry: PaperIndexEntry,
  selection: DailyPaperSelection,
): { status: PaperStatus; priority: PaperPriority } | null {
  if (selection.highlight || selection.watch) {
    if (entry.status !== "inbox" && entry.status !== "to_read") return null;
    if (selection.highlight) return { status: "to_read", priority: "high" };
    return { status: "to_read", priority: "normal" };
  }

  if (entry.status === "to_read") {
    return { status: "inbox", priority: "normal" };
  }
  if (entry.status === "inbox" && entry.priority !== "normal") {
    return { status: "inbox", priority: "normal" };
  }
  return null;
}

export class DailySelectionSyncService {
  private timers = new Map<string, ReturnType<typeof setTimeout>>();

  constructor(
    private opts: {
      storage: StorageAdapter;
      getOutput: () => OutputSettings;
      // Test-only until this sync service is wired into the plugin lifecycle.
      buildPaperIndex: () => PaperIndexStore;
      logger: Logger;
      debounceMs?: number;
      getLookbackDays?: () => number;
      getTimezone?: () => string;
      now?: () => Date;
    },
  ) {}

  schedule(file: { path?: string } | null | undefined): void {
    const path = this.opts.storage.normalizePath(file?.path ?? "");
    if (!this.isDailyPath(path)) return;
    const existing = this.timers.get(path);
    if (existing) clearTimeout(existing);
    const timer = setTimeout(() => {
      this.timers.delete(path);
      this.syncPath(path).catch((e) =>
        this.opts.logger.error(`daily-selection: sync failed for ${path}`, e),
      );
    }, this.opts.debounceMs ?? 750);
    this.timers.set(path, timer);
  }

  async syncPath(path: string): Promise<DailySelectionSyncResult | null> {
    const norm = this.opts.storage.normalizePath(path);
    if (!this.isDailyPath(norm)) return null;
    const content = await this.opts.storage.readText(norm);
    const result = await this.syncMarkdown(content);
    if (result.changed > 0) {
      this.opts.logger.info(
        `daily-selection: synced ${result.changed}/${result.found} selections from ${norm}`,
      );
    }
    return result;
  }

  async syncRecentDailyFiles(): Promise<DailySelectionStartupSyncResult> {
    const paths = this.recentDailyPaths();
    const existingPaths: string[] = [];
    const allSelections: DailyPaperSelection[] = [];

    for (const path of paths) {
      if (!(await this.opts.storage.exists(path))) continue;
      existingPaths.push(path);
      const content = await this.opts.storage.readText(path);
      allSelections.push(...parseDailySelections(content));
    }

    const selections = mergeSelections(allSelections);
    const result =
      selections.length > 0
        ? await applyDailySelections(this.opts.buildPaperIndex(), selections)
        : { found: 0, changed: 0, missing: [] };

    if (result.changed > 0) {
      this.opts.logger.info(
        `daily-selection: startup synced ${result.changed}/${result.found} selections from ${existingPaths.length} daily files`,
      );
    }

    return {
      ...result,
      scanned: existingPaths.length,
      paths: existingPaths,
    };
  }

  clear(): void {
    for (const timer of this.timers.values()) clearTimeout(timer);
    this.timers.clear();
  }

  private async syncMarkdown(
    markdown: string,
  ): Promise<DailySelectionSyncResult> {
    const selections = parseDailySelections(markdown);
    if (selections.length === 0) return { found: 0, changed: 0, missing: [] };
    return applyDailySelections(this.opts.buildPaperIndex(), selections);
  }

  private recentDailyPaths(): string[] {
    const timezone = this.opts.getTimezone?.() ?? "UTC";
    const now = this.opts.now?.() ?? new Date();
    const today = todayInTz(now, timezone);
    const dailyDir = this.opts.storage.normalizePath(
      this.opts.getOutput().dailyDir,
    );
    const paths: string[] = [];
    for (let i = LOOKBACK_DAYS - 1; i >= 0; i--) {
      paths.push(
        this.opts.storage.normalizePath(
          `${dailyDir}/${formatDate(daysBefore(today, i, timezone))}.md`,
        ),
      );
    }
    return paths;
  }

  private isDailyPath(path: string): boolean {
    if (!path.endsWith(".md")) return false;
    const dailyDir = this.opts.storage.normalizePath(
      this.opts.getOutput().dailyDir,
    );
    return path.startsWith(`${dailyDir}/`);
  }
}

function mergeSelections(
  selections: DailyPaperSelection[],
): DailyPaperSelection[] {
  const merged = new Map<string, DailyPaperSelection>();
  for (const selection of selections) {
    const cur =
      merged.get(selection.arxivId) ?? {
        arxivId: selection.arxivId,
        watch: false,
        highlight: false,
      };
    cur.watch = cur.watch || selection.watch || selection.highlight;
    cur.highlight = cur.highlight || selection.highlight;
    merged.set(selection.arxivId, cur);
  }
  return Array.from(merged.values());
}
