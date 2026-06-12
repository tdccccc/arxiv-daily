import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { daysBefore, formatDate, todayInTz } from "../utils/time";
import type { Logger } from "./logger";
import type {
  PaperIndexEntry,
  PaperInbox,
  PaperPriority,
  PaperStatus,
  PaperIndexStore,
} from "./paper-index";

export const DAILY_SELECTION_MARKER = "arxiv-daily";

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

export function selectionControlsForPaper(
  arxivId: string,
  entry?: PaperIndexEntry | null,
): string {
  const watch = entry?.status === "to_read";
  const highlight = entry?.status === "to_read" && entry.priority === "high";
  return [
    `- [${watch || highlight ? "x" : " "}] 关注 <!-- ${DAILY_SELECTION_MARKER}:${arxivId}:watch -->`,
    `- [${highlight ? "x" : " "}] 重点 <!-- ${DAILY_SELECTION_MARKER}:${arxivId}:highlight -->`,
  ].join("\n");
}

export function parseDailySelections(markdown: string): DailyPaperSelection[] {
  const selections = new Map<string, DailyPaperSelection>();
  const re =
    /^[ \t]*[-*][ \t]+\[([ xX])\][^\n]*?<!--\s*arxiv-daily:(\d{4}\.\d{4,5}):(?:selection:)?(watch|highlight)\s*-->/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(markdown)) !== null) {
    const checked = m[1].toLowerCase() === "x";
    const arxivId = m[2];
    const kind = m[3] as "watch" | "highlight";
    const cur =
      selections.get(arxivId) ?? { arxivId, watch: false, highlight: false };
    if (kind === "watch") cur.watch = checked;
    if (kind === "highlight") cur.highlight = checked;
    selections.set(arxivId, cur);
  }
  return Array.from(selections.values());
}

export function injectSelectionControls(
  markdown: string,
  papers: Array<{ id: string; indexEntry?: PaperIndexEntry }>,
): string {
  let out = markdown;
  for (const paper of papers) {
    if (out.includes(`${DAILY_SELECTION_MARKER}:${paper.id}:watch`)) continue;
    const controls = selectionControlsForPaper(paper.id, paper.indexEntry);
    out = insertControlsForPaper(out, paper.id, controls);
  }
  return out;
}

function insertControlsForPaper(
  markdown: string,
  arxivId: string,
  controls: string,
): string {
  const lines = markdown.split("\n");
  const arxivLine = new RegExp(
    String.raw`arxiv\.org/(?:abs|pdf)/${escapeRegExp(arxivId)}|${escapeRegExp(`[${arxivId}]`)}`,
  );
  for (let i = 0; i < lines.length; i++) {
    if (arxivLine.test(lines[i])) {
      lines.splice(i + 1, 0, controls);
      return lines.join("\n");
    }
  }

  const linkedHeading = new RegExp(
    String.raw`^###\s+.*\[\[${escapeRegExp(arxivId)}\]\]`,
  );
  for (let i = 0; i < lines.length; i++) {
    if (linkedHeading.test(lines[i])) {
      lines.splice(i + 1, 0, controls);
      return lines.join("\n");
    }
  }

  return markdown;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export async function applyDailySelections(
  store: PaperIndexStore,
  selections: DailyPaperSelection[],
): Promise<DailySelectionSyncResult> {
  const index = await store.load();
  const result = applySelectionsToIndex(index, selections);
  if (result.changed > 0) await store.save(index);
  return result;
}

export function applySelectionsToIndex(
  index: PaperInbox,
  selections: DailyPaperSelection[],
): DailySelectionSyncResult {
  let changed = 0;
  const missing: string[] = [];
  for (const selection of selections) {
    const entry = index.papers[selection.arxivId];
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
    const lookbackDays = Math.max(
      1,
      Math.floor(this.opts.getLookbackDays?.() ?? 1),
    );
    const timezone = this.opts.getTimezone?.() ?? "UTC";
    const now = this.opts.now?.() ?? new Date();
    const today = todayInTz(now, timezone);
    const dailyDir = this.opts.storage.normalizePath(
      this.opts.getOutput().dailyDir,
    );
    const paths: string[] = [];
    for (let i = lookbackDays - 1; i >= 0; i--) {
      paths.push(
        this.opts.storage.normalizePath(
          `${dailyDir}/${formatDate(daysBefore(today, i))}.md`,
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
