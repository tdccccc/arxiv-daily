import type {
  PaperIndexEntry,
  PaperPriority,
  PaperStatus,
  PaperSummary,
} from "../services/paper-index";

export type DashboardTab = "starred" | "all";

export type DashboardSortKey =
  | "priority"
  | "status"
  | "title"
  | "topic"
  | "published"
  | "firstSeen";

export type DashboardSortDirection = "asc" | "desc";

export interface DashboardQuery {
  tab?: DashboardTab;
  search?: string;
  topics?: string[];
  statuses?: PaperStatus[];
  priorities?: PaperPriority[];
  dateFrom?: string;
  dateTo?: string;
  hasNote?: boolean;
  detail?: boolean;
  sort?: {
    key: DashboardSortKey;
    direction?: DashboardSortDirection;
  };
}

export interface DashboardRow {
  entry: PaperIndexEntry;
  arxivId: string;
  title: string;
  authors: string;
  topic: string;
  firstSeen: string;
  hasNote: boolean;
}

export interface DashboardStats {
  total: number;
  topicCounts: Record<string, number>;
  statusCounts: Record<PaperStatus, number>;
  priorityCounts: Record<PaperPriority, number>;
  weekAdded: number;
  starred: number;
}

export interface DashboardResult {
  rows: DashboardRow[];
  stats: DashboardStats;
  tabCounts: Record<DashboardTab, number>;
}

export type DashboardAction =
  | {
      type: "set_status";
      arxivIds: string[];
      status: PaperStatus;
    }
  | {
      type: "set_priority";
      arxivIds: string[];
      priority: PaperPriority;
    }
  | {
      type: "set_mark";
      arxivIds: string[];
      status: PaperStatus;
      priority: PaperPriority;
    }
  | {
      type: "create_notes";
      arxivIds: string[];
    };

export interface DashboardPatch {
  arxivId: string;
  status?: PaperStatus;
  priority?: PaperPriority;
  ensureNote?: boolean;
}

export interface DashboardActionPlan {
  patches: DashboardPatch[];
  missingIds: string[];
  requiresConfirmation: boolean;
}

const DEFAULT_TAB: DashboardTab = "starred";
const STATUS_ORDER: Record<PaperStatus, number> = {
  to_read: 0,
  saved: 1,
  inbox: 2,
  reading: 3,
  read: 4,
  ignored: 5,
};
const PRIORITY_ORDER: Record<PaperPriority, number> = {
  high: 0,
  normal: 1,
  low: 2,
};

export function queryDashboard(
  entries: PaperIndexEntry[],
  query: DashboardQuery = {},
  opts: { now?: Date } = {},
): DashboardResult {
  const filtered = entries.filter((entry) => matchesDashboardQuery(entry, query));
  const rows = sortRows(filtered.map(toDashboardRow), query.sort);
  return {
    rows,
    stats: buildDashboardStats(rows.map((row) => row.entry), opts),
    tabCounts: buildDashboardTabCounts(entries),
  };
}

export function matchesDashboardQuery(
  entry: PaperIndexEntry,
  query: DashboardQuery = {},
): boolean {
  if (!matchesDashboardTab(entry, query.tab ?? DEFAULT_TAB)) return false;
  if (!matchesSearch(entry, query.search ?? "")) return false;
  if (!matchesTopic(entry, query.topics ?? [])) return false;
  if (!matchesAny(entry.status, query.statuses ?? [])) return false;
  if (!matchesAny(entry.priority, query.priorities ?? [])) return false;
  if (!matchesDateRange(entry, query)) return false;
  if (query.hasNote != null && Boolean(entry.paperPath) !== query.hasNote) {
    return false;
  }
  if (query.detail != null && entry.detail !== query.detail) return false;
  return true;
}

export function matchesDashboardTab(
  entry: PaperIndexEntry,
  tab: DashboardTab,
): boolean {
  switch (tab) {
    case "starred":
      return entry.status !== "ignored" && entry.priority === "high";
    case "all":
      return entry.status !== "ignored";
  }
}

export function buildDashboardStats(
  entries: PaperIndexEntry[],
  opts: { now?: Date } = {},
): DashboardStats {
  const weekRange = getLocalWeekRange(opts.now ?? new Date());
  const stats: DashboardStats = {
    total: entries.length,
    topicCounts: {},
    statusCounts: emptyStatusCounts(),
    priorityCounts: emptyPriorityCounts(),
    weekAdded: 0,
    starred: 0,
  };

  for (const entry of entries) {
    const topic = displayTopic(entry);
    stats.topicCounts[topic] = (stats.topicCounts[topic] ?? 0) + 1;
    stats.statusCounts[entry.status] += 1;
    stats.priorityCounts[entry.priority] += 1;
    if (entryHasSeenDateInRange(entry, weekRange.start, weekRange.end)) {
      stats.weekAdded += 1;
    }
    if (matchesDashboardTab(entry, "starred")) stats.starred += 1;
  }

  return stats;
}

export function buildDashboardTabCounts(
  entries: PaperIndexEntry[],
): Record<DashboardTab, number> {
  return {
    starred: entries.filter((entry) => matchesDashboardTab(entry, "starred")).length,
    all: entries.filter((entry) => matchesDashboardTab(entry, "all")).length,
  };
}

export function planDashboardAction(
  entries: PaperIndexEntry[],
  action: DashboardAction,
): DashboardActionPlan {
  const byId = new Map(entries.map((entry) => [entry.arxivId, entry]));
  const patches: DashboardPatch[] = [];
  const missingIds: string[] = [];
  const ids = uniqueIds(action.arxivIds);

  for (const arxivId of ids) {
    const entry = byId.get(arxivId);
    if (!entry) {
      missingIds.push(arxivId);
      continue;
    }

    if (action.type === "set_status") {
      if (entry.status === action.status) continue;
      const patch: DashboardPatch = { arxivId, status: action.status };
      if (action.status === "saved" && !entry.paperPath) patch.ensureNote = true;
      patches.push(patch);
      continue;
    }

    if (action.type === "set_priority") {
      if (entry.priority === action.priority) continue;
      patches.push({ arxivId, priority: action.priority });
      continue;
    }

    if (action.type === "set_mark") {
      if (entry.status === action.status && entry.priority === action.priority) {
        continue;
      }
      const patch: DashboardPatch = {
        arxivId,
        status: action.status,
        priority: action.priority,
      };
      if (action.status === "saved" && !entry.paperPath) patch.ensureNote = true;
      patches.push(patch);
      continue;
    }

    if (!entry.paperPath) {
      patches.push({ arxivId, ensureNote: true });
    }
  }

  return {
    patches,
    missingIds,
    requiresConfirmation: patches.some((patch) => patch.ensureNote),
  };
}

function toDashboardRow(entry: PaperIndexEntry): DashboardRow {
  return {
    entry,
    arxivId: entry.arxivId,
    title: entry.title,
    authors: entry.authors.join(", "),
    topic: displayTopic(entry),
    firstSeen: firstSeenDate(entry),
    hasNote: Boolean(entry.paperPath),
  };
}

function sortRows(
  rows: DashboardRow[],
  sort: DashboardQuery["sort"],
): DashboardRow[] {
  const key = sort?.key ?? "priority";
  const direction = sort?.direction ?? "asc";
  const dir = direction === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const primary = compareBySortKey(a, b, key);
    if (primary !== 0) return primary * dir;
    return compareDefault(a, b);
  });
}

function compareBySortKey(
  a: DashboardRow,
  b: DashboardRow,
  key: DashboardSortKey,
): number {
  switch (key) {
    case "priority":
      return PRIORITY_ORDER[a.entry.priority] - PRIORITY_ORDER[b.entry.priority];
    case "status":
      return STATUS_ORDER[a.entry.status] - STATUS_ORDER[b.entry.status];
    case "title":
      return a.title.localeCompare(b.title);
    case "topic":
      return a.topic.localeCompare(b.topic);
    case "published":
      return a.entry.published.localeCompare(b.entry.published);
    case "firstSeen":
      return a.firstSeen.localeCompare(b.firstSeen);
  }
}

function compareDefault(a: DashboardRow, b: DashboardRow): number {
  const priority = PRIORITY_ORDER[a.entry.priority] - PRIORITY_ORDER[b.entry.priority];
  if (priority !== 0) return priority;
  const firstSeen = b.firstSeen.localeCompare(a.firstSeen);
  if (firstSeen !== 0) return firstSeen;
  const published = b.entry.published.localeCompare(a.entry.published);
  if (published !== 0) return published;
  return a.title.localeCompare(b.title);
}

function matchesSearch(entry: PaperIndexEntry, search: string): boolean {
  const tokens = search
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  if (tokens.length === 0) return true;
  const haystack = searchableText(entry);
  return tokens.every((token) => haystack.includes(token));
}

function searchableText(entry: PaperIndexEntry): string {
  return [
    entry.arxivId,
    entry.title,
    ...entry.authors,
    entry.primaryTopic,
    ...entry.topics,
    entry.category,
    ...(entry.categories ?? []),
    ...summaryText(entry.summary),
  ]
    .join(" ")
    .toLowerCase();
}

function summaryText(summary: PaperSummary | undefined): string[] {
  if (!summary) return [];
  return [
    summary.sourceSections,
    summary.coreProblem,
    summary.keyMethod,
    summary.mainResult,
    summary.whyRelevant,
    summary.limitations,
  ].filter((value): value is string => Boolean(value));
}

function matchesTopic(entry: PaperIndexEntry, topics: string[]): boolean {
  const wanted = topics.map((topic) => topic.trim()).filter(Boolean);
  if (wanted.length === 0) return true;
  const have = new Set([entry.primaryTopic, ...entry.topics].filter(Boolean));
  return wanted.some((topic) => have.has(topic));
}

function matchesAny<T extends string>(value: T, allowed: T[]): boolean {
  return allowed.length === 0 || allowed.includes(value);
}

function matchesDateRange(
  entry: PaperIndexEntry,
  query: DashboardQuery,
): boolean {
  const from = query.dateFrom?.trim() ?? "";
  const to = query.dateTo?.trim() ?? "";
  if (!from && !to) return true;
  return entryHasSeenDateInRange(entry, from, to);
}

function entryHasSeenDateInRange(
  entry: PaperIndexEntry,
  from: string,
  to: string,
): boolean {
  const dates = entry.seenDates.length ? entry.seenDates : [entry.published];
  return dates.some((date) => dateInRange(date, from, to));
}

function dateInRange(date: string, from: string, to: string): boolean {
  if (!date) return false;
  if (from && date < from) return false;
  if (to && date > to) return false;
  return true;
}

function displayTopic(entry: PaperIndexEntry): string {
  return entry.primaryTopic || entry.topics[0] || "(none)";
}

function firstSeenDate(entry: PaperIndexEntry): string {
  return [...entry.seenDates].sort()[0] ?? entry.published;
}

function uniqueIds(ids: string[]): string[] {
  const out: string[] = [];
  for (const id of ids) {
    const arxivId = id.trim();
    if (arxivId && !out.includes(arxivId)) out.push(arxivId);
  }
  return out;
}

function emptyStatusCounts(): Record<PaperStatus, number> {
  return {
    inbox: 0,
    to_read: 0,
    reading: 0,
    read: 0,
    saved: 0,
    ignored: 0,
  };
}

function emptyPriorityCounts(): Record<PaperPriority, number> {
  return {
    low: 0,
    normal: 0,
    high: 0,
  };
}

function getLocalWeekRange(now: Date): { start: string; end: string } {
  const end = new Date(now);
  const start = new Date(now);
  const day = start.getDay();
  const daysSinceMonday = day === 0 ? 6 : day - 1;
  start.setDate(start.getDate() - daysSinceMonday);
  return {
    start: formatLocalDate(start),
    end: formatLocalDate(end),
  };
}

function formatLocalDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}
