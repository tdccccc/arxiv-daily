import type {
  PaperIndexEntry,
  PaperPriority,
  PaperStatus,
  PaperSummary,
} from "../services/paper-index";
import type {
  DiscoveryDirectionProvenance,
  PersonalizedDiscoveryRepresentative,
} from "../pipeline/personalized-paper-filter";
import type {
  PersonalNovelty,
  PersonalNoveltyDifferenceType,
} from "../pipeline/personalized-novelty";
import type { SummaryLanguage, Topic } from "../settings/types";
import {
  PaperSearchIndex,
  type PaperSearchReason,
  type PaperSearchResult,
} from "./paper-search-index";

export type DashboardTab = "starred" | "all";

export type DashboardSortKey =
  | "relevance"
  | "priority"
  | "title"
  | "topic"
  | "published";

export type DashboardSortDirection = "asc" | "desc";

export interface DashboardQuery {
  tab?: DashboardTab;
  search?: string;
  topics?: string[];
  statuses?: PaperStatus[];
  priorities?: PaperPriority[];
  dateFrom?: string;
  dateTo?: string;
  detailSummary?: boolean;
  sort?: {
    key: DashboardSortKey;
    direction?: DashboardSortDirection;
  };
}

export interface DashboardDiscoveryRepresentative {
  paperKey: string;
  title: string;
  evidenceDepth: PersonalizedDiscoveryRepresentative["evidenceDepth"];
}

export interface DashboardDiscoveryDirection {
  id: string;
  name: string;
  representatives: DashboardDiscoveryRepresentative[];
}

/** Durable metadata from the latest applicable committed report occurrence. */
export interface DashboardOccurrenceProvenance {
  reportPath: string;
  reportDate: string;
  source: "manual" | "library" | "both";
  manualTopics: Array<{ tag: string; name?: string }>;
  directions: DashboardDiscoveryDirection[];
  evidenceDepth?: PersonalizedDiscoveryRepresentative["evidenceDepth"];
}

/**
 * One representative prior paper of a Dashboard novelty comparison basis.
 * Only the persisted paperKey is carried: titles are never persisted in the
 * index (the marker/index payload stays minimal), so display titles are never
 * invented from other sources.
 */
export interface DashboardPersonalNoveltyBasisPaper {
  paperKey: string;
}

/**
 * Durable validated personal-novelty metadata from the latest applicable
 * committed report occurrence, selected with the same deterministic
 * report-date/path rules as DashboardOccurrenceProvenance.
 */
export interface DashboardPersonalNovelty {
  reportPath: string;
  reportDate: string;
  differenceType: PersonalNoveltyDifferenceType;
  comparisonBasis: DashboardPersonalNoveltyBasisPaper[];
  evidenceDepth: PersonalNovelty["evidenceDepth"];
  explanation: string;
}

export interface DashboardRow {
  entry: PaperIndexEntry;
  arxivId: string;
  title: string;
  authors: string;
  topic: string;
  firstSeen: string;
  hasDetailSummary: boolean;
  occurrenceProvenance?: DashboardOccurrenceProvenance;
  personalNovelty?: DashboardPersonalNovelty;
  relevanceScore?: number;
  matchReasons?: PaperSearchReason[];
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

export interface DashboardQueryOptions {
  now?: Date;
  detailSummaryIds?: ReadonlySet<string>;
  searchIndex?: PaperSearchIndex | null;
  topics?: readonly Pick<Topic, "tag" | "name">[];
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
    };

export interface DashboardPatch {
  arxivId: string;
  status?: PaperStatus;
  priority?: PaperPriority;
}

export interface DashboardActionPlan {
  patches: DashboardPatch[];
  missingIds: string[];
  requiresConfirmation: boolean;
}

const DEFAULT_TAB: DashboardTab = "starred";
const PRIORITY_ORDER: Record<PaperPriority, number> = {
  high: 0,
  normal: 1,
  low: 2,
};

/**
 * Localized difference-type labels for the Dashboard novelty block, matching
 * the daily-report novelty line labels exactly (en: "new method"; zh: "新方法").
 * The Dashboard UI itself renders in English; the zh map exists for host-neutral
 * label coverage and is verified against the difference-type enum in tests.
 */
export const DASHBOARD_PERSONAL_NOVELTY_DIFFERENCE_TYPE_LABELS: Record<
  SummaryLanguage,
  Record<PersonalNoveltyDifferenceType, string>
> = {
  zh: {
    "new-task": "新任务",
    "new-method": "新方法",
    "new-dataset": "新数据集",
    "new-experiment": "新实验",
    "efficiency-result": "效率结果",
    "counter-evidence": "反例证据",
  },
  en: {
    "new-task": "new task",
    "new-method": "new method",
    "new-dataset": "new dataset",
    "new-experiment": "new experiment",
    "efficiency-result": "efficiency result",
    "counter-evidence": "counter-evidence",
  },
};

/** Localized Dashboard label for one validated novelty difference type. */
export function personalNoveltyDifferenceTypeLabel(
  differenceType: PersonalNoveltyDifferenceType,
  language: SummaryLanguage = "en",
): string {
  return DASHBOARD_PERSONAL_NOVELTY_DIFFERENCE_TYPE_LABELS[language][differenceType];
}

export function queryDashboard(
  entries: PaperIndexEntry[],
  query: DashboardQuery = {},
  opts: DashboardQueryOptions = {},
): DashboardResult {
  const search = query.search?.trim() ?? "";
  const searchResults = search ? runIndexedSearch(entries, search, opts.searchIndex) : null;
  const byId = searchResults
    ? new Map(searchResults.map((result) => [result.entry.arxivId, result]))
    : null;
  const filtered = entries.filter((entry) =>
    matchesDashboardQuery(entry, query, opts, byId),
  );
  const rows = sortRows(
    filtered.map((entry) => toDashboardRow(entry, opts, byId?.get(entry.arxivId))),
    query.sort,
    Boolean(search),
  );
  return {
    rows,
    stats: buildDashboardStats(rows.map((row) => row.entry), opts),
    tabCounts: buildDashboardTabCounts(entries),
  };
}

export function matchesDashboardQuery(
  entry: PaperIndexEntry,
  query: DashboardQuery = {},
  opts: DashboardQueryOptions = {},
  indexedResults?: ReadonlyMap<string, PaperSearchResult> | null,
): boolean {
  if (!matchesDashboardTab(entry, query.tab ?? DEFAULT_TAB)) return false;
  if (
    indexedResults
      ? !indexedResults.has(entry.arxivId)
      : !matchesSearch(entry, query.search ?? "")
  ) {
    return false;
  }
  if (!matchesTopic(entry, query.topics ?? [])) return false;
  if (!matchesAny(entry.status, query.statuses ?? [])) return false;
  if (!matchesAny(entry.priority, query.priorities ?? [])) return false;
  if (!matchesDateRange(entry, query)) return false;
  if (
    query.detailSummary != null &&
    hasDetailSummary(entry, opts) !== query.detailSummary
  ) {
    return false;
  }
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
      patches.push({ arxivId, status: action.status });
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
      patches.push(patch);
      continue;
    }
  }

  return {
    patches,
    missingIds,
    requiresConfirmation: false,
  };
}

function toDashboardRow(
  entry: PaperIndexEntry,
  opts: DashboardQueryOptions,
  searchResult?: PaperSearchResult,
): DashboardRow {
  return {
    entry,
    arxivId: entry.arxivId,
    title: entry.title,
    authors: entry.authors.join(", "),
    topic: displayTopic(entry),
    firstSeen: firstSeenDate(entry),
    hasDetailSummary: hasDetailSummary(entry, opts),
    ...projectDashboardOccurrenceProvenance(entry, opts.topics),
    ...projectDashboardOccurrenceNovelty(entry),
    ...(searchResult
      ? { relevanceScore: searchResult.score, matchReasons: searchResult.reasons }
      : {}),
  };
}

function hasDetailSummary(
  entry: PaperIndexEntry,
  opts: DashboardQueryOptions,
): boolean {
  return Boolean(opts.detailSummaryIds?.has(entry.arxivId));
}

/**
 * Selects the latest provenance-bearing committed occurrence by the ISO date in
 * its daily-report filename, then by normalized report path as a deterministic
 * repeated-report tie-breaker, restricted to reports present in
 * entry.dailyReports. Legacy entries and reports without provenance
 * intentionally project no Dashboard metadata.
 */
export function projectDashboardOccurrenceProvenance(
  entry: Pick<PaperIndexEntry, "dailyReports" | "discoveryProvenanceByReport">,
  topics: readonly Pick<Topic, "tag" | "name">[] = [],
): Partial<Pick<DashboardRow, "occurrenceProvenance">> {
  const committedReports = new Set((entry.dailyReports ?? []).map(normalizeReportPath));
  const byPath = new Map(
    Object.entries(entry.discoveryProvenanceByReport ?? {}).map(([rawReportPath, provenance]) => [
      normalizeReportPath(rawReportPath),
      provenance,
    ]),
  );
  const selected = selectLatestCommittedReport(byPath.keys(), committedReports);
  const provenance = selected ? byPath.get(selected.reportPath) : undefined;
  if (!selected || !provenance) return {};

  const topicNames = new Map(topics.map((topic) => [topic.tag, topic.name]));
  const manualTopics = provenance.manualTopicTags.map((tag) => ({
    tag,
    ...(topicNames.get(tag) ? { name: topicNames.get(tag) } : {}),
  }));
  const directions = provenance.directions.map(projectDirection);
  const hasManual = manualTopics.length > 0;
  const hasLibrary = directions.length > 0;
  if (!hasManual && !hasLibrary) return {};

  return {
    occurrenceProvenance: {
      reportPath: selected.reportPath,
      reportDate: selected.reportDate,
      source: hasManual && hasLibrary ? "both" : hasManual ? "manual" : "library",
      manualTopics,
      directions,
      ...(hasLibrary ? { evidenceDepth: "metadata-and-abstract" as const } : {}),
    },
  };
}

/**
 * Projects the validated personal novelty of the latest committed occurrence
 * onto a Dashboard row using exactly the same deterministic report-date/path
 * selection as projectDashboardOccurrenceProvenance, over noveltyByReport ∩
 * dailyReports. Only persisted minimal payload is projected: the comparison
 * basis carries paperKeys (titles are never persisted in the index and are
 * never invented), and the exact validated evidence depth and bounded
 * explanation pass through. Legacy/no-novelty rows project no metadata.
 */
export function projectDashboardOccurrenceNovelty(
  entry: Pick<PaperIndexEntry, "dailyReports" | "noveltyByReport">,
): Partial<Pick<DashboardRow, "personalNovelty">> {
  const committedReports = new Set((entry.dailyReports ?? []).map(normalizeReportPath));
  const byPath = new Map(
    Object.entries(entry.noveltyByReport ?? {}).map(([rawReportPath, novelty]) => [
      normalizeReportPath(rawReportPath),
      novelty,
    ]),
  );
  const selected = selectLatestCommittedReport(byPath.keys(), committedReports);
  const novelty = selected ? byPath.get(selected.reportPath) : undefined;
  if (!selected || !novelty) return {};

  return {
    personalNovelty: {
      reportPath: selected.reportPath,
      reportDate: selected.reportDate,
      differenceType: novelty.differenceType,
      comparisonBasis: novelty.comparisonBasis.map((paperKey) => ({ paperKey })),
      evidenceDepth: novelty.evidenceDepth,
      explanation: novelty.explanation,
    },
  };
}

/**
 * Deterministic latest committed occurrence selection shared by the provenance
 * and novelty projections: newest ISO report date in the daily-report filename
 * first, then normalized report path as a same-date repeated-report
 * tie-breaker, restricted to committed reports (dailyReports ∩ candidate
 * paths). Invalid or uncommitted report paths never project metadata.
 */
function selectLatestCommittedReport(
  rawReportPaths: Iterable<string>,
  committedReports: ReadonlySet<string>,
): { reportPath: string; reportDate: string } | null {
  const candidates = [...rawReportPaths]
    .map((rawReportPath) => ({
      reportPath: normalizeReportPath(rawReportPath),
      reportDate: dailyReportDate(rawReportPath),
    }))
    .filter((candidate) =>
      candidate.reportDate && committedReports.has(candidate.reportPath)
    )
    .sort((a, b) =>
      b.reportDate!.localeCompare(a.reportDate!) ||
      b.reportPath.localeCompare(a.reportPath)
    );
  const selected = candidates[0];
  return selected?.reportDate
    ? { reportPath: selected.reportPath, reportDate: selected.reportDate }
    : null;
}

function projectDirection(
  direction: DiscoveryDirectionProvenance,
): DashboardDiscoveryDirection {
  return {
    id: direction.id,
    name: direction.name,
    representatives: direction.representatives.map((representative) => ({
      paperKey: representative.paperKey,
      title: representative.title,
      evidenceDepth: representative.evidenceDepth,
    })),
  };
}

function normalizeReportPath(path: string): string {
  return path.trim().replace(/\\/g, "/").replace(/\/{2,}/g, "/").replace(/^\.\//, "");
}

function dailyReportDate(path: string): string | null {
  const filename = normalizeReportPath(path).split("/").pop() ?? "";
  return /^(\d{4}-\d{2}-\d{2})\.md$/.exec(filename)?.[1] ?? null;
}

function sortRows(
  rows: DashboardRow[],
  sort: DashboardQuery["sort"],
  hasSearch: boolean,
): DashboardRow[] {
  const key = sort?.key ?? (hasSearch ? "relevance" : "priority");
  const direction = sort?.direction ?? (key === "relevance" ? "desc" : "asc");
  const dir = direction === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const primary = compareBySortKey(a, b, key);
    if (primary !== 0) return primary * dir;
    if (key === "relevance") {
      return a.arxivId < b.arxivId ? -1 : a.arxivId > b.arxivId ? 1 : 0;
    }
    return compareDefault(a, b);
  });
}

function compareBySortKey(
  a: DashboardRow,
  b: DashboardRow,
  key: DashboardSortKey,
): number {
  switch (key) {
    case "relevance":
      return (a.relevanceScore ?? 0) - (b.relevanceScore ?? 0);
    case "priority":
      return PRIORITY_ORDER[a.entry.priority] - PRIORITY_ORDER[b.entry.priority];
    case "title":
      return a.title.localeCompare(b.title);
    case "topic":
      return a.topic.localeCompare(b.topic);
    case "published":
      return a.entry.published.localeCompare(b.entry.published);
  }
  return 0;
}

function compareDefault(a: DashboardRow, b: DashboardRow): number {
  const priority = PRIORITY_ORDER[a.entry.priority] - PRIORITY_ORDER[b.entry.priority];
  if (priority !== 0) return priority;
  const published = b.entry.published.localeCompare(a.entry.published);
  if (published !== 0) return published;
  return a.title.localeCompare(b.title);
}

function runIndexedSearch(
  entries: PaperIndexEntry[],
  search: string,
  cachedIndex?: PaperSearchIndex | null,
): PaperSearchResult[] | null {
  if (cachedIndex === null) return null;
  try {
    return (cachedIndex ?? new PaperSearchIndex(entries)).search(search);
  } catch {
    // Host integrations can still render using the legacy substring matcher if
    // derived-index construction or querying fails for unexpected input.
    return null;
  }
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
    entry.abstract,
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
