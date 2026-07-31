import type {
  DashboardSortKey,
  DashboardTab,
} from "@arxiv-daily/core";

export const ARXIV_DAILY_DASHBOARD_VIEW = "arxiv-daily-dashboard";
export const RECENT_DATES_FOREGROUND_TIMEOUT_MS = 3000;
export const DASHBOARD_SEARCH_DEBOUNCE_MS = 250;

export const DASHBOARD_TABS: Array<{ id: DashboardTab; label: string }> = [
  { id: "all", label: "All" },
  { id: "starred", label: "Starred" },
];

export const PAGE_SIZE_OPTIONS: Array<{ value: number; label: string }> = [
  { value: 20, label: "20" },
  { value: 50, label: "50" },
  { value: 100, label: "100" },
  { value: Infinity, label: "All" },
];

export const SORT_LABELS: Record<DashboardSortKey, string> = {
  relevance: "Relevance",
  priority: "Starred first",
  published: "Published",
  topic: "Topic",
  title: "Title",
};

export const DEFAULT_SORT_KEY: DashboardSortKey = "priority";
