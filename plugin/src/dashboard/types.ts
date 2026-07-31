/** Shared dashboard page and day types (moved out of view.ts). */
export interface DashboardPage<T> {
  rows: T[];
  total: number;
  totalPages: number;
  currentPage: number;
  start: number;
  end: number;
  pageSize: number;
}

export interface DailyReportDay {
  date: string;
  path: string;
  papers: number;
  starred: number;
}
