import type { DashboardPage } from "./types";

export function showingText(page: DashboardPage<unknown>): string {
  if (page.total === 0) return "Showing 0 of 0 papers";
  if (!isFinite(page.pageSize)) return `Showing all ${page.total} papers`;
  return `Showing ${page.start}-${page.end} of ${page.total} papers`;
}

export function paginateDashboardRows<T>(
  rows: T[],
  currentPage: number,
  pageSize: number,
): DashboardPage<T> {
  const total = rows.length;
  // Infinity means "show all" — bypass arithmetic to avoid 0 * Infinity = NaN.
  if (!isFinite(pageSize)) {
    return {
      rows: rows.slice(),
      total,
      totalPages: 1,
      currentPage: 0,
      start: total === 0 ? 0 : 1,
      end: total,
      pageSize,
    };
  }
  const safePageSize = Math.max(1, Math.floor(pageSize));
  const totalPages = Math.ceil(total / safePageSize) || 1;
  const clampedPage = Math.max(
    0,
    Math.min(Math.floor(currentPage), totalPages - 1),
  );
  const offset = clampedPage * safePageSize;
  const pageRows = rows.slice(offset, offset + safePageSize);
  return {
    rows: pageRows,
    total,
    totalPages,
    currentPage: clampedPage,
    start: pageRows.length === 0 ? 0 : offset + 1,
    end: offset + pageRows.length,
    pageSize: safePageSize,
  };
}
