import { portablePathCollisionKey } from "@arxiv-daily/core";
import type { DashboardMarkdownFile } from "@arxiv-daily/core";

/** Vault path helpers shared by dashboard modules (moved out of view.ts). */
export function normalizeVaultPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
}

export function filterDashboardMarkdownFiles<T extends DashboardMarkdownFile>(
  files: T[],
  dailyDir: string,
  papersDir: string,
): T[] {
  const normalizedDailyDir = pathKey(dailyDir);
  const normalizedPapersDir = pathKey(papersDir);
  return files.filter((file) => {
    const path = pathKey(file.path);
    return (
      path.startsWith(`${normalizedDailyDir}/`) ||
      path.startsWith(`${normalizedPapersDir}/`)
    );
  });
}

export function dashboardHistoryPathSet(
  files: DashboardMarkdownFile[],
  dailyDir: string,
  papersDir: string,
): Set<string> {
  const normalizedDailyDir = pathKey(dailyDir);
  const normalizedPapersDir = pathKey(papersDir);
  return new Set(
    files
      .map((file) => normalizeVaultPath(file.path))
      .filter(
        (path) =>
          pathKey(path).startsWith(`${normalizedDailyDir}/`) ||
          isDirectChildMarkdown(path, normalizedPapersDir),
      ),
  );
}

export function shouldSkipDashboardHistorySync(
  previousHistoryPaths: ReadonlySet<string> | null,
  currentHistoryPaths: ReadonlySet<string>,
  currentEntryCount: number,
): boolean {
  if (!previousHistoryPaths || currentEntryCount === 0) return false;
  if (previousHistoryPaths.size !== currentHistoryPaths.size) return false;
  for (const path of currentHistoryPaths) {
    if (!previousHistoryPaths.has(path)) return false;
  }
  return true;
}

function isDirectChildMarkdown(path: string, dir: string): boolean {
  path = pathKey(path);
  const prefix = `${pathKey(dir)}/`;
  if (!path.startsWith(prefix) || !/\.md$/i.test(path)) return false;
  return !path.slice(prefix.length).includes("/");
}

export function markdownPathFromLeaf(leaf: unknown): string | null {
  const candidate = leaf as {
    getViewState?: () => { state?: { file?: unknown } };
    view?: { file?: { path?: unknown } };
  };
  const stateFile = candidate.getViewState?.().state?.file;
  if (typeof stateFile === "string") return stateFile;
  const viewPath = candidate.view?.file?.path;
  return typeof viewPath === "string" ? viewPath : null;
}

export function dailyDateFromPath(path: string, dailyDir: string): string | null {
  const normalized = pathKey(path);
  const prefix = `${pathKey(dailyDir)}/`;
  if (!normalized.startsWith(prefix)) return null;
  const rest = normalized.slice(prefix.length);
  const match = /^(\d{4}-\d{2}-\d{2})\.md$/i.exec(rest);
  return match?.[1] ?? null;
}

function pathKey(path: string): string {
  return portablePathCollisionKey(normalizeVaultPath(path));
}
