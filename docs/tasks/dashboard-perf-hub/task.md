# Codex Task: dashboard-perf-hub

task_id: dashboard-perf-hub
target_project: /home/tiandc/Documents/code/arxiv-daily
task_kind: implementation
mode: semi-auto
sandbox: workspace-write
provider: bnu
artifact_policy: keep-report-only
source: claude-code

## Goal

Five improvements to the arXiv Daily dashboard: (1) tabbed hub modal replacing 3 separate modals, (2) skip vault-wide file scan, (3) pagination (20/page), (4) in-memory daily file parse cache, (5) log buffer increase.

## Context

Working directory: /home/tiandc/Documents/code/arxiv-daily
Source: plugin/src/

Key files:
- plugin/src/dashboard/view.ts — dashboard + existing LogModal
- plugin/src/dashboard/history-sync.ts — syncDashboardHistory & helpers
- plugin/src/services/logger.ts — Logger buffer
- plugin/src/services/run-history.ts — RunHistoryStore
- plugin/src/services/diagnostics.ts — diagnostics report

Preserve all uncommitted changes. Do NOT run git add/commit.

## Tasks

### A: Three-tab Hub Modal (replaces LogModal, RunHistoryModal, DiagnosticsModal)

File: plugin/src/dashboard/view.ts

Delete the old `LogModal` class (around line 2318). Replace with a new `HubModal` class that has three tabs:

```
┌─────────────────────────────────┐
│  arXiv Daily — Logs & History    │
│  [ Logs ] [ Run History ] [ Diag ] │  ← tabs
├─────────────────────────────────┤
│  (content area)                 │
│                                 │
│  [Refresh] [Clear] [Copy] [Close] │  ← common footer
└─────────────────────────────────┘
```

Implementation details:
- Use regular DOM buttons for tabs (not Obsidian TabComponent, which is more complex). Style `.hub-modal-tab-active` to highlight.
- Clicking a tab hides the other content panels and shows the selected one.
- The **Logs** tab: same as current LogModal — show `logger.getBuffer()`, Refresh/Clear/Copy buttons. Keep `userSelect: text`.
- The **Run History** tab: load records from `this.plugin.runHistoryStore.readLatest(100)`, show them formatted (use `formatRunHistoryRecords` from `run-history.ts`). Add a Refresh button.
- The **Diagnostics** tab: build diagnostics from `buildDiagnosticsReport(...)` (imported from `./diagnostics`), show as text. Add a Refresh button.
- The **Close** button closes the modal.
- Keep the Copy button functional for all tabs (copy the visible tab's content).
- Remove the old `LogModal` class.

### B: Only scan dailyDir and papersDir, not entire vault

File: plugin/src/dashboard/view.ts, in `reloadIndex()`

Current code:
```ts
const markdownFiles = this.plugin.app.vault.getMarkdownFiles();
```

Replace with filtering to only the two directories that matter:
```ts
const allFiles = this.plugin.app.vault.getMarkdownFiles();
const dailyDir = normalizeVaultPath(this.plugin.settings.output.dailyDir);
const papersDir = normalizeVaultPath(this.plugin.settings.output.papersDir);
const markdownFiles = allFiles.filter((f) => {
  const p = normalizeVaultPath(f.path);
  return p.startsWith(dailyDir + "/") || p.startsWith(papersDir + "/");
});
```

But wait — `normalizeVaultPath` already exists in `view.ts`? Check — if not, define it locally. Actually looking at the codebase, `normalizeVaultPath` is in `history-sync.ts`, not in `view.ts`. So you need to either import it or do inline:

```ts
const norm = (p: string) => p.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
const dailyDir = norm(this.plugin.settings.output.dailyDir);
const papersDir = norm(this.plugin.settings.output.papersDir);
const markdownFiles = this.plugin.app.vault.getMarkdownFiles().filter((f) => {
  const p = norm(f.path);
  return p.startsWith(dailyDir + "/") || p.startsWith(papersDir + "/");
});
```

This alone eliminates scanning the entire vault. Add a logger.info line showing the reduction:
```ts
this.plugin.logger.info(
  `dashboard: scanning ${markdownFiles.length}/${allFiles.length} files (${dailyDir}, ${papersDir})`,
);
```

### C: Pagination (20 rows per page)

File: plugin/src/dashboard/view.ts

The dashboard table renders from `this.entries` (all papers). Add pagination:

1. Add properties to ArxivDailyDashboardView:
   - `private pageSize = 20`
   - `private currentPage = 0`
   
2. Before rendering the table, slice entries:
   ```ts
   const totalPages = Math.ceil(this.entries.length / this.pageSize) || 1;
   this.currentPage = Math.min(this.currentPage, totalPages - 1);
   const pageEntries = this.entries.slice(
     this.currentPage * this.pageSize,
     (this.currentPage + 1) * this.pageSize,
   );
   ```

3. Render pageEntries instead of this.entries in the table.

4. Add pagination controls below the table (find where the table is rendered). Look for existing "table" rendering code. It might be in `renderContent()` or similar. The pagination should look like:
   ```
   ◀ Page 3 / 15 ▶   Show 20 per page
   ```
   With prev/next buttons that call `this.setPage(n)`.

5. Add a `setPage(n: number)` method that updates `this.currentPage` and calls `this.renderContent()` or the equivalent re-render method.

6. `loadDetailSummaries` and `loadDailyReports` should still work on `this.entries` (full set), not just the page — these are background data loads, not rendering. Only the table render needs slicing.

7. Update the paper count label to show `"Showing 1-20 of 315 papers"` format.

### D: Daily file parse cache (dedup reloadIndex parsing)

File: plugin/src/dashboard/view.ts + plugin/src/dashboard/history-sync.ts

The problem: every `reloadIndex()` re-reads and re-parses all daily markdown files via `syncDashboardHistory()`, even if they haven't changed. We can cache the parsed results in memory.

Approach:
1. Add a property to ArxivDailyDashboardView:
   ```ts
   private parsedDailyCache: { path: string; hash: string; candidates: any[] }[] | null = null;
   ```
   
2. Replace the `syncDashboardHistory` call with a wrapper that:
   - Generates a quick content hash for each daily file (simple approach: compute a hash from path + file size + file mtime if available, or just track which paths we've already parsed)
   
   Actually simpler: **use `import { createHash } from "node:crypto"` won't work in browser/obsidian context**. Instead, just track the set of already-parsed file paths and their content length:

3. The simplest correct approach: in `reloadIndex`, before calling `syncDashboardHistory`, compute which daily files are new or changed since last cache:
   - Store `this._dailyFileCache: { path: string; length: number }[]`
   - On next reload, get daily files, compare path + length with cache
   - Only unchanged files can use cached results

   But `syncDashboardHistory` doesn't accept pre-parsed data, so this requires either:
   - (a) A caching wrapper around syncDashboardHistory
   - (b) Or just skip the cache if it's too invasive

   Let's go with a pragmatic approach: **move the daily file reading to happen once, store results in a member variable, and skip re-read if files haven't changed**:

   ```ts
   private cachedSyncResult: {
     dailyReportPaths: Set<string>;
     parsedReports: Set<string>;
     paperIdsByReport: Map<string, Set<string>>;
     dailyCandidates: DailyCandidate[];
   } | null = null;
   ```

   Actually, this is getting complex. Let me simplify: the cheapest and most effective optimization is the file count filter (task B). For the re-parse cache, do something lightweight:

   After `syncDashboardHistory` runs, store the list of known daily files with their sizes. Next time, check if the list changed; if not, skip the sync entirely if the paper index looks current:

   ```ts
   private lastSyncFileMap: Map<string, number> | null = null;
   
   // In reloadIndex, before calling syncDashboardHistory:
   const dailyFiles = markdownFiles.filter(f => f.path.startsWith(dailyDir + "/"));
   const fileMap = new Map(dailyFiles.map(f => [f.path, f.stat?.size ?? 0]));
   const unchanged = this.lastSyncFileMap && 
     fileMap.size === this.lastSyncFileMap.size &&
     [...fileMap].every(([k, v]) => this.lastSyncFileMap!.get(k) === v);
   
   if (unchanged && this.entries.length > 0) {
     // Skip full sync, just re-render
     this.renderContent();
     return;
   }
   this.lastSyncFileMap = fileMap;
   ```

   But `DashboardMarkdownFile` doesn't have `stat`. So use a simpler heuristic: just store the set of file paths. If the set is the same as last time, skip sync.

   ```ts
   // In reloadIndex, use this approach:
   private lastSyncedDailyPaths: Set<string> | null = null;
   ```

   If `lastSyncedDailyPaths` matches current daily file paths AND `this.entries.length > 0`, skip `syncDashboardHistory` and all data loading. Just re-render the existing data.

   This is the simplest and most effective cache. Add it.

### E: Log buffer 5000

File: plugin/src/services/logger.ts

Change `MAX_BUFFER_SIZE` from 1000 to 5000.

## Verification

```bash
cd /home/tiandc/Documents/code/arxiv-daily/plugin && npx tsc --noEmit
cd /home/tiandc/Documents/code/arxiv-daily/plugin && npm run build
```

Both must succeed.

## Report

Write to `docs/tasks/dashboard-perf-hub/codex-report.md`:
- Status for each section (A-E)
- Files changed
- Verification results
- Suggested commit message
