# arXiv Daily Reading Dashboard — Design

**Date:** 2026-06-12
**Scope:** `plugin/` (Obsidian plugin), paper review and management GUI.

## Background

The plugin has moved beyond a daily Markdown generator. It now keeps a
paper-level index with status, priority, seen dates, daily report links and
optional paper notes. Daily reports are the right place for same-day triage:
the user scans new papers and marks a few as `关注` or `重点`.

That does not solve later review. After a few weeks, the user should not need
to open daily reports one by one to find highlighted papers, saved papers, or
papers still waiting to be read.

The next UI layer should therefore be an Obsidian-native Reading Dashboard:
a searchable, filterable, batch-editable view backed by `papers.json`.

This is intentionally not a standalone desktop app. A standalone app would
need to reimplement Markdown reading/editing, vault navigation, links, sync
boundaries and conflict behavior. The first GUI should reuse Obsidian for
Markdown and provide only the paper-management surface that Obsidian lacks.

## Product Decision

Build a **Reading Dashboard inside the Obsidian plugin** before considering a
standalone GUI app.

The workflow split is:

- Daily report: today's discovery and quick positive selection.
- Reading Dashboard: cross-date search, review, status changes and summaries.
- Paper note: long-form reading notes, still edited in Obsidian Markdown.

## Requirements

1. The dashboard is opened from a command and from the ribbon menu.
2. The dashboard is an Obsidian custom view, not a generated `inbox.md` page.
3. The dashboard reads from `PaperIndexStore` and writes status/priority
   updates back to `papers.json`.
4. The dashboard must list selected papers without opening any daily report.
5. The dashboard must provide tabs:
   - `关注`: `status === "to_read"` and `priority !== "high"`.
   - `重点`: `priority === "high"`.
   - `正在读`: `status === "reading"`.
   - `已收藏`: `status === "saved"`.
   - `已读`: `status === "read"`.
   - `全部`: all non-ignored papers.
   - `忽略`: `status === "ignored"`.
6. The dashboard must provide filters:
   - topic / primary topic.
   - status.
   - priority.
   - date range, using first seen or published date.
   - has paper note.
   - detail paper.
   - missing Zotero key / citation key.
7. The dashboard must provide text search over:
   - arXiv ID.
   - title.
   - authors.
   - topic tag.
   - status / priority.
8. The dashboard must show a dense table with stable columns:
   - selection checkbox.
   - priority.
   - status.
   - title.
   - topic.
   - first seen / published date.
   - note.
   - arXiv / PDF actions.
9. Row actions must support:
   - set status.
   - set priority.
   - create paper note.
   - open paper note.
   - open daily report.
   - open arXiv / PDF URL.
10. Batch actions must support:
    - mark selected as ignored.
    - mark selected as read.
    - mark selected as saved.
    - set selected priority.
    - create lightweight notes for selected papers after confirmation.
11. A summary area must show:
    - total rows in current filter.
    - counts by status.
    - counts by priority.
    - counts by topic.
    - this-week new / watched / highlighted / saved counts.
    - saved papers missing Zotero key / citation key.
12. The dashboard must not directly edit paper note bodies. It opens notes in
    Obsidian's Markdown editor.

## Non-goals

- Standalone desktop app.
- Reimplementing Markdown editor behavior.
- Replacing Obsidian search, backlinks or file explorer.
- External sync beyond the user's vault sync.
- Direct Zotero API integration in this release.
- A database. In-memory filtering over `papers.json` is sufficient initially.
- LLM-generated weekly summaries. The first version is deterministic counts
  and filtered lists.

## UI Shape

```text
┌───────────────────────────────────────────────────────────────┐
│ arXiv Daily                                                   │
│ [关注] [重点] [正在读] [已收藏] [已读] [全部] [忽略]           │
│ Search...        Topic ▾  Status ▾  Priority ▾  Date range    │
├───────────────────────────────────────────────────────────────┤
│ 42 papers · high 8 · saved 5 · missing Zotero 3               │
├───────────────────────────────────────────────────────────────┤
│ □ !  to_read   2606.12345 Title...   photo-z   2026-06-12    │
│ □    saved     2606.12346 Title...   clusters  2026-06-11    │
│ ...                                                           │
├───────────────────────────────────────────────────────────────┤
│ Selected: 3   [Mark read] [Save] [Ignore] [Set priority ▾]    │
└───────────────────────────────────────────────────────────────┘
```

The first implementation can be a single-pane table. A right-side preview is
useful later, but it should not block v0.3.0.

## Data Flow

1. `DashboardView.onload` calls `plugin.buildPaperIndex().load()`.
2. Raw entries are converted to view rows by pure query helpers.
3. Search, tab filtering, sorting and aggregations run in memory.
4. Single-row and batch actions call `PaperIndexStore` update methods.
5. After saving, the dashboard reloads the index or applies a local patch.
6. Opening a note or daily report uses Obsidian `workspace.openLinkText`.

## Query Model

Add pure helper functions so the UI remains thin:

```ts
interface PaperQuery {
  tab: DashboardTab;
  search: string;
  topics: string[];
  statuses: PaperStatus[];
  priorities: PaperPriority[];
  dateFrom?: string;
  dateTo?: string;
  hasNote?: boolean;
  detail?: boolean;
  missingZotero?: boolean;
  missingCitation?: boolean;
}

interface DashboardSummary {
  total: number;
  byStatus: Record<PaperStatus, number>;
  byPriority: Record<PaperPriority, number>;
  byTopic: Record<string, number>;
  thisWeek: {
    new: number;
    watched: number;
    highlighted: number;
    saved: number;
  };
  missingZotero: number;
  missingCitation: number;
}
```

These helpers should live outside the DOM view and be covered by unit tests.

## State Update Rules

- `status` and `priority` are user-controlled fields.
- Dashboard actions may update them directly.
- Creating a note sets `paperPath` only after the file exists.
- Marking as `saved` should ensure a lightweight note exists, matching the
  current command behavior.
- Batch note creation must ask for confirmation because it can create many
  files.
- Dashboard must not overwrite manually edited note content.

## Acceptance Criteria

- Command palette opens the dashboard.
- Ribbon menu opens the dashboard.
- Dashboard lists all `to_read` papers under `关注`.
- Dashboard lists all high-priority papers under `重点`, regardless of status.
- Search filters by arXiv ID, title and author.
- Topic/status/priority/date filters can be combined.
- Summary counts reflect the current filtered result set.
- Single-row status and priority changes persist to `papers.json`.
- Batch status and priority changes persist to `papers.json`.
- Creating/opening a paper note works from the dashboard.
- Opening the daily report works from the dashboard when `dailyReports` is set.
- Empty, loading and malformed-index states are visible and actionable.
- Unit tests cover query helpers and batch action semantics.

## Risks

- The dashboard depends on the paper index path being stable. The current
  direction is `arxiv-daily/.index/papers.json` with legacy reads from
  `arxiv-daily/index/papers.json`; verify code, tests and docs agree before
  implementing the view.
- Large paper indexes could make full re-render slow. Start simple; add
  pagination or virtualization only when real data requires it.
- Obsidian DOM APIs make UI tests expensive. Keep core query/action logic pure
  and test that thoroughly.
