# arXiv Daily Reading Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Track progress by updating the checkboxes below.

**Goal:** Add an Obsidian-native GUI dashboard for cross-date paper review: search, filters, summaries, single-row actions and batch status/priority updates, all backed by `papers.json`.

**Product decision:** Build a plugin custom view first. Do not convert the project into a standalone desktop app in this release.

**Architecture:** Keep the UI thin. Put query/filter/summary/action semantics into testable helpers and stores. The custom view renders those results and delegates Markdown reading/editing back to Obsidian.

**Tech Stack:** TypeScript, Obsidian plugin API, existing `PaperIndexStore`, vitest. DOM-level tests should be minimal; pure query/action helpers need full unit coverage.

**Spec:** `docs/superpowers/specs/2026-06-12-reading-dashboard-design.md`

---

## Task 0: Confirm Paper Index Path Consistency

**Why:** Dashboard should build on one stable storage contract. The intended path is `arxiv-daily/.index/papers.json`, with legacy reads from `arxiv-daily/index/papers.json`.

**Files:**
- Modify: `plugin/src/services/paper-index.ts`
- Modify: `plugin/tests/paper-index.test.ts`
- Optional docs: `README.md`, `plugin/README.md`

- [ ] Confirm `derivePaperInboxPaths()` returns:
  - active path: `arxiv-daily/.index/papers.json`.
  - legacy path: `arxiv-daily/index/papers.json`.
- [ ] Confirm `load()` reads the active path first and falls back to legacy.
- [ ] Confirm `save()` writes the active path and handles legacy cleanup intentionally.
- [ ] Confirm tests cover fresh active index and legacy migration.
- [ ] Update docs so README, plugin README, `PLAN.md`, tests and code agree.

**Acceptance:**
- Fresh stores use `arxiv-daily/.index/papers.json`.
- Existing users with the legacy path do not lose state.
- Diagnostics report the active path and legacy path if applicable.

---

## Task 1: Add Dashboard Query Helpers

**Files:**
- Create: `plugin/src/services/dashboard-query.ts`
- Create: `plugin/tests/dashboard-query.test.ts`

- [ ] Define types:
  - `DashboardTab`
  - `PaperQuery`
  - `DashboardSummary`
  - `DashboardRow`
- [ ] Implement pure helpers:
  - `matchesDashboardTab(entry, tab)`
  - `filterDashboardEntries(entries, query)`
  - `sortDashboardEntries(entries, sort)`
  - `summarizeDashboardEntries(entries, now)`
- [ ] Search fields:
  - `arxivId`
  - `title`
  - `authors`
  - `primaryTopic`
  - `topics`
  - `status`
  - `priority`
- [ ] Date filtering:
  - use first `seenDates[0]` when present.
  - fallback to `published`.
- [ ] Unit tests:
  - tab semantics for `关注`, `重点`, `正在读`, `已收藏`, `已读`, `全部`, `忽略`.
  - combined filters.
  - search by id/title/author/topic.
  - summary counts.

**Acceptance:**
- Query helpers have no Obsidian imports.
- All dashboard business logic is unit-testable without DOM.

---

## Task 2: Add Bulk Paper Index Actions

**Files:**
- Modify: `plugin/src/services/paper-index.ts`
- Modify: `plugin/tests/paper-index.test.ts`

- [ ] Add methods:
  - `setManyStatus(arxivIds, status)`
  - `setManyPriority(arxivIds, priority)`
  - optional `updateMany(arxivIds, patch)`
- [ ] Ensure missing IDs are reported but do not fail the whole operation.
- [ ] Preserve all fields not explicitly updated.
- [ ] Add tests for:
  - bulk status update.
  - bulk priority update.
  - mixed existing/missing IDs.
  - no duplicate writes in a single batch.

**Acceptance:**
- Dashboard can update selected rows with one load/save cycle.
- User-controlled fields are changed only when explicitly requested.

---

## Task 3: Register Dashboard View and Open Command

**Files:**
- Create: `plugin/src/views/dashboard-view.ts`
- Modify: `plugin/main.ts`
- Modify: `plugin/src/commands.ts`
- Modify: `plugin/tests/__mocks__/obsidian.ts`

- [ ] Add view type constant, for example `ARXIV_DAILY_DASHBOARD_VIEW`.
- [ ] Implement `ReadingDashboardView extends ItemView`.
- [ ] Register the view in `main.ts` during `onload`.
- [ ] Add plugin method `openReadingDashboard()`.
- [ ] Add command:
  - `arXiv Daily: Open reading dashboard`
- [ ] Add ribbon menu item:
  - `Open reading dashboard`
- [ ] Update Obsidian test mock with minimal `ItemView` / workspace surface if needed.

**Acceptance:**
- Command palette opens the dashboard in a workspace leaf.
- Ribbon menu opens the same view.
- Opening the command twice reveals the existing view instead of creating duplicates.

---

## Task 4: Build Read-Only Dashboard UI

**Files:**
- Modify: `plugin/src/views/dashboard-view.ts`
- Optional: `plugin/styles.css`

- [ ] Render loading state while reading `papers.json`.
- [ ] Render malformed-index error state with path and diagnostics hint.
- [ ] Render empty state when there are no indexed papers.
- [ ] Render tabs:
  - `关注`
  - `重点`
  - `正在读`
  - `已收藏`
  - `已读`
  - `全部`
  - `忽略`
- [ ] Render filter controls:
  - search input.
  - topic dropdown.
  - status dropdown.
  - priority dropdown.
  - date range inputs.
  - has-note toggle.
  - detail toggle.
  - missing Zotero / citation toggles.
- [ ] Render summary strip.
- [ ] Render table rows with stable columns.
- [ ] Keep table dense and operational; avoid decorative card layouts.

**Acceptance:**
- User can find all watched/highlighted papers without opening daily files.
- Filters update the table and summary immediately.
- Text fits within table cells and does not overlap at narrow Obsidian pane widths.

---

## Task 5: Add Single-Row Actions

**Files:**
- Modify: `plugin/src/views/dashboard-view.ts`
- Modify if needed: `plugin/src/commands.ts`

- [ ] Add status dropdown or compact action menu per row.
- [ ] Add priority dropdown or compact action menu per row.
- [ ] Add open arXiv button.
- [ ] Add open PDF button.
- [ ] Add open daily report action using first `dailyReports` entry.
- [ ] Add create/open paper note action:
  - if `paperPath` exists, open it.
  - if missing, create lightweight note through existing writer behavior.
  - if status is set to `saved`, ensure note creation stays consistent with command behavior.
- [ ] Refresh row after mutation.

**Acceptance:**
- Single paper status/priority changes persist to `papers.json`.
- Create/open note works without duplicating existing files.
- Daily report opens when available and shows a clear disabled state when unavailable.

---

## Task 6: Add Multi-Select and Batch Actions

**Files:**
- Modify: `plugin/src/views/dashboard-view.ts`
- Modify if needed: `plugin/src/services/paper-index.ts`

- [ ] Add row selection checkboxes.
- [ ] Add select-all for current filtered rows.
- [ ] Add batch action bar visible only when rows are selected.
- [ ] Implement:
  - mark selected ignored.
  - mark selected read.
  - mark selected saved.
  - set selected priority.
  - create lightweight notes for selected papers.
- [ ] Confirm before batch note creation.
- [ ] Clear selection or retain only still-visible rows after filter changes.

**Acceptance:**
- Batch actions update exactly the selected IDs.
- Batch note creation cannot run accidentally.
- Summary counts update after batch mutation.

---

## Task 7: Dashboard Diagnostics and Consistency Checks

**Files:**
- Modify: `plugin/src/services/diagnostics.ts`
- Modify: `plugin/src/commands.ts`
- Modify tests: `plugin/tests/diagnostics.test.ts`

- [ ] Extend paper index diagnostics:
  - active index path.
  - legacy index path if supported.
  - invalid statuses.
  - invalid priorities.
  - missing paper paths.
  - invalid `seenDates`.
  - note frontmatter `arxiv_id` mismatch when feasible.
- [ ] Surface dashboard-relevant counts:
  - total.
  - watched.
  - highlighted.
  - saved without Zotero.
  - saved without citation key.
- [ ] Add tests for new diagnostics fields.

**Acceptance:**
- Diagnostics can explain why dashboard data is missing or inconsistent.

---

## Task 8: Documentation and Manual Verification

**Files:**
- Modify: `README.md`
- Modify: `plugin/README.md`
- Optional: `docs/superpowers/plans/PROGRESS.md`

- [ ] Document the new dashboard command and ribbon entry.
- [ ] Explain the workflow split:
  - daily report for triage.
  - dashboard for review.
  - paper note for long-form reading.
- [ ] Mention that `papers.json` is internal state and should not be hand-edited.
- [ ] Manual verification:
  - create several indexed fixture papers.
  - mark some watched/highlighted via daily checkbox.
  - open dashboard.
  - test tabs, filters, search, row action, batch action.
  - test opening notes and daily reports.

**Acceptance:**
- README describes how a user reviews watched/highlighted papers without opening daily reports.
- Manual smoke test covers the main research workflow.

---

## Suggested Implementation Order

1. Task 0: path consistency.
2. Task 1: pure query model.
3. Task 2: bulk store actions.
4. Task 3: view registration.
5. Task 4: read-only UI.
6. Task 5: row actions.
7. Task 6: batch actions.
8. Task 7: diagnostics.
9. Task 8: docs and smoke test.

This sequence keeps risk low: the data/query layer is tested first, then the UI becomes a thin shell over known behavior.
