# P4 — paper-workspace

goal_ref: ../goal.md
created: 2026-10-02T12:40:00+08:00
updated: 2026-10-02T12:40:00+08:00
revision: 1

## Outcome

The right-side area switches between a unified paper list and reading content while the left provides calendar/filter navigation and adjustable space.

## Assumptions

- The latest user-approved placement supersedes both a left-side paper list and full Dashboard replication.
- Index entries provide identity, summaries, recommendation reasons and marks; absent index information remains explicit without hidden repair.
- Favorite maps to Obsidian priority=high independently from inbox/to_read/read status. Other existing states are preserved.
- Browser localStorage alone cannot remember layout across random localhost ports; small non-secret layout preferences belong beside the CLI config.

## Approach

Protected APIs query existing core data and mutate individual mark fields atomically with preconditions. The client preserves list context when reading and reuses existing Markdown/generation services. A separate sidebar controller owns resizing, collapse and persisted layout preferences; no duplicate business storage.

Visual thesis: preserve the warm split reader, give the right-side list the full working width, and keep navigation restrained.
Content plan: left calendar/scopes/search; right title/filters/order/rows or article; explicit back-to-list and original Markdown/file access.
Interaction thesis: stable list/reading history, persisted-only mark feedback, pointer/keyboard resizing and responsive mobile filter access.

## Chunks

### Chunk 1 — Unified paper and preference APIs

- change kind: behavior change
- strategy: strict Red-Green at actual HTTP and persistence boundaries
- Red: new query/detail/mark/preferences routes fail before implementation
- Green: index-only discoveries, date/search/filter/order/pagination, safe note availability, persistent independent marks, stale conflicts, invalid/foreign requests, idempotency and unchanged source files pass; layout preferences survive a fresh service
- regressions: existing calendar/server/inspection tests, CLI typecheck/boundaries
- [ ] implementation and tests accepted

### Chunk 2 — Right-side list and reading navigation

- change kind: behavior change
- strategy: strict Red-Green DOM/API tests plus browser layout review
- Red: default right-side list, back-state preservation and mark controls absent
- Green: date list→paper/full report→back preserves date/search/filter/page/scroll; stored-only mark success and conflicts; explicit detail generation refresh; standalone Markdown files; stale-response and mobile navigation tests pass
- regressions: existing reader/calendar interaction contracts updated only where the new approved placement supersedes old expectations; renderer/server regressions and CLI typecheck
- [ ] implementation and tests accepted

### Chunk 3 — Adjustable sidebar and integrated acceptance

- change kind: behavior change plus visual styling
- strategy: strict Red-Green for preference/resize controller; browser verification for widths
- Red: separator, pointer/keyboard resize, remembered width and collapse controls absent
- Green: bounded resize, remembered preferences, late-load protection, mobile adaptation and failure feedback pass
- regressions: actual copied CLI persists paper marks and preferences; calendar/read/generate flow remains functional; desktop/mobile screenshots and zero normal console errors
- [ ] implementation and tests accepted

## Abort / reshape triggers

- If a right-side route loses its return context, fix navigation before adding more controls.
- If an index write requires direct raw JSON manipulation outside core mutation, keep the operation blocked until it uses the existing transaction.
- If a missing report/index cannot be identified, show an explicit unavailable state rather than infer zero papers or overwrite a user note.
