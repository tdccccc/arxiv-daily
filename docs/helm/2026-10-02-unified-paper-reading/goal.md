# Split workbench with a unified paper workspace

status: done
created: 2026-10-02T11:59:48+08:00
updated: 2026-10-02T13:59:30+08:00
revision: 4
owner: /root

## Intent

Keep the accepted split workbench: calendar and filters on the left, a paper list by default on the right, switching that same area to a daily report or paper summary when opened. Support persistent reading marks using the existing Paper Index and a user-adjustable sidebar.

## Success criteria

- [x] Right-side list includes all indexed discoveries, including papers without detailed notes; date/search/topic/reading/favorite filters and sorting/pagination use actual core data.
- [x] Selecting a date shows its paper list. Explicit actions open the whole report or a single paper in the same right-side area; returning restores filters, page and scroll position. No duplicated paper list remains on the left.
- [x] To-read/read/unmarked and independent favorites persist in the existing Paper Index, reject stale conflicting edits and preserve unrelated metadata and Markdown.
- [x] Existing full Markdown reports and standalone summary files remain accessible; unknown/missing index data is not mislabeled as zero selected papers.
- [x] Sidebar width is draggable/keyboard-adjustable, remembered across service launches and responsive to viewport constraints; desktop collapse and mobile navigation preserve reading space.
- [x] Protected HTTP/persistence, DOM, portable bundle and real browser acceptance verify the full workflow with isolated fixtures and no real paid model calls.

## Non-goals

- Full Obsidian Dashboard replication, new bulk/delete/scheduler actions, GUI configuration editing, PDF annotation or a third permanent content column.
- A second paper database, automatic history reindexing on GET, or treating the legacy saved status as favorite.

## Constraints

- Existing `.worktree/claude-code-research-plugin` only; preserve the original checkout and real user data.
- Reuse core Dashboard search/query and PaperIndexStore transactions. Favorites use priority=high; existing legacy reading states remain intact until explicit changes.
- Reading/querying never generates content or writes durable state. Detail generation uses the existing CLI operation and protections.
- Source containment, capability/origin checks and configuration revision checks remain in force for mutations.

## Phases

1. P1 — Reader-style unified paper workflow — status: superseded
2. P2 — Original Dashboard parity inventory — status: superseded
3. P3 — Full Dashboard port — status: superseded
4. P4 — Right-side paper workspace, persistent marks and adjustable navigation — status: done
