# P2a — Split dashboard view.ts into modules

<!-- Filename must be NN-<slug>.md with NN = N (e.g. P1 → 01-auth.md). -->
goal_ref: ../goal.md
status: active

## Outcome

`plugin/src/dashboard/view.ts` (3360 lines) shrinks to roughly half: pure
helpers and the self-contained `HubModal` live in sibling modules, `view.ts`
keeps the `ArxivDailyDashboardView` class, `registerDashboardView`, and
re-exports. Exported API unchanged; all 46 dashboard-view tests and the full
suite stay green.

## Assumptions

- The exported functions are already covered by tests
  (`plugin/tests/dashboard-view.test.ts` and dashboard/*) that import from
  `../dashboard/view`; re-exporting from view.ts keeps those imports working
  with zero test churn
- Pure helpers have no hidden dependency on the view class's instance state
  (they take params); the class body is left intact this phase — splitting
  the class itself is out of P2a scope
- `HubModal` is self-contained (Modal subclass + panel types) and moves
  cleanly to its own file

## Approach

Extract bottom-up by topic, one module per commit, running the dashboard
tests after each extraction. view.ts re-exports (`export { x } from "…"`) so
the public surface stays identical; remove the re-export only if a later
phase wants to narrow it.

## Tasks

- [ ] T1: `constants.ts` — view id, timeouts, tabs, sort keys, page sizes
- [ ] T2: `calendar.ts` — calendar cell types/state, whitelist, empty-reason,
      aria labels, month/date helpers, buildCalendarDailyReportMap
- [ ] T3: `files.ts` + `pagination.ts` — vault path/markdown file filters,
      history path set, showingText, paginateDashboardRows
- [ ] T4: `log-format.ts` + `detail-refs.ts` — log entry formatting; indexed
      detail summary refs/expectations
- [ ] T5: `actions.ts` — open/trash/star/command helpers, deferred actions,
      topic options, status text
- [ ] T6: `hub-modal.ts` — HubModal class + HubModalTab/HubPanel types
- [ ] T7: slim view.ts to class + register + re-exports; full verification

## Verification

- `npm test --workspace plugin` — 46 dashboard-view tests + 236 plugin tests pass after every task
- `npm run typecheck`, `npm run lint` (0 errors), `npm run check:boundaries`
- `wc -l plugin/src/dashboard/view.ts` — ≤ ~1700 lines at T7
- `npm test` (full workspace) — green at the end

## Abort / reshape triggers

- If extracting a function breaks a test or changes behavior: revert that
  extraction (L1), don't push through
- If a helper turns out to depend on class state: leave it in view.ts (L1,
  scope adjust) and note it for a future class-split phase
- If view.ts re-export chains create circular imports: consolidate re-exports
  into one barrel or inline the few cross-dependencies (L1)
