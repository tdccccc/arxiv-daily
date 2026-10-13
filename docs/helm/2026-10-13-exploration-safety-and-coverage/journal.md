# Journal — exploration safety and coverage

## 2026-10-13 — note (intake from review)

- evidence: read-only review of the merged model-driven release exploration (PR #70) on main 025c585. (1) The `press` action skips the restricted/sensitive/disabled checks that `click`/`fill`/`select` get: `parseExplorationAction` only checks ref-bearing actions (scripts/acceptance/exploration.mjs:46), the workbench executes `page.keyboard.press` with no check (exploration-browser.mjs:183), and Obsidian only checks that focus is inside the plugin surface (exploration-obsidian.mjs:564-568). Tab then Enter can reach the settings "Report a bug" / "Documentation" / "Repository" buttons that open real external URLs (plugin/src/settings/tab.ts:2215-2254), the scheduler toggle, "Clear run state…", and the API key reveal buttons. Obsidian screenshots have no secret masking (scripts/desktop-acceptance/screenshots.mjs), unlike the workbench (exploration-browser.mjs:234); the Obsidian fixture currently only holds a stub key. (2) Goal-task verifiers are sound, but several gate on exact UI labels ("Run History", "Force run for date…"), so a copy change shows up only as a budget-exhausted block. "Retry failed dates" and "Run pending" (plugin/src/dashboard/view.ts:1906,1927) have no task. (3) The personal literature library has no exploration task on either host. (4) Offline checks on main: 177 exploration tests, 242 acceptance-tool tests and 660 release-tool tests pass; fake-model runs on both hosts finish cleanly; docs match the code; CI never calls a model API.
- change: new initiative; the user chose to fix the press bypass and screenshot masking, add the two menu tasks and stale-label reporting, and add personal-library tasks (option C), in a new worktree.
- disposition: the 2026-10-10 initiative stays `done`; this is follow-up work, not a revisit of its phases.
- next: P1 — refuse `press` on restricted, sensitive, disabled or out-of-scope focus on both hosts; mask secret fields in Obsidian screenshots.

## 2026-10-13 — L3 steer (narrow P2)

- evidence: the user questioned the two "More" menu tasks. Single-date retry is already reachable from the calendar and covered by the run-for-date and auth-failure-retry tasks; "Retry failed dates" and "Run pending" only batch the same run over a set of dates, which ordinary tests suit better than a paid model run. For stale UI labels, a blocked reason in the report would surface only after a paid real-model run, while a plain fixed test catches a rename when it happens.
- change: user decisions (2026-10-13): drop the two menu tasks (now a non-goal); replace the "task out of date" report reason with a fixed test that checks every label a goal task depends on still exists in the product. Success criteria 3 and 4 merged into one; P2 outcome rewritten. P1, P3–P5 unchanged.
- disposition: no code existed for P2; P1 work in progress is unaffected and continues.
- next: continue P1.
