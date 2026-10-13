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

## 2026-10-13 — note (P1 done)

- evidence: chunks 388a471 (Obsidian press), fb93a8b (workbench press; owner follow-up lets Tab, Shift+Tab and Escape leave a restricted control so keyboard navigation cannot get stuck), a17880d (Obsidian screenshot mask), 65711ab (shared secret rule and navigation keys). Each fix had new tests fail before and pass after; `npm run test:release-tools` 679/679. Real-host check with a scripted local fake model (/tmp/p1-verify): workbench Tab×20 onto the schedule switch then Space was refused ("external service and scheduling settings are fixed by the fixture"), the switch stayed off, Tab and Escape still worked; Obsidian Shift+Tab onto "Show LLM API key" then Enter was refused, the key field stayed a password field, fixture-requests had no external URL, and the step screenshots show the key field covered by a black box (owner viewed daily-paper-limit-step-3.png).
- change: success criteria 1 and 2 checked; P1 done, P2 started. New observation recorded as an open question: after opening Obsidian settings, keyboard focus stays outside the settings pane, so every press is refused as out of scope until the model clicks inside; the keyboard-only persona may stall there. The scope check predates P1. Workbench library scope (npm ci at index build) recorded as an open question for P4.
- disposition: keep all P1 work.
- next: P2 — a fixed test that checks every UI label a goal task depends on still exists in the product.

## 2026-10-13 — note (P2 done)

- evidence: d8d230e. All 14 Obsidian and 6 workbench goal tasks declare `uiLabels`; verifiers match the same declared strings through `exactLabelPattern`. The new label test failed before labels existed and passes after (7/7); renaming every "Run History" in the plugin source makes it report `auth-failure-retry`. `npm run test:release-tools` 686/686. Owner removed "（Star paper）" from the workbench star task goal: it is Obsidian-only text that never existed in the workbench. Known limits: the check is case-insensitive and substring-based, so renaming one of several occurrences of a label (e.g. the plugin also has a "Run history" command) does not trip it; runtime-built text such as "Page 2 / N" is not declared.
- change: success criterion 3 checked; P2 done, P3 started.
- disposition: keep.
- next: P3 — Obsidian personal-library goal tasks.

## 2026-10-13 — note (workbench library scope decided)

- evidence: research.md — workbench `library-build` always runs `npm ci` for the pinned PDF runtime against the real registry, so index-dependent library flows cannot run under the fixture as-is.
- change: user decision (2026-10-13): option B — the harness may pre-install the PDF runtime into a test-only cache before the model acts (network on first run only); model-time requests stay on fixtures. Recorded as a constraint; the open question is removed.
- disposition: no code yet; applies to P4.
- next: continue P3 Chunk 1.
