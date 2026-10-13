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

## 2026-10-13 — note (tighten the label check after P3)

- evidence: P2's label check is case-insensitive and searches the whole host source, so a rename survives when the same text remains elsewhere (plugin has "Run History" in dashboard/hub-modal.ts:39 and "Run history" in commands.ts:904).
- change: user decision (2026-10-13): tighten it after P3 so P3's new tasks are covered in one pass. New P6: each label declares its source file and must appear there as a complete, case-sensitive string literal; runtime-built text is listed as an explicit exception. P2 stays done (its outcome holds; P6 is incremental).
- disposition: no code yet.
- next: continue P3 Chunk 1 (embedding-mode policy, local-embedding network question).

## 2026-10-13 — L1 adjust (P3 network guard before more library tasks)

- evidence: 525a7c5 landed the library fixture start, `library-revoke-consent`, and the anchored embedding policy (the bare "Embedding" mode select was swept up by an unanchored `embedding` rule; sub-fields stay restricted). Probe on the real isolated host: a connected library in local Embedding mode plus "Retry preparation" requests `https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/config.json` directly; transformers.js uses its own fetch, not the plugin HttpClient the harness proxies, so fixture-requests.json cannot see it. The probe blocked it with a CDP blocklist and stopped the session; nothing was downloaded. Any library task that leaves local mode reachable (including revoke, which ends in local mode) is exposed under an unscripted model.
- change: owner decision: add P3 Chunk 4 — exploration-only network guard at the Chromium layer (non-loopback requests fail), opt-in so fixed acceptance is unchanged — and finish it before adding `library-remote-consent` and the Chunk 2 tasks. The goal's constraint "外部请求全部走固定响应" already requires this; no goal change.
- disposition: keep 525a7c5; no real-model run of library tasks until Chunk 4 is accepted.
- next: P3 Chunk 4.

## 2026-10-13 — L1 adjust (plugin reload race)

- evidence: a4cc797 added `library-remote-consent` (scripted real-host run passed). Its wait-only run failed at the first observation with "Plugin failure: … Attempting to register an existing view type \"arxiv-daily-dashboard\"". Investigation (/tmp/reload-race): Obsidian awaits a plugin's async `onload()` on enable and unloads synchronously on disable; sequential awaited reloads were clean (0 errors in 40), concurrent reloads reproduced the exact error (38 in 20 iterations). Overlap sources: `withDeadline` stops waiting but does not cancel, so an abandoned reset/prepare keeps reloading; possibly also the first reset racing Obsidian's own startup load (that run had a single task, so the exact trigger is not confirmed). Product gap: `plugin/main.ts` `onload()` registers the Dashboard view and commands at its end without checking `this.unloading`, so a plugin disabled mid-onload still registers. Users can hit it by toggling the plugin quickly in Community plugins (no debounce).
- change: user decision (2026-10-13): fix both. P3 gains Chunk 5 (product: skip registration when unloaded during onload) and Chunk 6 (harness: serialize reloads, wait for startup load). Chunk 6 starts after Chunk 2 because both touch the Obsidian exploration driver.
- disposition: keep a4cc797.
- next: Chunk 5 in parallel with Chunk 2; then Chunk 6.

## 2026-10-13 — note (P3 chunks 1, 2, 5 accepted)

- evidence: a4cc797 (`library-remote-consent`), e4b942f (`library-build-index`, `library-search`, `library-similar-papers`; real index build in `prepare()` ≈3.2 s, far under the 60 s deadline; remote library fixtures carry a stub embedding key because `validateEmbeddingConfig` rejects an empty one), 0740019 (plugin skips registration and scheduler start when unloaded during onload). Scripted fake-model real-host runs passed for every new task; four runs showed no reload flake. The plugin fix closes the stale-instance leak but not the duplicate-view error under concurrent reload stress (37 before, 38 after): overlapping enables give two live instances, which only serialized reloads avoid. Whether users can reach that through the plugin toggle is unconfirmed; not pursued further.
- change: Chunks 1, 2 and 5 checked.
- disposition: keep.
- next: Chunk 6 (serialize harness reloads) and Chunk 3 (Direction candidate review tasks) in parallel with disjoint files.

## 2026-10-13 — L1 adjust (P3 chunks 3 and 6 accepted, observer gaps)

- evidence: a99e230 serializes plugin reloads (concurrent-reload stress on real Obsidian 37–38 errors in 20 → 0). eefbeff adds `library-accept-direction` and `library-edit-candidate` (rename only); scripted real-host runs passed, wait-only runs blocked. Real clustering needs two papers on the same fixture vector, so a third fixture PDF is seeded. Two observer gaps found on the real host: controls inside a collapsed `<details>` are reported as clickable (clicks then time out at hit-testing — safe, but wastes model steps), and with a confirmation dialog stacked on the review modal the observer resolves the first `.arxiv-daily-modal` in DOM order, so the confirm buttons are never observed (discard cannot be completed). Also, a button's `title` attribute wins over its text as the observed name.
- change: Chunks 3 and 6 checked; new Chunk 7 fixes the two observer gaps, restores discard in `library-edit-candidate`, and reuses the fixture PDF builder.
- disposition: keep a99e230 and eefbeff.
- next: P3 Chunk 7, then phase verification.
