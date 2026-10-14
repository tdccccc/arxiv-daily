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

## 2026-10-13 — L1 adjust (P3 verification found a library state leak)

- evidence: bfb09e6 (observer: topmost dialog, collapsed content hidden; discard restored) accepted. Phase verification (/tmp/p3verify): the 21-task run with an always-blocked model ended 20 tasks by the model's blocked action, but `library-edit-candidate` failed in `prepare()` with "no-evidence: clustering produced no evidence groups" (both runs). The 7 library tasks with scripted plans in one run: first five passed; `library-accept-direction` failed and `library-edit-candidate` blocked with the same error (both runs identical); alone, each passes. `reset()` restores data.json and reloads the plugin but leaves library PDFs and `.index` state from earlier tasks. No unexpected network requests; release tools 751/751. This is the phase's abort trigger "library state leaks … through reset()".
- change: Chunk 7 checked; new Chunk 8 finds the root cause first — a harness leak, or a plugin bug where a paper added after an earlier index is not picked up — then fixes it.
- disposition: keep bfb09e6. No new library tasks until Chunk 8 is accepted.
- next: P3 Chunk 8.

## 2026-10-13 — note (P3 done)

- evidence: d0e3fdc. Root cause of the leak was the harness: "Retry preparation" scans a library folder only the first time a catalog exists (plugin/main.ts ~1547-1554, intentional), and library catalog/index/profile state lives under `arxiv-daily/.index/personal-library-*` keyed by the library root, not data.json, so later tasks reused the earlier catalog and the seeded third paper never reached clustering. Users pick up new papers through "Scan library" in the review dialog, so this is not a product bug (possible UX note for P5: "Retry preparation" alone does not rescan). `reset()` now clears personal-library state and non-fixture library files inside the fixture vault. Phase verification on real Obsidian: the 21-task always-blocked run ends every task by the model's own blocked action; the 7 library tasks with scripted plans pass back to back twice; no unexpected network requests; release tools 753/753.
- change: P3 done (Obsidian half of success criterion 4; the workbench half is P4). P6 started.
- disposition: keep all P3 work.
- next: P6 — label check by source file and exact string.

## 2026-10-14 — note (P6 done)

- evidence: 575b7be. Labels are `{ text, source }`; each source must contain the text as a complete, case-sensitive string literal. Owner widened each host's allowed sources to include `packages/core/src/`, which removed the "Retry preparation" exception and lets "Revoke" and "Embedding" check both their plugin and core sources. One exception remains: the workbench `sidebar-toggle` purpose tag (a test hook inside a longer className, not display text). The old check passed a hub-modal-only rename, a comment-only label and a substring-only label; the new one fails all three. "Reading Dashboard" corrected to the product's "Reading dashboard". acceptance-exploration* 274/274; release tools 761/761.
- change: P6 done; P4 started.
- disposition: keep.
- next: P4 intake — workbench library flows, PDF runtime pre-install, and whether every workbench server request goes through the fixture.

## 2026-10-14 — handoff (P4 chunks 1 and 2 accepted)

- evidence: 65f3ce2 (version-keyed PDF runtime cache copied into each fixture, npm forced offline; real CLI connect/authorize/prepare/scan/index spawned no npm and sent only the fixture embeddings request) and 9d79ba6 (library seed through the real CLI plus server resync; `reset()` restores library config, accepted library directions, index state and extra PDFs inside the fixture; older workbench tasks still pass after a seeded library). Both were committed by the previous session with their validation but not yet recorded as accepted. Re-run in this session: acceptance-exploration* 297/297, matching the commit record. The worktree was clean and the last edit was 2026-10-13 20:28, so no other session was active.
- change: owner main-session-3a9f449b → claude-code-session-2026-10-14; P4 Chunks 1 and 2 checked.
- disposition: keep 65f3ce2 and 9d79ba6.
- next: P4 Chunk 3 — build, revoke, catalog filter and index search tasks.

## 2026-10-14 — note (user approves the P5 real-model run)

- evidence: user decision (2026-10-14): approve the P5 real-model full run, on condition that a small sample first estimates the token cost and the estimate is not excessive. Reference from the last release runs (2026-10-12, README "实测成本与耗时"): workbench ≈575k input tokens for 6 goal tasks + 3 personas, Obsidian ≈514k for 14 goal tasks + 3 personas; per goal task ≈38k (workbench) / ≈20k (Obsidian), per persona 55k–160k. The 7 P3 library tasks and the P4 workbench library tasks have no real-model data yet.
- change: none to goal or phases; the cost-consent constraint is satisfied for P5, subject to the sample estimate.
- disposition: sample = Obsidian `library-build-index`, `library-search`, `library-accept-direction` with the real model, run while P4 Chunk 3 is in progress (Obsidian side only); a workbench library sample follows once P4 tasks exist.
- next: report the sample-based estimate to the user; continue P4.

## 2026-10-14 — note (P4 chunk 3 accepted; Obsidian library sample cost)

- evidence: cf2c8ad, 469b209. Four workbench tasks — `library-build-index`, `library-revoke`, `library-catalog-filter`, `library-index-search` — with 16 verifier tests; the seed gained a `scanned` option (the CLI's `library scan` never requires consent). An observer bug was found on the real host: a settings row's `data-setting-name` won over a plain button's own text, so all library row buttons were named "Library" (and topic-row buttons the topic name); fixed in the harness only, no restriction changed. Test-first exception: the delegate's session dropped (ECONNRESET) while writing tests and implementation together, so the Red was reconstructed rather than observed in order (a missing task id fails with "task … must exist in goalTasks()"); the 16 verifier tests' negative cases and wait-only/always-blocked real-host runs are the compensating evidence. Owner re-run: acceptance-exploration* 314/314. Delegate runs: test:acceptance:tools 379/379, test:release-tools 801/801, fixed workbench suite 12/12; scripted real-host runs pass alone and back to back twice; wait-only and always-blocked runs end blocked; the 10-task always-blocked run ends every task by the model's blocked action; no npm spawn, fixture requests only (/tmp/p4c3-real-host). Separately, the real-model sample of three Obsidian library tasks (/tmp/p5-sample-obsidian-1791946942) passed 3/3 with 8 calls and 37,810 input tokens in total (≈12.6k per task), cheaper than the older Obsidian goal-task average; estimate for P5 ≈1.4–1.6M input tokens across both hosts, reported to the user as within their "not excessive" condition.
- change: P4 Chunk 3 checked.
- disposition: keep cf2c8ad and 469b209.
- next: P4 Chunk 4 — Direction candidate review tasks on the workbench.

## 2026-10-14 — note (P4 done, P5 started)

- evidence: 36a4815 adds `library-accept-direction` and `library-edit-candidate` on the workbench (the proposal is generated by the real CLI over two seeded clusters). The delegate found that workbench `reset()` left a topic emptied of its library-origin direction, which collided by name with the next generated proposal ("reference-out-of-scope"); its fix dropped every empty topic, which the owner narrowed to topics the reset itself emptied (Red with the broad filter, Green with the narrow one). Phase verification: the 12-task workbench library with an always-blocked model ends every task by the model's blocked action (delegate, /tmp/p4c4-real-host/all-12-blocked-*); the 6 workbench library tasks with scripted plans pass back to back twice (owner re-run after the narrowing, /tmp/p4c4-owner-all-library.log); no unexpected fixture request or npm spawn; acceptance-exploration* 324/324 and test:release-tools 811/811 (owner).
- change: P4 Chunk 4 checked; P4 done; success criterion 4 ticked (both hosts); P5 active with its phase plan.
- disposition: keep 36a4815.
- next: P5 Chunk 1 — real-model release runs, workbench then Obsidian.

## 2026-10-14 — note (P5 chunk 1 results and triage plan)

- evidence: first release runs with the real model (same CLI `[llm]` config as 2026-10-12). Workbench (/tmp/p5c1-workbench-1791959177): 12/12 goal tasks passed, 3 personas explored, 105 calls, 764,905 input / 4,223 output tokens, 2 findings. Obsidian (/tmp/p5c1-obsidian-1791959177): 20/21 passed, 1 blocked, 3 personas explored, 138 calls, 751,721 input / 5,322 output tokens, 1 finding. Total ≈1.52M input tokens, inside the 1.4–1.6M estimate. All 13 library tasks passed on the first real-model run. Dispositions so far: (1) Obsidian `completed-date-idempotent` blocked on "Model API call timeout after 45000ms" before any action; rerun alone passed (/tmp/p5c1-obsidian-retry-1791960287) — transient provider latency, no change. (2) Obsidian `library-edit-candidate` passed but used 18 of 20 steps: observed `<summary>` controls carry no open/closed state, so the model collapsed already-open sections repeatedly; `home` opens the dashboard behind an open review modal and reports success, so the model repeated it 5 times; and the Add/Generate buttons are named by their tooltips (the P3 note "title wins over text"). Owner decision: harness fixes on the Obsidian observer (delegated). (3) Workbench first-time-user finding "reading status not kept after reopening the paper" (step 70) — under investigation (delegated) before classifying. (4) First-time-user finding on both hosts: the first-run settings page is long and does not point to where to generate a report or read papers — a product UX question, put to the user.
- change: P5 Chunk 1 checked; Chunk 2 items as above.
- disposition: keep run evidence in /tmp; no code yet.
- next: P5 Chunk 2 — read-status classification, Obsidian observer fixes, user decision on the first-run guide.

## 2026-10-14 — note (first-run settings finding closed as false positive)

- evidence: screenshots /tmp/p5c1-workbench-1791959177/exploration/exploration-first-time-user-step-63.png and /tmp/p5c1-obsidian-1791959177/exploration/first-time-user-step-105.png. The persona's fixture is already configured with papers (not a true first run); the model opened settings itself late in the persona; the workbench "生成" button and the Obsidian dashboard stay available outside settings.
- change: user decision (2026-10-14): option A — no product change; finding (4) of the P5 chunk 1 triage is a false positive.
- disposition: no code.
- next: remaining P5 Chunk 2 items — read-status classification, Obsidian observer fixes.

## 2026-10-14 — handoff (paused by the user to save quota)

- evidence: read-status finding (workbench first-time-user step 70) is a model misreading: step 67 set paper A ("Reproducible evidence…", arxiv:2610.10002) to read; in observation s81 the list already showed A's status select as `read` and B's as `inbox`, and step 69 opened paper B (arxiv:2610.10001), whose untouched `inbox` the model reported. The workbench saves the select on `change` immediately (apps/cli/src/workbench/web/app.ts saveMark) and the server writes synchronously (apps/cli/src/workbench/papers.ts mark); the harness `selectOption` fires the product's events. Existing coverage (apps/cli/tests/workbench-papers.test.ts "persists independent reading and star marks…") already covers persistence, so the delegate's extra passing test was discarded. The Obsidian observer-fix delegate dropped its connection (ECONNRESET) before editing any file; the worktree is clean.
- change: finding (3) closed as a false positive. User asked to pause after this stage because quota use is high.
- disposition: no code from either delegate is kept.
- next: P5 Chunk 2 remaining item — Obsidian observer fixes from the 2026-10-14 triage entry: (a) observed `<summary>` controls report their `<details>` open state; (b) `home` with a plugin modal open must close it before opening the dashboard (now it reports success while the modal stays on top); (c) a button's own text before its `title` as the observed name (keep `aria-label` first), mirroring cf2c8ad. Then Chunk 3: confirmation release run on both hosts (≈1.5M input tokens; user already approved the cost of P5 runs) and README task tables/cost table.

## 2026-10-14 — handoff (resumed after the quota pause)

- evidence: worktree clean at 1da8869; the last helm edit was 14:57 and the user asked to continue the latest initiative at 16:33, so no other session is active.
- change: owner claude-code-session-2026-10-14 → claude-code-session-2026-10-14b.
- disposition: keep all P1–P5 work so far.
- next: P5 Chunk 2 — Obsidian observer fixes (a) summary open state, (b) `home` closes a plugin modal first, (c) own text before `title`.

## 2026-10-14 — note (P5 chunk 2 accepted)

- evidence: d600a2e fixes the three Obsidian observer gaps from the chunk 1 triage: (a) a `<summary>` reports `expanded` from its parent `<details>`; (b) `home` closes an open plugin modal or menu with the same Escape `back` uses and returns `ok:false` if it stays open, instead of opening the dashboard behind it; (c) a button's own text before `title`, `aria-label` still first. 7 new fixed tests with observed Red then Green. The existing host test that expected no input on `home` encoded the bug and now expects exactly one Escape and no mouse input. Delegate real-host runs without a model: scripted `library-edit-candidate` passed in 6 steps (/tmp/p5c2-obsidian-edit-scripted-1791970602); a home-with-modal probe went from view=modal to view=dashboard with no dialog (/tmp/p5c2-obsidian-edit-home-probe-1791970761); all 7 scripted library tasks passed (/tmp/p5c2-obsidian-library-all-scripted-1791970915); always-blocked and wait-only runs ended blocked. Owner re-run: acceptance-exploration* 331/331, test:release-tools 818/818. Not covered: the menu branch of (b) on a real host (fixed tests only). The delegate's first session dropped after writing the tests; it was resumed and finished.
- change: P5 Chunk 2 checked; every chunk 1 failure and finding now has a disposition (1 transient timeout, 2 false positives, 1 harness fix).
- disposition: keep d600a2e. The workbench observer has the same `<summary>` gap, and `<a>`/`<summary>` still put title before their own text there (buttons already fixed in cf2c8ad); it was not seen in a run, so it is left alone unless the confirmation run shows it.
- next: P5 Chunk 3 — confirmation release run on both hosts, then README task tables and cost table.

## 2026-10-14 — note (close)

- evidence: confirmation real-model release runs after d600a2e. Workbench (/tmp/p5c3-workbench-1791971488): 12/12 passed, 3 personas explored, 0 findings, 0 errors, 108 calls, 796,908 input / 4,289 output tokens, 6.1 min. Obsidian (/tmp/p5c3-obsidian-1791971488): 21/21 passed, 3 personas explored, 0 findings, 0 errors, 130 calls, 612,500 input / 4,933 output tokens, 9.5 min. Obsidian `library-edit-candidate` passed in 7 steps (18 in chunk 1); `completed-date-idempotent` passed (the chunk 1 timeout did not recur). c856d7c adds the 7 Obsidian library tasks to the README table (14 → 21; both tables match `--list`) and replaces the cost table with the 2026-10-14 runs. Peak use across the four P5 runs is 138 calls and 10.9 min, inside the unchanged `--release` defaults (220 steps, 440 calls, 20 min); the step margin is the tightest at about 1.6×, recorded in the README. test:acceptance:tools 396/396, test:release-tools 818/818.
- change: P5 Chunk 3 checked; P5 done; success criterion 5 ticked; goal done. Open question 1 closed: in both release runs the keyboard-only persona's presses were never refused (Obsidian run: one invalid-JSON reply, no scope refusals). Open question 2 kept as a follow-up, not a success criterion: workbench goal tasks and the Obsidian library tasks have untouched/leftover-state verifier tests; the original 14 Obsidian tasks do not, and no instance was found.
- disposition: keep all work. Branch fix/exploration-safety-coverage is unpushed; pushing and a PR need the user's approval. The workbench observer's `<summary>` state gap and title-before-text for `<a>`/`<summary>` remain unfixed (not seen in a run).
- next: none in this initiative; ask the user about pushing and a PR.

## 2026-10-14 — revisit (workbench observer parity)

- evidence: the close entry left the workbench observer's `<summary>` state gap and its title-before-own-text naming for `<a>`/`<summary>` unfixed because no run showed them. The user questioned this. The reason does not hold: it is the same gap that cost Obsidian `library-edit-candidate` 11 extra steps, the workbench uses `<details>`/`<summary>` in the library review dialog, settings and the main app, and the fix is small and needs no model call.
- change: goal reopened (done → active); P7 added as follow-up. P5 stays done (its accepted outcome holds). Intent and success criteria unchanged.
- disposition: keep all P1–P6 work.
- next: P7 Chunk 1 — workbench summary state and own-text naming, delegated.

## 2026-10-14 — note (P7 done, goal closed again)

- evidence: 822912c. A workbench `<summary>` reports `expanded` from its parent `<details>` and is named by its own text like a button; a link's own text now beats `title`. L1 during acceptance: the delegate's version also put a link's own text before its settings row, which renamed the four external-link rows ("Report a bug", "Documentation", "Repository", "Request a feature") to four identical "Open" names. No workbench `<a>` has a `title`, so that order fixed nothing there; the owner moved link own text after the row (Red: names were `['Open','Open']`, Green after the change). Restriction is by href, so safety was never affected. Delegate Red→Green for 6 tests (expanded true/false, titled link and summary, two guards); owner re-run: P7 tests 7/7, acceptance-exploration* 338/338, test:release-tools 825/825. Owner real-host runs without a model after the final change: 12 workbench goal tasks with scripted plans passed (/tmp/p7-runs/p7-owner-scripted-all12-1791979703710); always-blocked ended all 12 blocked (/tmp/p7-runs/p7-owner-blocked-all12-1791979745593); observations show `expanded` true on open review topics and false on "查看运行详情"/"Research", row names on the external links, and own text on paper links. The delegate's session dropped once (ECONNRESET) before any edit and was resumed. Its throwaway runner and scripted plans for all 12 workbench tasks live in /tmp/p7-run, outside the repo.
- change: P7 Chunk 1 checked; P7 done; goal done again. Success criteria unchanged and all ticked.
- disposition: keep 822912c.
- next: none; branch fix/exploration-safety-coverage is unpushed, pushing and a PR need the user's approval.
