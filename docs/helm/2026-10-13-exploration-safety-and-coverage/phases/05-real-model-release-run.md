# P5 — real-model-release-run

goal_ref: ../goal.md
created: 2026-10-14T14:26:11+08:00
updated: 2026-10-14T14:46:57+08:00
revision: 2

## Outcome

Both hosts' full task libraries (workbench 12 goal tasks, Obsidian 21, plus 3 personas each) have run in release mode with the real model; every failure and finding has a disposition; the README task counts, cost table and budget defaults match the observed runs.

## Assumptions

- The user approved the cost (journal 2026-10-14); the Obsidian library sample cost ≈12.6k input tokens per task, so both hosts together are estimated at ≈1.4–1.6M input tokens.
- The existing `--release` budgets (220 steps, 440 calls, 20 minutes per run) still fit the larger libraries; the runs will show whether they do.
- The real model uses the existing CLI `[llm]` config; keys never appear in output.

## Approach

Run both hosts once in release mode and keep the evidence; triage every failure and finding with a fixed reproduction first; rerun to confirm after fixes; then update the README counts, cost table and, if needed, budget defaults.

## Chunks

### Chunk 1 — first real-model release runs

- change: evidence of both hosts' full release runs, with per-task status, calls, tokens and duration, and every finding listed.
- kind: non-behavioral (evidence collection)
- first signal: n/a
- checks: `npm run test:acceptance -- --suite exploration --host workbench --release` and `--host obsidian --release`, each with a fresh `--output`.
- [x] accepted on observed evidence — see journal 2026-10-14 (P5 chunk 1 results).

### Chunk 2 — triage and fixes

- change: every failure and finding has a recorded disposition (product fix, task or prompt fix, harness fix, or false positive); confirmed problems are fixed with a fixed regression.
- kind: bug fix / test tooling (per finding)
- first signal: a reproduction (test or scripted run) fails before each fix.
- checks: acceptance-exploration*; test:acceptance:tools; related workspace tests; fixed workbench and Obsidian acceptance as affected.
- [ ] accepted on observed evidence

### Chunk 3 — confirmation run and release documentation

- change: a confirmation release run on both hosts after fixes; README task tables, cost table and (if the runs require it) `--release` budget defaults updated from the observed numbers.
- kind: documentation (+ behavior change if defaults move: Red-Green on the argument parsing test)
- first signal: n/a for docs; a failing parse test if defaults change.
- checks: `--list` matches the README tables; test:acceptance:tools.
- [ ] accepted on observed evidence

## Phase verification

- Both hosts have a release run after the last fix; goal-task failures in it are explained (fixed or recorded).
- Every finding has a disposition in the journal.
- `npm run test:release-tools` passes.

## Abort / reshape triggers

- If a single release run exceeds ≈3M input tokens or 30 minutes, stop and report the numbers to the user before running more.
- If the real model fails most library tasks because observation or action support is insufficient, stop and discuss instead of loosening verifiers.
