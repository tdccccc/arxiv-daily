# P1 — retain-filter-decisions

goal_ref: ../goal.md
created: 2026-10-15T10:35:00+08:00
updated: 2026-10-15T11:08:00+08:00
revision: 2

## Outcome

After a daily report commits, a compact filter decision record for that date — every candidate including `skip`, with category, directions, relevance score and whether it entered the report, plus the direction snapshot, prompt contract version and generation identity — stays in Vault data under the index directory, on both CLI and plugin, pruned by date.

## Assumptions

- At commit time the pipeline can reach the full validated `FilterRecord[]` (including `skip`) for the date — either from `filterPapers` or from the filter checkpoint document read before `cleanupCommittedCheckpoints` removes it. Verify which before building; prefer passing records through the pipeline over re-reading a file that may be absent (e.g. no checkpoint store wired in tests).
- The visible-set decision (`pipeline.ts` ranking + `maxDailyPapers` slice + `ignored` exclusion) is computed in one place, so "entered the report / cut by limit / excluded as ignored / skipped by model" can be derived without duplicating the ranking.
- CLI (`apps/cli/src/runtime.ts`) and plugin (`plugin/main.ts`) both construct the pipeline with core storage adapters, so one core store serves both hosts.
- A day's compact record (id, title, decision fields) is on the order of tens of KB; 180 days stays well under typical vault sizes. Measure on a real filter checkpoint if one is available.
- Data portability / export / "clear run state" code may enumerate Vault-data directories; a new directory may need to be listed there.

## Approach

Add a core store for filter decision records (one JSON document per report date under `<indexDir>/filter-decisions/`, atomic write like the existing checkpoint stores), have the pipeline write it after the daily report is committed and before transient checkpoints are cleaned, prune records older than the retention window on each write, and wire it in both hosts. Record the new Vault-data term in `CONTEXT.md`.

## Chunks

### Chunk 1 — filter decision record store

- change: core store can write, read and list per-date filter decision records with a versioned schema; invalid or corrupt documents read as absent with a warning; pruning removes records older than the retention window and leaves others.
- kind: behavior
- first signal: Red — new `packages/core/tests/filter-decision-record-store.test.ts` fails because the store does not exist.
- checks: focused test Green; `npm run -w packages/core test` (or the workspace's equivalent) stays Green; typecheck.
- [x] accepted on observed evidence — delegate observed Red (module missing) → 15/15 Green; owner re-ran `vitest run tests/filter-decision-record-store.test.ts` Green; workspace tests + typecheck Green per delegate report

### Chunk 2 — pipeline writes the record on commit

- change: a successful daily run writes the record for its date, covering every candidate the model classified (including `skip`) with its outcome (in report / cut by daily limit / excluded as ignored / skipped by model), direction snapshot, prompt contract version and generation identity (no API key, no raw endpoint); a failed write only warns; a rerun of the same date replaces the record; a run that never calls the filter writes nothing.
- kind: behavior
- first signal: Red — new pipeline test (mock LLM, same style as existing pipeline tests) fails because no record exists after commit.
- checks: focused test Green; existing pipeline, paper-filter and checkpoint tests stay Green.
- L1 (2026-10-15): `no_matches` completions (every candidate skipped; every kept paper ignored) also write the record — they are the days recall is judged on. Failure paths still write nothing.
- [x] accepted on observed evidence — delegate observed Red 7/9 then, after the L1 addition, Red 3/13 (no record file on no-match days) → 13/13 Green; owner re-ran record + paper-filter + pipeline tests (154 passed) and core `tsc --noEmit` clean

### Chunk 3 — hosts wire the store

- change: CLI runtime and Obsidian plugin pass the store to the pipeline; any Vault-data enumeration (portability, clear run state, backups) handles the new directory consistently with the existing filter checkpoints.
- kind: behavior
- first signal: Red — a CLI-level test (existing runtime/daily-run test harness) shows no record after a mocked daily run.
- checks: focused test Green; `npm run verify` or the repo's root verification stays Green; plugin build passes.
- [x] accepted on observed evidence — delegate observed CLI and plugin Red → Green; no production code enumerates `filter-checkpoints` by name, data export/import copies `.index` whole (round-trip test added, Green from the start as regression guard); owner re-ran CLI suite (457 passed) and plugin suite (1048 passed)

### Chunk 4 — glossary

- change: `CONTEXT.md` gains a *Filter decision record* entry next to *Filter checkpoint*, stating it is durable, compact, not reused as a cache, and exists for evaluation.
- kind: non-behavioral
- first signal: n/a
- checks: read-through for consistency with the existing checkpoint terms.
- [ ] accepted on observed evidence

## Phase verification

- Root verification command (as used by CI's "Root workspace verification") passes.
- One real CLI daily run against `plugin_test/arxiv-daily` (or a temp vault) with the user's consent if it calls the model, or with the existing fixed-response acceptance setup otherwise: the record file exists after the run, contains skipped candidates, and the filter checkpoint is gone as before. Note the observed file size per day.

## Abort / reshape triggers

- If the full record set (including `skip`) is not available at commit without re-running or re-reading fragile state, stop and reshape how records are captured (L1/L2) rather than storing only kept papers.
- If retaining records requires changing filter checkpoint reuse semantics, stop and steer (L3): the goal assumes the checkpoint stays transient.
