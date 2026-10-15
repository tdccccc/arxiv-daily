# P3 — first-baseline

goal_ref: ../goal.md
created: 2026-10-15T12:35:00+08:00
updated: 2026-10-15T12:35:00+08:00
revision: 1

## Outcome

The researcher has labelled at least 30 papers from real filter decision records, and the first baseline report exists for both the recorded decisions and a real-model rerun.

## Assumptions

- The researcher's real interests are the two topics used in the trial runs (Photo-z; Galaxy Cluster, one direction each), on astro-ph.CO + astro-ph.GA — confirmed by the user 2026-10-15.
- Backfilling five announcement days with real CLI runs in a temp vault (user consented to the cost of about five daily runs) yields roughly 10–20 selected papers; with 40 skipped samples that is a sheet of 50–60 rows.
- arXiv may keep throttling the query API, so backfilled records are likely titles-only (`hadAbstract=false`); the rerun with sheet abstracts then doubles as a titles-vs-abstracts comparison.

## Approach

Backfill 2026-10-08, 09, 12, 13 into the trial vault that already holds 2026-10-14, export one sheet per day range (single direction snapshot, so rerun works by default), hand it to the researcher, then run `evaluate --recorded` and a `--runs 3` rerun after the user confirms the cost line.

## Chunks

### Chunk 1 — backfill records and export the sheet

- change: five days of filter decision records exist in the trial vault; one sheet with every selected candidate and 40 sampled skipped ones is exported with abstracts.
- kind: non-behavioral (operation)
- first signal: n/a
- checks: each run completes; record count per day; sheet row count, non-empty abstracts, single direction snapshot.
- [ ] accepted on observed evidence

### Chunk 2 — researcher labels

- change: the researcher fills `label` / `label_direction` for at least 30 rows.
- kind: non-behavioral (user work)
- first signal: n/a
- checks: `evaluate --recorded` accepts the sheet without row errors; labelled count ≥ 30.
- [ ] accepted on observed evidence

### Chunk 3 — baseline reports

- change: `report-recorded` and a `--runs 3` rerun report exist for the labelled sheet; findings summarised for the user.
- kind: non-behavioral (operation, real model cost — confirm the printed cost line with the user first)
- first signal: n/a
- checks: both reports written; numbers spot-checked against the sheet.
- [ ] accepted on observed evidence

## Phase verification

- Goal success criterion 5 holds: ≥ 30 labelled papers and a first baseline report from a real-model run.

## Abort / reshape triggers

- If backfilled days still give fewer than about 8 selected papers in total, widen the date range before handing over the sheet rather than asking for labels on a sheet with too few positives.
