# P2 — abstract-fallback-visibility

goal_ref: ../goal.md
created: 2026-10-15T18:30:00+08:00
updated: 2026-10-16T00:50:00+08:00
revision: 8

## Outcome

When the arXiv query API (export.arxiv.org) fails or is throttled, candidates
still get abstracts from their `arxiv.org/abs/<id>` pages before filtering, a
header-less 429 no longer causes short retries, and any candidates that were
still filtered on title only are counted in the daily report header and
run-history (CLI and Obsidian) without failing the run.

## Assumptions

- Recon: `tmp/helm-reports/p2-recon.md` (git-ignored). Adapter lives in
  `packages/core/src/sources/arxiv-source-adapter.ts` (`enrichAbstracts`
  L196-226, only caller `pipeline.ts:861`); retries in
  `arxiv-fetcher.ts` `fetchHtml` L450-487; one process-wide
  `ArxivRequestCoordinator` serialises all hosts with one `cooldownUntil`.
- arxiv.org abs pages stay reachable while export.arxiv.org throttles
  (observed 10-14/10-15; probe 2026-10-15 18:20 shows the API back at 200,
  so throttling is intermittent).
- The abs cache (`abs/<id>` in `HtmlCache`) is read later only when rendered
  HTML and source both fail; the fallback's main value is the abstract text
  for filtering, cache reuse is a bonus.
- Hosts build the pipeline without a custom `sourceAdapter` and already pass a
  `PaperContentFetcher`, so a new public method there needs no host wiring.
- `RunHistoryRecord` validation is not exact-key: an optional field needs no
  schema bump. Run state (`run-state.json`) is untouched.

## Decisions (user, 2026-10-15; see journal)

- Cooldowns are per host (export.arxiv.org vs arxiv.org); pacing (serial,
  ≥ 3 s) stays shared across hosts.
- 429 without `Retry-After` on the query API: no retries; fail fast into the
  abs-page fallback and keep the query API cooled down in-process for a
  while. On arxiv.org (list/abs pages): wait at least 30 s before retrying.
- No cross-process cooldown marker in this initiative (reruns hit the abs
  cache; one daily run per day; extra persisted state not worth it yet).
- Abs-page fallback cap: 500 network fetches per run (cache hits don't
  count), stop after 3 consecutive failures, cancellable. Fixed constant, no
  setting.
- Count semantics: candidates offered to the filter with an empty abstract
  after the fallback (= `hadAbstract: false` in the decision record).

## Chunks

### Chunk 1 — per-host cooldown and header-less 429 handling

- change: the coordinator keeps cooldowns per host; a 429/503 with
  Retry-After on export.arxiv.org no longer delays arxiv.org requests (and
  vice versa). A header-less 429 from the query API is not retried and puts
  that host in an in-process cooldown that fails fast; a header-less 429 on
  arxiv.org retries after ≥ 30 s instead of 2–4 s.
- kind: behavior
- first signal: Red — new `arxiv-fetcher.test.ts` cases (fake clock/HTTP):
  API Retry-After cooldown currently delays an abs fetch; header-less API 429
  currently retried after ~2 s; header-less abs 429 retried after ~2 s.
- checks: same tests Green; existing fetcher tests (Retry-After 5 s, long
  deferral, 503 shared cooldown — reinterpreted per host) Green; core suite.
- [x] accepted on observed evidence — Red 6/44 for expected reasons
  (cross-host wait/deferral, API header-less 429 retried, 2–4 s page retry;
  one self-inflicted 503-guard assertion corrected); Green 44/44, core 2550,
  root typecheck 0, export script 15/15, CLI runtime 7/7; no existing
  expectation changed. Query-API header-less 429 → one request, plain
  `ArxivHttpError(429)`, then 10 min in-process fail-fast
  (`ArxivRetryDeferredError`, message still says "arXiv requested" — cosmetic).
  Page header-less 429 → 30 s, 60 s.

### Chunk 2 — abs-page abstract fallback

- change: after the query API call (ok, partial, or failed), papers still
  lacking an abstract get one via a new public `PaperContentFetcher` method
  that reads/writes the `abs/<id>` cache and uses `extractAbsPageAbstract`;
  sequential, ≤ 500 network fetches per run, stops after 3 consecutive
  failures, `throwIfCancelled` per paper and cancellation rethrown, an active
  per-host cooldown counts as failure, one summary log line. Partial query
  results are warned, not just info-logged.
- kind: behavior
- first signal: Red — adapter tests: query API throws (and partial map) →
  candidates still have empty abstracts; cancellation mid-fallback; cap and
  consecutive-failure stop.
- checks: same tests Green; existing adapter/pipeline tests with partial fakes
  updated deliberately; paper-content and section-extractor tests Green.
- [x] accepted on observed evidence — Red 14/80 in the three target files for
  expected reasons; Green: core 2568, root typecheck 0, CLI 457/457, plugin
  settings-lifecycle 47/47 (full plugin suite left to phase verification).
  New `PaperContentFetcher.fetchListingAbstract` shares the `abs/<id>` cache
  (adapter ids are canonical, same key as `fetchContent`); `fetch()`
  unchanged. 4 existing tests/fixtures gained a fake method. Known limit: once
  the 500 cap is hit, remaining papers are skipped even if cached.

### Chunk 3 — title-only count on the pipeline result

- change: the pipeline counts candidates with empty abstracts after
  enrichment and exposes it on completed results (including no_matches /
  no_updates where candidates existed) and in a log line; the run stays
  `completed`.
- kind: behavior
- first signal: Red — pipeline test (decision-record harness with
  `abstractOf`) expecting the count on the result.
- checks: same tests Green; pipeline suites Green.
- [x] accepted on observed evidence — Red 4/5 new decision-record tests
  (field missing; empty-day absence passes by design); Green 23/23, core
  2573, root typecheck 0, CLI 457/457, plugin commands + history-sync 74/74.
  `titleOnlyCandidates: {count, total}` on completed results (papers_written
  and both no_matches paths), absent on empty days and repairs; warn line
  when count > 0. Summarizer threading left to Chunk 4.

### Chunk 4 — daily report header line

- change: when the count > 0, the report header shows one line in zh/en
  (e.g. "筛选时 M 篇候选中有 N 篇没有摘要，只依据标题判断。"), in normal,
  emergency and rescue rendering, validated like `omittedByTopic`; absent
  when 0.
- kind: behavior
- first signal: Red — assembler, rescue and summarizer tests expecting the
  line (and its absence at 0).
- checks: same tests Green; report-rendering suites Green.
- [x] accepted on observed evidence — Red 30/213 in four target files;
  Green: core 2608, root typecheck 0, lint 0 errors, CLI 457/457, full plugin
  1048/1048; no existing expectation changed. Line sits after the fallback
  line, before the omitted line; byte-identical output at 0. Rescue
  `fixedPrefix`/`counts` carry it (whole-string postflight). Report parsers
  return identical results with and without the line (tested).

### Chunk 5 — run-history and host surfaces

- change: run-history records the count as an optional field and the shared
  formatter shows it (plugin history views); the CLI run output line mentions
  it; nothing marks the run failed.
- kind: behavior
- first signal: Red — run-history/recorder/scheduler-driver tests and CLI
  run-output test expecting the count; plugin history display test.
- checks: same tests Green; CLI and plugin suites Green.
- [x] accepted on observed evidence — Red core 17/46, CLI 2/38, plugin 3/46;
  Green: core 2628, CLI 459/459, plugin 1053/1053, root typecheck 0, lint 0
  errors (16 baseline warnings). History records `{count,total}` whenever
  present (0 included; formatter hides 0 → `titleOnly=3/40`); malformed
  values dropped, record kept. CLI `(2 papers written; 3/40 candidates judged
  by title only)`, plugin `describeResult` same suffix (English-only, as
  before); unchanged at 0.

### Chunk 6 — verification fixes (added after phase verification, L1)

- change: (1) reword the new `arxiv-fetcher.ts` comments so
  `check:boundaries` (regex on `process|Buffer`) passes; (2) any 429 from the
  query API, with or without Retry-After, is not retried and later API
  requests during that host's cooldown fail fast instead of waiting inline,
  so the abs-page fallback starts at once (verification showed a Retry-After
  120 s API 429 was retried at 0/120/240 s, delaying runs ~4 min, up to
  30 min per wait). arxiv.org Retry-After handling is unchanged. Commit the
  e2e test `packages/core/tests/pipeline/abstract-throttle-e2e.test.ts`
  with scenario (b) updated.
- kind: bug fix
- first signal: Red — fetcher test: API 429 with Retry-After makes exactly one
  request and a second API call within the cooldown fails fast without
  waiting; e2e (b) expects one API request; `npm run check:boundaries` fails.
- checks: same tests Green; `check:boundaries` passes; core suite.
- [x] accepted on observed evidence — Red: new fetcher test + e2e (b) (3 API
  requests vs 1), `check:boundaries` failing; Green: 51/51 on both files
  (re-run by owner), `check:boundaries` OK, core 2635, typecheck 0, lint 0
  errors, export script 15/15 (re-run by owner; its old "429 is retried"
  expectation now expects one API call plus abs fallback), CLI runtime 7/7.
  A Retry-After > 30 min on the API still surfaces as deferred on the first
  call.

## Phase verification

- Core end-to-end test through the pipeline with fake HTTP: query API 429
  (with and without Retry-After) and abs pages 200 → filter sees abstracts,
  decision record `hadAbstract` true, no header line; abs pages failing →
  header line and run-history count present, run completed.
- Local "Root workspace verification" steps from `.github/workflows/lint.yml`.
- With user consent (costs model calls): one real daily run in a temp config
  with the query API forced to fail (e.g. fake/blocked host), checking
  `hadAbstract` is mostly true and the header line matches expectations.

Results (2026-10-16):

- E2E (`tests/pipeline/abstract-throttle-e2e.test.ts`, real fetcher/adapter/
  content fetcher, fake HTTP): 6/6 — header-less and Retry-After API 429 →
  one API request, abs abstracts in the filter prompt, `hadAbstract` all true,
  no header line, abs cache hit afterwards; abs pages failing → stop after 3,
  header line with correct N/M, run-history `{6,6}`, run completed.
- Local root verification at 0657b5e: all 13 steps passed (release-tools 883,
  core 2635, node-runtime 73, CLI 459, plugin 1053, settings UI 172,
  boundaries OK, lint 0 errors, build/submission/smoke OK).
- Real daily run 2026-10-15 (user-approved; temp config, CO+GA, cap 3,
  gpt-6.1-sol, export.arxiv.org forced to 429): exit 0, 268 s; 1 API request,
  68 real abs requests all 200; decision record 68/68 `hadAbstract` true;
  report has 3 structured papers and no title-only line; run-history
  `titleOnlyCandidates {0,68}`; 4 LLM calls; temp key deleted, saved files
  key-free. The failing-fallback header path was exercised only by the e2e
  test. The "abs-page fallback filled X/Y" line is info-level and hidden in
  default CLI output (follow-up candidate).

## Abort / reshape triggers

- If abs pages also get throttled under the fallback's pace, stop and
  reshape (L2) rather than raising retries.
- If per-host cooldown requires reworking the coordinator beyond a keyed
  cooldown map, reconsider the approach before continuing.
