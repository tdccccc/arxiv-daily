# Execution Record

## Baseline

- Base commit: `e8aab4b`
- Branch: `feat/operations-metrics-search`
- Worktree: `.worktree/operations-metrics-search`
- Initial dependency install: `npm ci` passed, 0 vulnerabilities.

## Progress

- [x] Created isolated worktree.
- [x] Created task documentation.
- [x] Implemented secret safety.
- [x] Implemented generation metrics.
- [x] Implemented unified operations.
- [x] Implemented BM25 search and similar papers.
- [x] Updated user documentation.
- [x] Completed automated validation.
- [x] Completed isolated Obsidian verification.

## Implementation notes

### Phase 1 — Secret safety

- Added Core text, URL, value, and error redaction with exact-secret replacement plus common bearer/key patterns.
- Logger now sanitizes console arguments, notices, and buffered entries; changing configured secrets also re-sanitizes the existing buffer.
- LLM status/model errors, diagnostic reports and fallback presentation, and CLI output are sanitized with the configured key.
- The settings key control renders only a `Configured` sentinel for saved keys. Replace/Save/Cancel/Clear are explicit actions; rendering and cancellation do not persist, Clear requires confirmation, and logger secrets update after replacement or clearing.
- Kept `llm.apiKey`, plugin `data.json`, and CLI config/environment behavior unchanged.

### Phase 3 — Unified operations

- Added a host-neutral `OperationRegistry` with unique operation IDs, immutable active snapshots, cancellation-requested state, safe subscriptions, individual/global cancellation, and idempotent finish.
- Scheduler batches now expose one daily-run operation while retaining per-date `RunLock`, state, and history behavior. Signals cover recent-date refresh, arXiv HTML/Atom/source/PDF/binary requests, rate and retry delays, paper-content fallbacks, six-worker content fetch, manual detail generation, and PDF downloads.
- Manual detail and PDF tasks are registry-owned in the plugin, reject duplicate same-ID work, and retain cancelling state until cooperative unwind. PDF write plus index update is treated as a commit boundary after the final cancellation check.
- Obsidian `requestUrl` performs cooperative pre/post checks without changing transport. The stable `cancel-current-run` command is presented as `Cancel active tasks`; Dashboard activity/menu state and unload use the shared registry.
- CLI runtime shares the registry. First SIGINT/SIGTERM requests cooperative cancellation and waits; a second signal exits immediately.

### Phase 4 — Local retrieval

- Added a host-neutral, dependency-free `PaperSearchIndex` with deterministic NFKC/lowercase English, technical-hyphen, Han-bigram, and mixed-language tokenization plus canonical modern arXiv ID/version/URL handling.
- Search uses weighted BM25F-style field scoring, AND query clauses, exact/partial ID priority, deterministic reasons and ties, deduplicated topic/category fields, and bounded high-value OR terms for local similarity.
- Dashboard blank-search behavior is unchanged. Nonblank searches default to relevance, explicit sorts remain primary, and compact match reasons render only for active relevance search. The index is cached after history reload and reused across page/sort/filter/star/status changes; legacy substring matching remains the failure fallback.
- Added a separate accessible Similar Papers modal and row action. It ranks at most ten non-ignored local candidates, excludes the source, applies no category gate, displays textual local reasons rather than percentages, and delegates detail/daily/arXiv/PDF actions to existing Dashboard methods.
- Kept Paper Index schema and persistence unchanged; ranking performs no storage or network work.

### Phase 2 — Generation metrics

- Added host-neutral token/call metrics types and collector. OpenAI `prompt_tokens`/`completion_tokens` and common input/output aliases are parsed from SSE usage chunks.
- Streaming requests ask for `stream_options.include_usage`; fallback removes only that option and only for explicit unsupported-option HTTP 400/422 responses.
- Metrics distinguish logical calls, HTTP attempts, LLM elapsed time, usage completeness, and pipeline wall time. Retried generations are marked incomplete because failed-attempt usage is unavailable.
- Daily metrics aggregate filter, every daily batch, and newly generated details. Detail notes receive only their own detail call metrics; manual and pipeline detail paths use the same writer option.
- Writers append a folded callout at the absolute Markdown end under `<!-- arxiv-daily:generation-metrics -->`. Omitting metrics keeps prior writer output unchanged and frontmatter is untouched.
- Daily summary extraction and detail-summary detection strip the metrics suffix before parsing.

## Integration review findings and fixes

- HIGH cancellation state: confirmed that user cancellation persisted `skipped`, which is terminal under the existing schema. Cancelled runs now restore `pending` with the cancellation reason, while policy/guard skips retain existing `skipped` semantics. History remains schema 1 and records `resultKind: cancelled` as a `pending` event, so normal scheduler and manual retry remain available without adding an incompatible status.
- HIGH daily note/index boundary: confirmed that an existing daily Markdown file bypassed Paper Index synchronization after cancellation or an index write failure. Existing daily notes are now treated as the durable Markdown commit and trigger an idempotent repair that reads old or new note content, derives arXiv IDs and summaries, and reruns both `addDailyReports` and `setSummaries`. This covers cancellation after `writeDaily`, either index operation failing, and partial prior synchronization without a new marker or schema change.
- MEDIUM RecentDatesCache signals: confirmed that the first caller's signal owned the shared refresh. Shared refresh lifetime is now signal-independent; every caller gets a prompt cancellable wait, in either caller ordering, with listener cleanup and observed background promises. Host `requestUrl` work remains non-interruptible and may continue after all callers stop waiting.
- MEDIUM manual frontmatter-only replacement: confirmed a cancellation check between removal and writing could delete the original. Removal was eliminated; the final cancellation check now precedes a non-interruptible writer commit, and the writer's existing atomic temp/backup path explicitly replaces the frontmatter-only note.
- LOW Similar Papers rejection: confirmed action callbacks could reject without a handler. Modal dispatch now catches synchronous throws and rejected promises and delegates to an injected Dashboard handler that uses the redacting logger and a Notice.

### Phase 5 — Documentation

- Updated the root English README, Chinese README, plugin README, and English/Chinese getting-started guides for the verified Phase 1-4 behavior.
- Documented cancellation scope and cooperative `requestUrl` limits; the configured-key sentinel and plaintext local `data.json`; redacted logs/diagnostics/errors; folded timing/call/provider-token metrics and missing-usage semantics; local ranked search and explicit sort behavior; local Similar Papers retrieval and actions; and compatibility without a Paper Index migration.
- Corrected stale public wording that implied ordinary keyword/full-text search, omitted relevance sorting and Similar Papers, or described the shared Dashboard model as intended for a future VS Code Webview. No VS Code or generated bundle documentation was changed.
- Documentation validation passed: local Markdown links in the seven changed documentation files resolve, required Phase 1-4 concepts are present across the public docs, and `git diff --check` passes.
- Final automated validation and isolated Obsidian verification were completed after documentation updates.

## Deviations

None recorded.

## Validation

Phase 1/2 validation:

- Core full suite: 46 files, 395 tests passed.
- Focused Core safety/metrics/parser/writer/pipeline/manual-fetch suite: 9 files, 78 tests passed; final safety/metrics subset: 6 files, 42 tests passed.
- Focused plugin settings/commands/dashboard presentation suite: 3 files, 48 tests passed.
- Focused CLI config/runtime/main suite: 3 files, 13 tests passed; final CLI main rerun: 7 tests passed.
- Core, plugin, and CLI TypeScript checks passed after final edits.
- `git diff --check` passed.

Phase 3 validation:

- Focused Core operation/scheduler/manual-fetch/PDF suite: 4 files, 62 tests passed.
- Full Core suite: 47 files, 398 tests passed.
- Full plugin suite: 16 files, 153 tests passed; affected command/status/dashboard rerun: 3 files, 45 tests passed.
- Full CLI suite: 3 files, 13 tests passed.
- All workspace TypeScript checks and `check:boundaries` passed.
- `git diff --check` passed; no VS Code extension files changed.
- Build, audit, release-version, smoke, and isolated Obsidian checks remain for later phases.

Phase 4 validation:

- Focused Core retrieval suite: 1 file, 12 tests passed; full Core suite: 48 files, 410 tests passed.
- Focused plugin Dashboard/retrieval/modal suite: 3 files, 42 tests passed; full plugin suite: 17 files, 159 tests passed.
- All workspace TypeScript checks and `check:boundaries` passed.
- `git diff --check` passed; no generated bundle or VS Code files changed.

Integration-review fix validation:

- Focused Core scheduler/pipeline/manual-fetch/recent-dates/writer suite: 5 files, 117 tests passed; final pipeline/recent-dates rerun: 2 files, 34 tests passed.
- Focused Similar Papers modal suite: 1 file, 3 tests passed.
- Full Core suite: 48 files, 419 tests passed; full plugin suite: 17 files, 160 tests passed; full CLI suite: 3 files, 13 tests passed.
- All workspace TypeScript checks, `check:boundaries`, `check:release-version -- 0.2.1`, production builds, and `smoke:build` passed.
- `git diff --check` passed. Generated bundles and VS Code files are not in the working-tree diff.
- Dependency audit was not rerun during the read-only review; the final validation below reran it successfully.

Final feature-worktree validation:

- `npm audit`: 0 vulnerabilities.
- Workspace boundaries and release-version consistency for `0.2.1`: passed.
- All workspace TypeScript checks: passed.
- Core: 48 files, 419 tests passed.
- Node runtime: 1 file, 8 tests passed.
- CLI: 3 files, 13 tests passed.
- Plugin: 17 files, 160 tests passed.
- Total: 69 files, 600 tests passed.
- Production plugin/CLI builds and `smoke:build`: passed; plugin bundle size was 315.0 kB.
- `git diff --check`: passed.
- No VS Code extension files or generated bundles are present in the working-tree diff.

Isolated Obsidian 1.12.7 verification:

- Installed only into `/home/tiandc/Documents/code/arxiv-daily/plugin_test`; previous assets were backed up at `/tmp/arxiv-daily-plugin-test-before-operations-metrics-search-20260716-230650`.
- Plugin `0.2.1` reloaded, Dashboard opened, 20 commands registered, and `Cancel active tasks` was present under the stable command ID.
- Custom categories remained `astro-ph.CO` and `astro-ph.GA` after reload.
- API-key settings rendered one read-only `Configured` sentinel, no password input contained a value, Replace/Clear controls were present, and merely opening Settings did not change the `data.json` checksum.
- Dashboard rendered the Similar Papers action; the modal opened locally with ten results and no active operations or network task.
- `dev:errors` reported `No errors captured.`
- No paid LLM/arXiv/PDF operation was launched for this verification.
