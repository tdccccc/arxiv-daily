# Implementation Plan

## Phase 1 — Secret safety

- Add host-neutral text, URL, and error redaction.
- Sanitize Logger console/buffer output, LLM provider errors, diagnostics, and CLI stderr.
- Replace the API-key field with explicit unchanged/edit/save/cancel/clear states.
- Never assign the real saved key to an input value.
- Keep the existing settings schema and CLI environment/config behavior.

## Phase 2 — Generation metrics

- Parse OpenAI-compatible streamed usage and common token aliases.
- Request `stream_options.include_usage` with a narrow unsupported-option fallback.
- Record logical calls, HTTP attempts, LLM elapsed time, and reported token totals.
- Aggregate filter, daily batches, and generated details for a daily run.
- Append folded callouts to daily and detail Markdown without changing frontmatter.
- Add a stable marker and keep parsers from consuming the callout as summary text.

## Phase 3 — Unified operations

- Add a Core `OperationRegistry` for daily runs, detail summaries, and PDF downloads.
- Register one operation for each user-visible batch/task.
- Propagate signals through recent-date refresh, manual fetch, paper content, arXiv HTML/source/PDF, retries, delays, and content workers.
- Preserve coherent storage commit boundaries.
- Rename the visible command to `Cancel active tasks` without changing its ID.
- Make Dashboard/status/unload lifecycle operation-aware.
- Add cooperative SIGINT/SIGTERM handling to the CLI.

## Phase 4 — Local retrieval

- Add deterministic English/Chinese tokenization and canonical arXiv-ID handling.
- Build a weighted, host-neutral BM25-style in-memory index.
- Use it for Dashboard filtering and relevance ranking while preserving explicit sorts.
- Add compact match explanations.
- Add a separate Similar Papers modal and row action over local non-ignored entries.
- Do not change Paper Index schema or persist the derived index.

## Phase 5 — Documentation and validation

- Update user-facing English and Chinese documentation where behavior changes.
- Record implementation details and deviations in `execution.md`.
- Run dependency audit, boundaries, release consistency, typechecks, tests, builds, smoke tests, and whitespace checks.
- Install only into the isolated `plugin_test` Vault for Obsidian reload and UI/error verification.

## Compatibility requirements

- Existing `data.json` and `llm.apiKey` load without migration.
- Paper Index schema 1/2 remains unchanged.
- Existing reports and detail notes remain valid.
- Metrics are omitted when not supplied and never alter old writer outputs.
- Command ID and release assets remain stable.
- VS Code is untouched.

## Test matrix

- Redaction patterns, exact secrets, URLs, errors, Logger, diagnostics, CLI output.
- API-key DOM safety and Replace/Cancel/Clear persistence.
- Usage parsing, provider omission, unsupported stream options, retries, aggregation, writer output, parser compatibility.
- Operation registry ownership, batch cancellation, manual detail, PDF consistency, signal propagation, unload, CLI signals.
- Tokenization, ID normalization, BM25 weighting, AND/OR semantics, relevance sorting, explicit sorting, similarity filtering/reasons.
- Similar Papers modal callbacks, accessibility, and responsive styles.

## Validation commands

```bash
npm audit
npm run check:boundaries
npm run check:release-version -- 0.2.1
npm run typecheck
npm test
npm run build
npm run smoke:build
git diff --check
```
