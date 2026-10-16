# Daily report quality fixes

status: done
created: 2026-10-15T16:46:40+08:00
updated: 2026-10-16T01:10:00+08:00
revision: 5
owner: current-session

## Intent

Fix two verified causes of poor daily reports: structured summaries that fall
back to the English abstract because the model writes single-backslash LaTeX
inside JSON, and filtering that silently sees titles only when the arXiv query
API is throttled.

## Success criteria

- [x] A summary response that writes LaTeX with single backslashes (the real
      2610.16741 responses in `summary-repro/out-A.json`) yields a structured
      summary with its formulas intact instead of `validation-exhausted`.
- [x] `\times`, `\rm`, `\beta`, `\frac`, `\nu` written with one backslash keep
      their LaTeX instead of turning into control characters.
- [x] An `invalid-json` failure logs the concrete parse error, and the retry
      tells the model why it failed and to double backslashes while keeping
      LaTeX.
- [x] A real-model rerun of 2610.16741 (with user consent) gives a Chinese
      structured summary that keeps its formulas.
- [x] When the arXiv query API fails or is throttled, papers still lacking an
      abstract get one from their `arxiv.org/abs/<id>` page, within a daily cap,
      cancellable, stopping after repeated failures, and reusing the abs-page
      cache.
- [x] When some papers were filtered without an abstract, the daily report
      header and run-history say how many, in both CLI and Obsidian, without
      marking the run failed; nothing is shown when none are missing.
- [x] A 429 without `Retry-After` no longer triggers short 2–4 s retries.
- [x] Each phase passes the local "Root workspace verification" steps from
      `.github/workflows/lint.yml`.

## Non-goals

- Switching to `response_format` / structured outputs.
- Routing CLI requests through `HTTPS_PROXY`.
- Changing which papers are candidates or how the filter decides.

## Constraints

- Work only in `/home/tiandc/Documents/code/arxiv-daily-quality-fixes` on
  `fix/daily-quality-fixes` (based on `feat/quality-evaluation` for
  `extractAbsPageAbstract`). Do not touch `arxiv-daily-quality-eval` or the main
  checkout.
- A future PR stacks on `feat/quality-evaluation`; retarget it to `main` before
  that base merges.
- Real model calls or real daily runs cost money: ask the user each time. Use a
  temporary `XDG_CONFIG_HOME` config (mode 0600) with the Photo-z and Galaxy
  Cluster topics; delete temporary keys afterwards; never print API keys.
- Prompt changes bump the summary prompt contract version.
- Implementation by subagents (no nested agents, report files); owner accepts.
- No push, no PR without per-instance user consent.
- Commits: checkpoint commits authorized (one per accepted chunk, multi `-m`).

## Phases

1. P1 — single-backslash LaTeX in summary JSON is repaired and diagnosable, so formula-heavy papers get structured summaries — status: done
2. P2 — abstracts survive query-API throttling via abs-page fallback, and title-only filtering is visible in report and run-history — status: done
3. P3 — today's daily run gets candidate abstracts from arXiv RSS, so the query API is only a gap filler — status: superseded (dropped by user before start; see journal)
