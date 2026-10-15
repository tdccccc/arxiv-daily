# P1 — summary-json-backslashes

goal_ref: ../goal.md
created: 2026-10-15T16:46:40+08:00
updated: 2026-10-15T16:46:40+08:00
revision: 1

## Outcome

Summary responses whose JSON strings carry single-backslash LaTeX parse into
structured summaries with intact formulas, and any remaining `invalid-json`
failure is logged with its cause and retried with a reasoned correction.

## Assumptions

- Semantic summary fields never legitimately contain newline, tab, carriage
  return, backspace or form feed: `canonicalizeScientificMarkdownMath` rejects
  `\n`/`\r` (MULTILINE → `invalid-math`) and rendering compacts all whitespace
  to one line. So a JSON escape `\n`, `\t`, `\r`, `\b`, `\f` immediately
  followed by an ASCII letter is LaTeX (`\nu`, `\neq`, `\nabla`, `\times`,
  `\rm`, `\beta`, `\frac`), not a control character. Trade-off: a genuine
  newline directly followed by a word becomes a literal `\n…` text instead of
  a MULTILINE rejection; that is acceptable because such newlines were already
  invalid and the math validator still checks the resulting TeX.
- The three attempts in `~/quality-eval/investigations/summary-repro/out-A.json`
  contain no secrets (grep for key/bearer/authorization patterns: 0 hits); the
  fixture keeps only the model's raw content strings.
- `JSON.parse` messages can quote model text, so only a trusted category plus
  position goes into the retry prompt; the raw message goes to the warn log.

## Approach

Add a string-literal-aware backslash repair in front of `JSON.parse` in
`parseDailyPaperSummary` (always applied, because the `\t`/`\r`/`\b`/`\f`
cases parse "successfully" into corrupted text). Keep the parse error detail on
the validation error, surface it in the summarizer's warn log, and make the
retry correction name the cause and ask for doubled backslashes with LaTeX
kept. Prompt-file change is chunk 3 and depends on the user's decision.

## Chunks

### Chunk 1 — repair single-backslash LaTeX before JSON.parse

- change: inside JSON string literals only, a backslash not followed by
  `" \ / b f n r t u` is doubled; `\b \f \n \r \t` followed by an ASCII letter
  are doubled; `\u` not followed by 4 hex digits is doubled; correct escapes
  (`\\`, `\"`, `\n` before non-letters, `±`) and everything outside strings
  are untouched. The three real out-A responses become structured summaries.
- kind: bug fix
- first signal: Red — new tests in `packages/core/tests/daily-paper-summary.test.ts`
  fail: out-A fixture responses end as `validation-exhausted`; `\times`/`\rm`
  cases yield control characters or MULTILINE; run
  `npx vitest run tests/daily-paper-summary.test.ts` in `packages/core`.
- checks: same tests Green; edge cases (already-doubled, `\"` inside strings,
  backslash at end of string, `\u` valid/invalid, `\n` + digit/space, escape
  pairs `\\\\times`) Green; `packages/core` daily-summary tests and typecheck
  Green.
- [ ] accepted on observed evidence

### Chunk 2 — diagnosable invalid-json and reasoned retry

- change: an `invalid-json` validation error keeps the parse error message;
  the summarizer's warn log includes it; the retry correction (zh and en) states
  a trusted cause category (e.g. bad escape / truncated / unexpected token,
  with position) and explicitly says to write each backslash as two inside JSON
  strings, keep LaTeX, and not rewrite formulas as plain text. Persisted result
  shapes stay unchanged (or the result contract is bumped with tests).
- kind: bug fix
- first signal: Red — tests asserting (a) the second LLM call's user content
  contains the cause and the doubling/keep-LaTeX instruction after an
  unrepairable first response, (b) the warn log line contains the parse error
  detail.
- checks: same tests Green; existing correction-guidance and summarizer tests
  Green.
- [ ] accepted on observed evidence

### Chunk 3 — prompt hint and contract bump (pending user decision)

- change: if approved, one escape-rule line in
  `daily-paper-summary.system.md` and `.en.system.md` that keeps LaTeX, plus
  `DAILY_SUMMARY_PROMPT_CONTRACT_VERSION` 2 → 3 with old checkpoints treated as
  misses.
- kind: behavior
- first signal: Red — a checkpoint test showing a v2-contract checkpoint is no
  longer reused after the bump.
- checks: checkpoint and prompt tests Green.
- [ ] accepted on observed evidence

## Phase verification

- Local "Root workspace verification" steps from `.github/workflows/lint.yml`
  (audit, release-tools, boundaries, lint, typecheck, workspace tests,
  settings UI test, build, Obsidian submission, smoke build/install).
- With user consent: real-model rerun of 2610.16741 through
  `summarizeDailyPaperWithValidationRetry` (temp config, 0600) → structured
  Chinese summary with formulas kept, no fallback.

## Abort / reshape triggers

- If the repair rule would corrupt any correctly escaped real response
  (e.g. vault summaries that already parse), stop and reshape (L2).
- If real-model output still falls back for a non-escape reason, record it and
  steer instead of widening the repair.
