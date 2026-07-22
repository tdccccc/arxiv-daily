# P1 — math contract

goal_ref: ../goal.md
status: done

## Outcome

A pure, deterministic scanner canonicalizes unambiguous single-line math and returns stable validation diagnostics for unsafe or malformed scientific Markdown.

## Assumptions

- Daily structured semantic fields need inline math only; display environments are invalid in this one-line format.
- High-confidence bare TeX should trigger an LLM correction rather than be automatically wrapped in guessed delimiters.
- Existing CommonMark code-span behavior can be matched without changing the shared report renderer.

## Approach

Implement an escape-aware left-to-right scanner local to the structured daily-summary boundary. Treat code spans, links, autolinks, escaped dollars, and ordinary currency as protected or non-math content; validate delimiters and basic TeX structure only where mathematical intent is explicit.

## Tasks

- [x] Add the pure canonicalization/validation module with stable issue codes and safe diagnostics.
- [x] Preserve valid `$...$`; convert valid `\(...\)` and simple one-line `\[...\]` to `$...$`.
- [x] Reject display math, bare TeX, malformed delimiters, unbalanced grouping, and unmatched `\left`/`\right`.
- [x] Protect code spans, Markdown destinations, autolinks, escaped dollars, ordinary currency, and valid interval notation without opening validation bypasses.
- [x] Add focused unit tests, including reduced July 20/21 regressions, idempotence, multi-currency prose, intervals, broader bare TeX, and exact Markdown boundaries.
- [x] Run the focused test and core typecheck, then checkpoint P1 after audit findings are resolved.

## Verification

- `npm test -w @arxiv-daily/core -- --run packages/core/tests/scientific-markdown-math.test.ts`
- `npm run typecheck -w @arxiv-daily/core`
- `git diff --check`
- Valid and protected inputs remain byte-stable; invalid inputs return deterministic issue codes and do not receive guessed repairs.

## Abort / reshape triggers

- If currency cannot be distinguished conservatively from numeric math without semantic guessing, leave currency unchanged and narrow dollar validation rather than rewriting it.
- If integration concerns require renderer/parser changes, stop and reshape because P1 must remain a pure structured-field contract.
