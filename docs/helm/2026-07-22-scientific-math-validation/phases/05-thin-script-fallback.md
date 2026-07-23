# P5 — thin script fallback

goal_ref: ../goal.md
status: done

## Outcome

The script acceptance boundary is explicitly thin: valid `$...$` is byte-stable, only agreed unambiguous delimiter rewrites run, everything else is reject-with-diagnostics into the existing retry/fallback path with short trusted correction.

## Assumptions

- P1/P2 already implement the right recovery boundary (max 3 validation attempts, per-paper fallback).
- P4 already supplies prompt-primary examples; P5 must not re-grow the scanner as a quality engine.
- Documenting and testing the rewrite policy is enough to “thin” the contract without deleting useful safety checks (bare TeX, raw HTML in math, unbalanced grouping).

## Approach

Encode the accept-or-reject policy in the scanner module as an explicit contract (allowed rewrites only: `\(...\)` and simple one-line `\[...\]` → `$...$`). Add regression tests for byte-stability, reject-without-rewrite, correction brevity/trust, and idempotence of accepted values. Do not add new issue codes or smart boundary guessing.

## Tasks

- [x] Document the thin acceptance policy and allowed rewrite class on the scanner module.
- [x] Add tests locking: accepted `$...$` byte-stable; only explicit `\(...\)`/`\[...\]` rewrites on ok; reject returns original bytes; idempotence of accepted values.
- [x] Confirm correction guidance stays short, language-matched, example-bearing, and free of untrusted model text / foreign error dumps.
- [x] Confirm retry max=3 and fallback kinds unchanged via existing daily-paper-summary tests.
- [x] Run focused math + daily-paper-summary tests, core typecheck, and `git diff --check`.

## Verification

- Focused scanner and daily-paper-summary tests pass (106).
- No new scanner issue codes; no change to attempt count or fallback reason taxonomy.

## Abort / reshape triggers

- If thinning requires removing a safety check that currently prevents report corruption, stop and record the check as required reject-only rather than deleting it.
- If tests force expanding rewrite classes beyond the agreed `\(...\)` / simple `\[...\]` pair, stop and reshape with the user.
