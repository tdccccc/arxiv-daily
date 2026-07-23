# P4 — prompt-primary generation

goal_ref: ../goal.md
status: done

## Outcome

Bilingual daily-summary system and correction prompts make first-pass Obsidian `$...$` the happy path via short rules and concrete good/bad formula examples, without changing acceptance semantics.

## Assumptions

- The uncommitted prompt-contract-consistency split into short bullet rules is a useful base and can be extended in place.
- A few domain-realistic examples (astro / ML-style quantities, expectations, comparisons) teach better than more prohibition text.
- Correction guidance should restate only the failed contract in short trusted language; it should not become a second system prompt or dump internal issue codes.
- Detail-note prompts may stay lightly aligned on delimiters/examples, but daily structured fields remain the acceptance target for this phase.

## Approach

Treat prompts as the primary quality lever. Extend bilingual daily system prompts and `buildCorrectionGuidance` with compact good/bad examples that encode: inline `$...$` only; no split of one formula; `\langle...\rangle` for averages/expectations; bare comparisons OK; no bare TeX; no `\(...\)` / `\[...\]` / `$$...$$` in daily semantic fields. Keep scanner and parse/retry behavior unchanged in P4 so we can attribute gains to the prompt contract. Add focused prompt/message-capture tests before editing prompts.

## Tasks

- [x] Add failing tests that assert daily system prompts (zh/en) contain short math rules **and** concrete good/bad examples for `$...$`, expectation brackets, comparisons, and “do not split one formula.”
- [x] Add failing tests that assert correction guidance for `invalid-math` is short, language-matched, example-bearing or rule-restating, and does not dump untrusted raw model text.
- [x] Rewrite bilingual daily-summary system math sections to prompt-primary form: keep the existing short rules; add 3–5 tight good/bad examples; avoid long prose.
- [x] Tighten `buildCorrectionGuidance` math branch the same way (short trusted reason + one corrective example pattern), without changing reason-code trust boundaries.
- [x] Optionally mirror the same example set into detail prompts for delimiter consistency only; do not add a detail math scanner.
- [x] Run focused daily-paper-summary / summarizer / prompt tests, core typecheck, and `git diff --check`. Leave scanner code untouched unless a test forces a pure prompt-path fix.

## Verification

- Focused tests prove system + correction prompts carry the example-driven contract in both languages.
- `npm test -w @arxiv-daily/core -- --run tests/daily-paper-summary.test.ts tests/summarizer.test.ts` — 56 passed.
- `npm run typecheck -w @arxiv-daily/core` and `git diff --check` pass.
- No change to `canonicalizeScientificMarkdownMath` acceptance semantics, attempt count, or fallback reasons in this phase.

## Abort / reshape triggers

- If examples bloat the prompt past usefulness or fight injection-guard / paper_data boundaries, stop and reshape to a smaller shared math-examples fragment.
- If making correction “example-bearing” requires replaying model output or untrusted diagnostics, stop and keep trusted reason codes only.
- If work starts needing scanner rewrites to pass tests, stop — that belongs in P5, not P4.
