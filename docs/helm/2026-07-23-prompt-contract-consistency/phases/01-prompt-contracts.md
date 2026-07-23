# P1 — prompt contracts

goal_ref: ../goal.md
status: done

## Outcome

Detail and daily prompt paths give models concise, language-appropriate security and math-rendering rules without changing downstream behavior.

## Assumptions

- Prompt text and captured LLM messages are sufficient to verify language-specific guard selection.
- Detail-note display math remains valid as `$$...$$`, while `\(...\)` and `\[...\]` are unsupported output forms.

## Approach

Add focused failing prompt-contract tests, introduce an English guard and language-aware selection, then restructure the detail and daily math rules without changing scanner or renderer code.

## Tasks

- [x] Add failing tests for detail math rules, bilingual guard selection, and separated daily correction rules.
- [x] Add `injection-guard.en.md` and select guards by prompt language at all four call sites.
- [x] Strengthen both detail prompts while retaining standalone display math.
- [x] Split daily system/correction math contracts into short rules in both languages.
- [x] Run focused tests, core suite/typecheck, and diff checks.

## Verification

- Focused prompt, detail-selector, paper-filter, daily-paper-summary, and summarizer tests pass.
- `npm test -w @arxiv-daily/core`, `npm run typecheck -w @arxiv-daily/core`, and `git diff --check` pass.
- English prompt captures contain the English guard and exclude the Chinese guard; Chinese paths retain the Chinese guard.

## Abort / reshape triggers

- If a shared language choice cannot be made without changing public APIs, keep fixed-language prompts explicit rather than widening interfaces.
- If prompt restructuring changes scanner, renderer, retry, fallback, or detail output formats, stop and reshape instead of expanding scope.
