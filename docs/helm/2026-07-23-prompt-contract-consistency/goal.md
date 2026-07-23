# Prompt contract consistency

status: done
updated: 2026-07-23

## Intent

Align detail-note and daily-summary prompt contracts so scientific math renders reliably, English prompts receive an English injection guard, and dense math instructions are presented as short, independently testable rules.

## Success criteria

- [x] Chinese and English detail-note prompts prohibit unsupported math delimiters and bare HTML-shaped angle brackets while retaining supported display math.
- [x] Every prompt path injects a language-appropriate guard without weakening the existing paper-data boundary.
- [x] Daily-summary system and correction prompts express the existing math contract as short separate rules with unchanged validation behavior.
- [x] Focused tests, core typecheck, and diff checks pass.

## Non-goals

- Adding a detail-summary math scanner or rewriting existing Vault notes.
- Changing paper-filter language, daily-summary validation semantics, rendering, retry, fallback, or deployment behavior.

## Constraints

- Work only in the existing feature worktree and do not read or modify test-Vault notes, indexes, settings, caches, or reports.
- Preserve the current detail-note allowance for standalone `$$...$$`; daily-summary semantic fields remain inline-only.
- Use prompt-contract tests before implementation and do not push.

## Phases

1. P1 — Detail and daily prompt contracts are language-consistent, math-safe, and regression-tested — status: done

## Current focus

Done — focused prompt tests (107), the full core suite (732), core typecheck, and diff checks passed.
