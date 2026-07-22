# Daily summary fallback

status: done
updated: 2026-07-22

## Intent

Keep daily reports complete and useful when individual summary generation fails, while preserving deterministic output, trusted metadata, and explicit machine-readable fallback state.

## Success criteria

- [x] Every selected paper is represented in input order by either a validated structured summary or a safe localized fallback block.
- [x] Invalid daily inputs fail before generation, and retryable validation/generation failures follow bounded recovery paths.
- [x] A final rescue path can still produce and commit a complete daily report without misindexing fallback content as generated summaries.
- [x] Focused regressions, core typecheck, and repository verification pass without weakening cancellation or metrics behavior.

## Non-goals

- Changing paper filtering, detail-note generation, or source extraction.
- Introducing concurrent daily paper generation.
- Treating fallback abstracts as generated `PaperSummary` data.

## Constraints

- Daily paper processing remains sequential and deterministic.
- Titles, authors, IDs, categories, source labels, links, and fallback abstracts come only from trusted pipeline data.
- Untrusted fallback text cannot inject Markdown structure or machine-readable markers.
- Implement and activate one phase at a time; do not modify the closed sequential-daily-summaries initiative.

## Phases

1. P1 — Paired paper slots support validated deterministic structured or safe fallback rendering and parsing — status: done
2. P2 — Per-paper validation uses max total 3 logical calls and yields typed fallback on validation/transport exhaustion — status: done
3. P3 — Unexpected normal deterministic assembler failure invokes a bounded strict Rescue LLM assembler using compact paired slots — status: done
4. P4 — Rescue exhaustion or transient failure falls back to a minimal deterministic emergency report with verified pipeline commit/index behavior — status: done

## Current focus

Done — P4 follow-up audit hardening and complete repository verification passed.
