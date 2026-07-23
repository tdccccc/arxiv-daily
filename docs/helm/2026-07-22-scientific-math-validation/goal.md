# Scientific math validation

status: done
updated: 2026-07-23

## Intent

Make the LLM emit Obsidian-renderable scientific formulas on the first pass through a clear prompt contract. Keep the script as a thin acceptance net: reject unsafe or malformed math, apply only a few unambiguous normalizations, and use the existing per-paper retry/fallback only when acceptance fails.

## Success criteria

- [x] Bilingual daily-summary system and correction prompts use short rules plus concrete good/bad formula examples so first-pass `$...$` is the intended happy path (not scanner-driven teaching).
- [x] The acceptance boundary is thin and explicit: valid `$...$` stays byte-stable; only agreed unambiguous rewrites run; everything else is reject-with-diagnostics (no boundary guessing, no thick auto-repair).
- [x] Retry (max 3) and per-paper fallback remain the only recovery path, fire only on real acceptance failures, and correction text stays short trusted guidance rather than a dump of scanner internals.
- [x] Tests (and lightweight attempt/fallback signals where already available) show the design is prompt-primary: invalid-math is rare relative to first-pass accept, and over-rejection is preferred to silent mis-rewrite.
- [x] Full core verification passes; only `main.js`, `manifest.json`, and `styles.css` are deployed to the isolated test Vault when this initiative is closed.

## Non-goals

- Growing the scanner with open-ended edge-case issue codes as the primary quality strategy.
- Rewriting existing Vault reports or guessing math boundaries around bare TeX.
- Changing the scientific Markdown renderer contract, assembler architecture, concurrency, transport retries, detail-note display-math allowance (`$$...$$`), canonical IDs, links, or closed Helm initiatives.
- Replacing the existing three-attempt validation / per-paper fallback boundary with a new recovery system.

## Constraints

- Work only in the existing feature worktree (`sequential-daily-summaries`); preserve unrelated uncommitted work; do not commit or push unless explicitly asked.
- Validation failures must remain `DailyPaperSummaryValidationError` so permanent provider/configuration errors and cancellation keep current behavior.
- Do not read or overwrite test-Vault `data.json`, `.cache/`, indexes, notes, or daily reports; deploy only the three approved plugin assets after verification.
- Prefer prompt and contract changes over new scanner heuristics; any scanner change must shrink or clarify responsibility, not expand “smart” repair.
- Use subagents for substantial implementation and independent review when execution starts.

## Phases

1. P1 — A tested conservative single-line scientific-math scanner exists as the acceptance primitive — status: done
2. P2 — Per-paper parse boundary, retries, assembly, and persistence enforce and preserve accepted math — status: done
3. P3 — Thick scanner-audit-as-quality-path and deploy under the old primary strategy — status: superseded
4. P4 — Prompt-primary generation: short bilingual rules + concrete good/bad examples drive first-pass renderable math — status: done
5. P5 — Thin script fallback: documented accept-or-reject policy, minimal unambiguous rewrites, short trusted correction — status: done
6. P6 — Measure, independently audit, fully verify, and deploy the three approved plugin assets — status: done

## Current focus

Done

## Open questions

- None. Rewrite policy remains: only explicitly delimited `\(...\)` and simple one-line `\[...\]` → `$...$`; daily semantic fields inline-only; detail `$$...$$` out of daily scanner.
