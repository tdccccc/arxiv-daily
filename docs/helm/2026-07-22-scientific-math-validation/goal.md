# Scientific math validation

status: active
updated: 2026-07-22

## Intent

Ensure every accepted per-paper daily-summary field uses Obsidian-compatible, structurally valid inline mathematics before it reaches report assembly or the paper index. Canonicalize unambiguous delimiters and use the existing validation retry/fallback boundary for malformed or ambiguous TeX.

## Success criteria

- [x] A conservative single-line scanner preserves valid `$...$`, canonicalizes unambiguous `\(...\)` and `\[...\]`, and rejects malformed or bare TeX without corrupting currency, code spans, links, autolinks, or escaped dollars.
- [x] Per-paper validation and bilingual prompts enforce the math contract; invalid responses use the existing maximum-three validation retries and per-paper fallback.
- [x] Canonical values remain consistent through deterministic/rescue assembly, Markdown persistence, parser projection, and PaperIndex storage.
- [ ] Full repository verification and an independent audit pass, and only the three approved plugin assets are deployed to the isolated test Vault.

## Non-goals

- Rewriting existing Vault reports or guessing math boundaries around bare TeX.
- Changing the scientific Markdown renderer, fallback abstracts, detail summaries, canonical IDs, links, concurrency, transport retries, or assembly architecture.
- Modifying closed Helm initiatives.

## Constraints

- Work only in the existing feature worktree; preserve all prior uncommitted work and do not commit or push.
- Validation must fail as `DailyPaperSummaryValidationError` so permanent provider/configuration errors and cancellation retain their current behavior.
- Do not read or overwrite test-Vault `data.json`, `.cache/`, indexes, notes, or daily reports; deploy only `main.js`, `manifest.json`, and `styles.css` after verification.
- Use subagents for substantial implementation and independent review.

## Phases

1. P1 — A tested conservative single-line scientific-math canonicalization and validation contract exists — status: done
2. P2 — Per-paper prompts, retries, assembly, and persistence enforce and preserve the canonical contract — status: done
3. P3 — The complete change is independently audited, fully verified, and deployed to the isolated test Vault — status: active

## Current focus

P3
