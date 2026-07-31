# Review issue fixes

status: active
updated: 2026-07-31

## Intent

Fix the issues found in the 2026-07-31 code review, batch by batch, without
changing product behavior — each batch stays independently mergeable.

## Success criteria

- [ ] P1 landed: dead code removed, legacy shim corrected, README security note added, `obsidian` devDep pinned — all checks green
- [ ] P2 landed: dashboard `view.ts` split into modules; settings tab migrates to `getSettingDefinitions` with a 1.4.0 fallback
- [ ] P3 resolved: CLI `detailSelection` is either configurable via TOML or explicitly documented as a fixed preset
- [ ] No pipeline output change: daily reports and paper notes are byte-identical before/after each batch

## Non-goals

- No product behavior changes (except whatever P3 decides explicitly)
- No fixing the 53 sentence-case / UI-text lint warnings (pure churn)
- No UI redesign, no new features

## Constraints

- The main worktree is in use by another agent (bug fixes): every change goes
  through a separate worktree branch; nothing touches the main checkout
- P2 waits for `fix/arxiv-failure-recovery` and `fix/arxiv-request-resilience`
  to merge into `main` (they touch pipeline.ts and dashboard code — widest
  conflict surface)
- commit-msg hook: Conventional Commits (`feat|fix|refactor|docs|test|chore|perf`),
  body with why/what/validation for multi-file or >20-line changes
- Checks must stay green: `npm run typecheck`, `npm test`, `npm run lint`
  (0 errors), `npm run check:boundaries`

## Phases

<!-- Mirror status from phases/NN-*.md. PN ↔ filename NN. Outcomes only — no steps. -->
1. P1 — Zero-risk cleanup: dead code, legacy shim, README note, pinned obsidian — status: done
2. P2 — Refactors: split dashboard view.ts; declarative settings API with fallback — status: blocked
3. P3 — Decision: CLI detailSelection configurable or documented-as-fixed — status: done

## Current focus

P2 (blocked: waits for fix/arxiv-failure-recovery and fix/arxiv-request-resilience to merge into main)

## Resolved

- P3 (2026-07-31): closed as documented-as-fixed — the CLI deliberately
  omits `detail_selection` (schema: cli-toml-schema.md lines 59–60, 178,
  186, 348, 455, 465; decision journaled 2026-07-28 in
  cli-product-config-and-data-portability)
