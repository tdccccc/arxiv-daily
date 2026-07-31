# Review issue fixes

status: active
updated: 2026-07-31

## Intent

Fix the issues found in the 2026-07-31 code review, batch by batch, keeping
each batch independently mergeable. P2b may include the user-approved settings
UX changes and a Medium reasoning default for new or incomplete configurations.

## Success criteria

- [x] P1 landed: dead code removed, legacy shim corrected, README security note added, `obsidian` devDep pinned — all checks green
- [x] P2a landed: dashboard `view.ts` split into modules; view class + re-exports remain; all checks green
- [ ] P2b landed: settings tab migrates to `getSettingDefinitions` with a 1.4.0 fallback
- [x] P3 resolved: CLI `detailSelection` documented-as-fixed (schema + journal)
- [x] Deterministic pipeline rendering unchanged: with identical fetched data and LLM responses, daily reports and paper notes remain byte-identical; P2b's user-approved Medium default may change live LLM responses for new/incomplete configurations

## Non-goals

- No product behavior changes beyond P3 decisions and the user-approved P2b settings UX / Medium new-install reasoning default
- No fixing the 53 sentence-case / UI-text lint warnings (pure churn)
- No UI redesign, no new features

## Constraints

- Every change goes through a separate worktree branch; nothing touches the
  main checkout directly
- P2a waits for `fix/arxiv-failure-recovery` and `fix/arxiv-request-resilience`
  merged into main — **done 2026-07-31 (be4c705)**; P2a started on
  `refactor/settings-declarative-api`
- commit-msg hook: Conventional Commits (`feat|fix|refactor|docs|test|chore|perf`),
  body with why/what/validation for multi-file or >20-line changes
- Checks must stay green: `npm run typecheck`, `npm test`, `npm run lint`
  (0 errors), `npm run check:boundaries`

## Phases

<!-- Mirror status from phases/NN-*.md. PN ↔ filename NN. Outcomes only — no steps. -->
1. P1 — Zero-risk cleanup: dead code, legacy shim, README note, pinned obsidian — status: done
2. P2a — Split dashboard view.ts into modules (helpers + HubModal out; class + re-exports stay) — status: done
3. P2b — Settings tab getSettingDefinitions migration with 1.4.0 fallback — status: done (pending PR merge; landed criterion below)
4. P3 — Decision: CLI detailSelection configurable or documented-as-fixed — status: done

## Current focus

P2b (started 2026-07-31; branch re-cut 2026-07-31 to
  refactor/settings-declarative-api after PR #4 merged; T1–T7 committed
  2026-07-31, second manual-test UI follow-up in progress — pending another 1.13+
  manual pass, then PR review + merge; the "P2b landed" success criterion
  checks only after the merge, mirroring P2a)

## Resolved

- P3 (2026-07-31): closed as documented-as-fixed — the CLI deliberately
  omits `detail_selection` (schema: cli-toml-schema.md lines 59–60, 178,
  186, 348, 455, 465; decision journaled 2026-07-28 in
  cli-product-config-and-data-portability)
- Cleanup 2026-07-31: deleted abandoned Rust-rewrite branches
  (`feat/ui-polish` ⊂ `refactor/rust-standalone`, local + origin); no main
  traces, no helm tracking — dead experiment from 2026-07-13..15
