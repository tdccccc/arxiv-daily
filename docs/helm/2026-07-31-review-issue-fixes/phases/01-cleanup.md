# P1 — Zero-risk cleanup

<!-- Filename must be NN-<slug>.md with NN = N (e.g. P1 → 01-auth.md). -->
goal_ref: ../goal.md
status: done

## Outcome

Dead code and the legacy shim are cleaned up, the README carries the
`config.toml` permissions note, `obsidian` is pinned, and every workspace
check is green with unchanged pipeline output.

## Assumptions

- `dateWindowNote` (pipeline.ts:152) and `appendUnique` (pipeline.ts:664) are
  dead — confirmed by grep, but re-verify after the other agent's branch
  merges (they may touch pipeline.ts)
- The writer-side `dateWindowNote` option is a **tested dormant capability**
  ("submitted-date fallback notes", markdown-writer.test.ts:351) that the
  pipeline never wires; it stays in place (L1 decision, 2026-07-31)
- Removing the `dateWindowNote` option from `MarkdownWriter.writeDaily` does
  not change any produced file (callers always pass `undefined`)
- Pinning `obsidian` in the root devDependencies to the lockfile's resolved
  version is safe for CI (`npm ci` ignores the caret)

## Approach

One worktree branch (`chore/cleanup-review-issues`), one commit per item,
full check suite at the end. No behavior changes.

## Tasks

- [x] T1: delete the dead `dateWindowNote` binding and its call-site property
      in pipeline.ts (the writer option stays: tested dormant capability)
- [x] T2: delete unused `appendUnique` in pipeline.ts (keep the copies in
      paper-index.ts / manual-fetch.ts untouched)
- [x] T3: fix `arxiv_daily.py` shim — point at `apps/cli/dist/arxiv-daily-cli.cjs`,
      drop the removed `run-pending` mention
- [x] T4: README — note `chmod 600 ~/.config/arxiv-daily/config.toml` next to
      the uninstall instructions
- [x] T5: pin root devDependency `obsidian: latest` → `^1.13.1` (resolved
      lockfile version), lockfile root entry synced
- [x] T6: verify (typecheck / tests / lint / boundaries), commit per item,
      open PR #3

## Verification

- `npm run typecheck` — passes
- `npm test` — all green (core pipeline + markdown-writer tests exercise the
  touched code)
- `npm run lint` — 0 errors
- `npm run check:boundaries` — OK
- `grep -rn "dateWindowNote\|appendUnique" packages/core/src` — no residue
- `node --test scripts/tests/*.test.mjs` — release tools unaffected

## Abort / reshape triggers

- If markdown-writer tests fail or generated markdown changes: restore the
  parameter (L1) instead of pushing the removal — **fired 2026-07-31**: the
  writer option is a tested dormant capability; kept in place, only the
  pipeline's dead binding was removed
- If grep reveals an unspotted consumer of `dateWindowNote`: reshape scope (L2)
- If the other agent's bug-fix branch conflicts with pipeline.ts lines at
  merge time: wait for their merge and rebase (L1, sequencing)
