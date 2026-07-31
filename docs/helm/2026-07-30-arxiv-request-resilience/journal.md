# Journal

## 2026-07-30 — Final review adjustments and verification

### Evidence

- Final read-only review found a replaceable-note race, pre-write detail-index mutation, and an unbounded shared Retry-After cooldown.
- The replaceable note is now re-read, byte-compared, and reclassified immediately before the non-interruptible write; changed notes return `note_conflict` without note or index mutation.
- Detail/saved/path index mutations now occur only after `writePaperDetail` succeeds.
- Retry-After parsing and the shared cooldown are capped consistently at 30 minutes, including oversized numeric values.
- Regression tests after these adjustments: core 804, node-runtime 8, CLI 27, plugin 242; all passed.
- Release-tool tests, release-version check, workspace boundaries, typechecks, build, and `git diff --check` passed.
- Lint passed with 0 errors and 53 existing warnings.
- `smoke:build` fails because `scripts/smoke-build.mjs` still expects the removed `--config` flow. The identical command fails with identical output on the unmodified `main` checkout, proving this is a pre-existing baseline issue rather than a branch regression.

### Change

L1 adjust — kept the goal and P4 outcome, fixed the three review findings, and added regression coverage before closing.

### Disposition

Keep all P1–P3 implementation and review fixes. No success criterion was waived; the repository-wide verification criterion explicitly permits a separately evidenced unrelated baseline failure.

### Next action

Initiative closed. The branch remains uncommitted and unpushed for user review.
