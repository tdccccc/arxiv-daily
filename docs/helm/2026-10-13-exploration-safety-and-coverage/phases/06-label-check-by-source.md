# P6 — label-check-by-source

goal_ref: ../goal.md
created: 2026-10-13T22:30:00+08:00
updated: 2026-10-13T22:30:00+08:00
revision: 1

## Outcome

Each declared UI label names the product source file it comes from and must appear there as a complete, case-sensitive string literal, so renaming it fails the label test even when similar text survives elsewhere; labels that cannot be checked this way are listed as explicit exceptions with a reason.

## Assumptions

- Most labels are literal strings in one product file (e.g. "Run History" in `plugin/src/dashboard/hub-modal.ts`); a few are built at runtime or interpolated and need an exception.
- Some labels appear in more than one file legitimately (e.g. "Revoke" on the library row and its confirmation); a label may name several files when the task depends on each.

## Approach

Change `uiLabels` entries from plain strings to `{ text, source }` (source = one or more repo paths), keep the verifier constants as the single source of the text, and change `scripts/tests/acceptance-exploration-labels.test.mjs` to check that each source file contains the text as a complete quoted string literal (single, double or template quotes), case-sensitively. Exceptions carry `{ text, exception: "reason" }` and are reported, not checked.

## Chunks

### Chunk 1 — source-scoped exact label check

- change kind: behavior change (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: new tests show the current check passes when "Run History" is renamed only in `hub-modal.ts` (because "Run history" survives in `commands.ts`) and when a label survives only inside a comment.
- Green check: both cases fail with host, task, label and source file; all 23 Obsidian and 6 workbench goal tasks pass with source-scoped labels; exceptions are listed with reasons.
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npm run test:release-tools`.
- [ ] implementation and tests accepted

## Phase verification

- `npm run test:release-tools` passes.
- Owner reviews the exception list and accepts each reason.

## Abort / reshape triggers

- If more than a handful of labels need exceptions, stop and reconsider the matching rule (L1).
