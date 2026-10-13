# P2 — ui-label-check

goal_ref: ../goal.md
created: 2026-10-13T11:55:00+08:00
updated: 2026-10-13T11:55:00+08:00
revision: 1

## Outcome

Every goal task declares the exact UI labels it depends on, and an ordinary fixed test (no model, no app launch) fails with the task id and the missing label when any of them no longer appears in that host's product source.

## Assumptions

- Each label a task depends on appears literally in the host's product source (Obsidian: `plugin/src/**`; workbench: `apps/cli/src/workbench/**`), including Chinese workbench strings. Labels built at runtime from parts would need a documented exception.
- A plain substring search over source files is precise enough; false passes from a label surviving only in a comment or test are acceptable for this purpose.

## Approach

Add a `uiLabels` list to every goal task in `scripts/acceptance/exploration-tasks/{obsidian,workbench}.mjs`, covering labels named in the goal text and labels the verifier matches on. Where a verifier matches a label, it uses the same declared string so the two cannot drift. A new fixed test loads both task libraries with a minimal fixture, asserts every goal task has a non-empty `uiLabels`, and asserts every label occurs in the host's product source.

## Chunks

### Chunk 1 — declared labels and the source check

- change kind: behavior change (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: new test file (e.g. `scripts/tests/acceptance-exploration-labels.test.mjs`) fails because goal tasks have no `uiLabels`; a second assertion run against a deliberately renamed label (test-local copy of the check with an injected missing label) shows the failure message names the task id and the label.
- Green check: `node --test scripts/tests/acceptance-exploration-labels.test.mjs` passes for all 14 Obsidian and 6 workbench goal tasks.
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npm run test:acceptance:tools`; the test is picked up by `npm run test:release-tools`.
- [x] implementation and tests accepted

## Phase verification

- Temporarily rename one label in a scratch copy (not committed) or via the test's injection path and observe the failure message naming the task and label.
- `npm run test:release-tools` passes.

## Abort / reshape triggers

- If several labels can only be found by reconstructing runtime string building, stop and decide whether those tasks get an explicit exception list or the check moves to a different source (L1).
- If loading the task libraries in a plain test requires launching a host, stop and reshape the test seam.
