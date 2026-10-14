# P7 — workbench-observer-parity

goal_ref: ../goal.md
created: 2026-10-14T18:50:00+08:00
updated: 2026-10-14T19:55:00+08:00
revision: 2

## Outcome

The workbench observer reports a `<summary>`'s open state and names links and summaries by their own text before `title`, matching the Obsidian observer after d600a2e.

## Assumptions

- The workbench uses `<details>`/`<summary>` in the library review dialog, settings and the main app (apps/cli/src/workbench/web/library-review.ts, settings.ts, settings-setup.ts, app.ts), so the gap can cost steps there just as it did on Obsidian.
- No goal task's verifier or scripted plan depends on a link or summary being named by its `title`; the scripted real-host runs will show it.
- The Obsidian `home` fix (b) has no workbench counterpart: the workbench observes only the topmost `dialog[open]`, and there is no `home` action that opens a page behind it.

## Approach

Same shape as d600a2e, in scripts/acceptance/exploration-browser.mjs only: add `expanded` from the parent `<details>` next to `checked`, and extend the own-text-before-row-and-title rule from BUTTON to SUMMARY; a link's own text goes before `title` but after its settings row (L1: the external-link rows each render a generic "Open" link). No restriction, masking or scope rule changes.

## Chunks

### Chunk 1 — summary state and own-text naming on the workbench

- change: an observed workbench `<summary>` carries `expanded: true|false` from its parent `<details>`; `<summary>` controls are named like buttons (aria-label, labels, own text, row, `title`); `<a>` controls by aria-label, labels, row, own text, then `title`.
- kind: bug fix (test tooling)
- first signal: Red — new fixed tests through `browserObservation` on a DOM fixture fail because `expanded` is undefined and a titled link/summary is named by its tooltip.
- checks: the new tests Green; `node --test scripts/tests/acceptance-exploration*.test.mjs` (baseline 331); `npm run test:acceptance:tools` (baseline 396); `npm run test:release-tools` (baseline 818).
- [x] accepted on observed evidence — see journal 2026-10-14 (P7 done).

## Phase verification

- Scripted real-host runs of the 12 workbench goal tasks pass, and the always-blocked run ends every task blocked, without a real model.
- Observations from a real workbench run contain `expanded` on summaries.

## Abort / reshape triggers

- If a goal task or scripted plan relies on the old title-first name, stop and decide whether the task or the order should change instead of editing verifiers to match.
