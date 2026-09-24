# P2 — truthful-guide

goal_ref: ../goal.md
created: 2026-09-24T21:10:00+08:00
updated: 2026-09-24T21:10:00+08:00
revision: 1

## Outcome

The guide only reports a step complete when it really is, says what blocks the first report, includes a fifth step that turns on daily runs, shows progress while the first report runs, names sections as they appear on the page, renders as a full-width card on 1.13+, and never offers a delete button that does nothing.

## Assumptions

- Users who finished the old four steps but never enabled the schedule will see the guide again with only step 5 open; this is the intended effect of the 2026-09-24 decision.
- `plugin.setScheduleEnabled(true)` (validation + Skip/Run today modal) is the right action for step 5 — the same path as the Enable toggle.
- The email guide row's `name: ""` + host class pattern gives a full-width row on 1.13+ (it does for the email guide today).

## Approach

Change `getSetupStatus` so the topic step matches what `validateFilterConfig` checks for topics (no duplicate tags) and so the status carries the schedule state; the guide renders step 5 and, when the first three steps are done but the run is still blocked, prints the blocking reasons in step 4. New topics start with a tag derived from the name and made unique. Busy state for the first report lives on the tab so re-renders keep it.

## Chunks

### Chunk 1 — unique topic tags and visible blocking reasons (F4)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests: adding a topic after deleting the first one never duplicates an existing tag; typing a name into a new topic derives the tag from it, with a suffix on collision; `getSetupStatus` marks topics not ready when tags collide; the guide's step 4 shows the `validateFilterConfig` reasons when steps 1–3 are done but the run is blocked — all fail today
- Green check: `cd plugin && npx vitest run tests/onboarding.test.ts tests/settings-declarative-tab.test.ts`
- regression checks: `cd plugin && npm run typecheck && npm test`
- [ ] implementation and tests accepted

### Chunk 2 — step 5 turns on daily runs (F10)

- change kind: behavior change (user decision 2026-09-24)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests: `shouldRenderSetupGuide` stays true while the schedule is off even after a completed report; the guide lists a fifth step whose button calls `plugin.setScheduleEnabled(true)` and refreshes the guide; the step is complete when the schedule is on — fail today
- Green check: `cd plugin && npx vitest run tests/onboarding.test.ts tests/settings-declarative-tab.test.ts tests/settings-tab.test.ts`
- regression checks: `cd plugin && npm test`; dashboard empty-state copy still accurate (`tests/dashboard-view.test.ts`)
- [ ] implementation and tests accepted

### Chunk 3 — first report busy state (F7)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: test: while `generateFirstReport` is pending the guide (including after a guide re-render) shows a disabled "Generating…" button and a second click does not start another run — fails today
- Green check: focused vitest on the new test
- regression checks: `cd plugin && npm test`
- [ ] implementation and tests accepted

### Chunk 4 — copy, full-width card, last-category delete (F6, F8, F9)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests: guide copy names "LLM" and "arXiv categories"; the declarative guide row has an empty name and the host class; the category list has no `onDelete` when only one category remains — fail today
- Green check: focused vitest
- regression checks: `cd plugin && npm test`; real Obsidian screenshot of the guide shows one "Getting started" title and a full-width card
- [ ] implementation and tests accepted

## Phase verification

- Real Obsidian 1.13.7 probe: guide screenshot (full-width, single title, five steps); single category row has no Delete control
- Full check set from goal.md constraints

## Abort / reshape triggers

- If the reappearing guide for existing users turns out to be disruptive in a way the user did not intend, stop and ask before continuing chunk 2.
- If Obsidian keeps a delete affordance even without `onDelete`, fall back to a notice in `deleteCategory` (L1).
