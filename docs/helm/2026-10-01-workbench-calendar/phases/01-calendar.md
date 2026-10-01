# P1 — calendar

goal_ref: ../goal.md
created: 2026-10-01T23:40:36+08:00
updated: 2026-10-01T23:40:36+08:00
revision: 1

## Outcome

Users can navigate daily reports and their generation states by date from the packaged workbench.

## Assumptions

- Existing report files are readable even if their index/run record is missing.
- Historical date eligibility remains the existing CLI's responsibility. Unknown availability is shown as ungenerated, not falsely labeled unpublished.
- Completed zero-match runs may have no Markdown file; permanent failures and completed-but-missing reports cannot be fixed by offering a rerun that the scheduler would skip.

## Approach

Add a read-only month API backed by the document listing and full StateStore snapshot, with the active owned daily job as an overlay. Reuse only calendar-date arithmetic from the plugin; its schedule-window-specific action eligibility does not apply to the CLI. Add month/date selection and status controls above the daily list, with explicit date-prefilled generation and task-driven refresh.

Visual thesis: a compact, quiet calendar in the existing warm sidebar, using the blue selection accent and small status marks.
Content plan: collection tabs, month controls, date grid, selected-day status/actions, search and report list; preserve the article pane.
Interaction thesis: keyboard date navigation, clear hover/selection, mobile collapse, existing restrained article transition with reduced motion.

## Chunks

### Chunk 1 — Calendar data and shared arithmetic

- change kind: behavior-preserving extraction followed by behavior change
- strategy: Green characterization for shared date helpers; strict Red-Green for HTTP month-state behavior
- baseline: existing plugin calendar tests before/after extracting date helpers
- Red: actual HTTP calendar request fails before endpoint exists
- Green: month API tests pass across month/leap boundaries, timezone, files without index, >20 historical states, zero matches, missing reports, failures, future and in-flight jobs; byte snapshots remain unchanged
- regressions: focused plugin calendar tests, CLI server/inspection tests, workspace typecheck and boundaries
- [ ] implementation and tests accepted

### Chunk 2 — Calendar reading interaction

- change kind: behavior change
- strategy: strict Red-Green at DOM/API boundaries; visual layout through browser review
- Red: UI tests fail to show/navigate month cells and date-specific states/actions
- Green: tests cover selecting a report, inspecting empty/no-match/failure days without generation, explicit date-prefilled run, stale-month responses and keyboard interaction
- regressions: CLI tests, packaged workflow test, browser desktop/mobile review and screenshots
- [ ] implementation and tests accepted

## Phase verification

- Real Chromium reads a saved report via calendar, changes month, inspects failed/no-match days and opens a date-prefilled generation form.
- Small screen shows a collapsible calendar and readable article without horizontal overflow.
- Existing generation and Markdown reading regression tests pass; production and source fixture files remain unchanged by browsing.

## Abort / reshape triggers

- If a day status cannot be established from authoritative records, display uncertainty rather than invent availability or alter persistence.
- If sharing date arithmetic changes Obsidian behavior, preserve its original contract and limit extraction.
