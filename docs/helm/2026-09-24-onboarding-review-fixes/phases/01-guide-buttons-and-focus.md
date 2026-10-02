# P1 — guide-buttons-and-focus

goal_ref: ../goal.md
created: 2026-09-24T20:50:00+08:00
updated: 2026-09-24T21:10:00+08:00
revision: 2

## Outcome

On Obsidian 1.13+ the guide's "Connect AI", "Choose sources" and "Describe interests" buttons scroll to their section and focus the first field that still needs input, and typing in a topic after setup is complete keeps the input and its focus.

## Assumptions

- Obsidian 1.13 applies a group/list `cls` to the `.setting-group` element inside the tab's `containerEl` (seen in the 1.13.7 probe: groups render as `.setting-group` inside `containerEl`).
- The 1.13 settings page lives in its own window, so lookups must use `containerEl` / its `ownerDocument`, never the global `document`.
- "Describe interests" with no topics may create one via the existing `addTopic` (which already focuses the new name input).

## Approach

Give the three declarative groups stable classes and mark the legacy headings with the same class, so one lookup serves both render paths. `scrollToSection` finds the section, scrolls, then focuses the first unfinished field (API key / base URL / model; first category select; first incomplete topic field, expanding its card, or a new topic when there are none). A missing target shows a Notice instead of returning silently. For F2, the declarative guide refresh does a full update only when the guide has to reappear.

## Chunks

### Chunk 1 — guide buttons locate and focus their section (F1)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: new tests in `plugin/tests/settings-declarative-tab.test.ts`: (a) `buildSettingDefinitions` gives the LLM group, categories list and topics list the section classes — fails today (no `cls`); (b) with a container holding `.setting-group.<llm class>` and an empty API key input, clicking the guide's "Connect AI" button calls `scrollIntoView` and focuses the API key input — fails today (no-op); (c) "Describe interests" with an incomplete topic expands the card and focuses its first empty field; with no target present a Notice is shown
- Green check: `cd plugin && npx vitest run tests/settings-declarative-tab.test.ts tests/settings-tab.test.ts`
- regression checks: `cd plugin && npm run typecheck && npm test`; `npm run lint`
- [x] implementation and tests accepted

### Chunk 2 — topic edits after setup never trigger a full re-render (F2)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: new test: settings complete + a completed run state, no guide row connected; `tab.refreshSetupGuide()` must not call `update()`/`refreshSettings`; and when setup becomes incomplete again (guide must reappear) it does call it — first assertion fails today
- Green check: `cd plugin && npx vitest run tests/settings-declarative-tab.test.ts`
- regression checks: `cd plugin && npm run typecheck && npm test`
- [x] implementation and tests accepted

## Phase verification

- Rebuild (`npm run build`) and rerun the `/tmp/arxiv-probe` real-Obsidian probe: `guide` scenario shows scroll position change and the API key input focused after "Connect AI"; `topic` scenario shows the input still attached and focused
- Full check set from goal.md constraints

## Abort / reshape triggers

- If Obsidian ignores `cls` on groups/lists or renders them outside `containerEl`, switch to locating the group by its heading row (L1), and note it in the journal.
- If avoiding the full update leaves the guide stale in a user-visible way (e.g. guide needs to appear and does not), stop and reshape P1.
