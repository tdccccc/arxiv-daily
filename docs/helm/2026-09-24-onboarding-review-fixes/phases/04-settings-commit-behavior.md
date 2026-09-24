# P4 — settings-commit-behavior

goal_ref: ../goal.md
created: 2026-09-24T22:05:00+08:00
updated: 2026-09-24T22:05:00+08:00
revision: 1

## Outcome

Settings text fields commit once when the user finishes editing (change / Enter), never per keystroke, on both render paths; the PDF sidecar address can be moved; the model can be typed by hand and Get models never replaces it silently; picking a duplicate category is refused visibly.

## Assumptions

- Obsidian 1.13's declarative `text` control commits on every `input` event (observed in 1.13.7: three keystrokes → three `setControlValue` calls), so the fix is to render those rows ourselves.
- Real Obsidian `TextComponent.onChange` fires per `input`; the test mock binding it to `change` is inaccurate and may be corrected.
- The 1.13+ category row intentionally has no free-text input (existing test "renders each category as a fixed dropdown without custom input"); not adding one.
- Model decision (user, 2026-09-24): model field accepts free text, Get models only suggests, never replaces.

## Approach

Render rows for output folders and From email/name that validate while typing and commit on change. Moving one sidecar URL to a new origin moves the other to the same origin in the same transaction. Legacy text fields with side effects listen to `change` instead of `onChange`. The model becomes a text input with a suggestion list.

## Chunks

### Chunk 1 — 1.13+ text rows commit on change (F11)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests: typing (`input` events) into Daily reports folder / Paper notes folder / From email / From name rows commits nothing; `change` commits once; an invalid folder draft is marked invalid and not committed — fail today (plain `text` controls, no render rows)
- Green check: focused vitest in `tests/settings-declarative-tab.test.ts`, `tests/settings-definitions.test.ts`
- regression checks: `cd plugin && npm run typecheck && npm test`; real Obsidian probe `textcontrol` shows zero commits during typing
- [ ] implementation and tests accepted

### Chunk 2 — sidecar address can move (F12)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: test: with the sidecar enabled, changing the capability URL to port 5002 saves both URLs on 5002 — fails today (same-origin rejection)
- Green check: focused vitest
- regression checks: `cd plugin && npm test`
- [ ] implementation and tests accepted

### Chunk 3 — legacy text fields stop acting per keystroke (F13)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: make the `TextComponent` mock fire `onChange` on `input` like Obsidian; tests: typing into the legacy custom category, embedding base URL/model, sidecar URLs does not save or re-render until `change`; a rejected legacy sidecar toggle restores the toggle and reports instead of an unhandled rejection — fail today
- Green check: focused vitest in `tests/settings-tab.test.ts`
- regression checks: `cd plugin && npm test`
- [ ] implementation and tests accepted

### Chunk 4 — model can be typed, Get models only suggests (F14, C3)

- change kind: behavior change (user decision 2026-09-24)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests: the model row is a text input that commits a typed model on change; Get models with a list lacking the current model keeps it and offers the list as suggestions — fail today
- Green check: focused vitest in both settings test files
- regression checks: `cd plugin && npm test`; guide "Connect AI" still focuses the model field when only the model is missing
- [ ] implementation and tests accepted

### Chunk 5 — duplicate category choice is refused visibly (F16)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: test: choosing another row's category shows a notice, keeps both rows and restores the select — fails today (row silently disappears)
- Green check: focused vitest
- regression checks: `cd plugin && npm test`
- [ ] implementation and tests accepted

## Phase verification

- Real Obsidian 1.13.7 probe: typing into Daily reports folder makes no commit until change
- Full check set from goal.md constraints

## Abort / reshape triggers

- If correcting the `TextComponent` mock breaks many unrelated tests, keep the mock and assert on the listener type instead (L1).
