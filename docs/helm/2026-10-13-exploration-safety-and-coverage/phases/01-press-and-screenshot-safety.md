# P1 — press-and-screenshot-safety

goal_ref: ../goal.md
created: 2026-10-13T10:10:00+08:00
updated: 2026-10-13T10:10:00+08:00
revision: 1

## Outcome

On both hosts a keyboard `press` is sent only when the focused element would itself be allowed for a `click`, and Obsidian exploration screenshots hide secret fields the way the workbench screenshots already do.

## Assumptions

- Each host's existing `classifyControlRestriction` plus the sensitive/disabled rules are the right policy for the focused element; `press` needs no separate denylist.
- The focused element can be described with the same control descriptor the observation already builds (Obsidian: in the renderer via CDP; workbench: via Playwright `page.evaluate`).
- Enter in a text field can submit its form; refusing `press` when the form's submit control is restricted covers implicit submission. Verify against the real settings / email forms before relying on it.
- Obsidian screenshots can hide secret fields by temporarily styling them inside the renderer before `Page.captureScreenshot` and restoring afterwards; `createScreenshotWriter` does not need a generic mask API.

## Approach

Add a "focused-element check" step to each host's `press` executor that describes `document.activeElement` with the host's existing control descriptor and refuses with `ExplorationActionError` when it is restricted, sensitive, disabled, or (workbench) outside the workbench surface — zero key events sent. Then add Obsidian screenshot masking around `capture()` for the same selectors the workbench masks plus the plugin's sensitive-input heuristic.

## Chunks

### Chunk 1 — Obsidian press refuses restricted focus

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: new tests in `scripts/tests/acceptance-exploration-obsidian.test.mjs` (happy-dom + fake CDP, same style as the existing "press is rejected … outside the plugin's scope" test) fail because `press Enter` with focus on an external-link settings button, the scheduler toggle, the "Clear run state…" menu item, and a "Show … API key" reveal button still sends `Input.dispatchKeyEvent`.
- Green check: `node --test scripts/tests/acceptance-exploration-obsidian.test.mjs` — the new tests pass with zero `Input.*` calls and an `ExplorationActionError` naming the restriction; existing press tests (allowed key in scope, Shift+Tab, out of scope, non-allowlisted key) still pass.
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npx eslint` on changed files with the repo config's ignore overridden only if the existing phase practice did so (otherwise record "not run").
- [x] implementation and tests accepted

### Chunk 2 — workbench press refuses restricted or out-of-scope focus

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: new tests (in the workbench host test file that already drives the browser driver, `scripts/tests/acceptance-exploration-host.test.mjs` or its existing equivalent) fail because `press Space` with focus on the `schedule.enabled` / `email.enabled` switch, and `press Enter` with focus outside `.workbench`, still reach `page.keyboard.press`.
- Green check: focused test file passes; the refused cases produce `ExplorationActionError` and no key event; an allowed press on an ordinary control still works.
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npm run test:acceptance:tools`.
- [x] implementation and tests accepted

### Chunk 3 — Obsidian screenshots hide secret fields

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: a new test shows that the Obsidian driver's `capture()` sends `Page.captureScreenshot` while a revealed (`type="text"`) API key input or a password input is visible unmasked.
- Green check: the test observes the mask applied to every secret field before the capture call and removed afterwards (also when capture throws).
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npm run test:acceptance:tools`.
- [x] implementation and tests accepted

## Phase verification

- `npm run test:release-tools` passes (baseline on main: 660/660).
- Fake-model exploration run on each host with a scripted model that Tabs onto a restricted control and presses Enter: the run records the refusal as an invalid action, no external URL is opened, and the Obsidian screenshots of the settings page show masked secret fields.
- `goal.md` success criteria 1 and 2 can be checked.

## Abort / reshape triggers

- If describing the focused element needs a second, divergent control-description path per host (rather than reusing the observation descriptor), stop and reconsider (L1 at minimum).
- If Obsidian screenshot masking cannot be applied without changing the shared desktop-acceptance screenshot writer used by the fixed acceptance suite, stop and decide with the user before touching shared infrastructure.
- If the keyboard-only persona can no longer do anything useful after the fix (every press refused in ordinary navigation), stop and reshape.
