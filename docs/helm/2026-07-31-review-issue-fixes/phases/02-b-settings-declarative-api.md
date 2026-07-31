# P2b — Settings tab declarative API (getSettingDefinitions)

<!-- Filename must be NN-<slug>.md with NN = N (e.g. P1 → 01-auth.md). -->
goal_ref: ../goal.md
status: done

## Outcome

On Obsidian 1.13+, the arXiv Daily settings tab renders through the
declarative `getSettingDefinitions()` API (searchable in Settings search);
`display()` stays as the <1.13 fallback with zero behavior change. The tab
wires the three overrides — `getSettingDefinitions` (host callbacks bound
to the shared row renderers and the tab's mutation methods),
`getControlValue`, `setControlValue` (dotted-path resolvers, persisting
via `saveSettings`, with display()-parity trimming for email to/fromEmail
but not fromName). All checks stay green: full workspace 1163 tests,
typecheck, lint (0 errors), boundaries, esbuild build.

## Assumptions

- Obsidian 1.13+ prefers `getSettingDefinitions()` when implemented and
  falls back to `display()` otherwise (type docs + the
  obsidianmd/settings-tab lint rule); runtime behavior cannot be exercised
  locally — structure tests and types are the safety net
- The declarative surface (control / action / render / group / page / list)
  covers the current interactions; anything it cannot express uses the
  `render` escape hatch
- `display()` is left untouched, so <1.13 behavior is unchanged by this
  phase

## Approach

New `plugin/src/settings/definitions.ts`: flat key constants (e.g.
`llm.apiKey`), `getControlValue`/`setControlValue` path mapping over the
nested settings object (persist via `saveSettings`), and
`buildSettingDefinitions(tab)` returning `SettingDefinitionItem[]`. Complex
rows (API-key sentinel, test connection, get models, onboarding guide,
topics, email verify) use `action`/`render` callbacks that reuse the
existing helper functions already exported from tab.ts. `tab.ts` overrides
`getSettingDefinitions` + both resolvers; `display()` stays.

## Tasks

- [x] T1: key constants + getControlValue/setControlValue path mapping
- [x] T2: basic blocks — LLM (base URL, model dropdown, thinking, reasoning),
      output (dirs, link style, language), advanced (log level, delays,
      cache expiry)
- [x] T3: complex rows via action/render — API-key sentinel input, test
      connection, get models, onboarding guide strip
- [x] T4: topics as a `list` (add/delete/reorder) + categories block —
      plus quick start, detail notes, timezone to keep the arXiv section
      complete on 1.13+; shared mutations live on the tab
- [x] T5: email block (to/from/mode/verify/test) + schedule block
      (enabled/time window/interval) — enable toggle re-rendered through
      setScheduleEnabled so validation + modal flow survive
- [x] T6: wire tab.ts (three overrides) + structure tests (every key maps,
      key items present) + full verification
- [x] T7: manual-test UI follow-up (1.13+ only) — hide completed setup and
      Quick start, compact/reorder LLM with masked revealable API key and
      None-integrated reasoning, fixed category dropdowns, and concise
      collapsible topic cards; new-install reasoning default is medium
- [x] T8: second manual-test UI follow-up — remove framework drag layers from
      category/topic lists, keep topic tag beside its title, align the email
      guide, move send actions into right-side buttons, and replace the hosted
      verification sentinel actions with masked reveal/save-on-blur input

## Verification

- `npm test` (full workspace) green — settings-tab tests unchanged
- `npx tsc -p plugin/tsconfig.json --noEmit`, `npm run lint` (0 errors),
  `npm run check:boundaries`
- New structure tests: `getSettingDefinitions()` returns items; every
  control key resolves through getControlValue; key settings present
  (api key row, model, topics list, email to)
- `npm run build` (esbuild) passes with the new module

## Abort / reshape triggers

- If a block cannot be expressed declaratively: use `render` (L1) — do not
  drop settings
- If structure tests or the full suite fail: revert that block (L1)
- If the resolver mapping proves ambiguous (nested keys): prefix keys with
  section names and keep the mapping table explicit (L1)
- Runtime 1.13 behavior is unverifiable locally; if a later manual test
  shows a mismatch, adjust the affected block, not the whole design (L2)
