# P5 — core-fixes

goal_ref: ../goal.md
created: 2026-09-24T22:40:00+08:00
updated: 2026-09-24T22:40:00+08:00
revision: 1

## Outcome

The confirmed core defects that can be fixed without live provider access are fixed: paper-note frontmatter stays valid YAML for any topic tag, refreshing a paper note's frontmatter keeps the user's own properties and tags, and a date that has scrolled out of arXiv's recent listing fails permanently instead of being retried ten times.

## Assumptions

- Slug-shaped values (the common case) must keep their current unquoted output, so existing notes and byte-identical rendering tests stay unchanged; only YAML-unsafe values get quoted.
- Frontmatter the plugin writes is line-based (`key: value`, flow list for `tags`); preserving unknown top-level keys verbatim, with their indented continuation lines, is enough to keep user properties.
- The review agent's remaining batches may add chunks here (L1) once their High findings are verified.

## Approach

Small, local fixes in `packages/core`, each with a core test written first. The thinking-parameter finding (`extra_body` never flattened) changes what is sent to every provider and cannot be verified without real API calls; it is recorded in the review and left for a user decision.

## Chunks

### Chunk 1 — frontmatter values stay valid YAML (review F20)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core/tests/markdown-writer.test.ts`: a topic tag `AI: Robotics` or `ml,dl` produces frontmatter that a YAML parser reads back as the same `primary_topic` and tag — fails today (unquoted)
- Green check: `cd packages/core && npx vitest run tests/markdown-writer.test.ts`
- regression checks: core markdown / pipeline tests; slug tags render byte-identically
- [ ] implementation and tests accepted

### Chunk 2 — refreshing frontmatter keeps user properties (review F21)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: a paper note with extra keys (`rating: 5`, block `aliases`) and an extra tag keeps them after `refreshPaperNoteFrontmatter` — fails today (whole block replaced)
- Green check: focused vitest
- regression checks: existing `refreshPaperNoteFrontmatter` and manual-fetch tests
- [ ] implementation and tests accepted

### Chunk 3 — dates older than the recent listing fail permanently (review F22)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `listForDate` for a date older than the oldest `/recent` bucket returns `failed_permanent`; a date newer than the newest stays `failed_transient` — first fails today
- Green check: focused vitest on the source adapter tests
- regression checks: core scheduler / pipeline tests (`NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1` in packages/core)
- [ ] implementation and tests accepted

## Phase verification

- Full check set from goal.md constraints

## Abort / reshape triggers

- If quoting changes any existing byte-identical rendering fixture for slug tags, stop and narrow the quoting rule (L1).
- If the scheduler relies on `failed_transient` for old dates in some catch-up path (e.g. a fallback source), stop and reconsider chunk 3.
