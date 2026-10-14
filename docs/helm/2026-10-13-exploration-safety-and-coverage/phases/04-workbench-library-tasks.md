# P4 — workbench-library-tasks

goal_ref: ../goal.md
created: 2026-10-14T01:00:00+08:00
updated: 2026-10-14T10:43:43+08:00
revision: 2

## Outcome

The workbench exploration library has goal tasks for the Personal literature library — build index, revoke, catalog filter, index search, and Direction candidate review — each with a program verifier and an untouched-state test, starting from a fixture-seeded connected library with remote embeddings, a pre-installed PDF runtime, and no non-fixture request possible from the model's actions.

## Assumptions

- research.md "P4 intake": the pinned PDF runtime can be prepared once with network into a version-keyed test cache and copied (not symlinked at the leaf) into each fixture's `cache_dir`, after which `library-build` spawns no npm.
- Remote embeddings against the fixture (`dimension = 3`, stub key) work on the CLI as on Obsidian; local mode is never used.
- Connecting by typing a path stays out of scope: the path field is restricted on purpose (a model could connect a real directory); the embedding fields, including mode, stay restricted on the workbench.
- The real CLI (`library connect`, `status`, `authorize`) with `fixture.env` is the deterministic seed, followed by a server resync.

## Approach

First make library work safe and offline at model time (runtime pre-install, npm forced offline), then a library seed and a reset that clears library state, then tasks in order of setup cost. Each task declares source-scoped `uiLabels` (P6 check) and gets verifier tests in the workbench test file the existing workbench tasks use.

## Chunks

### Chunk 1 — pre-installed PDF runtime, npm offline at model time

- change kind: behavior change (test tooling, safety)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: a test shows the workbench fixture has no prepared runtime and its environment lets npm reach the registry; a real-host probe of `library-build` on a seeded library spawns `npm ci`.
- Green check: before the server starts, the harness copies a version-keyed prepared runtime (prepared once with network, reused afterwards) into the fixture cache; the workbench environment sets npm offline so any accidental install fails fast; a real-host `library-build` completes with no npm spawn and only fixture requests.
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npm run test:acceptance:tools`; the fixed workbench acceptance suite still passes (its launch is unchanged unless it opts in).
- [x] implementation and tests accepted

### Chunk 2 — library seed and reset isolation

- change kind: behavior change (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests show `reset()` leaves TOML `[library]`, `[embedding]`, library-accepted topics, `.index/personal-library-*` state and extra library PDFs.
- Green check: a `startWorkbenchLibraryFixture` seeds connected (optionally authorized, optionally indexed) state through the real CLI plus a server resync; `reset()` restores library-related config and files to the baseline inside the fixture only; older workbench tasks still pass.
- regression checks: as Chunk 1.
- [x] implementation and tests accepted

### Chunk 3 — build, revoke, catalog filter and index search tasks

- change kind: behavior change (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: verifier tests for the new tasks fail before they exist (untouched state, partial actions, state without observed actions).
- Green check: tests pass; scripted fake-model real-host runs pass, wait-only runs end not passed.
- regression checks: as Chunk 1; label test.
- [ ] implementation and tests accepted

### Chunk 4 — Direction candidate review tasks

- change kind: behavior change (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: verifier tests for generate-and-accept and rename/discard fail before they exist.
- Green check: as Chunk 3.
- regression checks: as Chunk 1; label test.
- [ ] implementation and tests accepted

## Phase verification

- All workbench goal tasks in one run with an always-blocked fake model end by the model's own blocked action.
- The workbench library tasks with scripted plans pass back to back.
- No non-fixture request and no npm spawn during model-time actions.
- `npm run test:release-tools` passes.

## Abort / reshape triggers

- If the prepared runtime cannot be reused without network at model time, stop and decide with the user.
- If a needed library control is restricted, report; do not loosen the policy silently (connect-by-path stays excluded).
- If library state leaks into older workbench tasks, fix reset before adding tasks.
