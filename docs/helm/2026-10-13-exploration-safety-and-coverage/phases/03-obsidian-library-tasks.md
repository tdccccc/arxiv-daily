# P3 — obsidian-library-tasks

goal_ref: ../goal.md
created: 2026-10-13T12:40:00+08:00
updated: 2026-10-13T16:30:00+08:00
revision: 3

## Outcome

The Obsidian exploration library has goal tasks for the Personal literature library — consent, revoke, index build, Dashboard library search, Library similarity, and reviewing Direction candidates — each with a program verifier, a "fails on untouched state" test, and a deterministic fixture start that uses no native folder picker and never touches a real library.

## Assumptions

- Facts from research.md: a connected library is a `libraryConnection` in the fixture `data.json` built with `connectedLibraryFixture`/`readRootIdentity` (`scripts/desktop-acceptance/settings-fixture.mjs`) over the fixture's `vault/library` PDFs; consent can be granted in `prepare()` through the plugin's `authorizeLibraryProcessing` over CDP.
- Remote Embedding mode against the fixture's `/embeddings` (3-dim vectors, so `embedding.dimension = 3`) indexes 2 PDFs within the 60 s `prepare()` deadline; local embedding (model download) is not used.
- The fixture answers the direction-proposal prompt; incremental direction suggestions (unanswered diff prompt) stay out of scope.
- The HTTP proxy installed by the harness still covers plugin requests after `reloadPlugin()`.
- Library controls in settings and the review dialog are observable and not classified restricted; anything restricted (e.g. the embedding API key) is left as fixture data.

## Approach

First build a small library fixture helper for the exploration host that reuses the desktop-acceptance builders and the exploration driver's settings patch / reload path, then add tasks in order of setup cost. States that need an index are produced by running the plugin's real scan and index in `prepare()`, not by hand-writing index files. Every task declares `uiLabels` (P2 check) and gets an untouched-state test in `scripts/tests/acceptance-exploration-host.test.mjs` or `acceptance-exploration-obsidian.test.mjs`, following existing task tests.

## Chunks

### Chunk 1 — library fixture start, consent and revoke tasks

- change kind: behavior change (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: new task ids (e.g. `library-remote-consent`, `library-revoke-consent`) are absent from `GOAL_TASK_IDS`; new tests for their verifiers fail. Consent task: user switches Embedding mode to remote and confirms the disclosure; verify `settings.embedding.mode === "remote"` and `libraryConnection.authorization` present, and the model clicked the confirm control. Revoke task: start authorized; verify authorization removed after a Revoke click.
- Green check: focused task tests pass, including "verify fails on untouched state"; label test passes.
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npm run test:acceptance:tools`.
- exception: the fixture start runs inside a real Obsidian; compensating check is a scripted fake-model run of both tasks on the real host (pass when the scripted steps are done, not passed when the model only waits/finishes).
- [x] implementation and tests accepted

### Chunk 4 — block non-fixture network in Obsidian exploration (runs before Chunk 1's consent task and Chunk 2)

- change kind: bug fix (safety)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: with a connected library in local Embedding mode, clicking "Retry preparation" in the isolated Obsidian sends `https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/config.json` straight to the internet (observed 2026-10-13 with CDP `Network.requestWillBeSent`; the HttpClient proxy and fixture-requests.json never see it). A fixed test shows the exploration launch has no network guard.
- Green check: exploration Obsidian sessions launch with a guard that fails every request not addressed to loopback (fixture server, CDP) at the Chromium network layer for all processes, opt-in so the fixed acceptance suite is unchanged; the same real-host probe shows the Hugging Face request failing without any CDP blocklist, and ordinary fixture traffic (runs, remote embeddings) still works.
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npm run test:acceptance:tools`; scripted real-host run of `library-revoke-consent` still passes.
- [x] implementation and tests accepted

### Chunk 2 — index build, library search and similar-paper tasks

- change kind: behavior change (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests for new tasks (e.g. `library-build-index` via "Retry preparation", `library-search` via the Dashboard search "Library matches" block, `library-similar-papers` via "Find similar papers" → Library similar) fail before the tasks exist. Verifiers: index `current.json` exists and fixture saw embeddings requests after task start; library match rows shown and the model opened one; similar modal library list non-empty after the model opened it.
- Green check: focused tests pass, including untouched-state failures.
- regression checks: as Chunk 1.
- exception: as Chunk 1; the real-host run also measures `prepare()` time for the index build against the 60 s deadline.
- [x] implementation and tests accepted

### Chunk 3 — Direction candidate review tasks

- change kind: behavior change (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests for new tasks fail before they exist: accept one proposed direction into a research topic (verify a direction with `origin: "library"` in `data.json` plus the acceptance receipt), and rename or discard a candidate (verify `direction-proposal.json` changed accordingly).
- Green check: focused tests pass, including untouched-state failures.
- regression checks: as Chunk 1.
- exception: as Chunk 1.
- [ ] implementation and tests accepted

### Chunk 5 — plugin ignores a late onload after it was disabled (product)

- change kind: bug fix (product)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: a plugin unit test that disables the plugin while `onload()` is still awaiting and then lets `onload()` finish shows it still registers the Dashboard view (and commands) on the stale instance; on the real host, two overlapping reloads produce "Attempting to register an existing view type \"arxiv-daily-dashboard\"" (/tmp/reload-race, 38 errors in 20 iterations).
- Green check: the unit test shows a stale instance skips registration; the overlapping-reload repro shows no plugin failure.
- regression checks: plugin test suite (`npm --prefix plugin test`), plugin typecheck/lint as the plugin's normal checks; `npm run test:release-tools`.
- [x] implementation and tests accepted

### Chunk 6 — harness never overlaps plugin reloads

- change kind: bug fix (test tooling)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: a test shows two `reloadPlugin()` calls (e.g. one abandoned by a `withDeadline` timeout and the next task's reset) run concurrently; first exploration reset can also start while Obsidian's own startup load of the plugin is still running.
- Green check: reloads are serialized per session (a later reload waits for an earlier one, including an abandoned one), and the first reset waits for the startup load; tests prove ordering.
- regression checks: `node --test scripts/tests/acceptance-exploration*.test.mjs`; `npm run test:acceptance:tools`; scripted real-host runs of the library tasks.
- [ ] implementation and tests accepted

## Phase verification

- Scripted fake-model run of every new task on the real Obsidian host passes; the same tasks with a model that only waits end not passed.
- Whole Obsidian goal library under a fake model that immediately reports blocked: every task's `prepare()` and `reset()` succeed in order (no cross-task leakage from library state into older tasks).
- `npm run test:release-tools` passes.

## Abort / reshape triggers

- If remote indexing of the fixture PDFs cannot finish inside `prepare()`'s deadline, stop and decide (raise the deadline for library tasks vs seed index state) before adding Chunk 2 tasks.
- If library state leaks into older tasks through `reset()` (it restores `data.json` but leaves vault files), stop and fix reset before adding more tasks.
- If a core library control is classified restricted by the existing policy, stop and decide whether the policy or the task is wrong (do not loosen the policy silently).
