# Research — intake recon (2026-10-10)

Disposable intake notes from two read-only recon subagents. Decisions live in goal.md / phase files.

# Recon: exploration framework (helm intake)

## 1. Architecture

**Loop** (`scripts/acceptance/exploration.mjs:117` `runExplorationLoop`): per task — `driver.observe()` → if `observation.errors` non-empty, mark `failed` and stop (exploration.mjs:150-157) → run trusted `task.verify({driver,observation,observations,actions,startedAt,signal})` (exploration.mjs:158) → if verified/failed, screenshot + record + break (160-167) → else call model with `{task,observation,checks,history(last 8),remaining}` (171-177) → `parseExplorationAction` validates reply (182) → if browser action, `driver.execute()` (191); if `finding`, push to findings as "needs-review" (198-200, uses literal 待复核 status string `needs-review`); if `done`/`blocked`, scenario → `blocked` (201-207). Every step is appended to `exploration-events.jsonl` (134-138, record()). Tasks run sequentially; loop `break`s the whole run on first non-`passed` scenario (227) — i.e. no further tasks run after a block/fail.

**Driver interface** (consumed generically by the loop, concretely implemented in exploration-browser.mjs:91 `createExplorationDriver`): `observe()` → observation object; `execute(action, observation, {signal})` → `{ok, error?}`; `capture(name)` → optional screenshot path (`driver.capture?.`, optional); `dispose()` cleanup; task-specific extras `inspect()` and `dailyEvidence(date)` used only by the three default tasks' `verify()`.

**Observation format** (`browserObservation`, exploration-browser.mjs:52-89, executed in-page via `page.evaluate`): `{id, title, text (innerText, 16000-char cap), controls[], layout:{rootMounted,view,dialogOpen,sidebarCollapsed,viewport}}`. `controls[]` capped at 220 elements, each `{ref, tag, type, role, name, field, disabled, readOnly, sensitive?, value?, checked?, href?, options? (selects), dataAction, paperKey, purpose, rect, inViewport}`. Refs are per-observation (`s<version>-<n>`) and written onto DOM via `data-acceptance-ref` so `execute()` can re-find them (exploration-browser.mjs:139). `observe()` wraps the snapshot adding `url`, `errors` (page errors + diagnostics), `boundaryEvents` (popups/out-of-scope network blocked).

**Action schema/validation** (`parseExplorationAction`, exploration.mjs:20-59): single JSON object, ≤16000 chars, `type` ∈ `{click,fill,select,back,reload,wait,scroll,finding,done,blocked}` with a per-type whitelist of extra fields (`actionKeys`, line 8-12); unknown/extra fields rejected. `click/fill/select` require a `ref` resolvable in the **current** observation's `controls`, and reject `restricted`/`sensitive`/`disabled` controls (34-37). Type-specific bounds: fill ≤2000 chars on editable text/textarea only (38-43); select must be an enabled option value (44-47); wait 1-2000ms (48-50); scroll up/down 1-1200px (51-53); finding needs non-empty summary(≤1000)+evidence(≤2000) (54-56); blocked needs a reason (57). Restriction classification lives in `classifyControlRestriction` (exploration-browser.mjs:11-27): blocks credentials/password fields, file inputs, vault/folder/path fields, base-url/endpoint/provider/embedding/pdfParserSidecar/email/schedule fields, email/library-connect/reveal-secret actions, and any link leaving the workbench origin/pathname. Re-validated against **live** DOM state at execute time too (exploration-browser.mjs:140-147), not just the stale observation — defends against TOCTOU.

**Task shape** (`createDefaultExplorationTasks`, exploration-browser.mjs:201-257): `{id (kebab-case, enforced by regex at exploration.mjs:122), title, goal (natural-language Chinese instruction), verify({driver,observation,actions,...}) → {passed,checks:[{label,passed,actual?}], failed?, error?}}`. Optional `prepare({driver,signal})` hook exists in the loop (exploration.mjs:146) though none of the 3 default tasks use it. Verification is **independent of the model**: `runExplorationLoop` only accepts `verified` when `checks.length>0 && every(passed) && at least one non-"wait" action occurred` (exploration.mjs:160) — the model cannot self-report success.

**Budgets** — all enforced in `runExplorationLoop` via `limit()` (exploration.mjs:80-84, clamps + throws out-of-range): `maxSteps` (model decisions) 1-300 default 50 (118); `maxDurationMs` 1-3,600,000 default 600000, checked every iteration via `deadline` (119, checked at 149); `maxInvalidActions` 1-10 default 3, aborts task after N consecutive invalid/unexecutable actions (120, 220); `modelCallTimeoutMs` per-call 1-300000 default 45000 (121). Separately, `maxApiCalls` (HTTP requests incl. retries) is enforced inside the model adapter itself (exploration-model.ts:98-105), not the loop — permanent per-process counter, independent of step count. CLI flags map 1:1: `--max-steps`, `--max-api-calls`, `--max-duration-ms` (run.mjs:16-18, 61-63).

**Findings channel**: `action.type === "finding"` → pushed to `findings[]` with `status:"needs-review"` (exploration.mjs:198-200), attached per-scenario at the end (`scenario.findings`, line 230) and surfaced in the JSON report; never treated as pass/fail evidence. README calls this "待复核" (not auto-trusted as a confirmed defect) (README.md:60).

**Report/evidence**: `artifactDir/exploration-events.jsonl` (append-only, one JSON record per model step, mode 0600) and `artifactDir/exploration.json` (full structured result, written at end and also incrementally re-written on error/cleanup paths — exploration.mjs:134-138, 237, 276, 290). Result shape: `{suite:"exploration", status, scenarios[], findings[], artifacts[], errors[], metrics:{modelCalls,actions,elapsedMs}, coverage:{completed,incomplete}, model (describe), modelUsage}`.

**Model config / LlmClient**: `createConfiguredExplorationModel` (exploration-model.ts:79-138) reads only the `[llm]` table from the **CLI's own config path** (`resolveCliConfigPath()`, never the disposable fixture) — `base_url`,`api_key`,`model` required, `provider`/`thinking_mode`/`reasoning_effort` optional, validated strictly (readModelSettings, 48-76). Builds a real `LlmClient` (from `@arxiv-daily/core`) over `NodeHttpClient`, with an HTTP wrapper enforcing `maxApiCalls` (98-105) and a fixed `maxCompletionTokens:4096`/`maxOutputCodeUnits:16000`, `temperature:0`. Payload size capped at 100,000 chars before send (116). The TS model adapter is compiled on the fly via esbuild into a temp dir and loaded as CJS (`loadExplorationModel`, exploration.mjs:242-252) so the shipping CLI never bundles test code.

**Redaction**: `createExplorationRedactor(secrets)` (exploration.mjs:61-70) does literal secret substring removal + regex for `Bearer `, `sk/rk/pk/api-` prefixed tokens, and `...key=/token=/secret=/password=` query params. `sanitize()` (72-78) recursively redacts strings and blanks any object key matching `api-key|authorization|password|secret|access-token`. The model adapter additionally builds its own redactor from the resolved endpoint's credentials/query params and registers them with the shared `Logger` (exploration-model.ts:86-93, 108). Every write path (events.jsonl, exploration.json, error scenarios) passes through `safe()`/`redact()`.

## 2. Host coupling

Browser/Playwright-specific, entirely inside `exploration-browser.mjs`: `browserObservation()` (DOM query via `page.evaluate`), `data-acceptance-ref` tagging, Playwright `page.locator/click/fill/selectOption/reload/goBack/mouse.wheel/screenshot`, `context.route` network guard, `page.on("pageerror")`/`context.on("page")` popup guard, URL-origin scoping (`inScope`/`assertScope`). The two default-task helpers `inspect()` (reads fixture TOML) and `dailyEvidence()` (reads vault file + `api/runs/current` HTTP) are workbench-fixture-specific, not loop-specific.

Host-agnostic: everything in `exploration.mjs` — the loop, action schema/validation, redaction, budgets, report/events writing. It only calls `driver.observe/execute/capture/dispose` and `task.verify`, with no Playwright import.

For an Obsidian driver to plug into the same loop it would need, analogous to `createExplorationDriver`:
- `observe()` → same `{id,title,text,controls[],layout,errors,boundaryEvents,url?}` shape, but built from Obsidian's renderer DOM via CDP `Runtime.evaluate` (the existing `evaluate()`/`read()` helpers in `obsidian-ui.mjs:77` and `desktop-acceptance/cdp.mjs`) instead of Playwright's `page.evaluate`. `classifyControlRestriction` can likely be reused as-is (pure function on a control-shape object, exploration-browser.mjs:11), but the live-DOM query (`surface.querySelectorAll(...)`) must run inside the renderer process via `session.evaluate(expr)` returning JSON (pattern already used in `obsidian-ui.mjs:77,83-95` for click/fill), not `page.evaluate(fn, arg)` — CDP only accepts string expressions.
- `execute(action, observation, {signal})` → mapped to CDP: for `click`/`fill` reuse the `click()`/`fill()` building blocks in obsidian-ui.mjs:83-95 (querySelector + dispatch events) or raw `Input.dispatchMouseEvent` as `runDate()` already does for geometry-sensitive clicks (obsidian-ui.mjs:106-135); `back`/`reload`/`scroll` have no existing analog and would need new CDP calls (`Page.reload`, synthetic wheel/`Input.dispatchMouseEvent` type `mouseWheel`).
- `capture(name)` → reuse `createScreenshotWriter` already wired into `createObsidianDriver` (obsidian-ui.mjs:71, `screenshots.capture`).
- `dispose()` → no persistent listeners currently attached by `createObsidianDriver`, so likely a no-op or just closing eval handles.
- Session lifecycle: `createObsidianDriver({session, fixture, artifactDir})` already exists and is built on `runDesktopSession` (desktop-acceptance/session.mjs:36) which does full process-group launch/build-deploy/trust/cleanup — this is the "second driver" scaffold, currently wired only for the fixed `obsidian` suite (scripts/acceptance/obsidian.mjs:80-111), not for exploration.

**Minimal seam**: the loop already treats `driver` as an opaque interface (`exploration.mjs:117` signature) — no extraction needed there. The real work is a new `scripts/acceptance/exploration-obsidian.mjs` (mirroring `exploration-browser.mjs`) implementing `observe/execute/capture/dispose` over CDP plus an Obsidian-flavored `createDefaultExplorationTasks`-equivalent, and extending `exploration.mjs:254` `runExploration()` (currently hardcodes `startWorkbenchSession` + `createExplorationDriver` from exploration-browser.mjs, lines 263-266) to select a driver/session factory by target host. Estimated size: small-to-medium — most of `classifyControlRestriction` and the action-validation/loop code is reusable verbatim; new code is mainly the CDP-based `observe()` DOM harvester (~60-90 lines, adapting `browserObservation`) and CDP action execution (~40-60 lines), plus wiring `runExploration` to branch on a `host` option instead of unconditionally importing the browser driver.

## 3. Task kinds

Today the loop has **no path for a verify-less task**: `tasks.some(... typeof task.verify !== "function")` throws at entry (exploration.mjs:122), and a task only leaves the while-loop via a program-verified pass/fail, budget exhaustion, or `blocked`/`done`. There is no "free exploration, findings only" terminal state — `done` is always treated as *not* satisfying checks (201-203, `scenario.status = "blocked"`), it's never a legitimate success. For persona runs, either (a) add a task kind with `verify: undefined`/a sentinel and change exploration.mjs:122 and the `verified` check (160) to treat such tasks as "complete when model emits `done`" without requiring checks, while still collecting findings normally; or (b) give persona tasks a trivial always-true `verify` after some turn budget — hackier and would misleadingly show as "passed" in the scenario table. Likely needs a small branch: `task.freeform === true` tasks terminate on `done`/budget exhaustion with status `"explored"` (new, non-pass/fail) rather than `"blocked"`.

**Task selection**: `run.mjs`/`runExploration` always calls `createDefaultExplorationTasks(fixture)` unless the caller passes `tasks` explicitly (exploration.mjs:267); there's no CLI flag to pick a subset of a larger task library — only `tasks` as a programmatic override. A larger library (10-15 goal tasks + 2-3 persona tasks) would need a `--task` / `--persona` selection flag threaded through `run.mjs` parseArgs and `runExploration`.

**Per-task budgets**: none today — `maxSteps`/`maxDurationMs`/`maxInvalidActions`/`modelCallTimeoutMs` are all whole-run limits (exploration.mjs:118-121), consumed across however many tasks run; `metrics.modelCalls` is a single run-wide counter checked against `stepsLimit` per step (169) regardless of which task is executing. Extending to per-task step/call budgets needs new bookkeeping (a per-task counter plus still respecting the run-wide ones).

**Isolation/reset between tasks**: no reset exists. Tasks execute back-to-back against the *same* `driver`/page/fixture (the `for (const [index, task] of tasks.entries())` loop in exploration.mjs:140 shares one `driver`), and the three default tasks are explicitly written to depend on this (e.g. `sidebar-persistence`'s goal says "close settings if still open" — exploration-browser.mjs:224). Also the whole run `break`s after the first non-`passed` scenario (exploration.mjs:227), so later tasks never even get a chance — this alone blocks scaling to 10-15 independent goal tasks; it would need to continue past individual task failures (likely an intentional change, not just an oversight, given current 3-task design assumes a dependent linear story).

## 4. Cost data

From `docs/helm/2026-10-09-product-interaction-acceptance/verification.md:10`: real model exploration — **3 tasks passed, 19 API calls / 19 actions, 70.329s, 105,066 input + 640 output tokens**. That's ~6.3 model decisions/task, ~5,530 input tokens and ~34 output tokens per decision on average (history grows up to last-8 steps, so later steps in a task cost more than early ones). No local `output/playwright/acceptance/*/exploration*` evidence exists on disk currently (git-ignored, not present in this checkout) to get a finer per-step token breakdown; the only numbers available are this aggregate from the verification doc and the README's stated defaults (50 steps / 100 API calls / 600s, exploration.mjs:118-121, exploration-model.ts:83).

**Extrapolation for ~15 tasks × ~20 steps = 300 decisions**: at ~5,530 input + ~34 output tokens/decision ≈ 1.66M input + ~10K output tokens total per full exploration run — roughly 16x the measured 3-task run's token volume. Time: at ~3.7s/decision observed (70.3s/19), 300 decisions ≈ 18.5 minutes of model-call time alone (excludes action execution overhead), which exceeds today's default `maxDurationMs` (600000ms = 10 min) and would need that budget raised, consistent with the ticket's "token cost acceptable but should be bounded and reported" framing. The `maxApiCalls` default of 100 (exploration-model.ts:83) would also need raising for 300 decisions (each decision is ≥1 API call, more on provider retries).

## 5. Existing tests (`scripts/tests/acceptance-exploration.test.mjs`)

Covers: done-without-checks is blocked and stops remaining tasks (lines 40-48); only program checks after a real action can pass (50-62); vacuous verifier (empty checks) can never pass (64-69); invalid/unparseable/disallowed actions never reach the driver and bounded-retry blocks after `maxInvalidActions` (71-80); step-budget exhaustion (82-89); model-call deadline/abort via `AbortSignal` (91-100); API failure ⇒ `blocked` vs. independently-observed product error ⇒ `failed` (102-111); findings are kept separate from pass/fail and preserved with evidence in the events log (113-128); action-validation allowlist/boundary tests incl. secrets/paths/navigation (130-139, 230-243); real `LlmClient` wiring against a local HTTP stub — SSE streaming, model/field mapping, secret redaction in requests/describe, token usage accounting (168-184); missing config/key errors (186-192); budget-exhausting retries still counted (194-203); redaction survives into both events.jsonl and exploration.json (205-219); real-adapter deadline abort (221-228); parent-signal pre-abort and in-flight cancellation (246-256); `dailyEvidence()` race/symlink and failed-run verification edge cases (258-300).

**Fake-model injection pattern** (reusable for testing a new Obsidian driver without real API calls): `model` param to `runExplorationLoop` can be a **plain async function** `async (request) => '<json string>'` (not just the LlmClient-shaped object) — see `fixtureDriver()` (19-28) + inline `model: async () => '...'` lambdas throughout. This is the way to unit-test a new driver: supply a scripted/fake model function and a minimal `{observe,execute,capture}` driver stub, bypassing `loadExplorationModel`/esbuild/real HTTP entirely.

## 6. Unknowns

- No local exploration evidence JSON exists in this checkout to get finer per-step token/observation-size numbers (only the aggregated verification.md line); a real run would be needed for a precise per-step cost model (e.g. observation text/control-count size growth across a task).
- Whether `done` should ever be a legitimate terminal state for persona/free-exploration tasks is an open design decision — today it's unconditionally treated as a failure to satisfy checks (exploration.mjs:201-203); no existing code path distinguishes "free exploration done" from "goal task falsely claims done."
- No CDP equivalent currently exists for `scroll`/`back`/`reload` actions in the Obsidian harness (obsidian-ui.mjs has no such helpers); exact CDP calls to implement them are unresearched here.
- Not verified whether Obsidian's CDP target process model (one renderer, `session.evaluate`) can support the same "observation id ↔ data-attribute ref ↔ re-validate live DOM at execute time" staleness defense used in `exploration-browser.mjs:131,140-147` without adaptation — likely yes via `setAttribute` through `evaluate()`, but not confirmed against Obsidian's actual DOM (e.g. CSP, Electron sandboxing of `document` access).
- Per-task budget allocation strategy (equal split vs. priority) for 10-15 tasks is undecided; only run-wide budgets exist today.

# Recon: model-driven Obsidian plugin exploration — helm intake

## 1. Helm verdict

`grep -H "^status:" docs/helm/*/goal.md` shows 47 goals, 3 active besides this one:
`2026-08-11-email-relay-v2-production-cutover`, `2026-08-13-discovery-loop-and-library-insight`,
`2026-10-01-release-050-followups`, `2026-09-02-directions-inside-topics`,
`2026-10-03-workbench-core-parity`, plus `2026-08-27-obsidian-desktop-acceptance-harness` (active).
`2026-10-09-product-interaction-acceptance` is **done**.

- **`2026-08-27-obsidian-desktop-acceptance-harness` (active, P5 open)** built the CDP/launch/isolation
  layer this initiative must reuse: isolated XDG dirs, process-group lifecycle, vault-state
  backup/restore, build-deploy + version assertion, trust-dialog handling, app-usability guards,
  preflight (inotify headroom), screenshot pre-validation (visible/non-blank). Its remaining open
  criterion (P5, settings-page layout geometry/screenshots) is about **assertable geometry for
  human judgment**, not model-driven action — different shape of work, same substrate.
- **`2026-10-09-product-interaction-acceptance` (done)** shipped the actual pattern this initiative
  wants to extend: a model-API observe→act exploration loop with budgets, action whitelist, and
  program-checked tasks — but **only against the browser workbench** (`exploration-browser.mjs`).
  Its fixed Obsidian scenarios (`obsidian-scenarios.mjs`) are hand-coded CDP journeys, not model-driven.
- **Verdict: this is new work, not a resume.** No active goal's success criteria cover "LLM operates
  the Obsidian plugin via observe→act." It should be scoped as a new goal that explicitly **reuses**
  `scripts/desktop-acceptance/*` (session/launch/app-state/screenshots/preflight) and the exploration
  action/budget framework from `scripts/acceptance/exploration.mjs` + `exploration-model.ts`, and
  **extends** `obsidian-ui.mjs`/`obsidian-scenarios.mjs` rather than duplicating them. Recommend not
  touching `obsidian-desktop-acceptance-harness` or `product-interaction-acceptance` goal state per
  their own non-goals/constraints (both explicitly forbid altering parallel Helm state).

## 2. Vocabulary (root CONTEXT.md)

Must use: **Plugin product** (Obsidian-facing: settings UI, Dashboard, in-app scheduler), **Reading
workbench** (browser surface, CLI product), **Host** (Plugin/CLI are hosts, not two cores), **Core**,
**Vault** / **Vault data** / **Plugin settings store** (`data.json`, independent of CLI TOML),
**Research topic** / **Direction** / **Representative set**, **Daily report** (not "daily summary"),
**Paper note** (not "deep dive"), **Structured summary**, **Library processing consent**, **Send
yourself** / **Official delivery** for email modes, **Settings draft**. No conflicts found between
this initiative's draft framing and CONTEXT.md; just avoid "daily summary," "deep dive," "configuration
sync," and don't call `data.json` "plugin settings" interchangeably with CLI TOML.

## 3. Plugin user-facing surface inventory (plugin/src)

| Surface | Location | Program-checkable signal | Flag |
|---|---|---|---|
| Commands (14): run-now, run-for-date, run-all-pending, retry-failed, force-run-for-date, clear-run-state, cancel-current-run, summarize-by-id, set-paper-status, create-paper-note, mark-current-\* (per mark), open-today, send-test-email, open-reading-dashboard, review-personal-library-directions, review-incremental-suggestions, preview-personal-library-files, scan-personal-library, reload-personal-library-catalog, index-personal-library-fulltext | `plugin/src/commands.ts:306-510+` | `app.commands.executeCommandById`; check `run-state.json`/`papers.json`/vault files/Notice text | **send-test-email** sends via configured provider — denylist unless fixture-proxied. **clear-run-state** mutates bookkeeping — denylist for free-exploration. |
| Ribbon icon (dashboard) | `commands.ts:616` | DOM presence/click opens Dashboard view | — |
| Views: Reading Dashboard (tabs All/Starred/Detail-summary, toolbar Refresh/Run today/Summarize-by-ID/More menu/Settings) | `plugin/src/dashboard/view.ts` | `.arxiv-daily-dashboard__tab` counts, pagination state, `data.json`/index files | "More" menu toggles scheduler enable/disable — scheduler-enable is schedule-touching, treat as sensitive. |
| Modals: date picker, hub modal, similar-papers modal, interest-profile modal, paper-mark modal | `date-picker-modal.ts`, `dashboard/hub-modal.ts`, `dashboard/similar-papers-modal.ts`, `library/interest-profile-modal.ts`, `commands.ts` (mark modal) | input value + submit-button geometry/hit-test; resulting file/notice | — |
| Settings tab sections: LLM, arXiv categories (incl. Topics/Directions add/delete/reorder), Output & schedule, Personal library, Email delivery, Advanced | `plugin/src/settings/tab.ts:1422,1549,1699,1811,1971,2119`; `SettingsSection` type at `definitions.ts:107` | each field maps to a `data.json` key; verify via `pollJsonFile` pattern already used in `obsidian-scenarios.mjs` | **Email delivery** fields (api key/hosted token) + **LLM** api key are secrets — exclude from fill/read like workbench's `classifyControlRestriction`. |
| Personal library directory picker | `plugin/src/hosts/obsidian/library-directory-picker.ts` | native OS `showOpenDialog` — **outside CDP/DOM reach entirely** | Out-of-scope for autonomous model: touches real filesystem dialog, could pick a real directory. Must denylist/stub. |
| Resource opener: open note/daily report (in-vault `openLinkText`), open external URL (`window.activeWindow.open(..., "_blank")`) | `plugin/src/hosts/obsidian/resource-opener.ts` | vault navigation vs. external browser window | **openUrl** leaves the fenced environment — denylist external links. |
| PDF opening (`#page=N`) | `plugin/src/library/pdf-opener.ts` (not read in depth; confirmed exercised by existing `pdf.location` scenario and `scenarios.mjs` in desktop-acceptance) | page-location assertion already proven in P3 of the harness goal | reuse existing scenario, don't rebuild |
| Context menu ("More" toolbar menu — not a native right-click file-menu) | `dashboard/view.ts:1873` (`showMoreMenu`) | `Menu` items: scheduler toggle, Run for date…, Force run for date…, Retry failed dates… | scheduler toggle is schedule-touching, treat as denylist for free exploration |
| Onboarding/setup guide | `plugin/src/onboarding.ts` | `SetupStatus` fields (llmReady, categoriesReady, topicsReady, readyToRun, firstReportComplete, scheduleEnabled), `guideCompleted` marker in `data.json` | — |
| Email delivery service | `plugin/src/services/email-delivery.ts` | `testEmailResultMessage` / Notice text | sending real email is the clearest dangerous op; existing acceptance already proxies it through a fixture HTTP endpoint — same technique must gate any model-driven send-test-email task |
| Paper note creation / mark-as (star, to-read, etc.) | `commands.ts` mark-current-\* commands, `dashboard/actions.ts`, `files.ts` | vault file existence + `papers.json`/index mark field | writes real vault files under the test vault — fine inside the isolated fixture, must stay denylisted outside it |

No in-plugin data import/export UI was found (confirmed by CONTEXT.md: Data export/import and CLI
config are CLI-product concepts); plugin owns only `data.json` settings + vault data writes.

## 4. Existing Obsidian driving capability

- **Launch/isolation** (`scripts/desktop-acceptance/launch.mjs`, `session.mjs`): `xvfb-run` wraps
  Obsidian with `--remote-debugging-port`; every XDG dir (`XDG_CONFIG_HOME/DATA_HOME/CACHE_HOME`)
  redirected into a fresh `mkdtemp` sandbox so the user's real `~/.config/obsidian` and vault list are
  never touched. Process started via `spawnInProcessGroup`/reclaimed via `kill -PGID` (never by name).
  Vault mutable state is backed up/restored by `createVaultStateGuard`; current branch build is
  deployed and version-asserted (`deployBuildUnderTest`/`assertVersionUnderTest`) before every run.
- **CDP session**: `cdp.mjs` exposes `createCdpClient`, generic `evaluate(client, expr)` (Runtime.evaluate
  over the raw WebSocket), `selectVaultTarget`. `obsidian-ui.mjs`'s driver wraps this with `read`
  (JSON round-trip), `wait`/`poll`, `command` (executes registered plugin commands by id),
  `click(selector)`/`fill(selector, value)` (both just call JS `.click()`/set `.value` + dispatch
  input/change events — **not real input events**), plus one real `Input.dispatchMouseEvent` sequence
  used specifically for the date-submit button (`session.mjs:122-124`) because Electron's modal needs
  genuine hit-testing (`dateSubmitTargetExpression` + `waitForStableTarget`).
- **App usability / diagnostics**: `app-state.mjs` (`waitForAppUsable`/`withAppUsable`) brackets every
  walk with positive-capability checks (app/workspace/setting objects, leaf count) — a collapsed host
  aborts with zero reported assertions rather than false green. `diagnostics.mjs` collects
  console/pageerror from attach time (before the trust dialog, so plugin startup is in-window).
- **Screenshots**: `screenshots.mjs` validates target existence/visibility/size/viewport-fraction and
  rejects solid-color captures (custom PNG decoder) *before* writing to disk.
- **Settings/fixture seeding**: `createObsidianSettings({ endpoint })` in `obsidian.mjs` builds a
  complete `data.json` fixture (LLM pointed at the acceptance fixture server, one topic/direction,
  schedule disabled, email disabled, embedding local, pdf sidecar disabled); `installSettingsFixture`
  writes it before launch. `installHttpProxyExpression` rewrites the plugin's `HttpClient.request` to
  route through `/proxy` on the fixture server, so paper-source/LLM/email calls stay fixture-controlled
  without touching real endpoints — this is the reusable mechanism for whitelisting external calls
  during model-driven runs too.
- **Reusable for an observe→act loop**: the whole launch/isolation/diagnostics/screenshot/app-usability
  stack, the fixture-proxy technique, and the budget/action-schema layer in `exploration.mjs` (driver-
  agnostic: `parseExplorationAction(raw, observation)` only assumes an `observation.controls[].ref`
  shape) and `exploration-model.ts` (the model call loop itself).
- **Gaps versus the browser's `exploration-browser.mjs`**: Obsidian has **no generic DOM observation**
  function analogous to `browserObservation()` (which walks all visible interactive elements, tags
  `data-acceptance-ref`, and reports restricted/sensitive/rect/inViewport for *any* control) — the
  Obsidian driver today only has scenario-specific hardcoded selectors/commands. There is **no generic
  click-by-ref** (only `.click()` via evaluate, or the one hand-built real-mouse sequence for the date
  dialog) and **no generic fill/keyboard typing** (current `fill()` sets `.value` directly, bypassing
  CodeMirror-style editors if any settings field used one — none currently appear to). **No generic
  scroll.** Critically, **nothing fences off Obsidian's own chrome** (ribbon bar, file explorer,
  command palette, status bar, other community plugins) from the plugin's own UI the way
  `classifyControlRestriction`/`inScope` fence the workbench to one origin/path — this must be designed
  before a model can safely "observe the whole screen." Native dialogs (library directory picker) are
  invisible to CDP DOM evaluation entirely and must be denylisted/stubbed outright.

## 5. Unknowns

- Did not fully read `plugin/src/settings/tab.ts` (3430 lines) or `declarative-rows.ts`/`definitions.ts`
  field-by-field; section list (LLM / arXiv categories+Topics / Output & schedule / Personal library /
  Email delivery / Advanced) and a few anchors are confirmed, but the complete per-field inventory
  (every topic/direction add/delete/reorder control, every library sub-control) was not enumerated line
  by line.
- Did not read `plugin/src/library/pdf-opener.ts`, `library/modal.ts`, `library/connection.ts`, or
  `library/proposal-acceptance-state.ts` in depth — PDF opening and direction-review/proposal-acceptance
  surfaces are confirmed to exist (and partly exercised by `scenarios.mjs`/`direction-review.mjs` in
  `scripts/desktop-acceptance`) but not inventoried field-by-field.
- Did not read `scripts/acceptance/exploration-model.ts` (the model HTTP-call loop) or
  `scripts/acceptance/run.mjs`/`report.mjs` end-to-end — budget enforcement details (steps/API
  calls/duration) are described in `scripts/acceptance/README.md` but not verified against code.
- Did not verify whether `scripts/desktop-acceptance/direction-review.mjs`, `topic-directions.mjs`, or
  `library-settings.mjs` already cover some of the "library" settings-section interactions the new
  initiative would want as goal tasks (likely yes, given their names, but not opened).
