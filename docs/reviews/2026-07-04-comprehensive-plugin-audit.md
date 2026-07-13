# Comprehensive Plugin Audit — 2026-07-04

> Full review of the Arxiv Daily Obsidian plugin (branch: `refactor/scheduler-hybrid`)
> 28 specialized agents covering all subsystems, analyzing 80+ source files and 50+ test files

## Review Methodology

This review was conducted by **28 specialized agents** across 5 phases:

| Phase | Agents | Focus |
|-------|--------|-------|
| Explore Part 1 — Pipeline & LLM | 4 | Pipeline orchestrator, fetching/parsing, LLM/filter/summarizer, content writing |
| Explore Part 2 — Scheduling & State | 4 | Scheduler driver, run-gate/state-store, lock/cancellation, services |
| Explore Part 3 — Dashboard & Hosts | 4 | Dashboard view/model, Obsidian adapters, Node adapters, main entry/CLI |
| Explore Part 4 — Settings & Infra | 4 | Settings types, categories/tab, utilities, build config |
| Deep Review | 6 | Bugs/race conditions, TypeScript type safety, performance, security, data integrity, async patterns |
| Final synthesis | — | All findings consolidated below |

---

## Executive Summary

| Dimension | Critical | High | Medium | Low / Info | Overall |
|-----------|----------|------|--------|------------|---------|
| Bug / Race Conditions | 1 | 3 | 6 | 2 | ⚠️ **Moderate risk** |
| Type Safety | 0 | 0 | 3 | 4 | ✅ **Good** |
| Data Integrity | 1 | 3 | 5 | 2 | ⚠️ **Needs attention** |
| Security | 0 | 1 | 4 | 2 | ✅ **Acceptable** |
| Performance | 0 | 3 | 4 | 4 | ✅ **Good** |
| Async / Promises | 1 | 1 | 3 | 3 | ⚠️ **Needs attention** |
| Settings / Config | 0 | 1 | 4 | 3 | ✅ **Good** |
| Test Coverage | 0 | 0 | 4 | 2 | ⚠️ **Medium gaps** |

---

## Detailed Findings

### 1. Pipeline & LLM Layer

#### HIGH: Paper index mutated before daily file is committed to disk
- **File:** `plugin/src/pipeline/pipeline.ts:261-278` vs `:327`
- **Description:** `addDailyReports` and `setSummaries` update the paper index to reference `dailyPath` and store summaries, but `writeDaily` is only called at line 327 — **after** the detail-report loop. If anything between lines 270-327 throws (cancellation, detail-loop error, or `writeDaily` itself fails), the index is left pointing at a daily file that does not exist on disk. No rollback exists. On retry, `dailyExists` (line 99) returns `false` so the pipeline re-runs, but the index entries already carry stale `dailyReports` references.
- **Impact:** Stale index references to non-existent files; phantom "daily exists" state corruption
- **Recommendation:** Move `addDailyReports` + `setSummaries` to **after** `writeDaily` succeeds. Consider wrapping index mutations in a rollback wrapper (save pre-mutation state, revert on error).

#### HIGH: `addDailyReports` and `setSummaries` are non-atomic
- **File:** `plugin/src/pipeline/pipeline.ts:264-270`
- **Description:** Two separate `enqueueMutation` calls. If `addDailyReports` succeeds but `setSummaries` throws, the index has daily-report references but no summaries. The catch at line 271 returns `failed_transient`, but the partial mutation persists.
- **Impact:** Partial index state — daily-report references exist with no summaries, causing read errors in downstream consumers
- **Recommendation:** Wrap both mutations in a single queue job or add a compensating rollback in the catch block.

#### MEDIUM: `setPaperPath` in "already exists" branch not wrapped in try/catch
- **File:** `plugin/src/pipeline/pipeline.ts:288-297`
- **Description:** Unlike the write branch (lines 300-322) which has a try/catch, the `setPaperPath` call in the "detail already exists" path is unprotected. If it throws a non-cancellation error, the exception propagates out of the detail loop before `writeDaily` runs.
- **Impact:** Detail files exist on disk with index paper paths, but no daily file is written — inconsistent state
- **Recommendation:** Wrap in try/catch like the other branch, logging and continuing on error.

#### MEDIUM: Cancellation reported as `failed_transient`, indistinguishable from transient failure
- **File:** `plugin/src/pipeline/pipeline.ts:75`
- **Description:** `PipelineResult` has no `cancelled` variant. A user-cancelled run is returned as `failed_transient`. The scheduler will retry it, defeating the user's cancel intent.
- **Impact:** Cancel doesn't stick — scheduler retries cancelled runs
- **Recommendation:** Add a `cancelled` variant to `PipelineResult` and handle it distinctly in `scheduler-driver.ts`.

#### MEDIUM: `fetchRecent` and `fetchMetadataByIds` do not receive abort signal
- **File:** `plugin/src/pipeline/pipeline.ts:127,418`
- **Description:** Neither call passes `signal`. Cancellation during these network fetches won't take effect until the in-flight request completes; `throwIfCancelled` after the await only fires once the request returns.
- **Impact:** Network fetches continue even after user cancels, wasting quota and time
- **Recommendation:** Thread `signal` through to HTTP calls. Use `AbortSignal.timeout` or pass to `HttpClient.request`.

#### LOW: Per-category failures silently dropped when any category succeeds
- **File:** `plugin/src/pipeline/pipeline.ts:472-477`
- **Description:** `collapseCategoryFailures` is only called when `succeededCategories.length === 0`. If 2 of 5 categories have permanent parse failures but 3 succeed, the permanent failures are never reported — the pipeline continues with partial data and returns `completed`. Permanent parse errors (usually schema drift) should be surfaced to the user.
- **Impact:** Silent data loss — user doesn't know categories are failing
- **Recommendation:** Log permanent failures at `error` level (currently `warn`) and include them in the pipeline result reason.

---

### 2. Scheduling & State Management

#### CRITICAL: `cancelAll` is racy and can leave scheduler disabled until manual intervention
- **File:** `plugin/src/services/cancellation.ts:29-39`, `plugin/src/services/scheduling/scheduler-driver.ts:113,124,214,229,253,277`
- **Description:** `cancelAll` snapshots `controllers.keys()`, adds dates to `cancelledDates`, then aborts each controller. But `begin()` (lines 15-23) deletes the date from `cancelledDates` only when a **new** run starts for that date. After `cancelAll`, the `cancelledDates` set remains non-empty for dates where no run starts. Every subsequent tick checks `isCancellationRequested()` and short-circuits — **effectively disabling the scheduler permanently** until the user manually triggers runs for each cancelled date. The `isCancellationRequested` flag is global, so cancelling one batch taints all concurrent batches.
- **Impact:** Scheduler silently stops working after a cancellation; only discoverable via log inspection
- **Recommendation:** Redesign cancellation to be per-run or per-batch, not a global flag. Clear `cancelledDates` when all cancelled controllers have finished (not on `begin()`). Consider a token-based model instead.

#### HIGH: `tryRun` lock + cancellation interleave can leave a date stuck at `running` forever
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:386-394`, `plugin/src/services/scheduling/run-gate.ts:19`
- **Description:** In `tryRun`: lock acquired → `store.setRunning(date)` → `runForDate` runs. If `setRunning` succeeds but `setFailed` **itself** throws (e.g. disk write fails), the catch at line 386 swallows the error and logs it. The state remains `"running"` forever — the lock is released by `finally` at line 392, but the store entry is stuck. Every future tick hits `checkTickGate` → `reason: "running"` (run-gate.ts:19) and skips forever. `clearDate` is only invoked by `forceRunForDate`/`retryFailedInLookback`, not by the periodic tick. **No recovery path exists.**
- **Impact:** A date is stuck in "running" state permanently, never re-attempted
- **Recommendation:** Add a recovery path: if `setFailed` throws, also call `store.clearDate(date)` or `store.setFailed(date, "permanent")` via a fallback path. Consider a startup recovery pass that un-sticks dates stuck in "running" for >24h.

#### HIGH: `pending` result silently drops transient failure counter
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:368-373`, `plugin/src/services/state-store.ts:105`
- **Description:** When `result.kind === "pending"`, the driver calls `store.clearDate(date)` (line 370), removing all state including `attempts` and `lastAttempt`. No state transition `running → pending` is recorded. This resets the attempt counter to 0, defeating the `MAX_TRANSIENT_ATTEMPTS = 10` promotion logic in `setFailed` (state-store.ts:105). Failed attempts are lost; the date can be retried indefinitely never escalating to permanent failure.
- **Impact:** A date can cycle `running → pending → running` thousands of times with no escalation to permanent
- **Recommendation:** Record the attempt counter before `clearDate`, or don't clear date on pending — instead store `pending` status with the current attempt count.

#### MEDIUM: `tick` interval fires can overlap — no re-entrancy guard
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:64-67`
- **Description:** `setInterval(tick, interval)` fires every interval regardless of whether the previous `tick()` is still running. Per-date lock prevents corruption, but wasted iterations and progress-report races occur.
- **Impact:** Wasted cycles; progress bar flicker; log noise
- **Recommendation:** Skip tick if already running: `if (this.ticking) return; this.ticking = true; try { await this.tick(); } finally { this.ticking = false; }`

#### MEDIUM: `forceRunForDate` clears state outside lock — TOCTOU race
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:202`
- **Description:** `forceRunForDate()` calls `store.clearDate(date)` **outside** `withLock`. A concurrent tick can see the cleared state and start a duplicate pipeline run. Same pattern in `retryFailedInLookback` at line 220.
- **Impact:** Rare but possible duplicate runs for the same date
- **Recommendation:** Move `clearDate` inside `withLock`, or have `tryRun` accept a "force clear before start" flag.

#### MEDIUM: State store read-modify-write not atomic across instances
- **File:** `plugin/src/services/state-store.ts:158-170,264-273`
- **Description:** `enqueueMutation` serializes mutations only within a single `StateStore` instance. Two Obsidian windows or external modification to `run-state.json` between load and save will be silently overwritten. Additionally, `snapshot()`, `get()`, and `failedDates()` read from in-memory `this.state` which is only refreshed at the start of each queued mutation — so observers (including run-gate) can see stale state.
- **Impact:** Stale run-gate decisions; potential duplicate runs in multi-window Obsidian
- **Recommendation:** Document single-instance assumption. Add in-memory refresh interval or subscription for observers.

---

### 3. Dashboard & Hosting Layer

#### MEDIUM: Race condition in calendar month refreshes
- **File:** `plugin/src/dashboard/view.ts:1059-1072,616-622`
- **Description:** Rapid clicks on prev/next month trigger overlapping `refreshCalendarDailyReports(month)` calls. Each sets `this.calendarDailyReports` then calls `this.render()`. If month A's refresh resolves after month B's, month-A data overwrites month-B's `calendarDailyReports` while `calendarMonth` is already month-B. The next render shows month-B header with month-A report counts. No sequencing/abort guard.
- **Impact:** Calendar shows wrong paper counts for selected month after rapid navigation
- **Recommendation:** Use an abort/sequence token: increment a counter on each navigation, store the current token, and on resolution skip rendering if the token has changed.

#### MEDIUM: Full DOM rebuild on every state change
- **File:** `plugin/src/dashboard/view.ts:635-672`
- **Description:** `render()` calls `contentEl.empty()` and rebuilds everything — header, toolbar, tabs, filters, calendar, batch controls, results table, pagination — on every filter change, tab switch, sort change, page change, and month navigation. For each paper row (lines 1563-1632), creates 12+ DOM elements (checkbox, star, title, meta, topic, date, 5 action buttons). With 100 papers visible, that's 1200+ DOM nodes created and attached per render. Every render re-attaches click handlers with no event delegation.
- **Impact:** UI flicker; unnecessary layout recalculations; poor performance on low-end machines
- **Recommendation:** Use event delegation on table body. Only re-render the affected section (e.g., only tbody instead of the whole view). Consider document fragments for batch DOM insertion.

#### MEDIUM: Stale "Detail summary" filter count after tab switch
- **File:** `plugin/src/dashboard/view.ts:929-934`
- **Description:** Tab buttons call only `updateTabButtonState(tabs)` + `renderCurrentResults()`. The "Detail summary" filter button and its count badge (from `renderToolbarFilter`) are not refreshed, so the count reflects the previous tab.
- **Impact:** Misleading UI — user sees wrong paper counts
- **Recommendation:** Refresh filter counts on tab switch or recompute from current tab's data.

#### LOW: Node `writeText` does not create parent directories
- **File:** `plugin/src/hosts/node/storage-adapter.ts:20-22`
- **Description:** `appendText` (line 24) and `rename` (line 48) both create parent dirs with `mkdir({ recursive: true })`, but `writeText` and `writeBinary` do not. Obsidian's adapter creates parents automatically, so callers that work under Obsidian will throw `ENOENT` under Node when the parent directory doesn't exist.
- **Impact:** File operations fail under Node CLI with non-obvious `ENOENT` errors
- **Recommendation:** Add `path.dirname + mkdir` before `writeFile` in `writeText` and `writeBinary`.

#### LOW: Node `EnvSecretProvider` silently no-ops on `setSecret`/`deleteSecret`
- **File:** `plugin/src/hosts/node/secrets.ts:3-21`
- **Description:** The `SecretProvider` interface marks `setSecret`/`deleteSecret` as optional. The Node implementation omits both — `secrets.setSecret?.(...)` silently no-ops. Under Obsidian, these methods persist. Any code path that mutates API keys at runtime will appear to succeed under Node but lose the value on next restart.
- **Impact:** Silent data loss for runtime API key changes under Node CLI
- **Recommendation:** Implement with a throw (or store in-memory with a log warning).

---

### 4. Settings & Infrastructure

#### MEDIUM: `versions.json` missing current release
- **File:** `plugin/versions.json:22`
- **Description:** Manifest declares version `0.1.21`, but `versions.json` only contains entries up to `0.1.20`. Obsidian uses this map to verify `minAppVersion` per release.
- **Impact:** Plugin update validation may fail; users may not get update notifications
- **Recommendation:** Add `"0.1.21": "..."` entry to `versions.json`.

#### MEDIUM: Double change-listener on model dropdown
- **File:** `plugin/src/settings/tab.ts:941`
- **Description:** `showModelDropdown` adds a `select.addEventListener("change", ...)` handler on top of the `d.onChange` handler registered at line 223. After "Get Models" is clicked, every dropdown change fires both handlers → `s.llm.model` written twice and `saveSettings()` runs twice.
- **Impact:** Redundant saves; potential race if saves interfere with each other
- **Recommendation:** Remove the redundant event listener, or use a single `onChange` handler per dropdown.

#### MEDIUM: Quick-start template silently overwrites categories
- **File:** `plugin/src/settings/tab.ts:350`
- **Description:** `apply()` sets `s.arxiv.category = tpl.category` and `s.arxiv.categories = [tpl.category]`, but the confirm modal (line 361) only mentions replacing topics. Picking e.g. "NLP / LLMs" discards the user's existing multi-category list without warning.
- **Impact:** User loses customized multi-category configuration
- **Recommendation:** Show a warning in the confirm modal about categories being replaced.

#### MEDIUM: `isMinutesWithinWindow` silently returns `false` for cross-midnight windows
- **File:** `plugin/src/utils/time.ts:58`
- **Description:** `isMinutesWithinWindow` returns `false` whenever `startMinutes > endMinutes`, so cross-midnight configs (e.g. `"23:00"–"02:00"`) can never return `true` from `isTimeWithinLocalWindow`. This is a silent correctness bug, not a guard against invalid config.
- **Impact:** Overnight scheduler windows never fire; no error or warning shown to user
- **Recommendation:** Handle the `start > end` case as "overnight" (check if current time is >= start OR < end). Or validate the config and reject cross-midnight windows with an error message.

#### MEDIUM: `isWeekendDate` uses UTC arithmetic on tz-local dates
- **File:** `plugin/src/utils/time.ts:84-87`
- **Description:** `isWeekendInTz` correctly uses `Intl.DateTimeFormat` with `timeZone: tz`, but `isWeekendDate` computes the weekday via `Date.UTC(...).getUTCDay()`. A `{y,m,d}` triple from `todayInTz(now, tz)` is a tz-local calendar date, but `isWeekendDate` reads it as UTC. For timezones whose UTC offset crosses a day boundary (e.g. `-04:00` late at night), the weekend check is off by one day.
- **Impact:** Weekend detection can be wrong by ±1 day for timezones with large positive/negative UTC offsets
- **Recommendation:** Compute weekday from `Intl.DateTimeFormat({timeZone: tz, weekday})` or pass `tz` as a parameter.

#### LOW: `slugify` silently produces empty strings for non-ASCII input
- **File:** `plugin/src/utils/slugify.ts:5`
- **Description:** `[^a-z0-9-]` deletes all non-ASCII. `slugify("论文 总结")` returns `""`, `slugify("Café résumé")` returns `"caf-rsum"`. For arxiv-daily that may ingest CJK titles, empty slugs cause collisions/overwrites.
- **Impact:** Empty filenames or collisions when papers have non-Latin titles
- **Recommendation:** Use `\p{L}\p{N}` with `u` flag, or `normalize("NFD")` + strip combining marks, or transliterate.

#### LOW: `retry.ts` backoff can overflow to `Infinity`
- **File:** `plugin/src/utils/retry.ts:25`
- **Description:** `baseDelayMs * Math.pow(backoff, attempt-1)` with `backoff=10`, ~20 attempts yields `1e19` → `Infinity`. `Math.max(0, Infinity)` stays `Infinity`; `setTimeout(done, Infinity)` is clamped by Node to ~2.1e9 ms (~24 days). No `backoff` validation or jitter.
- **Impact:** Retry delays become astronomically long; no jitter causes thundering herd
- **Recommendation:** Cap maximum delay (e.g., 30 minutes). Validate `backoff > 1`. Add jitter (e.g., `delay * (0.5 + Math.random() * 0.5)`).

---

### 5. Race Conditions & Concurrency (Deep)

#### HIGH: `cancelAll` global flag taints all concurrent/scheduled batches
- **File:** `plugin/src/services/cancellation.ts:46-48` (duplicate from above, severity: HIGH)
- **Impact:** After any cancellation, all future scheduler ticks are blocked until manual recovery
- **Recommendation:** Make cancellation per-run/per-batch, not global. Add automatic drain of stale cancellation tokens.

#### MEDIUM: `tryRun` state stuck at "running" if `setFailed` itself fails
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:386-394`
- **Impact:** Date stuck permanently unreachable by scheduler
- **Recommendation:** Add startup recovery pass that un-sticks dates in "running" for >1 hour.

#### MEDIUM: No tick re-entrancy guard
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:64-67`
- **Impact:** Wasted iterations, log noise, progress flicker
- **Recommendation:** Add `this.ticking` boolean guard.

#### MEDIUM: `clearDate` outside lock — TOCTOU
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:202,220`
- **Impact:** Occasional duplicate runs for same date
- **Recommendation:** Move clear inside lock.

#### LOW: `cancelCurrentRun` returns without awaiting
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:77`
- **Description:** Returns immediately after firing abort signals — callers can never wait for cancellation to take effect.
- **Impact:** `onunload` may complete before in-flight pipelines have cleaned up
- **Recommendation:** Return a promise that resolves when all cancelled runs acknowledge.

---

### 6. TypeScript Type Safety

#### MEDIUM: PipelineResult discriminated union lacks `cancelled` variant
- **File:** `plugin/src/pipeline/pipeline.ts:35-39`
- **Description:** 4 variants exist (`completed`, `pending`, `failed_transient`, `failed_permanent`). Cancellation maps to `failed_transient`, which is semantically wrong. No downstream consumer can distinguish cancellation from infrastructure failure.
- **Impact:** Schedule logic cannot honor user cancellation intent
- **Recommendation:** Add `cancelled` variant, handle distinctly in all consumers.

#### MEDIUM: Missing `noUncheckedIndexedAccess` in tsconfig
- **File:** `plugin/tsconfig.json:6`
- **Description:** `strict: true` does NOT enable `noUncheckedIndexedAccess`. Without it, `arr[i]` and `obj[key]` are typed as `T` rather than `T | undefined`. Particularly relevant for the heavy index/key access in `paper-index.ts`, `state-store.ts`, and `cli/config.ts`.
- **Impact:** Runtime `undefined` bugs from unchecked array/object access not caught at compile time
- **Recommendation:** Add `"noUncheckedIndexedAccess": true` and fix resulting type errors.

#### MEDIUM: `parseRunState` accepts schema-invalid JSON silently
- **File:** `plugin/src/services/state-store.ts:240-248`
- **Description:** `JSON.parse` throws on malformed input (handled by `.bak` fallback), but syntactically-valid but schema-mismatched content — e.g. `status: 42` or `attempts: "3"` — is accepted without validation. Downstream code compares `entry.status === "running"` (run-gate.ts:19, 49); a corrupted status string lets the gate fall through to `allow: true`, potentially running a date that was already completed.
- **Impact:** Corrupted state file silently accepted; incorrect scheduler decisions
- **Recommendation:** Add JSON Schema or Zod validation on load, falling back to `.bak` on validation failure.

#### LOW: `writeAtomic` crash window silently loses state if primary file is missing
- **File:** `plugin/src/services/state-store.ts:222,229-236,287-311`
- **Description:** `writeAtomic` does: write `.tmp` → rename `path`→`.bak` → rename `.tmp`→`path` → remove `.bak`. If crash occurs between the two renames, the primary file is missing. `loadRunStateWithFallback` checks `exists(path)` first — if missing, it returns `{}` **without checking `.bak`** (line 222 returns before reaching the backup code in catch at line 225). The backup is unreachable.
- **Impact:** Complete state loss on crash in a ~10ms window
- **Recommendation:** When `exists(path)` is false, also check `exists(path + ".bak")` and load from it.

#### LOW: Run history `decodeRunHistoryLines` silently drops malformed lines
- **File:** `plugin/src/services/run-history.ts:180-205`
- **Description:** Lines that fail `JSON.parse` are silently skipped. A crash mid-append produces a truncated JSONL line, which is silently dropped. Data loss is invisible.
- **Impact:** Silent loss of run history entries
- **Recommendation:** Log the skipped line content (truncated to 200 chars) at `warn` level.

---

### 7. Performance

#### HIGH: Sequential content fetch — 20-50 papers fetched one-at-a-time
- **File:** `plugin/src/pipeline/pipeline.ts:191-234`
- **Description:** Each paper's content is fetched sequentially: `for (let i = 0; i < visiblePapers.length; i++) { await this.deps.paperFetcher.fetch(p.id) }`. For 20-50 papers, this is the pipeline's longest sequential tail latency.
- **Impact:** Pipeline takes proportionally longer as paper count grows (O(n) instead of O(1) parallel)
- **Recommendation:** Use a concurrency-limited pool (e.g., 5-10 concurrent with a semaphore). Each fetch is independent.

#### HIGH: Sequential detail summaries — each is an LLM call, done serially
- **File:** `plugin/src/pipeline/pipeline.ts:285-323`
- **Description:** Detail reports loop calls `summarizePaperDetail` + `writePaperDetail` per paper, sequentially. Each is an LLM call — the dominant cost. These are fully independent and could be parallelized.
- **Impact:** Detail report generation is O(n) wall-clock time for n detail papers
- **Recommendation:** Parallelize with a concurrency limit. The `dailyExists` check before each write (line 288) already guards against duplicate work.

#### HIGH: `openai` SDK dominates bundle size (~80%) when only chat completions used
- **File:** `plugin/package.json:16`
- **Description:** The OpenAI JS SDK is ~1.5-2.5 MB minified. The codebase only uses chat completions with streaming, which could be served by a ~50-line `fetch`-based wrapper. `fetchModels` already bypasses the SDK.
- **Impact:** Plugin `main.js` is >2 MB when it could be ~300 KB
- **Recommendation:** Replace `openai` SDK with raw `fetch` calls for chat completion streaming. This cuts plugin size by ~80%.

#### MEDIUM: Sequential category fetches could be parallelized
- **File:** `plugin/src/pipeline/pipeline.ts:414-470`
- **Description:** Categories are fetched one-at-a-time via `for (const category of categories) { await fetcher.fetchRecent(category) }`. With 5-10 configured categories, this is 5-10 sequential HTTP round-trips. arXiv's `/list/` pages are independent.
- **Impact:** Pipeline startup latency scales linearly with category count
- **Recommendation:** Use `Promise.allSettled` with the aggregation logic already being idempotent and race-safe.

#### MEDIUM: Full DOM rebuild on every state change (dashboard)
- **File:** `plugin/src/dashboard/view.ts:635-672`
- **Impact:** UI flicker, performance degradation with many papers
- **Recommendation:** Targeted re-renders, event delegation, document fragments.

---

### 8. Security

#### MEDIUM: Prompt injection defense relies on model obedience, not sanitization
- **Files:** `plugin/src/prompts/injection-guard.md`, `plugin/src/pipeline/paper-filter.ts:54`, `plugin/src/pipeline/summarizer.ts:168,457-462`
- **Description:** Paper titles, abstracts, and authors are interpolated raw into `<paper_data>` tags in the user message. The injection-guard system prompt instructs the model to treat content inside `<paper_data>` as data. However, there is **no character-level sanitization** — no stripping of control characters, no escaping of `</paper_data>` sequences. A paper whose metadata deliberately contained `</paper_data>` followed by instructions could break out of the tag wrapper. The defense relies entirely on the LLM obeying the guard instruction (empirical, with no guarantee).
- **Impact:** Potential LLM prompt injection from malicious paper metadata; model could be tricked into following instructions embedded in paper titles/abstracts
- **Recommendation:** Strip or encode `<` characters in user-supplied text before interpolation. At minimum, reject input containing `</paper_data>`. This is a low-cost hardening step.

#### LOW: `dangerouslyAllowBrowser: true` hardcoded in LLM client
- **File:** `plugin/src/llm/client.ts:95`
- **Description:** This disables the OpenAI SDK's browser-safety guard, which warns about API key exposure in browser contexts. For Obsidian plugins this is unavoidable — the API key is shipped to the browser context and sent directly to the LLM provider. Users with custom `baseUrl` pointing to a third-party proxy will leak their key to that proxy.
- **Impact:** API key exposed to any third-party proxy the user configures
- **Recommendation:** Document this in settings UI: "Your API key is sent directly to the configured LLM provider."

#### LOW: Stream idle timeout misclassified as user cancellation
- **File:** `plugin/src/llm/client.ts:280-284`
- **Security/Reliability impact:** Timeout-triggered abort creates an `AbortError` that `isCancellationError` treats as user cancellation, skipping all retries. A transient network hiccup becomes a permanent failure.
- **Recommendation:** Use a sentinel error class (not AbortError) for idle timeouts so they are distinguishable from user cancellation.

#### LOW: `fetchModels` swallows all errors silently
- **File:** `plugin/src/llm/client.ts:151-154`
- **Description:** `catch { continue }` inside a loop over candidate URLs means if all URLs 401 (bad key), the user sees only the generic "Failed to fetch models from any endpoint" — they cannot distinguish auth failure from network failure.
- **Impact:** Poor debugging experience when API key is invalid
- **Recommendation:** Preserve last error message in the thrown exception.

#### LOW: API key stored as plaintext in plugin settings
- **File:** `plugin/src/hosts/obsidian/secrets.ts:16-28`
- **Description:** Plugin stores API key in `PluginSettings.llm.apiKey` — a plain string in unencrypted JSON. Matches standard Obsidian plugin pattern but any process with filesystem access to the vault can read it.
- **Impact:** API key readable by any process with vault filesystem access
- **Recommendation:** Document this risk. Consider OS keychain integration for Node CLI.

---

### 9. Data Integrity

#### HIGH: File writes are not atomic — crash mid-write produces partial files
- **File:** `plugin/src/pipeline/markdown-writer.ts` (all write methods), `plugin/src/hosts/obsidian/storage-adapter.ts:16`, `plugin/src/hosts/node/storage-adapter.ts:20-22`
- **Description:** Every write method — `writeDaily`, `writePaperDetail`, `writePaperNote`, `refreshPaperNoteFrontmatter` — calls `storage.writeText(path, content)` directly with no temp-file + rename pattern. On Linux, `writeFile` is not atomic for files larger than ~4KB (common for daily reports with ~50 papers). Obsidian's `Vault.adapter.write()` is also not documented as atomic. If the plugin crashes mid-write, a truncated `.md` file remains. On next run, `paperDetailExists` sees the partial file and errors "paper already exists" — the paper is stuck in a half-written state with no recovery.
- **Impact:** Partial/corrupt files on crash; papers stuck in unrecoverable half-written state
- **Recommendation:** Implement write-temp-then-rename for all markdown file writes, matching the pattern already used in `state-store.ts:writeAtomic`. Add a recovery scan on startup that removes partial `.tmp` files and validates `.md` file integrity.

#### MEDIUM: Run history rotation crash window can lose history entries
- **File:** `plugin/src/services/run-history.ts:254-276`
- **Description:** `rotateHistoryFiles` removes `${path}.${maxRotations}` then renames files down. If crash occurs mid-rotation, history files are in an inconsistent state. The next append starts with `current = ""` and overwrites, losing the rotated tail.
- **Impact:** Run history loss on crash during rotation
- **Recommendation:** Perform rotation as an atomic operation: write new rotated state to temp files, then rename them into place.

#### MEDIUM: `setFailed` escalation produces internally inconsistent history record
- **File:** `plugin/src/services/state-store.ts:97-119`, `plugin/src/services/scheduling/scheduler-driver.ts:375-378`
- **Description:** `setRunning` increments `attempts`. `setFailed` checks `prev.attempts >= MAX_TRANSIENT_ATTEMPTS` (post-increment) and escalates to `failed_permanent`. But `recordFailed` at line 378 passes `result.kind` (still `"failed_transient"` from the pipeline) while the persistent `status` in the store is `"failed_permanent"`. The history record is internally inconsistent: `resultKind: "failed_transient"` + `status: "failed_permanent"`.
- **Impact:** Consumers that join on these fields see contradictory data
- **Recommendation:** Pass the actual persisted status to `recordFailed` instead of the pipeline's raw `result.kind`. Or recompute status at record time from the store.

#### MEDIUM: `daysBefore` uses UTC arithmetic on tz-local dates
- **File:** `plugin/src/utils/time.ts:62-73`
- **Description:** Subtracts `n * 86_400_000` from `Date.UTC(date.y, date.m-1, date.d)` and reads back via `getUTC*`. The arithmetic is calendar-correct only if input/output are both UTC-framed, but `todayInTz` produces tz-framed triples. `daysBefore(todayInTz(now, "Asia/Tokyo"), 1)` returns the previous UTC day, not the previous Tokyo-local day — wrong around the Tokyo/UTC day boundary.
- **Impact:** Date arithmetic off by one day for timezones near the UTC boundary
- **Recommendation:** Convert to epoch milliseconds using the timezone-aware `Date` constructor, then subtract days.

#### LOW: `onunload()` may complete before async cleanup finishes
- **File:** `plugin/main.ts:158-162`
- **Description:** `onunload()` fires abort signals via `scheduler.cancelCurrentRun()` but does not wait for in-flight pipelines to clean up. Obsidian may destroy the plugin context before the async error chain writes `failed_transient` to the state file. The file is left with `status: "running"`, blocking all scheduler paths (including `forceRunForDate`) on next load.
- **Impact:** Date stuck in "running" state after plugin unload/reload cycle
- **Recommendation:** Await cancellation completion in `onunload()`. Also add startup recovery for dates stuck in "running".

---

### 10. Async & Promise Handling

#### MEDIUM: `onunload()` has a floating promise with no error handling
- **File:** `plugin/main.ts:159`
- **Description:** `void this.app.workspace.detachLeavesOfType(...)` — no `.catch()` chain. If Obsidian's API rejects, the rejection is unhandled.
- **Impact:** Potential unhandled rejection during plugin unload
- **Recommendation:** Add `.catch(() => {})`.

#### MEDIUM: Stream idle timeout misclassifies as user cancellation (repeated from Security)
- **File:** `plugin/src/llm/client.ts:280-284`
- **Impact:** Single idle timeout ends the LLM call with no retry, defeating the 3-attempt policy
- **Recommendation:** Use distinct error type for timeouts vs user cancellation.

#### MEDIUM: `forceRunForDate` and `retryFailedInLookback` clear state outside lock
- **File:** `plugin/src/services/scheduling/scheduler-driver.ts:202,220`
- **Impact:** TOCTOU race with concurrent tick
- **Recommendation:** Move `clearDate` inside `withLock`.

#### LOW: CLI entry point has unhandled rejection
- **File:** `plugin/src/cli/main.ts:269`
- **Description:** `void runCli().then(...)` — no `.catch()` on the `void` expression.
- **Impact:** Unhandled rejection if CLI startup fails
- **Recommendation:** Add `.catch((err) => { console.error(err); process.exit(1); })`.

#### LOW: Multiple `AbortController` instances created without cleanup tracking
- **File:** `plugin/src/services/cancellation.ts:15-23`
- **Description:** Each `begin()` creates a new `AbortController`. If `begin()` is called for the same date twice (race), the first controller is leaked — its signal handlers never fire because nobody holds a reference.
- **Impact:** Memory leak on repeated begin/abort cycles
- **Recommendation:** Abort and remove existing controller for that date before creating a new one.

---

### 11. Test Coverage Gaps

#### Coverage gaps identified across the test suite:

| Gap | Severity | Details |
|-----|----------|---------|
| State store crash recovery | MEDIUM | No tests for `writeAtomic` crash recovery (`.bak` fallback, missing primary file) |
| Pipeline partial failure | MEDIUM | No tests for partial index mutations (index write succeeds, daily write fails) |
| Cancellation edge cases | MEDIUM | No tests for `cancelAll` then tick; no tests for overlapping cancel+begin |
| Date/time edge cases | MEDIUM | No tests for DST transitions, leap year (Feb 29), cross-midnight scheduler windows |
| Timezone boundary dates | LOW | `daysBefore` UTC/TZ mismatch not tested |
| Slugify with CJK input | LOW | Empty slug output not tested |
| Cross-instance state-store | LOW | No test for concurrent stores on the same file |
| Settings migration corner cases | LOW | Migration from non-existent old format not tested |

**General observations:**
- Tests extensively cover normal flow but significantly under-test error paths and edge cases
- Mock-heavy tests verify mock expectations rather than behavioral invariants
- No integration test that runs the full pipeline against a real (or recorded) arXiv response
- Vitest config lacks `isolate: true`, `restoreMocks: true`, and coverage configuration

---

## Consolidated Priority Matrix

### CRITICAL (must fix — data loss / crash / scheduler stuck)

| # | Finding | File | Line(s) |
|---|---------|------|---------|
| 1 | `cancelAll` global flag permanently disables scheduler after cancellation | `cancellation.ts`, `scheduler-driver.ts` | 29-39, 113/124/214/229/253/277 |
| 2 | State stuck "running" forever if `setFailed` itself fails — no recovery path | `scheduler-driver.ts`, `run-gate.ts` | 386-394, 19 |

### HIGH (should fix — functional bug affecting many users)

| # | Finding | File | Line(s) |
|---|---------|------|---------|
| 1 | Paper index mutated before daily file committed — stale index on failure | `pipeline.ts` | 261-278 vs 327 |
| 2 | `pending` result resets attempt counter, defeating permanent escalation | `scheduler-driver.ts`, `state-store.ts` | 368-373, 105 |
| 3 | File writes not atomic — crash mid-write produces partial files | `markdown-writer.ts`, `storage-adapter.ts` | all writes |
| 4 | Sequential content fetch (20-50 papers, one-at-a-time) | `pipeline.ts` | 191-234 |
| 5 | Sequential detail summaries (LLM calls) | `pipeline.ts` | 285-323 |
| 6 | `openai` SDK dominates bundle size (~80%) | `package.json` | 16 |

### MEDIUM (should fix — edge case bugs / maintainability)

| # | Finding | File | Line(s) |
|---|---------|------|---------|
| 1 | `addDailyReports` + `setSummaries` non-atomic | `pipeline.ts` | 264-270 |
| 2 | `setPaperPath` in "exists" branch unprotected | `pipeline.ts` | 288-297 |
| 3 | Cancellation not representable in PipelineResult union | `pipeline.ts` | 75 |
| 4 | No tick re-entrancy guard | `scheduler-driver.ts` | 64-67 |
| 5 | `clearDate` outside lock — TOCTOU | `scheduler-driver.ts` | 202, 220 |
| 6 | `fetchRecent`/`fetchMetadataByIds` don't receive abort signal | `pipeline.ts` | 127, 418 |
| 7 | Calendar month refresh race | `view.ts` | 1059-1072 |
| 8 | Full DOM rebuild on every state change | `view.ts` | 635-672 |
| 9 | Stale filter count after tab switch | `view.ts` | 929-934 |
| 10 | `versions.json` missing current release | `versions.json` | 22 |
| 11 | Double change-listener on model dropdown | `tab.ts` | 941 |
| 12 | Quick-start template silently overwrites categories | `tab.ts` | 350 |
| 13 | `isMinutesWithinWindow` fails for cross-midnight schedules | `time.ts` | 58 |
| 14 | `isWeekendDate` uses UTC on tz-local date | `time.ts` | 84-87 |
| 15 | State store not atomic across instances | `state-store.ts` | 158-170 |
| 16 | `parseRunState` accepts schema-invalid JSON | `state-store.ts` | 240-248 |
| 17 | `writeAtomic` crash window: `.bak` unreachable when primary missing | `state-store.ts` | 222 |
| 18 | Run history rotation crash window loses entries | `run-history.ts` | 254-276 |
| 19 | `setFailed` escalation + history record inconsistency | `state-store.ts`, `scheduler-driver.ts` | 97-119, 375-378 |
| 20 | `daysBefore` UTC arithmetic vs tz-local dates | `time.ts` | 62-73 |
| 21 | `onunload()` may complete before async cleanup | `main.ts` | 158-162 |
| 22 | Stream idle timeout misclassified as user cancellation | `client.ts` | 280-284 |
| 23 | Missing `noUncheckedIndexedAccess` in tsconfig | `tsconfig.json` | 6 |
| 24 | No `isolate: true` or `restoreMocks: true` in vitest config | `vitest.config.mts` | — |
| 25 | Prompt injection relies on LLM obedience only — no sanitization | `injection-guard.md`, `paper-filter.ts`, `summarizer.ts` | — |

### LOW (nice to have)

| # | Finding | File | Line(s) |
|---|---------|------|---------|
| 1 | Per-category failures silently dropped | `pipeline.ts` | 472-477 |
| 2 | Node `writeText` doesn't create parent directories | `node/storage-adapter.ts` | 20-22 |
| 3 | Node `EnvSecretProvider` silently no-ops setSecret | `node/secrets.ts` | 3-21 |
| 4 | `slugify` produces empty strings for non-ASCII | `slugify.ts` | 5 |
| 5 | `retry.ts` backoff overflow to `Infinity` | `retry.ts` | 25 |
| 6 | Run history `decodeRunHistoryLines` silent skip of malformed lines | `run-history.ts` | 180-205 |
| 7 | CLI entry point unhandled rejection | `cli/main.ts` | 269 |
| 8 | `onunload()` floating promise | `main.ts` | 159 |
| 9 | Multiple AbortController leaks | `cancellation.ts` | 15-23 |
| 10 | `fetchModels` swallows all errors | `client.ts` | 151-154 |
| 11 | `dangerouslyAllowBrowser` no alternative | `client.ts` | 95 |

### INFO (suggestions)

| # | Finding |
|---|---------|
| 1 | Consider adding OS keychain integration for Node CLI |
| 2 | Consider prompt caching for LLM calls |
| 3 | Consider adding token usage tracking (`stream_options: { include_usage: true }`) |
| 4 | Consider generating `--metafile` for bundle analysis in production builds |
| 5 | Consider adding coverage configuration to vitest |
| 6 | Consider adding LRU eviction policy to HTML cache |
| 7 | Document single-instance assumption for state-store |
| 8 | Consider event delegation for dashboard table rows |
| 9 | Consider concurrency-limited parallel fetches for papers |

---

## Recommendations by Priority

### Immediate (Critical — address within 1 week)

1. **Fix `cancelAll` race** — redesign the cancellation system to be per-run, not global. This is the most severe issue as it can silently disable the scheduler.
2. **Add recovery for dates stuck at "running"** — implement startup scan that un-sticks dates with `status: "running"` older than 1 hour (or add fallback in `setFailed` that catches persistence errors).

### Short-term (High — address within sprint)

3. **Move index writes after `writeDaily`** — prevent stale index references to non-existent files.
4. **Don't clear date on `pending` result** — preserve attempt counter for transient escalation logic.
5. **Implement atomic file writes** — use write-temp-then-rename for all markdown files (match `state-store.ts` pattern).
6. **Parallelize content fetches** — add concurrency-limited pool for paper content retrieval.
7. **Evaluate `openai` SDK replacement** — raw `fetch` wrapper for chat completions could cut bundle size ~80%.

### Medium-term (Medium — address next iteration)

8. Fix cross-midnight window bug in `isMinutesWithinWindow`.
9. Fix `isWeekendDate` timezone mismatch.
10. Add `noUncheckedIndexedAccess` to tsconfig and fix type errors.
11. Add prompt injection sanitization (encode `<` in user content).
12. Fix `daysBefore` UTC/TZ frame mismatch.
13. Add `versions.json` entry for current release.
14. Fix double change-listener in model dropdown.

---

*Generated by multi-agent workflow on 2026-07-04 — 28 specialized agents, 80+ source files analyzed*
*Branch: `refactor/scheduler-hybrid` | Commit: `b786d5c`*
