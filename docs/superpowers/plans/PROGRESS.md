# arxiv-daily Obsidian Plugin — Execution Progress

**Plan:** [`docs/superpowers/plans/2026-05-11-obsidian-plugin-mvp.md`](./2026-05-11-obsidian-plugin-mvp.md)
**Spec:** [`docs/superpowers/specs/2026-05-11-obsidian-plugin-design.md`](../specs/2026-05-11-obsidian-plugin-design.md)
**Result:** v0.1.0 shipped — https://github.com/tdccccc/arxiv-daily/releases/tag/v0.1.0
**Started:** 2026-05-11
**Finished:** 2026-05-12
**Execution mode:** Inline (subagent dispatch blocked by gateway issue)

## Status

**All 24 planned tasks complete plus 3 follow-up enhancements.**

- 11 test files / 68 vitest tests passing
- `tsc -noEmit` clean
- `npm run build` produces ~540 KB CJS bundle
- Merged to `main`, tagged `v0.1.0`, GitHub Release with assets uploaded
- BRAT-installable via repo slug `tdccccc/arxiv-daily`

## Task ledger

### Planned (24)

- [x] Task 1 — Scaffold plugin project (9df636a)
- [x] Task 2 — Settings types & defaults (03e6f71)
- [x] Task 3 — Time utility (TZ-aware) (c6012cc)
- [x] Task 4 — Async retry utility (7f22873)
- [x] Task 5 — RunLock (7935456)
- [x] Task 6 — StateStore (11fa3ff)
- [x] Task 7 — Logger (d2fa0e8)
- [x] Task 8 — arXiv parser (923a2ad) — note: uses `/recent?show=2000`; listings no longer include abstracts
- [x] Task 9 — ArxivFetcher + Atom abstract enrichment (9ab978a) — added atom-parser.ts for `/api/query` batch lookups
- [x] Task 10 — HtmlCache (9ea7c0d) — bonus: added unit tests
- [x] Task 11 — Section extractor (14956bd)
- [x] Task 12 — Paper content fetcher (39c416d)
- [x] Task 13 — LlmClient (f1c7416)
- [x] Task 14 — Paper filter (73e3ada)
- [x] Task 15 — Daily summarizer (bf1de3e)
- [x] Task 16 — Paper detail summarizer (bf1de3e — same file)
- [x] Task 17 — MarkdownWriter (996f6b0)
- [x] Task 18 — ArxivPipeline orchestrator (3bea617) — added obsidian stub for tests
- [x] Task 19 — SchedulerService (7f79829)
- [x] Task 20 — Settings UI tab (24b19eb)
- [x] Task 21 — Commands + ribbon (e94a42f)
- [x] Task 22 — main.ts lifecycle (82ed54d)
- [x] Task 23 — README + dev docs (48341b9)
- [x] Task 24 — Manual smoke test (verified on a live vault against DeepSeek V4)

### Follow-up enhancements (post-MVP, pre-release)

- [x] Ribbon → menu (today / all pending / specific date) (5a36fd5)
- [x] Summarize by arXiv ID (b5a1138)
- [x] LICENSE + BRAT install docs (c640e14), merge to main (4a21a07),
  tag + GitHub release with assets

---

## v0.1.1

**Plan:** [`docs/superpowers/plans/2026-05-12-scheduler-skip-and-progress.md`](./2026-05-12-scheduler-skip-and-progress.md)
**Spec:** [`docs/superpowers/specs/2026-05-12-scheduler-skip-and-progress-design.md`](../specs/2026-05-12-scheduler-skip-and-progress-design.md)
**Result:** v0.1.1 shipped — https://github.com/tdccccc/arxiv-daily/releases/tag/v0.1.1
**Started:** 2026-05-12
**Finished:** 2026-05-12

### Changes

**Scheduler gating (Block 1):**
- Default `schedule.enabled` → false (fresh installs don't auto-run)
- `setScheduleEnabled()` unified toggle for ribbon + settings
- `tickToday()` — today-only, weekend-aware, bypasses runAtLocal
- `start()` no longer fires immediate tick; callers trigger `tickToday()`
- Ribbon menu: status header + Enable/Disable toggle
- Settings tab: enable toggle at top with Running/Paused status

**Skip existing files (Block 2):**
- `MarkdownWriter.dailyExists()` / `paperDetailExists()` existence checks
- Pipeline pre-checks daily file before any network call
- Per-paper skip in detail loop
- Writers throw on pre-existing files (no more silent .bak rename)

**Status bar progress (Block 3):**
- `ProgressReporter` interface + `NoopProgressReporter`
- `StatusBarController` renders idle/run/disabled state
- Pipeline emits stage events (fetch-recent → enrich-abstract → filter → fetch-content → summarize-daily → write-detail)
- Scheduler emits batch/idle events

**LLM settings redesign:**
- Provider presets (DeepSeek/OpenAI/Anthropic/GLM/Custom)
- Auto-fill URL, model, thinking, reasoning effort on provider change
- All fields remain editable for any provider
- Latest models: GPT-5.2–5.5, Claude Opus 4.7, GLM-5.1
- Provider-specific reasoning effort options

**arXiv settings redesign:**
- Grouped category dropdown (~50 categories)
- Auto-generated tag/display maps from detail categories
- Comma-separated detail categories input
- Timezone dropdown with common presets

**UI:**
- All settings translated to English

### Test coverage

- 14 test files / 99 vitest tests passing
- `tsc -noEmit` clean
- `npm run build` produces ~550 KB CJS bundle

---

## Known limitations carried into v0.1.0

These were intentional MVP scoping choices, not bugs:

1. **arXiv listing has no abstracts** — mitigated by Atom-API enrichment.
   If Atom is unreachable the filter falls back to title-only judgement.
2. **Daily summary input = Abstract + Conclusion only**, not full text.
   Cost trade-off explained in the original brainstorming; full-text
   daily would explode token usage with marginal quality gain.
3. **`/html/<id>` not always available** — older / unrendered papers
   only have an Abstract page. The pipeline writes a short
   abstract-based block; manual `Summarize by arXiv ID` refuses to
   produce a detail summary in this case.
4. **Catch-up requires Obsidian to be open** at some point during /
   after `runAtLocal` on the target day. The 5-day rolling lookback
   covers short outages; longer gaps exceed arXiv's `/recent` window.
5. **State is per-machine** (lives in plugin data, not the vault).
   Two machines syncing the same vault may both run the same day.

---

## Future work

Roughly ordered by user impact / effort ratio. Pick one or two per
follow-up release.

### v0.1.x — small wins

- **Robust conclusion extraction** — widen keyword match
  (`discussion|outlook|prospect|final remark|implication`) + a "first
  N chars of Methods/Results" fallback when a Conclusion section is
  absent. Should noticeably improve daily-summary depth for papers
  whose authors don't literally call a section "Conclusion".
- **Daily summary depth setting** — let users pick
  `shallow | medium | deep` (current = shallow). Medium = Abstract +
  Intro + Conclusion + first N chars of Results. Deep = full sections
  (essentially same as detail).
- **Per-paper progress notice** — long detail runs are silent; show a
  ticker Notice like `arXiv Daily: detail 3/7 (2605.08068)`.
- **`Re-run for date` command** — currently if a date is `completed`
  the scheduler skips it forever. A force re-run command would let
  users regenerate after changing prompts or models.

### v0.2.0 — multi-profile

The spec's v2 section anticipates this. Lift `arxiv` / `output` /
`schedule` into a `ProfileSettings[]`, keep `llm` / `advanced` global.
UI gets a profile picker on the settings tab. Commands and ribbon menu
need a "for which profile" selector (or default to the active one).
Migration: existing v0.1.x settings auto-become the default profile.

### v0.3.0 — cron / CLI fallback

For users who can't keep Obsidian open every day:

- **Node CLI entry** — `npx arxiv-daily run --date YYYY-MM-DD` that
  shares the same TS pipeline modules (everything except Obsidian-
  specific I/O is already framework-agnostic). Writes Markdown to a
  configured path. crontab calls this.
- **State sync** — optionally write state to a vault file (`.state.json`
  inside the output dir) so the Obsidian-side scheduler sees what
  cron already did, and vice versa.

### v0.4.0 — quality of life

- **iOS support** — currently `isDesktopOnly: true` (we use Node `fs`
  for HTML cache via Electron `userData`). Mobile mode would need to
  store the cache inside the vault or in IndexedDB. Modest effort,
  unblocks a chunk of Obsidian's user base.
- **Submission to Obsidian Community Plugins** — requires the iOS
  path above, a `fundingUrl` field in manifest, removing any
  inline-style violations, and a clean PR to
  `obsidianmd/obsidian-releases`. 2–6 week review.
- **Cost estimator in settings** — before running, estimate today's
  token spend based on paper count × avg block size × model price.
  Especially valuable for users on metered DeepSeek tiers.

### Unscoped ideas (might or might not matter)

- **Co-citation graph** — extract references from detail papers and
  surface related papers in subsequent days' digests.
- **OPML / RSS export** — for users who want to share their filtered
  feed with collaborators outside Obsidian.
- **Hybrid filter** — pre-filter on cheap embedding similarity to
  user research interests, then only send borderline cases to the
  expensive LLM. Could halve token cost.
- **Web preview** — render the daily Markdown as a shareable HTML
  page (deployed to GitHub Pages off the vault).

---

## Process notes (for future sessions)

- The subagent dispatch path was blocked by a third-party gateway
  panic during this session. Inline execution was used instead.
- Two design adjustments emerged mid-implementation that aren't in
  the original spec:
  1. `/recent?show=2000` to defeat arXiv's default 50-entry pagination.
  2. Atom API enrichment for abstracts (listing pages no longer
     include them).
- The plan was followed task-for-task. Decisions taken outside the
  plan are noted in their commit messages and in the task ledger above.