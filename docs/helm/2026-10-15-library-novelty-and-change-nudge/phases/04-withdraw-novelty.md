# P4 — withdraw-novelty

goal_ref: ../goal.md
created: 2026-10-15T18:30:00+08:00
updated: 2026-10-15T18:30:00+08:00
revision: 1

# P4 plan: withdraw daily-report personal novelty, keep search improvements, keep old reports readable

Read-only plan, from worktree HEAD 1d4b296 plus uncommitted P3 edits. Paths are relative to /home/tiandc/Documents/code/arxiv-daily-library-novelty.
P3 is concurrently editing plugin/main.ts, apps/cli/src/library-cmd.ts, apps/cli/src/workbench/web/{i18n,library-review}.ts. P4 execution must start only after P3 lands (the merge zones are called out in Chunk 4).

## 1. Removal inventory (D = dormant at f784464, A = added by 243e5db / 85d923d)

Producer side
- A packages/core/src/pipeline/library-novelty-stage.ts (whole file), packages/core/src/library/prior-work.ts (whole; exported at core index.ts:41), plugin/src/library/prior-work-wiring.ts, apps/cli/src/library-prior-work.ts. Delete the tests too: core library-prior-work.test.ts, pipeline-library-novelty.test.ts, plugin library-prior-work-wiring.test.ts.
- A pipeline.ts: import 38 and `libraryPriorWork` dep 127 (LibraryPriorWorkPort import 35); stage block 300-322 (`stageStart/End("personal-novelty")`, `paper.personalNovelty =`).
- A plugin/main.ts: import 156, `libraryPriorWork:` 2946, `buildLibraryPriorWork` 2952-2985. apps/cli/src/runtime.ts: import 33, 148, 167 (cli-runtime.test.ts +209 lines is mostly novelty; trim).
- D progress stage `"personal-novelty"`: core/adapters.ts:139, plugin/src/services/status-bar.ts:9 and 21 (+ status-bar.test.ts). Existed before the initiative.
- D personalized-novelty.ts (1000+ lines: generator, `normalizePersonalNovelty*`, checkpoint port at :319, fingerprints; core index.ts:79) and prompts/personal-novelty.system.md. Delete with personalized-novelty.test.ts.
- D `NoveltyCheckpointPort` and the novelty half of DailyFilterCheckpointStore (services/daily-filter-checkpoint-store.ts:19-52 imports/re-exports, 74-86 paths and document type, 121-129, 201-270 load/lookup/save, 366-385 assertMutable, 419-465 read/persist, 544-565 decode, 690-745 replace-with-backup, 768-775). pipeline.ts:88-92 stops extending the port. The disk files it owns: `<indexDir>/filter-checkpoints/<date>.novelty.json`, plus `.bak`, `.tmp`, `.bak.tmp`.
  - The files live only for an in-flight run: `removeAll(date)` (store :317-336, called from `cleanupCommittedCheckpoints`, pipeline.ts:~641) deletes them at commit. They can be orphaned only by a run that crashed before commit.
  - Decision: delete all novelty read/write code but keep a 4-line best-effort `removeIfExists` of the legacy novelty paths inside `removeAll`. Rename the derived paths to `legacyNovelty*`. Adjust daily-filter-checkpoint-store.test.ts (A: +58 lines) and daily-summary-checkpoint-store.test.ts:278 ("never includes personal novelty ...", delete).

Writer side (all D, v2 is A)
- personal-novelty-marker.ts (whole, v1 + v2 writer/parser/`parseDailyReportPersonalNovelty`); core index.ts:69; re-exports at daily-summary-parser.ts:12-15.
- daily-summary-rendering.ts: imports 13-18, `PERSONAL_NOVELTY_DIFFERENCE_TYPE_LABELS` 49-, marker+visible lines 198-205, the blank-line condition at 210 (`paper.discoveryProvenance || paper.personalNovelty` becomes `paper.discoveryProvenance`), `renderVisiblePersonalNovelty` 275-300.
- FilteredPaper.personalNovelty paper-filter.ts:16 and 48-52; summarizer.ts:112; assembler daily-summary-assembler.ts:12-13, 45-51, 161-163 (preflight "invalid personal novelty").

Reader side (all D, v2 title plumbing A)
- paper-index.ts: imports 6-7, field 79-80, `reconcileDailyReportOccurrenceNovelty` 382-412, 932, decoder 1024, `normalizeNoveltyByReport` 1208-1220.
- dashboard/history-sync.ts: imports 13 and 19-20, 91, 157-161, 177-190, 370-383, 413, prune 739-766 (`invalidNoveltyReports` parameter). pipeline.ts commit projection 543-551 and repair projection 687-695.
- dashboard/model.ts: types 72-105 (`personalNovelty` on DashboardRow), label table and helper 170-215, projection ~430-460 (`projectDashboardOccurrenceNovelty`), import 12-15. plugin/src/dashboard/view.ts: import 15, `dashboardPersonalNoveltyLines` 147-175, row render 1568-1575; plugin/styles.css:1429 `.arxiv-daily-dashboard__novelty`.
- CLI workbench: apps/cli/src/workbench/papers.ts:2, 16 (`novelty` field), 101 (projection); web/papers.ts:94-95 (reading-pane section "与已有文献的关系" and its comparison line); packages/core/src/presentation/reading-translations.ts:312 (the translation pair for that heading).
- scripts/personalization/analyze-dogfood.ts:11-12, 94-103 (dormant dogfood tool reads `noveltyByReport`; drop the novelty columns/integrity check).
- Tests to trim: dashboard-model, paper-index, topic-direction-marker (keep legacy cases, see §2), assembler, pipeline.test, dashboard-view, dashboard-history-sync, plus CLI workbench-papers, -paper-ui, -navigation, -generation-footer, -reading-language, -settings-ui, -library-review-ui, cli-personalized-workflow.
- Do not touch: prompts "Contributions and Novelty" (paper-detail.en.system.md), detail-selector "Novelty" criterion, extensions/dsh fixtures. They are about paper reading, not personal novelty.

Consent wording (all A; revert to the 243e5db^ text)
- plugin/src/library/modal.ts:68-69 (the "The daily report also sends ..." sentence; library-modal.test.ts +18).
- apps/cli/src/workbench/web/i18n.ts:3-8 (`DAILY_NOVELTY_DISCLOSURE` and the extra message table), settings.ts:1 and ~512, library-review.ts:6 and 213 (P3 also edits i18n.ts and library-review.ts, so rebase on it).
- apps/cli/README.md:132 and README.zh-CN.md:132 (sentence + `library revoke` clause; keep the revoke instruction).
- The consent fingerprint was unchanged by 243e5db, so no re-authorization is triggered.

Acceptance (all A)
- scripts/acceptance/obsidian-scenarios.mjs:3 (import), 194-340 (`libraryNovelty` scenario and helpers; registered at ~341), obsidian.mjs:16 (catalog row `obsidian.library.novelty`). The fixed Obsidian suite goes 9 to 8 scenarios.
- fixtures.mjs:13-14 (`FIXTURE_NOVELTY_EXPLANATION_PREFIX`), 48-60 (novelty route; also remove the `personal-novelty` request kind if nothing else uses it).
- exploration-tasks/obsidian.mjs:12, 32-33, 244-255, 1320-1360 (`library-novelty-visible`); tests acceptance-fixtures.test.mjs:11, 49-71; acceptance-exploration-host.test.mjs:22, 316, 1994-2075.
- scripts/acceptance/README.md:135 (task row) and :197 (the "Obsidian also covers personal novelty" clause).

## 2. Compatibility after removal

- Old reports keep `<!-- arxiv-daily-personal-novelty:v1|v2:... -->` lines plus a visible `> Personal novelty:` / `> 个人新颖性：` line (rendered as a plain quote). After removal no code parses these markers: no warning, no "invalid" state, the report text is untouched.
- Trap: topic-direction-marker.ts:2, 77, 115 counts the novelty marker as a preceding marker family when it computes the canonical slot of the topic-direction callout (`canonicalSlot = start+1+preceding.length*2`). Dropping it would make every old report with novelty + topic markers return "topic direction marker placement is invalid", and history sync would silently lose direction data for them. Fix: keep a private `LEGACY_PERSONAL_NOVELTY_MARKER_PREFIX = "arxiv-daily-personal-novelty"` constant in topic-direction-marker.ts, still counted as a 2-line preceding family (marker + visible line), with a comment pointing at ADR 0015 withdrawal. Existing cases in topic-direction-marker.test.ts stay as legacy-layout tests. The provenance parser does not mention novelty (grep), so it is unaffected.
- paper-index.ts decoder (:1005-1030) is field-by-field normalization, not exact-key: an unknown `noveltyByReport` key on disk is ignored on load and absent on the next save. `PAPER_INBOX_SCHEMA_VERSION` stays 5 (:23); no migration, no bump (a bump would make older plugins refuse the file). Add a test with a v5 file carrying `noveltyByReport` (valid v1/v2 records and garbage): loads, entry has no such key, round-trip save drops it.
- history sync: remove `noveltyByReport` collection, reconcile, `invalidNoveltyReports`, and the "skipped invalid personal novelty" warning. Old reports are never flagged. Stale `noveltyByReport` data in an old index just disappears at the next mutate. Required test: dashboard-history-sync.test.ts syncing a report with v1 + v2 markers and the visible line, asserting logger.warn is never called and provenance and topic-direction projection are identical to the same report without novelty lines.
- Repair path (pipeline.ts:~660-700 rebuilds the index from committed markdown): just drop the novelty branch; `extractPaperSummaries` already runs on reports that contain the visible line, so no change. The assembler preflight only validates freshly generated papers; with the field gone it has nothing to check. A rerun of a day regenerates from source papers and never reads the old marker, so it simply writes no novelty.
- Checkpoint legacy files: see §1 (cleanup retained in `removeAll`).
- Older plugin versions reading a newer index/report are unaffected (we only stop writing).

## 3. Keep list and the BM25 query cap

- Keep (confirmed present, independent of novelty): `MAX_BM25_QUERY_TERMS` and `limitBm25QueryTerms` (generation-bm25-index.ts:18, 31), the per-block bucket hoisting (:~120), the 216114e sha256/bucket speedups (utils/digest.ts, generation-index-*.ts, digest.test.ts, generation-bm25-integrity-parity.test.ts).
- "Knowledge-base search support for file: documents": 243e5db did not change any search code for `file:` keys. The only `file:` support is the manifest lookup inside prior-work.ts (:51-54, ~102-122) and `isNoveltyFileBasisKey` in personalized-novelty.ts, both novelty-only and both go. Generic plugin/CLI search already shows `file:` hits through `projectLibraryFullTextMatches` (plugin/main.ts:~2058) and the CLI projection. Chunk 1 asserts this with a test that a file: document is returned by `searchFullTextKnowledgeBase` (if that test already exists, only cite it). Flag to caller: the user's "keep file: support" resolves to "nothing extra to keep".
- Dashboard "Find similar papers" is NOT hit today: view.ts:2342-2353 sends `queryText = title + abstract` for the dense leg but already passes `lexicalQueryText: entry.title` (since d71c46e). Titles stay under 64 unique terms.
- It IS hit by free-text callers (query reaches `searchGenerationBm25` unmodified, which throws "query has more than 64 unique terms" at generation-bm25-index.ts:92-93, wired at index-orchestration.ts:945 `input.lexicalQueryText ?? input.queryText`):
  - plugin Dashboard search box view.ts:2386 and palette command commands.ts:546 (`searchPersonalLibraryFullText(query)`, default hybrid, so the lexical leg throws on a pasted abstract),
  - CLI `library search --query` (library-cmd.ts:129, modes lexical|hybrid) and the workbench library search (workbench/library.ts:71; query up to 4096 chars, default lexical).
- Fix once in core instead of per host: at index-orchestration.ts:945 use `limitBm25QueryTerms([input.lexicalQueryText ?? input.queryText])`. The text is returned unchanged when within 64 unique terms, so existing rankings do not move; dense and RRF legs keep the full text. Leave the legacy non-generation path (:897) alone (it has no cap and truncating would change its ranking). Keep the explicit Dashboard `lexicalQueryText: entry.title`.
- Real-index test (core): reuse the scaffolding from library-prior-work-real-index.test.ts (memory storage, `synchronizeFullTextGenerationIndex`, 120-term abstract). Move it and the `limitBm25QueryTerms` cases into a new tests/fulltext-lexical-query-cap.test.ts that calls `searchFullTextKnowledgeBase` directly in lexical and hybrid modes. Red today (throws), Green after. Add a CLI case in apps/cli/tests/cli-library.test.ts near :208 with a >64-term `--query` if that fixture is a real index.

## 4. Docs
- ADR 0015: Status line becomes "Accepted; §1-2 withdrawn 2026-10-15, §3-5 in force". Mark §1 and §2 "Withdrawn" with the reasons: the comparison papers are chosen by retrieval, not by the researcher, so they cannot be confirmed as what the researcher cares about; the real evaluation (quality-report4: 5 useful lines out of 20, difference type almost always "new task") shows a line reduces to "a related library paper exists", which was judged not worth a line. Strike the novelty consequences (retrieval basis, marker v2, file: basis, "Every digest paper can carry a comparison"); keep a note that the BM25 cap and speedup stayed. Add a short "Withdrawal" section and keep Context text (history).
- ADR 0004 §4 (:56-62) add "Withdrawn: personal novelty is not part of the product (ADR 0015)". Also :71, 110, 126 mention novelty evidence; add a one-line pointer. ADR 0012 :43 ("Personal novelty loses its stated comparison basis"): append "Closed: withdrawn, see ADR 0015".
- CONTEXT.md:40 `Personal novelty`: replace the row with an "_Avoid_/retired term" style entry: not a product concept; marker text only appears in historical reports. CONTEXT.md:26 `Daily discovery`: drop "and explain what they appear to add relative to prior work the researcher already holds".
- docs/technical-report.md :42, 90, 259, 272 (novelty snapshot/inputs, Dashboard novelty, index `noveltyByReport` mention); both CLI READMEs; scripts/acceptance/README.md rows. docs/releases/0.5.0.md and 0.3.1 are history: leave. goal.md/journal.md are owner-maintained (tick criterion, record chunks).

## 5. Chunks (each leaves typecheck + its tests green; run after P3 lands)

Test command base: `cd packages/core && npx vitest run <files>`; plugin `cd plugin && npx vitest run <files>`; CLI `cd apps/cli && npx vitest run <files>` (confirm runner with `npm run test -w <workspace> -- <file>` once). Scripts: `node --test scripts/tests/<file>.test.mjs`.

1. Lexical query cap in core search. kind: fix + test. first signal: Red = new `packages/core/tests/fulltext-lexical-query-cap.test.ts` ("lexical and hybrid search accept a 120-term abstract") fails with "query has more than 64 unique terms"; also Green baseline for the moved `limitBm25QueryTerms` cases. Change index-orchestration.ts:945; move cap/real-index scaffolding out of library-prior-work-real-index.test.ts (the novelty half is deleted in Chunk 3). checks: `npx vitest run tests/fulltext-lexical-query-cap.test.ts tests/fulltext-index-orchestration.test.ts tests/fulltext-retrieval-evaluation.test.ts`, core typecheck.
2. Compatibility fixtures first (characterization). kind: test-only. first signal: Green baseline now; they must still pass after every later chunk. Add (a) topic-direction parse of an old report with provenance + novelty v1/v2 + topic marker valid, (b) dashboard-history-sync of the same, no warn, (c) paper-index v5 file with `noveltyByReport` loads and round-trips without it (this one is Red until Chunk 5 only if you assert absence of the key, so assert absence in Chunk 5 and loading in Chunk 2), (d) pipeline repair path on that report. Command: `npx vitest run tests/topic-direction-marker.test.ts tests/paper-index.test.ts tests/pipeline.test.ts` and `cd plugin && npx vitest run tests/dashboard-history-sync.test.ts`.
3. Stop producing novelty and drop its acceptance. kind: removal. first signal: Red = pipeline test "no personal-novelty stage runs and no model call is made even if a host passes libraryPriorWork" (TypeScript: `libraryPriorWork` no longer in PipelineDeps, so write it as typecheck failing when buildLibraryPriorWork still exists); Green after. Scope: pipeline.ts stage + deps, library-novelty-stage.ts, prior-work.ts, both host wirings + runtime.ts, ProgressStage + status-bar, and in the same chunk the acceptance scenario `obsidian.library.novelty`, fixture route, exploration task, their tests and README rows (the scenario would fail the moment the producer is gone). checks: root `npm run typecheck`, `npm run check:boundaries`, core/plugin/CLI affected tests, `npm run test:acceptance:tools`.
4. Consent wording back to pre-243e5db. kind: removal. first signal: Red = update plugin/tests/library-modal.test.ts and CLI i18n/settings/library-review UI tests to assert the sentence is absent (they currently assert it). Files: modal.ts:68-69, i18n.ts, settings.ts, library-review.ts, two READMEs. Merge note: P3 touches i18n.ts and library-review.ts. checks: plugin library-modal test, CLI workbench-settings-ui and workbench-library-review-ui tests.
5. Remove readers/projections. kind: removal. first signal: Red = Chunk 2(c) asserting `noveltyByReport` absent after save, plus dashboard-model test "row has no personalNovelty"; Green after. Scope: paper-index field/reconcile/normalizer, history-sync, pipeline commit+repair branches, dashboard model/view/CSS, CLI workbench papers + reading pane + translation, analyze-dogfood.ts. Schema version stays 5. checks: core dashboard-model, paper-index, pipeline, plugin dashboard-view + history-sync, CLI workbench-papers/paper-ui/navigation/generation-footer/reading-language, typecheck.
6. Remove writer + generator + checkpoint port. kind: removal. first signal: Green baseline (compat tests from Chunk 2 stay green; assembler test for a paper without novelty). Scope: personal-novelty-marker.ts (keep topic-direction legacy constant), rendering, FilteredPaper/summarizer/assembler fields, personalized-novelty.ts + prompt + tests, checkpoint store novelty code (keep legacy file cleanup in `removeAll`, add a test that an orphan `<date>.novelty.json` is removed on commit), core index.ts exports. checks: full core test run `npm run test -w packages/core` (watch heap: see docs/reviews/2026-08-30-core-test-heap-exhaustion.md), typecheck, boundaries, lint.
7. Docs. kind: docs. first signal: n/a (`git diff --check`, grep `Personal novelty` shows only ADR history, historical release notes and the legacy-compat comment). Scope per §4.

Phase verification (owner, after Chunk 7)
- `npm run typecheck`, `npm run check:boundaries`, `npm run check:product-units`, `npm test`, `npm run test:workspaces`, `npm run test:acceptance:tools`, `npm run lint`, `npm run build`.
- `npm run test:acceptance -- --suite obsidian`: fixed suite passes with 8 scenarios (no `obsidian.library.novelty`); the other library scenarios still pass with the reverted consent text.
- Real Obsidian old-report check: install the build into the test vault, drop a report generated at 243e5db (the artifact `obsidian-novelty-daily.md` or the vault's 2026-10-05 report, markers v2 + provenance + topic markers) into the daily dir, run Dashboard history sync (open Dashboard, All tab), confirm in the plugin log: no `dashboard: skipped invalid` and no novelty warning, rows and topic-direction chips unchanged, no novelty block, reading pane opens. Record or explicitly note if not done.
- Library search: run a pasted-abstract query in the Dashboard search box and in `library search --mode lexical` and `--mode hybrid` against the real test library; no "more than 64 unique terms" error.
- grep gate: `grep -rIn -i "novelty" packages/*/src apps/*/src plugin | grep -v "Contributions and Novelty\|LEGACY_PERSONAL_NOVELTY"` is empty.

Abort triggers
- Chunk 2 fixtures fail after any removal chunk (old reports flagged invalid or topic directions lost): stop and restore the legacy-slot handling.
- Any need to bump `PAPER_INBOX_SCHEMA_VERSION` or change the consent fingerprint (would force migration or re-authorization).
- The cap change moves an existing ranking (fulltext-retrieval-evaluation or orchestration tests change): restrict the cap to the throwing case only.
- A removal reaches code P3 still edits in a way that cannot be merged cleanly; wait for P3 rather than resolving by hand.
- A fixed-suite scenario other than the novelty one starts failing for a reason tied to removed code.

## Chunk 验收（owner）

- Chunk 1 — 词法查询截词进入核心检索
- [x] accepted on observed evidence
- Chunk 2 — 旧日报兼容性特征测试
- [x] accepted on observed evidence
- Chunk 3 — 停止生成新颖性（流水线、宿主接线、验收场景）
- [x] accepted on observed evidence
- Chunk 4 — 授权文案恢复
- [x] accepted on observed evidence
- Chunk 5 — 删除读取与显示
- [x] accepted on observed evidence
- Chunk 6 — 删除写入、生成器与检查点端口
- [x] accepted on observed evidence
- Chunk 7 — 文档
- [x] accepted on observed evidence
