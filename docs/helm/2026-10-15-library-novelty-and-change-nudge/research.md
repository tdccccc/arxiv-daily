# P2 Chunk 8 plan and Chunk 10 measurement (read-only study, 2026-10-15)

No repo file was edited. Code references are to the worktree /home/tiandc/Documents/code/arxiv-daily-library-novelty (HEAD f784464; another delegate's uncommitted edits to prior-work.ts / library-novelty-stage.ts exist, and nothing here depends on them). The business LLM was never called: the isolated config points at a dummy endpoint (http://127.0.0.1:9, nothing listens there) with a dummy key.

---------------------------------------------------------------------------

# Part 1 - Chunk 8 plan: file: library documents as novelty basis, titles shown for all basis papers

## 1.1 Findings (file:line, packages/core/src unless noted)

**A. Where `arxiv:` keys are hard-required**

- `pipeline/personalized-novelty.ts:1192` `isCanonicalArxivPaperKey` (prefix `arxiv:` and `paperKeyFromArxivId` round trip). It is used in six places, three of which must change:
  - `:391` daily paper key (stays arXiv: digest papers are arXiv only).
  - `:412` representative (basis paper) key in `prepareNoveltyRepresentatives` (change). The same function (`:408-421`) also demands `authors` 1..16, `published`, `categories` 1..16, which a `file:` document does not have (the manifest has only title/abstract). So accepting the key is not enough; the representative shape must be relaxed for file docs.
  - `:446-449` `preparePersonalNoveltyMatches` basis keys (change).
  - `:627` `normalizePersonalNovelty` comparisonBasis (change). This one function is the gate for everything persisted: the marker parser (`personal-novelty-marker.ts:70`), the paper index reconcile (`services/paper-index.ts:396`) and the index load (`:1217`), and `normalizePersonalNoveltyWithBasis` (`:644`, used by the assembler preflight `pipeline/daily-summary-assembler.ts:161`). If a `file:` key passes the pipeline stage but not this function, the daily assembly preflight throws and the whole day fails, so the predicate must change in lockstep everywhere.
  - `:825`, `:918` checkpoint entry paperKey = daily paper (stays arXiv). `:854` checkpoint `request.identity.basisPaperKeys` (change).
- Canonical file key form already exists as a precedent: `library/personal-library-interest-profile.ts:565-577` (`/^file:sha256:[0-9a-f]{64}$/`, `isCanonicalProposalPaperKey` = arxiv or file).
- **Prompt input: the model does not need arXiv-form keys.** `buildPersonalNoveltyRequest` (`personalized-novelty.ts:532`) serializes `{paper, basis}` as JSON; the model only copies basis `paperKey` strings back and `decodePersonalNovelty` (`:564`) checks them against the supplied set. `prompts/personal-novelty.system.md` says "new arXiv paper" for the daily paper only; the basis is described as "supplied basis paperKeys". Only a one-line note is wanted ("a basis paper may lack authors/published/categories") and, if the prompt text changes, bump `PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION` (plan messages feed the checkpoint fingerprint anyway, `:~740`).

**B. Marker** (`pipeline/personal-novelty-marker.ts`)
- Payload `{v:1, d:reportDate, a:arxivId(daily paper), n:PersonalNovelty}`, base64url, prefix regex `...:v1:...` (`:63`), exact-key checks (`:67-70`), and a canonical re-render equality check (`:77-79`): parse accepts only lines the renderer would produce.
- Bound `PERSONAL_NOVELTY_MARKER_MAX_CODE_UNITS = 10_000` (`:17`), derived for explanation (1000 units x6 escape) + 40 arXiv keys (read-back bound `PERSONAL_NOVELTY_MAX_COMPARISON_BASIS=40`, `personalized-novelty.ts:51`) x 4/3 base64. With 40 `file:` keys (79 chars each in JSON) the bound is exceeded (about 12.5k), and titles add more. Planning produces at most 5 basis papers (`PERSONAL_NOVELTY_MAX_RETRIEVED_BASIS`), so the bound must be re-derived.
- Parsing is all-or-nothing per report: any malformed marker makes `parseDailyReportPersonalNovelty` (`:92`) return `invalid` for the whole report.

**C. Daily Markdown visible line** (`pipeline/daily-summary-rendering.ts:280-297`)
- Already prints `Title (paperKey)` using `novelty.comparisonBasisTitles` (display shape `PersonalNoveltyWithBasis`, built by `attachPersonalNoveltyBasis`, `personalized-novelty.ts:701`). So the Markdown already shows titles for any basis whose title is known. Only change: for a `file:` key do not print the 76-char hash in parentheses (title only). The titles are dropped at the marker (`normalizePersonalNoveltyForMarker`, `:675`, "persisted marker payload stays minimal"), which is why the Dashboard cannot show them.

**D. Paper index** (`services/paper-index.ts`)
- `noveltyByReport: Record<string, PersonalNovelty>` (`:80`). There is no separate key whitelist: the only gate is `normalizePersonalNovelty` (exact 4 keys, arXiv keys, 1..40, code-unit sorted, exact depth, bounded explanation).
- Write path `reconcileDailyReportOccurrenceNovelty` (`:387-413`): a non-normalizable occurrence throws `PaperIndexError` and rejects the whole mutation; the caller at `pipeline/pipeline.ts:544-551` is inside a try whose catch returns `failed_transient` ("paper index daily report update failed", `:~558`). Callers: `pipeline.ts:544`, `pipeline.ts:688` (repair), `dashboard/history-sync.ts:179`. All of them feed it from `parseDailyReportPersonalNovelty`, so the index is purely a projection of the committed Markdown markers. Consequence: **display titles must be carried by the marker**, otherwise they vanish at the next history sync/repair.
- Read path `normalizeNoveltyByReport` (`:1212-1222`): structurally invalid entries are silently dropped (the documented discovery-provenance precedent), the document still loads.

**E. Compatibility with older plugin versions**
- Old plugin reading a new marker (extra payload field, `v2`, or a `file:` key): `parsePersonalNoveltyMarker` returns null (exact-key / `:v1:` regex / arXiv-key checks) so the whole report's novelty is `invalid`: `pipeline.ts:553` logs a warning, `history-sync.ts:~185` skips it and `pruneStaleMarkerReferences` (`history-sync.ts:736-766`) clears that report's index novelty. No crash, no data loss in the report Markdown, but the old plugin shows no novelty line data for **all** papers of that day. The visible Markdown line is plain text and unaffected.
- Old plugin reading a new index file: entries it cannot validate are dropped at load and lost on its next save (re-created later by a newer plugin's history sync).
- New plugin reading old markers (v1, no titles, arXiv keys): must remain valid; Dashboard falls back to the key.
- Real-world exposure is small: the generator was dormant between ADR 0012 and this feature (no release has written markers since), so this feature's reports are unreleased. Still make the design additive and versioned.

**F. Dashboard / workbench projection**
- `dashboard/model.ts:85-90` `DashboardPersonalNoveltyBasisPaper { paperKey }` (comment states titles are never persisted); projection `:434-451` (`comparisonBasis: novelty.comparisonBasis.map((paperKey)=>({paperKey}))`).
- Plugin view `plugin/src/dashboard/view.ts:157-168` `dashboardPersonalNoveltyLines` joins keys with ", ".
- CLI workbench: `apps/cli/src/workbench/papers.ts:101` passes `projectDashboardOccurrenceNovelty(...).personalNovelty` through unchanged; the web UI `apps/cli/src/workbench/web/papers.ts:95` renders `item.paperKey`.
- Tests that pin the key-only behavior: `plugin/tests/dashboard-view.test.ts:715-745`, `packages/core/tests/dashboard-model.test.ts`, `apps/cli/tests/workbench-papers.test.ts`, `plugin/tests/dashboard-history-sync.test.ts:211,263`.

**G. Port: where file: documents get title/abstract**
- `library/prior-work.ts` `basisPaper` returns null unless the key is in the catalog and `source==="arxiv"`; `file:` hits are dropped with a debug log. The catalog never contains file docs.
- The authoritative source is the knowledge-base manifest: `library/fulltext/knowledge-base.ts:92-118` `FullTextPaperKnowledgeRecord` has `status`, `title?`, `abstract?`, `titleVersion?`, `contentHash?`, `filePaths`. Precedent for reading it: `library/personal-library-direction-proposer.ts:425-436` `fallbackPaperFromManifest` (ready + title).
- Hosts already have the store: CLI `apps/cli/src/library-prior-work.ts` (`knowledgeBase` const), plugin `plugin/src/library/prior-work-wiring.ts` (`openSearchSource()` returns `store` / `sourceManifest`).
- **Measured on test_library (isolated index, local embedding):** manifest has 361 papers: 146 arXiv + 215 file (211 ready, 4 failed). Of 211 ready file docs, 211 have a title, **205 have title+abstract** (6 have bibcode-like titles such as `1982ARA&A..20..547F` or Chinese-book titles with no abstract). All 205 are inside the DTO bounds (abstract max 2399 <= 6000, title max 617 <= 2000). So the DTO bounds drop only about 3 percent; **but** extraction quality is mixed (some "abstracts" are author lists, e.g. "Transformers without Normalization Jiachen Zhu 1, 2 ..."; two abstracts are under 100 chars). No quality gate is proposed; the model's "genuinely comparable" refusal handles it (ADR 0015 consequence stands).
- Self-match risk: a digest paper whose PDF sits in the library under an unrecognized filename is a `file:` doc and would be its own top hit (the port only excludes the query's `arxiv:` key). Cheap guard in the port: drop a file hit whose normalized title equals the query title.

## 1.2 Decision (minimal, compatible)

1. **One basis-key predicate** in `personalized-novelty.ts`: `isCanonicalNoveltyBasisKey` = canonical `arxiv:` key OR `/^file:sha256:[0-9a-f]{64}$/`. Replace `isCanonicalArxivPaperKey` only at the basis sites (`:412, :449, :627, :854`). Daily paper keys (`:391, :446, :825, :918`) stay arXiv-only.
2. **Representative shape is a union by key form**, no invented data: arXiv basis keeps the exact 6 keys; a `file:` basis is exactly `{paperKey, title, abstract}` (same title/abstract bounds). The prompt payload is `JSON.stringify` of whatever exact shape, so the model sees only what exists. Do not fabricate authors/published/categories.
3. **No links** for `file:` basis papers. Today neither Markdown nor Dashboard links any basis paper, so nothing regresses; local path/PDF opening is out of scope. In the visible Markdown line a `file:` basis prints title only, an arXiv basis keeps `Title (arxiv:...)`.
4. **Titles for all basis papers via the marker (v2) and index**:
   - Marker payload gains `t: string[]` (display titles aligned with `n.comparisonBasis`); the marker version becomes 2 exactly when `t` is present. v1 markers (no titles) still parse. `n` stays the existing 4-key `PersonalNovelty`, so `normalizePersonalNovelty` keeps its meaning.
   - Titles are display data, capped at 300 code units in the marker (longer titles are shortened with an ellipsis when rendering the marker; the Markdown visible line keeps the full title). Titles are accepted only when `comparisonBasis.length <= 5`. `PERSONAL_NOVELTY_MARKER_MAX_CODE_UNITS` is re-derived (rough worst case: 6000 explanation + 5x(79 key + 300x6 title) + scaffolding, x4/3, so about 24_000) and a worst-case test pins it.
   - Index `noveltyByReport` stores `PersonalNovelty` (legacy 4-key) or the existing display shape `PersonalNoveltyWithBasis` (5 keys, with `comparisonBasisTitles`): a new `normalizePersonalNoveltyRecord` accepts either exact shape and is used by `reconcile...` and `normalizeNoveltyByReport`.
   - Dashboard: `DashboardPersonalNoveltyBasisPaper { paperKey; title?: string }`; lines show titles joined with "; " (titles can contain commas) and fall back to the key when no title exists. CLI web view renders `title ?? paperKey`.
5. **Port**: new dep `knowledgeBase: { loadManifest() }` on `createLibraryPriorWork`; manifest loaded lazily once per run and only if a `file:` hit lacks a catalog entry; a file hit is usable when `status==="ready"` and non-empty title and abstract and within DTO bounds (precedent `fallbackPaperFromManifest`). The 8a port change also needs the retrieval slot logic to count usable file docs (they are the majority of hits, see Part 2).
6. **Rejected alternative**: resolving titles at Dashboard render time from the catalog/manifest. It needs async library lookups in the view, breaks when the library changes or is detached, and the index is rebuilt from reports anyway.
7. **Compat stance for owner confirmation**: old plugins will not render novelty for any day written by the new version (whole-report `invalid`, no crash). Recommended to accept because the feature is unreleased; the alternative (keeping v1 byte-compatible) cannot also deliver titles for arXiv basis papers.

## 1.3 Chunk plan

### Chunk 8a - file: documents are valid novelty basis (port + DTO + persistence accept file: keys)
- change: key predicate and union representative shape (`personalized-novelty.ts`), optional one-line prompt note and contract-version bump if the prompt changes, `file:` key accepted by `normalizePersonalNovelty` / marker / index (so persistence cannot fail on a `file:` basis), Markdown visible line prints title only for `file:`, port reads title/abstract from the manifest, self-title guard, CLI and plugin wiring pass the manifest store. Marker bound raised enough for 5 `file:` keys (no titles yet).
- kind: behavior
- first signal (named failing tests, all red before the change):
  1. `packages/core/tests/personalized-novelty.test.ts` new cases: `preparePersonalizedNoveltyInput` accepts a `{paperKey:"file:sha256:<64 hex>", title, abstract}` representative and rejects malformed forms (uppercase hex, 63 hex, `file:abc`, a file rep carrying `authors`, an arXiv rep missing `authors`); `planPersonalNoveltyCalls` request payload for a file basis contains no `authors`; `decodePersonalNovelty` accepts a model answer citing the file key; checkpoint decode round-trips. Command: `npm test -- tests/personalized-novelty.test.ts`.
  2. `packages/core/tests/library-prior-work.test.ts` new cases: a ready file doc with title+abstract becomes a basis paper; failed / no-abstract / out-of-bound docs are dropped; a file doc whose title equals the query title is dropped; manifest loaded once and not at all when there are no file hits. Command: `npm test -- tests/library-prior-work.test.ts`.
  3. `packages/core/tests/daily-summary-assembler.test.ts`: file basis renders title without the hash in both languages and the marker renders/parses. `packages/core/tests/paper-index.test.ts`: reconcile accepts a file key; load keeps it.
  4. `packages/core/tests/pipeline-library-novelty.test.ts`: fake port returning a file basis => Markdown has marker + visible line, `noveltyByReport` has the file key, run `completed`.
  5. Hosts: `plugin/tests/library-prior-work-wiring.test.ts` (manifest source passed through), `apps/cli/tests/cli-runtime.test.ts` (an unrecognized-filename PDF indexed as `file:` produces a marker with a `file:` basis). Commands: `npm run test --workspace obsidian-arxiv-daily -- tests/library-prior-work-wiring.test.ts`, `npm run test --workspace arxiv-daily -- tests/cli-runtime.test.ts`.
- checks: the above Green; `tests/library-prior-work-real-index.test.ts` and `tests/lexical-search.test.ts` regression; `npm run typecheck`; `npm run check:boundaries`; `grep -n "isCanonicalArxivPaperKey" packages/core/src/pipeline/personalized-novelty.ts` leaves exactly the four daily-paper sites.

### Chunk 8b - basis titles in marker v2, index and Dashboard
- change: marker v2 with `t` (cap 300, <=5 basis), re-derived marker bound, `normalizePersonalNoveltyRecord` for index write/load, parse result carries titles, Dashboard model/view and CLI workbench show titles (fallback to key), update the stale comments (`dashboard/model.ts:85`, `plugin/src/dashboard/view.ts` header) and ADR 0015 Consequences (file docs as basis; titles persisted in marker v2).
- kind: behavior
- first signal (named failing tests):
  1. `packages/core/tests/daily-summary-assembler.test.ts` (marker section): v2 marker round-trips with titles; a literal v1 marker string captured from the current renderer still parses with no titles; title over 300 units is shortened and idempotent through render/parse; titles count mismatch or more than 5 basis rejected; worst-case marker (5 file keys, 5 max-length titles of escape-heavy characters, 1000-unit explanation of escape-heavy characters) is <= the new bound. Command: `npm test -- tests/daily-summary-assembler.test.ts`.
  2. `packages/core/tests/paper-index.test.ts`: reconcile stores titles; legacy 4-key entry survives load; invalid titles drop only that entry.
  3. `packages/core/tests/dashboard-model.test.ts`: basis `{paperKey,title}`; legacy => `{paperKey}`.
  4. `plugin/tests/dashboard-view.test.ts` (rewrite of the case at `:715-745`): "Personal novelty: new method vs. prior papers: Title A; Title B", key fallback when no title. `plugin/tests/dashboard-history-sync.test.ts`: a report with v2 markers restores titles into the index. Command: `npm run test --workspace obsidian-arxiv-daily -- tests/dashboard-view.test.ts tests/dashboard-history-sync.test.ts`.
  5. `apps/cli/tests/workbench-papers.test.ts` (+ the web UI test that pins "比较依据"): titles rendered. Command: `npm run test --workspace arxiv-daily -- tests/workbench-papers.test.ts`.
- checks: `npm run typecheck`; core `npm test`; `npm run test:workspaces`; `npm run build --workspace obsidian-arxiv-daily`; real Obsidian check of the Dashboard row (or journal "not done", as the phase doc requires for Dashboard changes). Acceptance tooling under scripts/acceptance that pins the key-only Dashboard text, if any, must be updated.
- order: 8a then 8b; 8b is technically independent but landing 8a alone would show 76-char hash keys in the Dashboard for file basis papers, so release them together and re-run the quality evaluation after both.

## 1.4 Risks / open points for the owner
- Old plugins lose novelty for whole days written by the new version (stance in 1.2.7).
- File-doc extraction noise (author-list "abstracts", bibcode titles): not gated; model refusal is the only filter, as for unrelated arXiv papers (see Part 2: two thirds of hits are weak or unrelated either way).
- The 300-unit marker title cap is a display decision; the full title still appears in the Markdown line.

---------------------------------------------------------------------------

# Part 2 - Chunk 10 measurement: lexical vs hybrid (local embedding), no business-LLM calls

## 2.1 Setup (all under /tmp/novelty-eval/measure/, library read-only)
- Config: /tmp/novelty-eval/measure/xdg/arxiv-daily/config.toml (dummy LLM base_url http://127.0.0.1:9/v1/chat/completions, key "dummy", embedding.mode=local), HOME and cache under measure/. Wrapper measure/cli.sh. Library: /home/tiandc/Desktop/plugin_test/test_library (376 files), connected by `library connect`. Embedding runtime/model copied from the earlier run cache (no download).
- **Authorization**: `library connect` reports `authorization-required` (fingerprint sha256:173d59f5...dd1a1, bound to the dummy endpoint). I never authorized. `library prepare`, `scan`, `index` and `search` (lexical, hybrid, dense) all ran without authorization in local-embedding mode. Authorization only matters for the daily-report novelty gate (`inspectCliLibraryConnection(...).status.kind`) and remote embedding.
- `library scan` hit arXiv HTTP 429 again (metadata fetch, 0 catalog papers). To avoid hammering arXiv I copied the previous catalog file (same library, same scope/identification fingerprints, 146 arXiv papers) into the isolated vault and then ran a **fresh** `library index`: 357 indexed, 4 failed, 86 s wall (local multilingual-e5-small-q8, 384 dim). Index composition: 146 `arxiv:` + 211 ready `file:`.
- Queries: the 20 digest papers of 2026-10-14 (/tmp/novelty-eval/digest-papers.json; title+abstract complete from the arXiv API). Lexical query = the same 60-unique-term truncation used by the earlier retrieval-only check (/tmp/novelty-eval/retrieval-trunc.json, title words first, then abstract order; stopwords count toward the cap because the BM25 tokenizer, `fulltext/bm25-retrieval.ts:36`, does not drop them). Top 6 per query (port asks limit+1=6). Same call path as the port (`searchFullTextKnowledgeBase`, `maxHitsPerPaper:1`, catalog titles as title scores). Benchmark programs: measure/bench/bench.ts (+bench2.ts, prof.ts) bundled from the worktree sources with esbuild; raw results measure/bench-results.json, bench-dense.json; per-hit judgments measure/judge.py / grades.json; hit listing measure/hits-view-num.txt.
- Modes: **lexical** (60-term query); **hybrid60** (60-term query on both legs = what `library search --mode hybrid` can express); **hybrid full-embed** (embed full title+abstract, 60-term lexical leg = the likely production call, `lexicalQueryText`); plus extra **dense** (full text / 60-term) for reference.
- Fact for Chunk 7: `library search --mode hybrid` with the full title+abstract fails with `searchGenerationBm25: query has more than 64 unique terms` (the lexical leg of hybrid), while `--mode dense` with the full text works. The cap must apply to the lexical leg in hybrid too.

## 2.2 Retrieval composition (top 6 x 20 papers = 120 hits per mode)

| mode | arxiv: hits | file: hits | file hits usable (title+abstract in manifest) | papers with no arxiv: hit |
|---|---|---|---|---|
| lexical 60 terms | 51 (43%) | 69 (57%) | 69/69 | 2/20 |
| hybrid 60 terms | 51 | 69 | 69/69 | 0/20 |
| hybrid, full embed | 48 | 72 | 72/72 | 1/20 |
| dense, full text | 52 | 68 | 68/68 | 1/20 |
| dense, 60 terms | 49 | 71 | 71/71 | 1/20 |

Every file: hit in these runs had a usable title and abstract. With the current arXiv-only port the usable basis is 2.5 papers per digest paper (51/20); after Chunk 8a it is the full 6 (5 after self-exclusion), i.e. file docs are 57-60 percent of the useful material for this library.

## 2.3 Relevance (my judgment from titles only: digest paper title+abstract vs hit title; related / weak / unrelated; single rater, 20 papers, so differences of a few points are noise)

Counts over usable hits assuming Chunk 8a (arxiv + file with abstract), top 6:

| mode | related | weak | unrelated | related % | related+weak % | papers with >=1 related | papers with >=1 related or weak |
|---|---|---|---|---|---|---|---|
| lexical 60 terms | 22 | 35 | 63 | 18 | 48 | 11/20 | 18/20 |
| hybrid 60 terms | 25 | 38 | 57 | 21 | 52 | 9/20 | 17/20 |
| hybrid, full embed | 27 | 37 | 56 | 22 | 53 | 9/20 | 17/20 |
| dense, full text | 21 | 35 | 64 | 18 | 47 | 8/20 | 18/20 |
| dense, 60 terms | 22 | 32 | 66 | 18 | 45 | 9/20 | 17/20 |

Top 5 only (what the model would see after self-exclusion): lexical 21 R / 29 W / 50 U; hybrid60 23 / 34 / 43; hybrid full-embed 23 / 32 / 45; dense full 19 / 30 / 51.

arXiv-only (today's port, before 8a), papers with >=1 related arxiv hit: lexical 8/20, hybrid60 8/20, hybrid full-embed 7/20, dense full 6/20. So 8a lifts lexical from 8 to 11 papers with at least one related basis paper.

Other observations: top-6 overlap between lexical and hybrid60 is 3.1 of 6, lexical and dense full 1.35 of 6, hybrid60 and hybrid full-embed 3.95 of 6. Papers with no related library paper at all (cosmic voids, likelihood-ratio statistics, globular-cluster ages, 4PCF parity violation, TDCOSMO x2, SKA radio dipole, BHB stars, SPHEREx photometry...) get 6/6 weak-or-unrelated hits in every mode, so about half of the digest depends entirely on model refusal (`noComparablePrior`), in all modes alike. Per-paper R/W/U (top 6):

| paper | title | lexical | hybrid60 | hybrid full-embed | dense full |
|---|---|---|---|---|---|
| 2610.16007 | TractorJAX: GPU-Accelerated SPHEREx Forced Photometry... | 1/2/3 | 0/2/4 | 0/3/3 | 0/3/3 |
| 2610.16742 | DES Y3 redMaPPer clusters in eROSITA region | 6/0/0 | 6/0/0 | 6/0/0 | 5/0/1 |
| 2610.18348 | Cosmic voids within DESI DR1 | 0/3/3 | 0/2/4 | 0/5/1 | 0/6/0 |
| 2610.17186 | Likelihood-ratio distribution as detectability measure | 0/0/6 | 0/0/6 | 0/1/5 | 0/1/5 |
| 2610.14169 | CARMA VI globular cluster ages | 0/2/4 | 0/1/5 | 0/0/6 | 0/0/6 |
| 2610.17647 | Promise of gravitational-wave astronomy | 3/0/3 | 3/0/3 | 3/0/3 | 1/1/4 |
| 2610.16741 | XMM Cluster Survey eROSITA cross-calibration | 3/3/0 | 5/1/0 | 6/0/0 | 6/0/0 |
| 2610.17200 | Stacked kSZ, satellites vs baryonic feedback | 1/1/4 | 2/4/0 | 2/2/2 | 1/2/3 |
| 2610.16574 | NFDiag MCMC convergence with normalizing flows | 1/0/5 | 2/0/4 | 2/1/3 | 2/1/3 |
| 2610.18707 | 4PCF parity violation, Quijote-Odd | 0/3/3 | 0/4/2 | 0/4/2 | 0/1/5 |
| 2610.14881 | TDCOSMO 2026 Hubble tension | 0/1/5 | 0/2/4 | 0/3/3 | 0/1/5 |
| 2610.17022 | Cosmic radio dipole, SKA | 0/3/3 | 0/3/3 | 0/2/4 | 0/2/4 |
| 2610.15999 | Shape degeneracies and H0 (cluster lenses) | 1/0/5 | 0/0/6 | 0/0/6 | 0/3/3 |
| 2610.14781 | TDCOSMO 2026 3% H0 | 0/0/6 | 0/0/6 | 0/0/6 | 0/0/6 |
| 2610.18099 | Radio ghosts, minihalo candidates | 3/1/2 | 3/3/0 | 3/2/1 | 3/2/1 |
| 2610.15031 | Blue horizontal branch stars photometric ID | 0/3/3 | 0/3/3 | 0/3/3 | 0/2/4 |
| 2610.16536 | CosmoAgent: AI agent for theoretical cosmology | 1/1/4 | 1/1/4 | 1/1/4 | 1/3/2 |
| 2610.16496 | DESI DR2 BAO-supernova sensitivity | 1/4/1 | 1/3/2 | 2/3/1 | 2/1/3 |
| 2610.15536 | H0 and baryon density from localized FRBs | 0/4/2 | 0/5/1 | 0/4/2 | 0/3/3 |
| 2610.17857 | Merging clusters weak lensing for CSST | 1/4/1 | 2/4/0 | 2/3/1 | 0/3/3 |

(The earlier title-only estimate of the lexical arXiv-only subset was 31 / 35 / 33 percent; this rerun has 13 R / 18 W / 20 U on the 51 arXiv hits, i.e. 25 / 35 / 39 percent, same ballpark; the file hits are noisier: 9 R / 17 W / 43 U.)

## 2.4 Time (this machine, CPU wasm/onnx, library of 357 docs; per query, in one process unless noted)

| mode | first query | later queries (median / min-max) | 20 queries total |
|---|---|---|---|
| lexical 60 terms | 4058 ms (includes opening the generation) | 3797 ms (3759-3875) | 76.3 s |
| hybrid 60 terms | 4513 ms (includes local model load) | 3993 ms (3953-4103) | 80.5 s |
| hybrid, full embed | 4049 ms | 4025 ms (3960-4082) | 80.5 s |
| dense, full text | 846 ms (includes model load) | 166 ms (138-177) | 3.9 s |
| dense, 60 terms | 131 ms | 132 ms (130-138) | 2.7 s |

Separate processes via the CLI (one query each, includes node start): lexical 4.4 s wall; hybrid 5.1 s wall (11.8 s CPU vs 4.6 s: the model load/embedding uses extra threads). Local model load is about 0.6-0.7 s the first time (130 MB model files, about 650 MB RSS in the benchmark process); embedding one query afterwards costs 15-130 ms.

**Key finding: the lexical leg, not embedding, dominates.** A BM25 query costs about 3.6-3.9 s regardless of query length (1 term: 3.8 s, 11 terms: 3.5 s). A CPU profile attributes about 70 percent of the time to the pure-JS `sha256Hex` (`utils/digest.ts`; probably the lexical-shard digest at `fulltext/generation-index-store.ts:2229`) on every query; `openCurrent` itself is 1-7 ms. Dense retrieval over the same index takes 0.13 s. Hybrid = BM25 leg + dense leg, so it costs about 5 percent more than lexical. For a daily run: 20 papers x about 3.8 s = about 76 s of CPU in the host process, in either mode. This is independent of the lexical-vs-hybrid choice but matters for Obsidian responsiveness (ADR 0007 lesson) and is not covered by any current chunk. I did not check whether the Obsidian plugin's BM25 path yields to the event loop.

## 2.5 Recommendation for local mode
- **Keep lexical for local embedding mode (no wiring change; Chunk 10 closes as "measured, no change").** Numbers: related share 18 percent (lexical) vs 21-22 percent (hybrid), papers with at least one related basis 11/20 vs 9/20, related+weak 48 vs 52-53 percent; with 20 papers and one title-only rater these are statistically the same. Hybrid costs +0.5 s one-time model load, +about 0.2 s per query (about 5 percent), and brings the 130 MB wasm model into the Obsidian renderer, the ADR 0007 hazard, for no demonstrated gain. Remote mode keeps hybrid as designed (ADR 0015 section 2).
- Be explicit that retrieval quality is mediocre in every mode (about 20 percent related, about 50 percent unrelated at top 6; half the papers have no related library paper), so the feature's value rests on the model refusing; no mode choice fixes that. Spending the 8a change (file docs as basis) is the largest measurable gain (usable basis 2.5 to 6 per paper; 8 to 11 papers with a related basis).
- Follow-ups worth a chunk (not done): (1) the BM25 per-query cost of about 3.8 s (avoid re-hashing per query, e.g. verify once per opened generation or per run); a dense-only local mode would be about 28x faster at similar relevance (R 18 percent, 8-9/20 papers with a related hit) but needs the embedding model, so it is a trade-off for the owner, not a recommendation from this data; (2) the 60-term cap currently spends slots on stopwords ("the", "we", "of"); dropping stopwords before capping is untested but cheap; (3) hybrid, if ever used, must pass a capped `lexicalQueryText` (confirmed above).
- Caveat on the judgment: titles only, one rater, 20 papers from one library and one day; not reproducible as a metric. Raw judgments are in /tmp/novelty-eval/measure/judge.py and grades.json for re-grading.
