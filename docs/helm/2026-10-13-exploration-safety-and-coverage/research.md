# Research — personal literature library exploration (intake recon, 2026-10-13)

Disposable intake notes for P3/P4. Decisions go to journal.md.

## Plugin product — visible library UI

- Settings "Personal library" row (`plugin/src/settings/tab.ts:580-641`): Choose/Change folder (native picker — not explorable), Retry preparation (shown when no completed run or a preparation error, `packages/core/src/library/library-connection.ts:175`), Cancel, Revoke.
- Embedding mode select; switching to remote opens the consent disclosure `.arxiv-daily-library-authorization-modal` (tab.ts:269, :786ff).
- "Topics from library" → "Review suggestions" (tab.ts:767-779; `definitions.ts:226`), disabled unless the index has a completed run with papers and no preparation error. Opens `.arxiv-daily-interest-review-modal` (`plugin/src/library/interest-profile-modal.ts`): Update/Generate (285), Scan library (312), Library overview (229), Refresh (238), candidate checkbox + "Add to research topics" (405), Rename topic (480), Change topic / representative papers / Preview matches / Save edits (768-837), Move (893), Discard (930), Done (512). Generation asks for Library processing consent first (`main.ts` `authorize` ~800).
- Dashboard search "Library matches" block (`plugin/src/dashboard/view.ts:571,2373`; `library-search-block.ts`), open-PDF icon per row.
- Dashboard row "Find similar papers" → `SimilarPapersModal` "Library similar" tab (view.ts:1576, 2311-2360; `similar-papers-modal.ts:78`).
- Command-palette only (out of scope): review directions (commands.ts:403), incremental suggestions (416), preview files (440), scan (464), reload catalog (485), index (509), search (537), diagnose full-text runtime (574).

## Reading workbench — visible library UI

- Sidebar "个人文献库" (`apps/cli/src/workbench/web/app.ts:176` → :384); library page (`web/library.ts:34-46`): 连接与索引, 方向审核, catalog filter (`GET api/library`), index search lexical/hybrid/dense (`POST api/library/search`).
- Settings library row (`web/settings.ts:105,218,494-524`): Choose folder = absolute-path text field (no native picker), Confirm and build index (disclosure → `library-build`), cancel, revoke.
- Review page (`web/library-review.ts`): generate/regenerate (234), preview (125), save (111), move (120), discard (126), rename topic (159), select all/none/accept selected (138-141), back (243). Server `server.ts:206-269`, `library.ts:94-131`.
- Blocker: `library-build` always runs prepare/scan/index (`settings-actions.ts:49`); prepare runs `npm ci` into `<cacheDir>/runtimes/pdfjs-5.4.624` (`packages/node-runtime/src/library-runtime.ts:24-37,75-84`) against the real registry (the fetch preload does not cover npm).

## Starting state without the folder picker

- Plugin connection: `data.json` `libraryConnection {schemaVersion:1, selectedRoot, rootIdentity:"dev:ino", eligibleExtensions:[".pdf"], processingDepth:"metadata-and-abstracts"}`; builder `scripts/desktop-acceptance/settings-fixture.mjs:32-74` (`connectedLibraryFixture`, `readRootIdentity`); seeded in `acceptance.mjs:95-108`. `createFixtureEnvironment` writes 2 PDFs to `vault/library` (`scripts/acceptance/fixtures.mjs:183-187`). Exploration can write it via `driver.applySettingsPatch` (`exploration-obsidian.mjs:992`).
- Consent: `authorization {fingerprint}`; plugin computes it with `authorizeLibraryProcessing(expectedFingerprint)` (`main.ts:646`) — call over CDP in `prepare()`. Local embedding indexing needs no consent; remote embedding and LLM generation do.
- On-disk under `arxiv-daily/.index/`: `personal-library-catalog.json`, `personal-library-search-index/<scopeHex>/<idHex>/current.json`, `personal-library-knowledge-base/...`, `personal-library-profiles/.../direction-proposal.json`, `incremental-suggestions.json`. Hex dirs are fingerprint-derived; produce them by running real scan/index in `prepare()` rather than hand-writing.
- Existing desktop journeys inject in-memory state rather than building: `direction-review.mjs:231-246`, `library-settings.mjs:362` (`setLastIndexRunExpression`), `direction-review.mjs:276`.
- Endpoints: fixture `/embeddings` returns 3-dim vectors (`fixtures.mjs:137-141`) → settings need `embedding.dimension = 3` (exploration default 384, `scripts/acceptance/obsidian.mjs:33`). Proposal prompt answered (`fixtures.mjs:58-63`); incremental diff prompt NOT answered. Plugin traffic proxied via `installHttpProxyExpression`. Local embedding downloads ~130 MB model — unsuitable.

## Program-checkable outcomes

- Consent: `libraryConnection.authorization` present / removed. Embedding mode: `settings.embedding.mode`.
- Index: `current.json` exists; `plugin.libraryIndexStatus.snapshot().lastRun.papers > 0`.
- Proposal: `direction-proposal.json` exists; fixture `kind:"proposal"` request.
- Accepted candidate: `settings.arxiv.topics[].directions[]` with `origin:"library"` (`packages/core/src/settings/accept-proposed-topics.ts:181`) + `libraryProposalAcceptances` receipt (`main.ts:916-930`). Workbench: TOML direction `origin = "library"`.
- Search / similar: `.arxiv-daily-dashboard__library-item` count, similar modal list, fixture `kind:"embeddings"` requests.

## Unknowns

- Whether the proxy wrapper survives `reloadPlugin()`; real index time for 2 PDFs vs the 60 s `prepare()` deadline (`exploration.mjs:325`); whether the observer exposes `<select>`/`<details>` in the review dialog; whether workbench `GET api/library` works without a prepared runtime.
