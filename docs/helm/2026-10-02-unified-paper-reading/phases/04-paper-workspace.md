# P4 — paper-workspace

goal_ref: ../goal.md
created: 2026-10-02T12:40:00+08:00
updated: 2026-10-02T13:59:30+08:00
revision: 3

## Outcome

The right-side area switches between a unified paper list and reading content while the left provides calendar/filter navigation and adjustable space.

## Assumptions

- The latest user-approved placement supersedes both a left-side paper list and full Dashboard replication.
- Index entries provide identity, summaries, recommendation reasons and marks; absent index information remains explicit without hidden repair.
- Favorite maps to Obsidian priority=high independently from inbox/to_read/read status. Other existing states are preserved.
- Browser localStorage alone cannot remember layout across random localhost ports; small non-secret layout preferences belong beside the CLI config.

## Approach

Protected APIs query existing core data and mutate individual mark fields atomically with preconditions. The client preserves list context when reading and reuses existing Markdown/generation services. A separate sidebar controller owns resizing, collapse and persisted layout preferences; no duplicate business storage.

Visual thesis: preserve the warm split reader, give the right-side list the full working width, and keep navigation restrained.
Content plan: left calendar/scopes/search; right title/filters/order/rows or article; explicit back-to-list and original Markdown/file access.
Interaction thesis: stable list/reading history, persisted-only mark feedback, pointer/keyboard resizing and responsive mobile filter access.

## Chunks

### Chunk 1 — Unified paper and preference APIs

- change kind: behavior change
- strategy: strict Red-Green at actual HTTP and persistence boundaries
- Red: new query/detail/mark/preferences routes fail before implementation
- Green: index-only discoveries, date/search/filter/order/pagination, safe note availability, persistent independent marks, stale conflicts, invalid/foreign requests, idempotency and unchanged source files pass; layout preferences survive a fresh service
- regressions: existing calendar/server/inspection tests, CLI typecheck/boundaries
- [x] implementation and tests accepted

### Chunk 2 — Right-side list and reading navigation

- change kind: behavior change
- strategy: strict Red-Green DOM/API tests plus browser layout review
- Red: default right-side list, back-state preservation and mark controls absent
- Green: date list→paper/full report→back preserves date/search/filter/page/scroll; stored-only mark success and conflicts; explicit detail generation refresh; standalone Markdown files; stale-response and mobile navigation tests pass
- regressions: existing reader/calendar interaction contracts updated only where the new approved placement supersedes old expectations; renderer/server regressions and CLI typecheck
- [x] implementation and tests accepted

### Chunk 3 — Adjustable sidebar and integrated acceptance

- change kind: behavior change plus visual styling
- strategy: strict Red-Green for preference/resize controller; browser verification for widths
- Red: separator, pointer/keyboard resize, remembered width and collapse controls absent
- Green: bounded resize, remembered preferences, late-load protection, mobile adaptation and failure feedback pass
- regressions: actual copied CLI persists paper marks and preferences; calendar/read/generate flow remains functional; desktop/mobile screenshots and zero normal console errors
- [x] implementation and tests accepted

## Abort / reshape triggers

- If a right-side route loses its return context, fix navigation before adding more controls.
- If an index write requires direct raw JSON manipulation outside core mutation, keep the operation blocked until it uses the existing transaction.
- If a missing report/index cannot be identified, show an explicit unavailable state rather than infer zero papers or overwrite a user note.

## Implementation contract recorded at continuation

### Paper API contract

- `GET api/papers?scope=all|inbox|to_read|read|starred&q=&topic=&date=YYYY-MM-DD&sort=published|title|priority|relevance&direction=asc|desc&offset=0&limit=20` (max100). Response: `{papers,total,offset,limit,nextOffset,counts:{all,inbox,to_read,read,starred},libraryCount,topics:string[],day:WorkbenchCalendarDay|null}`. Counts are after date/search/topic but before scope; libraryCount excludes ignored using existing core rules. Date must match daily-report references, not a manual note's incidental seenDate.
- `GET api/paper?key=paperKey` returns `{paper}`. Paper DTO: `{key,arxivId,title,authors:string[],published,topics:string[],category,status:PaperStatus,priority:PaperPriority,starred,abstract,summary:PaperSummary|null,detailPath:string|null,reports:[{path,date,title,available}],originalUrl,pdfUrl,provenance:DashboardOccurrenceProvenance|null,novelty:DashboardPersonalNovelty|null}`. Export types from server `workbench/papers.ts`; frontend imports them with `import type`.
- Availability/paths are checked against existing scoped document catalog, not blindly trusted from index strings. Canonical arXiv URLs or validated HTTP(S) links only. No hidden history reconciliation or parsing new papers into the index on GET. Core queryDashboard/PaperSearchIndex provide established search/filter/sort/provenance logic.
- `POST api/paper/mark`: `{key,action:'status',value:'inbox'|'to_read'|'read',expected:oldPaperStatus}` or `{key,action:'star',value:boolean,expected:oldPriority}`. Return `{paper}`. Use PaperIndexStore.mutate to atomically reload, reject stale field conflicts409, update only the chosen field, save via existing persistence. Desired value already current is an idempotent no-op. Favorite is high priority, independent of status; unstar high→normal, preserve low if already unstarred. Missing404, invalid400.
- Add optional `WorkbenchOptions.beforeWrite` gate, supplied by launch using the existing config-revision check; use for mark writes so an old UI cannot silently edit after a config change. Existing run behavior stays intact.
- `GET api/preferences` returns `{sidebarWidth:number|null,sidebarCollapsed:boolean}`, default null/false with no file write. POST same shape persists private/atomic non-secret layout preferences beside configPath (e.g. workbench-ui.json); width must be null or finite280..900. This survives new localhost ports. Do not edit CLI TOML or Vault files.

### Client structure

- Rework `web/app.ts` around right-side list/reading routes; optional `web/papers.ts` owns rendering. Preserve public mountWorkbench(root, options) and existing Markdown renderer, generation/settings dialog and run handling.
- Left: calendar plus scope/search/topic filters, secondary document-file access. No paper or Markdown result rows on the left. Right list includes metadata/recommendation, known note availability, independent persisted star/status controls and pagination/sort.
- Date click filters the right list. Explicit `read-day` opens the whole saved report. Paper click opens existing overview; explicit detail view or generation uses old operations. `back` restores date/search/scope/sort/offset and right-pane scroll. Retain standalone Markdown file list on the right via `browse-documents`.
- Root `data-view='list'|'reading'`; preserve `.workspace`, `.library-pane`, `.reading-pane`, `.toc-pane`, `.header-actions`. Mobile defaults to right list, `.show-filters` toggles the left; selecting date/filter returns to list. Desktop ToC appears only during reading.
- New `mountSidebar(root,{request})` in `web/sidebar.ts` returns a cleanup function. It adds the divider and desktop collapse control, persists layout through preferences APIs, and reacts to data-view/viewport changes. Add `sidebar.css` after main CSS in the existing build asset collector; it is not yet included or mounted.
- Persist-only mark UI: disable saving control, apply returned state only after success, refresh actual values on conflict/error without false success. Preserve legacy saved/reading/ignored labels until explicitly changed. On detail-generation completion refresh availability without stealing the reading selection.

### Incoming checkpoint evidence (before this implementation)

- Last committed code baseline remains the prior calendar/reader implementation; `8664203` committed the latest plan only. Current UI bundle still runs the old left-list layout.
- Uncommitted `web/sidebar.ts` + `web/sidebar.css` are implemented but not wired into app/build. `workbench-sidebar.test.ts`:4 expected failures against the stub, then4 Green after implementation. DOM checks cover remembered width, keyboard/pointer constraints, collapse, delayed-load protection and save-error/mobile handling. Integration/browser acceptance still required.
- `workbench-paper-ui.test.ts` (8 cases) is a useful initial fixture suite written by a failed delegate. `workbench-papers.test.ts` (5 actual HTTP/persistence cases) was added by owner. Their combined run has13 expected Red failures because APIs/right-side UI are absent. Log: `.artifacts/paper-workspace-red.log`. Do not commit these tests as accepted until Green.
- Two delegate turns failed on model token rate limits. No backend implementation was produced. The UI delegate wrote only the initial8-case test file; all other current work is owner-authored. No running test/server process needs resuming.
- Next concrete step: implement `workbench/papers.ts`, `preferences.ts`, wire routes in server.ts and the write gate in launch.ts to turn the5 HTTP tests Green; then implement the right-side client and integrate sidebar, preserving existing navigation/security/Markdown tests wherever their old placement assumptions are not superseded.


## Accepted implementation and verification

- Backend checkpoint `67f9c02`: protected paper list/detail/mark routes use the shared Dashboard query/search and PaperIndexStore transactions; preference reads are side-effect free and writes are private/atomic beside CLI config. Seven HTTP/persistence tests cover index-only discoveries, date occurrences, scope counts, persisted marks, concurrent conflict, invalid payloads and missing index/report access. Initial five missing-route failures and a malformed-status regression were observed before Green.
- UI checkpoint `b02dd89`: right-side paper list, overview/full Markdown routing, retained query/page/scroll, explicit generation and persisted-only mark feedback; sidebar pointer/keyboard resize, collapse and durable preferences are integrated. Eleven paper UI tests and four sidebar tests pass; the original reader/calendar suites retain their behavior coverage with only superseded placement and auto-open assumptions adapted. Initial eight UI and four sidebar Red cases were observed; later scroll restoration and mobile search regressions also went Red before fixes. The inherited overview fixture used arrays where the API specifies nullable provenance objects; it was corrected to the contract. Async navigation assertions now await the rendered list.
- Full CLI suite: 207 tests in 23 files pass. After the final file-mode topic-control correction, all 34 relevant UI tests pass again. CLI typecheck, workspace boundaries, product-unit inventory and strict Claude plugin validation pass.
- Portable integration: all seven checks pass. The copied-CLI workbench test additionally verifies saved marks/preferences in a fresh process, index-only papers, bundled assets and configuration revision conflicts. The final rebuilt bundle's workbench test passes again.
- Real headless Chromium, isolated `.artifacts/p4-demo` config/Vault and HTTP fixture: generate 2026-05-11 daily report -> two indexed papers -> mark to-read and favorite -> reload -> overview -> explicit detail generation -> saved detail -> back -> full report -> back. Independent Markdown/math files remain readable. No real paid provider or user data was used.
- Browser layout checks: 1440x1000 desktop and 390x844 mobile; pointer resize and keyboard adjustment restore 556px after reload, collapse restores after reload, and a new server/port reads the previous preference. Mobile search stays open while typing; returning shows the updated list with no horizontal page overflow. Final console: zero errors and warnings. An initial resize script reloaded before the preference request completed; corrected verification waits for the persisted response before reload and passes.
- Screenshots (local ignored artifacts): `output/playwright/p4-paper-list.png`, `output/playwright/p4-mobile-list.png`; final suite log: `.artifacts/p4-demo/final-tests.log`. UI assets have been rebuilt; already-running workbenches need a service restart.
- No success criterion was waived. Public network/provider integration and Obsidian-host acceptance were not run: P4 uses the standalone CLI surface, core contracts and isolated generation fixtures.
