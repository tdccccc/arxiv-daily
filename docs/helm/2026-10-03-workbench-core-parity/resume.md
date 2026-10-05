# Resume after context compression

## User-approved next work

P2/P3 and P20 are complete. The retained feature worktree now includes library browsing/review and automatic settings persistence; local DSH0.1.18 is packaged. The user's Enable issue was the daily-discovery switch itself: turning it off now persists before X/Escape/Done closes settings, and reopening stays off. Manual Save is removed. Local main remains the earlier P19 merge at678febe; these subsequent changes are not merged/pushed/installed. P4 richer run management stays pending; no active phase. Await user feedback rather than extending scope.

## Workspace

- /home/tiandc/Documents/code/arxiv-daily/.worktree/claude-code-research-plugin
- branch feat/claude-code-research-plugin
- Default Chinese. No Computer Use; screenshots supplied by user may be read. No real paid model calls/email or writes to user vault for testing.
- Main remains at the previous merge; separate Obsidian warning worktrees are untouched. All implementation stays in this worktree.
- Helm and code-change-discipline apply. Preserve staged/unrelated edits, observe Red/Green, inspect staged diff before scoped Conventional Commits.

## Product decisions

Literature workflows are primary; agent conversation auxiliary. DSH is the first integrated desktop/web host; future Claude Mod is not current implementation scope. One calendar/filter sidebar, right side paper list or reader. Markdown is durable record.

Settings follow Obsidian business semantics with user-approved extra Appearance group, left jump navigation, theme light/dark/system and UI language zh/en. UI language is independent of summary language. Pure arxiv-daily wordmark; all icon proposals rejected and archived, do not revive.

## Previous settings release and installed version

DSH0.1.12 installed into Web profile and bundle hash verified. User must restart dsh web/refresh to load new running code. Do not recommend --profile desktop. Electron uses its native plugin manager. Obsidian source adapted and built, NOT deployed to user's actual plugin.

Shared business settings now live in core settings/schema.ts (metadata/options/conditions/topic fields/time choices), settings/editing.ts (candidate normalization), services/settings-operations.ts (test/verification email). Model list already shares LlmClient. Obsidian definitions/change-service and CLI workbench/init consume these. Hosts retain storage, secret placeholder semantics, revision locks, authorization, and runtime transactions. No automatic value synchronization. Obsidian legacy complex topic editors retain their pre-existing live-draft rollback limit.

Last verified: CLI299 tests, Obsidian771, focusedcore34, DSH20; all typechecks, boundaries, inventory and builds pass. Implementation commits fef7915,16c4007,234d7e9; acceptance3099329. P15 results and its new package are recorded below.

## P15 result

Core source reports pending/awaiting_announcement for newer buckets; explicit empty announcements are no_updates. Pipeline reuses arXiv weekend policy for manual generation, preserving existing report repair. Completed zero filtering is no_matches. State/history persist optional outcome and failureAttempts; waiting does not exhaust the real-failure budget. Old records retain fallback compatibility.

Workbench child CLI results use typed IPC (main-types.ts, main.ts, workbench/launch.ts, server.ts). Do not replace this with exit-code or log-string inference. Obsidian calendar/cache handling and shared notices consume outcomes. No-update days do not finish first-report onboarding. Durable errors and saved reports take precedence over old owned-run state.

Validation evidence: core279 focused, CLI312 full plus subsequent75/27 focused regressions, Obsidian774 full, DSH20 including isolated actual Host; typechecks/boundaries/inventory/builds passed. No Electron visual walkthrough. Implementation commits c00cff9,1c1f33b. Package: extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.13.tgz (linux/x64); user Web profile remains0.1.12 until installed.

## Runtime pitfall

Never rpc.intercept('/api') in DSH: collides with Gateway. Working host uses ctx.connection.fetch.register for exact /api/arxiv-daily/open with existing RPC envelope. Preserve Host/Origin checks and exact frame ancestors (loopback or dsh-app://app). Do not replace with rpc.handle: installed host had injection failure.

## Verification

Use temporary XDG/DSH_HOME/vault. DSH fixtures and actual installed Host acceptance are in extensions/dsh-arxiv-daily/tests. Build via node extensions/dsh-arxiv-daily/build.mjs; package via npm pack dist/package. CLI vitest config apps/cli/vitest.config.mts, plugin equivalent; core script may need8GiB/single fork for broad runs. Do not conflate DOM/Host tests with an Electron visual walkthrough.

## P16 current integration

Main's schema6 proposals contain topics[].directions. Accepted directions are normal settings topics; standalone profiles and personalized daily snapshots were removed by main. CLI library confirm/review use atomic topicSettings read/change and root-TOML library_proposal_acceptances. Retired profile commands explicitly reject with migration guidance. Main's title/abstract index uses bounded page text and skips document sidecars. Do not reintroduce removed profile imports.

Workbench supports direction rows + daily paper limit, preserves direction id/origin and old-client omissions. CLI config normalization now gives legacy directions stable IDs across loads. Main download progress, setup completion marker, native settings actions, parser provenance and P15 waiting/failure budgets all retained. Checks: CLI339, plugin1007, focused core suites, DSH20, Claude7, native workflow5; builds/typechecks/boundaries/inventory pass. No whole-core/cross-platform/Electron-visual claim. Latest artifact0.1.14 (linux/x64); no production install or push. See phase16 for precise evidence.

## P17 current display fix

Workbench markdown.ts now exports safe inline rendering plus blank-line-aware display blocks. Browser overview, abstract, reason, title and TOC use the same reader as server Markdown. Heading math delimiters are preserved and duplicate first headings matched by metadata. Source files unchanged; code samples stay literal. DSH0.1.15 built, local fonts/CSS verified through actual Host; install/restart needed.63 focused+38 final regressions, typecheck/boundaries/build and20 DSH tests pass. No Electron visual walkthrough. See phase17.

## P18 current reading context

Core metrics now include durable generatedAt and versioned JSON inside the existing callout. splitGenerationMetrics returns {body,metrics}, preserving later notes and code examples. Workbench API/rendering exposes generationMetrics and scoped paper.generation; overview appendix starts at Summary sources and generation-footer is last. Old missing data stays unrecorded. Header adds session-bounded history back/forward with scroll/filter/hash restoration, no history.length inference. Tested core332 focused, CLI358 full, plugin1007 full, DSH20 actual Host; all typechecks/boundaries/builds pass. Latest local package0.1.16; no production install. See phase18.

## P19 merge state

Local main contains the first workbench/DSH iteration and7 updated user docs. No remote push/tag/publication/production installation occurred. Full acceptance: root3881 pass/2 explicit real-corpus skips; release-tools368; DSH20; Claude7; all types/build/submission/smokes/published-manifest pass; audit0, lint0 errors/22 permitted warnings. Fixed release checker nested dependency handling + DSH CI runtime input coverage. Subsequent GUI work P2/P3/P4 remains pending. Keep this distinction from a fully published release or complete library GUI parity.

## P2/P3 result after P19

LibraryWorkflow now exposes guarded renameTopic/moveDirection/removeCandidate/previewDirection, plus indexedPapers on review snapshots. WorkbenchLibrary reuses CLI context and core for catalog/search/PDF/review/actions; independent analysis authorization is at api/library/authorize. Long propose/preview jobs use the existing run tray, previews return through api/library/preview?runId, never parsed terminal logs. Review is at ?view=review and catalog at ?view=library; both have left-side entries.

Review supports proposed/overview tabs, representative evidence, drafts, renaming/moving/deleting candidates, weak-evidence selection, explicit partial acceptance and history-aware preview restoration. Proposal receipts keep processed candidates from being edited/re-added. Config/proposal revision checks, explicit endpoint authorization, scoped PDF reads and cancellation stay enforced. Atomic read views do not wait behind long model jobs; search and mutations keep their lease. PDF cap25MiB.

Full3950 pass/2 real-corpus opt-in skips (CLI422/Obsidian1007), all types/boundaries/inventory/root build; lint0 errors/22 pre-existing warnings; DSH20 actual isolated Host and Claude7 passed. Final0.1.17 package sha2567daeeecadb2802534f746bf187b17f531b9d884cea5fc3231657c7ce4838f294. No real-model/corpus/email or browser automation, no Electron visual check, no publication/install/main merge. See phase03 for evidence. P4 is the only pending product phase.

## P20 current settings behavior

bindSettings returns {flush,close,dispose}. Input debounce500ms, change saves immediately, root input waits for completion, and writes serialize using returned revisions. Explicit model/library/email operations flush first. X/Escape/Done wait for pending edits; failure preserves fields with retry and explicit discard confirmation. Displayed secrets are never written back as edits. Preferences save independently and redraw settings in place, retaining newer selections and business drafts; the shell updates after close. Opening settings never writes/enables scheduling. Completed manual discovery also retires the Enable guide requirement.

Final CLI443/45 files and DSH20 passed; types/boundaries/product inventory/build passed. Local0.1.18 sha25646d7d07de8df1d72a115d260bb7cfc82bacf4a4db0da436fa41e23b1b55f53cb. No production install or main merge. Relevant logs /tmp/arxiv-p20-{cli,dsh,types,build,http-green,guide-green}.log.
