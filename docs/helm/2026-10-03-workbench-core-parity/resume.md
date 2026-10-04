# Resume after context compression

## User-approved next work

Start P15 in phases/15-structured-run-status.md: strengthen structured run outcomes in core first, then adapt Obsidian, CLI and DSH. Afterward prioritize paper-list decision support, personal-library UX and direction review. Do not restart completed settings work.

## Workspace

- /home/tiandc/Documents/code/arxiv-daily/.worktree/claude-code-research-plugin
- branch feat/claude-code-research-plugin
- Default Chinese. No Computer Use; screenshots supplied by user may be read. No real paid model calls/email or writes to user vault for testing.
- Main checkout contains unrelated work. All implementation stays in this worktree.
- Helm and code-change-discipline apply. Preserve staged/unrelated edits, observe Red/Green, inspect staged diff before scoped Conventional Commits.

## Product decisions

Literature workflows are primary; agent conversation auxiliary. DSH is the first integrated desktop/web host; future Claude Mod is not current implementation scope. One calendar/filter sidebar, right side paper list or reader. Markdown is durable record.

Settings follow Obsidian business semantics with user-approved extra Appearance group, left jump navigation, theme light/dark/system and UI language zh/en. UI language is independent of summary language. Pure arxiv-daily wordmark; all icon proposals rejected and archived, do not revive.

## Latest completed work

DSH0.1.12 installed into Web profile and bundle hash verified. User must restart dsh web/refresh to load new running code. Do not recommend --profile desktop. Electron uses its native plugin manager. Obsidian source adapted and built, NOT deployed to user's actual plugin.

Shared business settings now live in core settings/schema.ts (metadata/options/conditions/topic fields/time choices), settings/editing.ts (candidate normalization), services/settings-operations.ts (test/verification email). Model list already shares LlmClient. Obsidian definitions/change-service and CLI workbench/init consume these. Hosts retain storage, secret placeholder semantics, revision locks, authorization, and runtime transactions. No automatic value synchronization. Obsidian legacy complex topic editors retain their pre-existing live-draft rollback limit.

Last verified: CLI299 tests, Obsidian771, focusedcore34, DSH20; all typechecks, boundaries, inventory and builds pass. Implementation commits fef7915,16c4007,234d7e9; acceptance3099329. Goal/P15 docs are the only new changes during compaction preparation.

## P15 investigation anchors

- packages/core/src/sources/types.ts: SourceListForDateResult currently only ok/error.
- packages/core/src/sources/arxiv-source-adapter.ts: missingRecentDateReason produces newer-than-newest diagnostic; missing buckets become failed_transient, older-than-oldest permanent; partial discovery rejected.
- packages/core/src/pipeline/pipeline.ts: PipelineResult + fetchPapersForDate bridge source outcomes.
- packages/core/src/services/scheduling/scheduler-driver.ts: automatic tick checks weekends; runForDateNowAt is manual and does not use that guard.
- packages/core/src/services/state-store.ts and run-history.ts: inspect existing status/history persistence before extending.
- apps/cli/src/main.ts: writeRunResult and manual vs scheduled entry.
- apps/cli/src/workbench/announcement-calendar.ts: interim weekend guard and exact legacy error-string matcher. Replace new-result guessing with core outcomes; keep safe old-record compatibility if needed.
- apps/cli/src/workbench/calendar.ts/server.ts/web/app.ts: day state, ephemeral job state and UI labels.
- plugin dashboard/status adapters: discover actual call chain rather than guessing.

User example: 2026-10-03 all categories newer than newest astro-ph.CO/GA recent bucket2026-10-02. Expected no update/waiting, not generation failed. Interim DSH guard uses core isWeekendDate and a narrow read-only legacy matcher; it is not the final architecture.

## Runtime pitfall

Never rpc.intercept('/api') in DSH: collides with Gateway. Working host uses ctx.connection.fetch.register for exact /api/arxiv-daily/open with existing RPC envelope. Preserve Host/Origin checks and exact frame ancestors (loopback or dsh-app://app). Do not replace with rpc.handle: installed host had injection failure.

## Verification

Use temporary XDG/DSH_HOME/vault. DSH fixtures and actual installed Host acceptance are in extensions/dsh-arxiv-daily/tests. Build via node extensions/dsh-arxiv-daily/build.mjs; package via npm pack dist/package. CLI vitest config apps/cli/vitest.config.mts, plugin equivalent; core script may need8GiB/single fork for broad runs. Do not conflate DOM/Host tests with an Electron visual walkthrough.
