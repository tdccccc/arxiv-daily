# Resume after context compression

## User-approved next work

P15 implementation and verification are complete; see phases/15-structured-run-status.md. DSH0.1.13 is packaged locally, not installed into the user profile. Future work remains paper-list decision support, personal-library UX and direction review. Do not restart completed settings/status work.

## Workspace

- /home/tiandc/Documents/code/arxiv-daily/.worktree/claude-code-research-plugin
- branch feat/claude-code-research-plugin
- Default Chinese. No Computer Use; screenshots supplied by user may be read. No real paid model calls/email or writes to user vault for testing.
- Main checkout contains unrelated work. All implementation stays in this worktree.
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
