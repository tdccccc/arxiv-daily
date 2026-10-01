# P1 — reading-service

goal_ref: ../goal.md
created: 2026-10-01T22:53:43+08:00
updated: 2026-10-01T23:03:00+08:00
revision: 2

## Outcome

Existing Markdown reports are available through a scoped local HTTP service with faithful, non-executable HTML rendering and existing product operation dispatch.

## Assumptions

- Configured daily/paper directories are the initial document scope; arbitrary Vault browsing is unnecessary.
- Browser Markdown needs ordinary formatting and scientific math, not full Obsidian plugin semantics.
- The UI service can remain a foreground process, started in the background by Claude; closing the service ends browser access but does not affect saved results.

## Approach

Implement in `apps/cli/src/workbench/`, preserving workspace boundaries. Use markdown-it with HTML disabled and KaTeX with untrusted rendering. The server binds to loopback and uses a random per-launch URL capability, origin/host checks, and realpath containment. Reuse existing inspection and runCli operations. Runtime assets will be embedded in the CLI in P2.

## Chunks

### Chunk 1 — Markdown rendering contract

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `npm exec --workspace apps/cli -- vitest run --config vitest.config.mts tests/workbench-markdown.test.ts`; missing rendering contract, then formatting/link/math assertions fail against minimal surface.
- Green check: same command passes for generated report formats, frontmatter, scientific math, stable headings, links, code and untrusted source content.
- regression checks: CLI typecheck and focused inspection tests.
- [x] implementation and tests accepted — renderer behavior Red observed; 8 renderer tests and CLI typecheck pass.

### Chunk 2 — Scoped read service and product actions

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: HTTP integration tests in `tests/workbench-server.test.ts` fail for missing document API, protected local-file reads and action dispatch.
- Green check: real HTTP requests against temporary output folders pass; read-only byte snapshots remain unchanged; generation callback is explicit and serialized.
- regression checks: CLI workspace tests, boundary and product-unit checks.
- [x] implementation and tests accepted — HTTP 404 baseline failed 7 tests as expected; all 7 HTTP tests pass. Full CLI: 156 tests; boundaries and product inventory pass.

## Phase verification

- A real loopback server lists and reads fixture Markdown and assets, resolves document links, rejects unknown token / origin / path traversal / symlink escapes, and never returns credentials from status.
- No generation runtime is built until an explicit action requests it; errors and progress are visible and bounded.

## Abort / reshape triggers

- If original document paths cannot be resolved without changing core schemas, revisit the adapter rather than adding parallel storage.
- If the CLI build cannot contain runtime assets, explicitly revise packaging before claiming portable launch.
