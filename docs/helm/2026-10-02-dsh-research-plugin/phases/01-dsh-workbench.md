# P1 — dsh-workbench

goal_ref: ../goal.md
created: 2026-10-02T14:56:36+08:00
updated: 2026-10-02T14:56:36+08:00
revision: 1

## Outcome

An installed DSH bundle opens the existing literature workbench from a native button and manages its process lifetime.

## Assumptions

- DSH ships the right-sidebar Browser type. Desktop uses webview; Web uses iframe.
- Existing CLI init/config remain the first-run setup; missing configuration must have actionable feedback.
- The local machine already provides DSH 0.1.7-alpha.1; its Connection RPC and slot registrations can be verified without launching a user's browser.

## Approach

Use a native composer-dock button, authenticated Connection RPC, and the shipped Browser tab. A small process controller spawns the exact CLI artifact lazily and shares it across sessions. For Web iframe use, an explicit CLI option permits only the current loopback DSH origin; host/origin/capability checks on API writes remain intact. Plugin RPC never accepts an executable, arbitrary URL or file path from the browser. Desktop and Web use the same product data/config.

## Chunks

### Chunk 1 — Scoped workbench embedding

- change kind: behavior change
- strategy: strict Red-Green at CLI and HTTP boundaries
- Red: explicit embedding option rejected; permitted ancestor absent
- Green: one validated loopback origin is allowed only when requested; wildcard, external, credentials/path/query origins rejected before config loading; API protections unchanged
- regressions: workbench HTTP/CLI tests, typecheck
- [ ] implementation and tests accepted

### Chunk 2 — DSH bundle and managed lifecycle

- change kind: behavior change
- strategy: strict Red-Green process and Host/Client contract tests
- Red: plugin files/start controller/entry absent
- Green: authentic RPC registry and slots, single concurrent startup, bounded startup/error, retry after exit, unload shutdown, no shell execution, validated returned URL, UI error and disposal behavior
- regressions: copied CLI generation/persistence; npm pack file inventory; real installed DSH profile composition and authenticated RPC with temporary DSH_HOME/XDG directories
- exception: user forbids Computer Use, so no automated Electron desktop clicking; compensate with real Host RPC, actual installed client API/slot contract and DOM event tests. Desktop visual placement remains manual verification, not claimed as observed.
- [ ] implementation and tests accepted

## Abort / reshape triggers

- If the installed Host lacks authenticated RPC or Browser slots, keep the integration visibly unavailable and revise the adapter before proceeding.
- If embedding needs a wildcard origin or loss of capability checks, stop that approach.
- If startup requires new business logic or a second store, reuse the existing CLI boundary instead.
