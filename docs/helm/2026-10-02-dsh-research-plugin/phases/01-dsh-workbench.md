# P1 — dsh-workbench

goal_ref: ../goal.md
created: 2026-10-02T14:56:36+08:00
updated: 2026-10-02T15:40:01+08:00
revision: 2

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
- [x] implementation and tests accepted

### Chunk 2 — DSH bundle and managed lifecycle

- change kind: behavior change
- strategy: strict Red-Green process and Host/Client contract tests
- Red: plugin files/start controller/entry absent
- Green: authentic RPC registry and slots, single concurrent startup, bounded startup/error, retry after exit, unload shutdown, no shell execution, validated returned URL, UI error and disposal behavior
- regressions: copied CLI generation/persistence; npm pack file inventory; real installed DSH profile composition and authenticated RPC with temporary DSH_HOME/XDG directories
- exception: user forbids Computer Use, so no automated Electron desktop clicking; compensate with real Host RPC, actual installed client API/slot contract and DOM event tests. Desktop visual placement remains manual verification, not claimed as observed.
- [x] implementation and tests accepted

## Abort / reshape triggers

- If the installed Host lacks authenticated RPC or Browser slots, keep the integration visibly unavailable and revise the adapter before proceeding.
- If embedding needs a wildcard origin or loss of capability checks, stop that approach.
- If startup requires new business logic or a second store, reuse the existing CLI boundary instead.


## Accepted evidence

- `16630e3`: explicit `ui --frame-origin` and canonical loopback validation. Two expected CLI/HTTP failures were observed after separating an initial sandbox listen EPERM; 26 focused checks and typecheck passed after implementation. Default frame denial and same-origin writes remain protected.
- `0606fc5`: DSH bundle, authenticated startup RPC, composer-dock entry, process lifecycle, package, docs and CI inventory. Four process and five registry/opener contracts failed against stubs before implementation. Later tests reproduced StrictMode effect cleanup, malformed URL ports and absent artifact platform metadata; all three were fixed before acceptance.
- Final plugin suite: 13 tests passed, none skipped. Real DSH 0.1.7-alpha.1 installed the packed bundle into a temporary DSH_HOME; HTML boot metadata included the client module; anonymous RPC returned 401 and foreign Origin returned 403. Concurrent authenticated opens returned one workbench capability. Existing Markdown/math, synthetic daily generation, two discoveries including an index-only paper, independent reading/star marks, layout preferences and listener shutdown all passed.
- React DOM tests verify the visible button, busy state, setup error and retry under StrictMode; distributed-client tests verify the DSH loader format and required services. The local DSH package bundles React internally instead of exposing a Node `react` package; DOM tests therefore use isolated dev dependencies, while real Host tests consume the installed DSH. An initial test-cleanup ordering error was corrected without changing production behavior.
- Regression: 209 CLI tests passed in 23 files; 7 existing Claude-integration portable checks passed; CLI typecheck and boundaries passed; 17 product inventory/workflow checks passed. The new manifest initially failed the closed product inventory, so an independent extension entry and scoped verification workflow were added. CI was validated locally, not executed remotely.
- Security scan had no findings on the new plugin. It also scanned unrelated global Claude configuration/skills: the reported critical hit in the existing brainstorming server combines WebSocket SHA-1/base64 with `process.env` references; inspection found no asserted .env exfiltration in those lines. No global configuration or existing plugins were changed. New package has no install scripts or runtime npm dependencies; startup accepts no browser-supplied executable/path.
- Desktop installation correction: installed DSH reserves its `desktop` profile for Electron. Use Desktop Plugins -> Add plugin with the absolute local tarball path, then Enable now. The CLI install command applies to Web profiles only; actual installed plugin-manager docs accept tarballs and absolute local paths.
- Artifact: `extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.0.tgz`, generated for linux/x64 and explicitly qualified in its manifest. Other platforms need their own build or the existing full native release matrix. The source package version is independent of the core/CLI version.
- Per the planned no-Computer-Use exception, Electron visual placement/webview navigation and macOS/Windows runtime behavior were not exercised. Do not describe the Host/DOM checks as desktop screenshots or a complete cross-platform release. First-run configuration still uses the existing CLI init; no GUI setup wizard is claimed.
