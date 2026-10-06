# P3 — literature-entries

goal_ref: ../goal.md
created: 2026-10-02T21:33:39+08:00
updated: 2026-10-02T22:36:21+08:00
revision: 2

## Outcome

Literature is available above Settings before any chat and in the right Sidebar guide, following dsh-context's host extension pattern.

## Assumptions

- dsh-context uses root sidebar.footer.action + shell.overlay, and sidebarRightTabs + keyed sidebar.right.pane.tab/title seats.
- Existing workbench HTML stays isolated inside a frame; both placements share the same product service and record.
- DSH Desktop's fixed dsh-app://app origin needs explicit frame permission; arbitrary custom origins remain forbidden.

## Chunks

### Chunk 1 — Narrow Desktop frame support

- kind: behavior change; strategy: strict Red-Green CLI/HTTP and DSH host tests
- Red: fixed dsh-app origin is rejected by current CLI/host validation
- Green: only dsh-app://app and existing exact local HTTP(S) origins are accepted; other schemes/hosts/paths remain rejected, API-origin checks stay intact
- regression: workbench/CLI tests and authenticated DSH integration
- [x] accepted

### Chunk 2 — Fixed navigation and shared frame view

- kind: behavior change; strategy: strict Red-Green registry and React DOM tests
- Red: footer/overlay/right-tab seats missing and blank-session entry unavailable
- Green: global button opens without session/model turn; guide registers named tab; shared view loads capability, reports failure/retries and cleans up; old composer entry removed
- regression: real DSH registry and installed Host tests, package checks, product inventory/boundaries
- exception: no Computer Use per user; verify native slot contracts and component behavior, not a claim of Electron visual acceptance
- [x] accepted

## Abort triggers

- Never require a prompt or create an artificial conversation to open the global workbench.
- Do not open frame ancestors to wildcards or discard capability and Origin checks.


## Acceptance

Implemented in `5cb8d9e`, packaged as 0.1.2. The global entry uses sidebar.footer.action and shell.overlay; the right guide uses sidebarRightTabs and keyed pane body/title seats. Both render the same isolated WorkbenchFrame. The composer entry and Browser dependency/enable patch were removed. Opening the global entry never reads a session or submits a prompt.

Observed Red: fixed Desktop origin rejected at CLI/HTTP and Host boundaries; footer and right-tab registrations absent; session-independent loading absent. An unrelated test fixture shadowed the URL constructor and was corrected before acceptance. Green: 19 DSH tests pass, including real installed SlotCore registration/disposal, React StrictMode root opening before a conversation, retry and close, collapsed-rail accessibility, dedicated tab loading, real Host authentication/Gateway coexistence, generation/marks and disable/re-enable. All 28 focused CLI/workbench regressions, CLI typecheck, boundaries and inventory pass.

The installed DSH was observed as 0.2.0-rc.2 during this work; this round's real Host and SlotCore evidence applies to that version. Historical 0.1.7-alpha.1 results remain history, not a claim that it was rerun. The CI pin remains the earlier baseline and was not executed remotely.

Desktop embedding explicitly permits only dsh-app://app in addition to existing exact loopback web origins. Host validation rejects other custom origins, and API writes still reject the embedding parent. No live user profile or data was changed. Electron layout/webview rendering was not visually exercised per the no-Computer-Use constraint; component and protocol tests are not screenshots. The generated local archive is linux/x64.

Reference implementation read: https://github.com/bowenliang123/dsh-context/blob/main/src/client/index.ts and https://github.com/bowenliang123/dsh-context/blob/main/src/client/sidebar.ts (footer/overlay and guide/tab extension patterns).
