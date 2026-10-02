# P3 — literature-entries

goal_ref: ../goal.md
created: 2026-10-02T21:33:39+08:00
updated: 2026-10-02T21:33:39+08:00
revision: 1

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
- [ ] accepted

### Chunk 2 — Fixed navigation and shared frame view

- kind: behavior change; strategy: strict Red-Green registry and React DOM tests
- Red: footer/overlay/right-tab seats missing and blank-session entry unavailable
- Green: global button opens without session/model turn; guide registers named tab; shared view loads capability, reports failure/retries and cleans up; old composer entry removed
- regression: real DSH registry and installed Host tests, package checks, product inventory/boundaries
- exception: no Computer Use per user; verify native slot contracts and component behavior, not a claim of Electron visual acceptance
- [ ] accepted

## Abort triggers

- Never require a prompt or create an artificial conversation to open the global workbench.
- Do not open frame ancestors to wildcards or discard capability and Origin checks.
