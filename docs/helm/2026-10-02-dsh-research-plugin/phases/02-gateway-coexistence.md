# P2 — gateway-coexistence

goal_ref: ../goal.md
created: 2026-10-02T19:45:14+08:00
updated: 2026-10-02T20:03:50+08:00
revision: 2

## Outcome

The plugin activates alongside the existing DSH gateway and leaves normal host requests working.

## Assumptions

- The user's screenshot reports the shared /api interceptor already occupied.
- DSH 0.1.7-alpha.1 has one shared interceptor, owned by its Gateway; extension transports must not claim it.
- The current client, workbench process and research data remain useful.

## Approach

Replace the shared interceptor with a plugin-owned transport supported by Connection. Reproduce against a real registry with the host gateway seat already occupied, retain authentication/Origin checks, verify ordinary Gateway replies alongside plugin startup, then package 0.1.1.

## Chunks

### Chunk 1 — Gateway coexistence fix

- change kind: bug fix
- strategy: strict Red-Green; regressions target occupied shared channel, normal host replies, malformed requests and lifecycle cleanup
- Red: activation collides when a real Connection shared interceptor is already registered; normal gateway health is absent in the old integration
- Green: host and plugin endpoints both respond, independent of registration order; unload only removes plugin contributions
- regressions: all DSH tests, packed Host integration, product inventory and boundaries
- [x] implementation and tests accepted

## Abort / reshape triggers

- If a dedicated transport bypasses Host authentication or Desktop forwarding, retain the authenticated /api exact Fetch-route extension instead.
- Do not change core paper data or the user's live profile to hide an activation failure.


## Accepted result

The plugin now registers `/api/arxiv-daily/open` through `connection.fetch.register`, not the Gateway's singleton interceptor. It accepts the existing client RPC envelope and returns the matching result after the same Host/Origin/authentication fence. No client routing or research data changes were necessary.

Observed Red: both gateway-first and plugin-first registrations reproduced the exact screenshot error against the installed real Connection class. An attempted separate RPC channel exposed `cannot get property webServer without inject` in the installed Host, so the planned exact Fetch-route fallback was used. The real Host regression now checks the Gateway's `pluginManager/listPlugins` response and active phases, rather than accepting our endpoint alone.

Observed Green: all 16 DSH tests pass, none skipped. Actual DSH 0.1.7-alpha.1 installs the packed plugin, keeps both Gateway and plugin active, rejects anonymous/foreign-Origin requests, serves synthetic daily generation and durable marks, and successfully disables/re-enables the plugin through its own Plugin Manager. Re-enabling opens a new capability and preserves marks; shutdown closes the current listener. Product inventory and workspace boundary checks pass. Core/CLI source was unchanged, so the unrelated full CLI suite was not repeated. No Computer Use or live-profile mutation was performed.

Version 0.1.1 is built and packed for linux/x64. The installed 0.1.0 must be replaced and DSH restarted to load the corrected module generation; upgrade steps are in the plugin README. P1's coexistence acceptance remains superseded, while P2 re-accepts the retained feature under the strengthened contract.
