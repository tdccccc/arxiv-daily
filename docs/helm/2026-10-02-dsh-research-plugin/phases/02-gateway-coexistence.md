# P2 — gateway-coexistence

goal_ref: ../goal.md
created: 2026-10-02T19:45:14+08:00
updated: 2026-10-02T19:45:14+08:00
revision: 1

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
- [ ] implementation and tests accepted

## Abort / reshape triggers

- If a dedicated transport bypasses Host authentication or Desktop forwarding, retain the authenticated /api exact Fetch-route extension instead.
- Do not change core paper data or the user's live profile to hide an activation failure.
