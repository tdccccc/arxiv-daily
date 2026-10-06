# P2 — parity-inventory

goal_ref: ../goal.md
created: 2026-10-02T12:17:52+08:00
updated: 2026-10-02T12:17:52+08:00
revision: 1

## Outcome

The original Dashboard's visible controls, behavior and host dependencies are listed as the concrete porting baseline.

## Assumptions

- The user means their plugin Dashboard and primary operations, not all Obsidian infrastructure.
- Browser reading/configuration/scheduling use equivalent standalone host mechanisms where Obsidian services do not exist.

## Approach

Read the original view, constants, styles, row/batch actions, More menu, similarity and log/history components. Map each item to existing core/CLI functionality before implementation. No new product layout is designed.

## Chunks

### Chunk 1 — Source-grounded parity inventory

- change kind: non-behavioral
- strategy: source inspection and cross-reference
- baseline: original `plugin/src/dashboard/view.ts`, constants/styles, hub and similar-paper components
- verification: parity.md enumerates layout, defaults, controls, data services and explicit host adaptations with file anchors
- [ ] inventory accepted

## Abort / reshape triggers

- If an apparent main Dashboard feature depends on a separate preferences page or Obsidian infrastructure, identify that boundary explicitly instead of claiming a copied button is complete.
