# P2 — reading-ui

goal_ref: ../goal.md
created: 2026-10-01T23:03:00+08:00
updated: 2026-10-01T23:03:00+08:00
revision: 1

## Outcome

One documented CLI or Claude plugin command opens the packaged local reading workbench, verified in a real browser.

## Assumptions

- Users already run the CLI init wizard; browser settings initially inspect this configuration and explain how to change it.
- The local service stays running in the terminal (or Claude background task); generated data remains available independently.
- Original PDF links can open a separate tab; full PDF annotation and extraction are not part of Markdown reading acceptance.

## Approach

Embed a small TypeScript/CSS browser client, KaTeX styles/fonts, and required dependency notices in the exact existing CLI build. Add `ui [--port N] [--no-open]`, a Claude `/arxiv-daily:open` skill, and generation subprocess dispatch to the same executable. Reuse P1 API contracts. Pin server configuration for its lifetime and reject actions after configuration changes.

Visual thesis: a quiet paper-like reading surface with warm neutrals, strong typography, one blue action accent, and a compact navigable report list.
Content plan: primary workspace (list + article), secondary table of contents, current settings dialog, visible generation task status; no marketing hero or decorative cards.
Interaction thesis: restrained document transition, clear list hover/selection and focus feedback, small dialogs; reduced-motion support.

## Chunks

### Chunk 1 — Browser reading surface

- change kind: behavior change
- strategy: strict Red-Green-Refactor for stateful UI behavior; visual layout uses manual browser verification.
- Red / baseline signal: DOM tests fail to show documents, navigate links, surface API errors or dispatch explicit jobs against an empty UI.
- Green check: DOM tests using service-shaped fixtures pass for list/search/read/link navigation/settings/run states and stale-response handling.
- regression checks: renderer/server tests, CLI typecheck; actual browser checks for typography, math fonts, narrow layout and keyboard navigation.
- [ ] implementation and tests accepted

### Chunk 2 — Portable launch and existing pipeline integration

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: CLI `ui` tests return unknown command; package test fails to launch a standalone copied CLI / load embedded assets.
- Green check: CLI dispatch and copied-bundle process tests pass; explicit UI actions use the existing run command with correct environment/cancellation.
- regression checks: CLI tests, existing plugin independent-process tests, workspace typechecks/build/boundaries/product inventory and strict Claude plugin validation.
- [ ] implementation and tests accepted

## Phase verification

- Real browser on isolated fixture data: daily list, paper link, formula/table/code/local image rendering, ToC, search, settings, raw Markdown and PDF/source links, missing/empty states, mobile layout.
- A fixture-backed generation request through the packaged server writes an actual daily report or note through the original CLI; reading leaves existing source bytes unchanged.
- User-facing README clearly distinguishes supported reading and CLI configuration from future Obsidian-level editing.

## Abort / reshape triggers

- If launch requires repository-only paths or remotely hosted frontend assets, correct packaging before acceptance.
- If an action needs a duplicate pipeline, preserve the CLI dispatch boundary rather than reimplementing it in the browser.
