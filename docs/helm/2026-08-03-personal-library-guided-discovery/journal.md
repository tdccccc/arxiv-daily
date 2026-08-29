# Journal — personal-library-guided daily discovery

## 2026-08-04 — waiver + handoff: dogfood comparison superseded by knowledge-base initiative

- evidence: P6 shipped identification v2 (25.6% → 55.8% on test_library), global-title-grouping direction generation (topics now span distinct themes), store policy-migration fix, scan range-read optimization, and the cue-ordering/consent-status fixes; all deployed to plugin_test and verified green (core 1336 / plugin 382). Three research agents (tech stack, incremental direction architecture, code evolution path) returned a unified knowledge-base design; user confirmed multilingual-e5-small embeddings and the staged plan. Direction generation is being replaced by the clustering-based knowledge-base initiative (2026-08-05-local-paper-knowledge-base P2/P3), so the P6 dogfood-comparison criterion (at least one researcher-accepted library-derived paper missed by manual topics) is waived: the comparison's purpose — proving direction quality — is now served by the knowledge-base direction pipeline.
- change: P6 goal phase 6 → done, goal status → done. New initiative goal.md created (2026-08-05-local-paper-knowledge-base): P1 full-text index + similarity search, P2 clustering-based directions, P3 incremental direction updates.
- disposition: keep all P6 shipped code (deployed); knowledge-base reuses P6 mechanisms (identification v2, profile CAS, host boundaries).
- next: plan and execute knowledge-base P1 (full-text extraction + local embeddings + similarity search).

## 2026-08-04 — note: P6 start + email decision

- evidence: goal.md open question "email digest projection" was left open by P4/P5; digest/email rendering (`packages/core/src/delivery/digest.ts`, `email-render.ts`) is a separate projection from daily Markdown, and occurrence provenance + personal novelty are already persisted in the Paper Index and daily-report markers. User confirmed dogfood library = `~/Nextcloud/work/Article` (526 files, 25 arXiv-identifiable, 352 author-year) and dogfood execution = user operates the real Obsidian vault while I prepare the run kit and analyze artifacts. User asked whether keeping email unchanged makes a later change costly; answered no — isolated digest-model + render change.
- change: goal.md phase index P6 → active; open question resolved (digest/email unchanged in this initiative, rationale recorded); created `phases/06-operational-hardening-and-dogfood.md`.
- disposition: keep phase plan; email decision recorded, no code change.
- next: T2 — headless real-library validation of catalog/reconciliation/reload against `~/Nextcloud/work/Article`.
