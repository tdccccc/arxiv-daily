## 2026-08-18 — note

- evidence: P4 runtime/distribution spike found that the repository supports Node `>=20.11.0`, the plugin supports older Obsidian hosts, Core forbids Node/database dependencies, and releases contain no per-platform native assets. Current search loads every paper JSON/base64 vector and creates a corpus-sized centered copy, while host storage lacks binary range read and atomic binary replacement.
- change: resolved the P4 backend choice in favor of a pure TypeScript immutable generation index with prebuilt BM25 postings and fixed-block exact dense scanning; SQLite, LanceDB, native vector extensions and ANN are excluded from P4. Activated `phases/04-transactional-derived-index.md` and removed the resolved backend question from the goal.
- disposition: keep all accepted P1–P3 parser, EvidenceChunk and ranking contracts; replace only the query data layout and access path. Keep the legacy per-paper JSON knowledge base as a migration/rebuild source and do not delete it in place.
- next: establish binary range/atomic storage capabilities with Red/Green contract tests before defining the generation codec.

## 2026-08-18 — L2 reshape

- evidence: range read and binary no-replace prototypes passed ordinary behavior tests but independent review repeatedly found an unavoidable path-validation-to-open/install race. Pure Node path APIs cannot provide the same descriptor-anchored root binding across the currently supported Linux, macOS, Windows and older Obsidian hosts; exposing the capability would widen the storage permission surface and still overstate its guarantee.
- change: superseded the range/exclusive primitive P4 plan with P4b. The new path stores vectors, metadata and postings as complete binary blocks with a strict per-object byte cap, writes them under a unique uncommitted generation namespace, validates their closure, and promotes only a small text pointer through existing atomic/recovery support.
- disposition: discarded the entire unaccepted adapter prototype and its tests; no production or test changes from that path remain. Keep the stable goal-level contracts: bounded query memory, transactional generation recovery, no query-time legacy paper loading, and P3 ranking equivalence.
- next: define the bounded block codec and generation descriptor with failing size/schema/checksum tests, without adding filesystem capabilities.
