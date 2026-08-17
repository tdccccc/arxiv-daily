## 2026-08-18 — note

- evidence: P4 runtime/distribution spike found that the repository supports Node `>=20.11.0`, the plugin supports older Obsidian hosts, Core forbids Node/database dependencies, and releases contain no per-platform native assets. Current search loads every paper JSON/base64 vector and creates a corpus-sized centered copy, while host storage lacks binary range read and atomic binary replacement.
- change: resolved the P4 backend choice in favor of a pure TypeScript immutable generation index with prebuilt BM25 postings and fixed-block exact dense scanning; SQLite, LanceDB, native vector extensions and ANN are excluded from P4. Activated `phases/04-transactional-derived-index.md` and removed the resolved backend question from the goal.
- disposition: keep all accepted P1–P3 parser, EvidenceChunk and ranking contracts; replace only the query data layout and access path. Keep the legacy per-paper JSON knowledge base as a migration/rebuild source and do not delete it in place.
- next: establish binary range/atomic storage capabilities with Red/Green contract tests before defining the generation codec.
