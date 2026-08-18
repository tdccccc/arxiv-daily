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

## 2026-08-18 — L1 adjust P4b.4

- evidence: the first unaccepted lexical candidate passed its fixtures by changing the P3 Han BM25 oracle, finalized papers before all 65,536-chunk windows were consumed, omitted body compact-alias matches, validated only postings that existed, and could reread selected evidence and metadata blocks many times. A bucket also had no paging path once its dictionary exceeded the 4 MiB object cap.
- change: keep schema-v3 generation foundations, fixed chunk windows, checksum/caps, metadata, routing, cancellation and late evidence. Rework lexical pages so query scoring preserves the accepted P3 tokenizer/length/float order, papers enter bounded top-k only after complete cross-window aggregation, compact aliases use routed gram candidates with exact compact-text verification, selected evidence blocks are read once, and closure compares evidence-derived canonical postings/alias streams in bounded windows.
- disposition: restore the accepted P3 scorer before using it as oracle; rewrite the unaccepted generation lexical reader, schema details and fixtures. Retain schema-v3 changes only where they satisfy the revised exact-equivalence and bounded-I/O contracts.
- next: observe Red for P3 oracle restoration, mixed Han, body alias, cross-window aggregation, paging/collision, closure omissions and selected/unselected corruption; then implement the minimum paged format and reader.

## 2026-08-18 — L1 adjust P4b.4 closure layout

- evidence: the paged term-list candidate could prove lexical completeness only by repeatedly scanning every dictionary page and rereading/re-tokenizing evidence for each posting. Independent review bounded its logical I/O at tens of TiB for legal layouts, found quadratic single-chunk work, and observed four simultaneously live 4 MiB objects. The correctness tests were Green, but the promotion path violated the phase's bounded-work outcome.
- change: replace the unaccepted lexical object layout with chunk-order authority postings. Each postings object carries an exact-permutation term catalog; each dictionary page carries posting-range route entries plus an exact-permutation query catalog and recomputed bucket mask. Promotion uses separate evidence↔postings and postings↔dictionary ordered zippers with exact exhaustion, keeping at most two fixed-size objects and linear $O(B+R)$ I/O.
- disposition: keep the accepted P3 oracle restoration, schema-v3 capability state, binary envelope/caps, compact alias derivation, fixed object limits and mutation contracts. Rewrite the unaccepted term-list codecs, repeated-scan closure, reader fixtures and associated tests around the linear stream representation.
- next: observe format Red for occurrence/catalog permutations and posting-range dictionaries, implement strict codecs, then implement the two linear closure passes before rewriting the query reader.

## 2026-08-18 — L1 split P4b.5 production cutover

- evidence: production generation construction requires descriptor checksums before store replay, so a single live paper iterator cannot feed `stageAndPromote`; a bounded host-storage spool and multiple linear passes are required. Online GC is also unsafe because opened handles read objects lazily and currently have no lifecycle claim—deleting an old generation after pointer promotion can break a pinned query. Existing descriptors lack per-block source digests, so revision-changing block reuse cannot be proven safely.
- change: split P4b.5 into builder, production migration/search cutover, and host-authorized quiescent maintenance. The builder pins a committed manifest snapshot, streams one paper and bounded blocks into a spool, derives dictionary objects by replay, then promotes from the completed descriptor. Exact same-revision generations are reused as a whole; changed revisions rebuild. GC and fixed-claim repair run only after the host stops admission and awaits active operations; claims are never stolen by age.
- disposition: keep schema v4, store transaction protocol and generation readers. Do not add generic filesystem permissions, guess block reuse, silently skip a manifest-ready corrupt paper, or claim cross-process online GC. Legacy KB remains the authoritative rebuild source and is not deleted after cutover.
- next: implement the committed-source builder and bounded spool contract with canonical order, source/document binding, block-cap, replay, stale-source and failure-isolation Red/Green evidence.

## 2026-08-18 — P4b.5 builder checkpoint

- evidence: the accepted builder snapshots the committed legacy manifest, loads and validates one ready paper at a time in canonical UTF-16 key order, emits deterministic capped generation objects through a bounded spool, and proves the emitted generation with the real transactional store and dense/BM25/RRF readers. Independent review found and the tests reproduced quadratic trial encoding, unclosed iterators on exact replay or pre-promotion rejection, untyped hostile spool returns, and cleanup retries lost after synchronous rejection.
- change: centralize real codec/lexical invocation instrumentation, budget rows incrementally and encode only on flush, make replay one-shot, and assign iterator-close ownership to `stageAndPromote` on every return or throw path. Spool cleanup now preserves the primary error, shares a concurrent attempt, retries after synchronous throw or immediate rejection, and is idempotent after success.
- disposition: accept the production generation builder and shared lexical derivation as a host-neutral Core capability. The executable plugin search path is unchanged, so the technical report handoff is `no-impact`; whole-generation reuse, durable storage-backed spool composition, migration/search cutover and maintenance remain in the following P4b.5 tasks.
- next: implement a generation-scoped `StorageAdapter` spool and synchronization orchestration that reuses an exact current generation or builds and promotes from a durable manifest snapshot while preserving the prior current on failure.

## 2026-08-18 — handoff to codex-main-session

- evidence: the user explicitly requested takeover of the active initiative and completion of every `goal.md` success criterion. The worktree is clean at `f4f97b9`; P4b remains the only active phase, and the accepted P4b.5 builder checkpoint is the latest implementation state.
- change: transferred initiative ownership from `zcode-main-session` to `codex-main-session` without changing the goal, phase outcome, accepted implementation, or verification record.
- disposition: retain all accepted P1-P4b.5 builder work. Continue isolating changes from the parallel `2026-08-13-discovery-loop-and-library-insight` initiative and do not modify its Helm state.
- next: observe Red for a generation-scoped durable `StorageAdapter` spool and production synchronization/search-cutover orchestration, then implement exact whole-generation reuse, snapshot rebuild and prior-current preservation on failure.
