# P3 — Atom metadata cache

goal_ref: ../goal.md
status: done

## Outcome

Daily enrichment and manual summaries reuse fresh, validated Atom metadata across fetcher instances and process runs, requesting the API only for cache misses.

## Assumptions

- The existing cache root and expiry setting are appropriate for Atom metadata as a separate namespace.
- Per-ID positive entries provide useful reuse without negative-cache correctness risks.
- `AtomPaperMeta` contains all metadata needed by both daily and manual paths.

## Approach

Add a host-neutral, versioned per-ID cache over `StorageAdapter`, integrate cache-first batching in `ArxivFetcher.fetchMetadataByIds()`, route manual lookup through that method, and wire the same persistent cache root into plugin and CLI construction and cleanup.

## Tasks

- [x] Add a strict versioned `AtomMetadataCache` with TTL, canonical identity validation, atomic writes, and cleanup.
- [x] Integrate cache hits/misses, canonical deduplication, positive writes, and existing batching into `ArxivFetcher`.
- [x] Route manual metadata lookup through `fetchMetadataByIds()` and remove duplicate Atom parsing.
- [x] Wire persistent cache construction and cleanup into plugin and CLI composition roots.
- [x] Add focused cache, fetcher, manual cross-flow, plugin, and CLI tests.
- [x] Run focused suites, affected typechecks, and boundary checks.

## Verification

- Fresh full hits make zero Atom HTTP requests; partial hits request only canonical misses.
- Missing API entries are not negatively cached and batches remain capped at 200.
- Expired, malformed, or mismatched entries are removed best-effort and refetched.
- Two fetcher/runtime instances over the same storage root reuse metadata.
- Manual fetch can consume metadata cached by a prior daily fetch.

## Abort / reshape triggers

- If existing storage adapters cannot support reliable atomic cache writes, tolerate only corruption-as-miss and never use a malformed entry as metadata.
- If plugin and CLI cache roots have incompatible path semantics, keep the cache host-neutral but instantiate host-specific roots rather than weakening validation.
