# P3 — metadata single-flight

goal_ref: ../goal.md
status: done

## Outcome

Concurrent overlapping metadata requests fetch each canonical miss once per process, while cache reads, writes, cleanup, and partial batches remain deterministic and recoverable.

## Assumptions

- Per-ID flights are the smallest model that handles overlapping ID sets without losing batching.
- Owner cancellation may fail joined waiters; registry cleanup must permit a later retry.
- Process-local cache serialization is acceptable because operations are small relative to network latency.

## Approach

Install per-ID flights before network work, recheck cache before owned batches, settle flights incrementally, and serialize cache operations so stale reads and cleanup cannot delete fresh writes.

## Tasks

- [x] Add process-wide canonical-ID metadata flights supporting overlapping sets and cache-free fetchers.
- [x] Recheck owned IDs before HTTP and persist joined positive results into each caller's cache.
- [x] Settle successful/omitted IDs per batch and clean failure/cancellation flights safely.
- [x] Serialize Atom cache get/set/cleanup and protect concurrent directory/write/delete operations.
- [x] Reject future timestamps and invalid TTL values consistently in get and cleanup.
- [x] Define and test multi-batch partial success semantics.
- [x] Run focused cache/fetcher/manual/runtime tests, typechecks, and boundary checks.

## Verification

- `[A,B]` and `[B,C]` issue B in exactly one API request, with and without persistent cache.
- Distinct cache roots share the network flight and each retain positive metadata.
- Omitted, failed, and cancelled IDs leave no negative cache or poisoned registry.
- A 201-ID call can reject on the second batch while first-batch cache and joined waiters remain successful.
- Future or invalid cache timestamps/TTL fail closed; concurrent reads and cleanup cannot delete fresh writes.

## Abort / reshape triggers

- If owner-controlled cancellation creates unacceptable shared-call semantics, add waiter reference counting rather than leaking or pinning flights.
- If a global cache queue causes measurable host contention, introduce namespaced/per-path locks only with a reliable storage identity.
