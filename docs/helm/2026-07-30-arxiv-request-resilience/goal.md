# arXiv request resilience

status: done
updated: 2026-07-30

## Intent

Make arXiv-backed manual summaries reliable under upstream throttling and outages while reducing avoidable Atom API traffic and protecting existing user notes.

## Success criteria

- [x] Every arXiv HTTP attempt is process-wide serialized, spaced by at least three seconds, and honors shared 429/503 Retry-After cooldowns.
- [x] Retry classification and user-facing errors distinguish transient rate limits/outages from permanent request failures.
- [x] Dashboard and manual fetch agree on detail-note identity and never overwrite handwritten or mismatched notes.
- [x] Fresh Atom metadata is reused across fetcher instances and runs through a bounded persistent cache.
- [x] Focused and repository-wide verification passes, or unrelated baseline failures are recorded with evidence.

## Non-goals

- Cross-process or cross-device arXiv request coordination.
- Caching missing arXiv IDs or changing the paper-index persistence schema.
- Automatically overwriting notes that contain user-authored content.

## Constraints

- Keep core host-neutral; do not introduce Node or Obsidian APIs into `packages/core`.
- Preserve existing public result and persisted-state compatibility where practical.
- Treat the server's Retry-After as a minimum and keep all waits cancellable.
- Do not commit or push without explicit user instruction.

## Phases

1. P1 — all arXiv attempts obey shared serialization, spacing, cooldown, and retry policy — status: done
2. P2 — manual summary errors and existing-note handling are safe and consistent — status: done
3. P3 — Atom metadata is persistently cached and reused by all fetch paths — status: done
4. P4 — full regression evidence supports closing the initiative — status: done

## Current focus

Done. P4 verification passed and the initiative is closed.
