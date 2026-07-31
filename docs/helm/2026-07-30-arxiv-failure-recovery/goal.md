# arXiv failure recovery

status: done
updated: 2026-07-30

## Intent

Complete end-to-end arXiv failure recovery so transport stalls, partial upstream failures, concurrent metadata demand, and retry exhaustion remain recoverable and accurately reported in both plugin and CLI flows.

## Success criteria

- [x] Hung or non-conforming HTTP adapters cannot permanently retain the process-wide arXiv queue.
- [x] Only typed transport failures and selected transient HTTP statuses are retried.
- [x] Partial category and content-endpoint failures do not become silent permanent data loss.
- [x] Existing verified detail notes and their index state reconcile safely without unnecessary network work.
- [x] Concurrent metadata misses are coalesced and cache operations remain correct under process-local concurrency.
- [x] Retry exhaustion, cancellation, cooldown, and user-visible state agree across scheduler, history, plugin, and CLI.
- [x] Monotonic timing and long Retry-After handling never request before the server minimum or retain hour-scale timers.
- [x] Focused and repository-wide verification passes.

## Non-goals

- Cross-process, cross-device, or CLI-to-Obsidian request locking.
- Automatic rewriting of historical daily reports that may already be incomplete.
- Persisted per-category progress or cross-run daily Markdown merge semantics.

## Constraints

- Core remains host-neutral.
- Obsidian requestUrl physical cancellation is unavailable; logical timeout and queue recovery take priority and the orphan-request limitation must be documented.
- Cancellation must never be retried.
- User-authored notes and existing daily reports must not be overwritten implicitly.
- Do not commit or push without explicit user instruction.

## Phases

1. P1 — transport deadlines release the queue and only explicit transient failures retry — status: done
2. P2 — category, detail-index, and content fallback workflows recover atomically — status: done
3. P3 — concurrent metadata demand is single-flight and cache-safe — status: done
4. P4 — scheduler state and monotonic Retry-After policy are consistent — status: done
5. P5 — adversarial review and full verification support closure — status: done

## Current focus

Complete
