# P5 — core-fixes

goal_ref: ../goal.md
created: 2026-09-24T22:40:00+08:00
updated: 2026-09-28T21:01:46+08:00
revision: 4

## Outcome

The confirmed core / CLI / relay defects that can be fixed without live provider access are fixed: paper-note frontmatter stays valid YAML for any topic tag, refreshing a paper note's frontmatter keeps the user's own properties and tags, a date that has scrolled out of arXiv's recent listing fails permanently instead of being retried ten times, a run interrupted by a crash is retried automatically, a filter answer wrapped in one code fence is accepted, hosts that cannot send automatic email say so, and the CLI / node-runtime / relay code has been reviewed.

## Assumptions

- Slug-shaped values (the common case) must keep their current unquoted output, so existing notes and byte-identical rendering tests stay unchanged; only YAML-unsafe values get quoted.
- Frontmatter the plugin writes is line-based (`key: value`, flow list for `tags`); preserving unknown top-level keys verbatim, with their indented continuation lines, is enough to keep user properties.
- The review agent died after batches A and B (API 503); batch C is reviewed in this session (user decision 2026-09-25). Its CONFIRMED High findings become further chunks here (L1).
- The existing transient-retry cap (`MAX_TRANSIENT_ATTEMPTS`) is the right bound for crash-recovered runs too.
- Allowing exactly one outer ```` ```json ```` fence does not weaken the 08-10 filter contract: everything inside is still validated strictly.

## Approach

Small, local fixes, each with a test written first. The thinking-parameter finding (`extra_body` never flattened) changes what is sent to every provider and cannot be verified without real API calls; it is recorded in the review and left for a user decision. Cross-platform automatic email (exclusive create on macOS / Windows) is out of scope; it is a follow-up initiative after this one (user decision 2026-09-25) — here we only make the gap visible. Cross-process locking (RunLock, PaperIndexStore) is recorded, not fixed.

## Chunks

### Chunk 1 — frontmatter values stay valid YAML (review F20)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core/tests/markdown-writer.test.ts`: a topic tag `AI: Robotics` or `ml,dl` produces frontmatter that a YAML parser reads back as the same `primary_topic` and tag — fails today (unquoted)
- Green check: `cd packages/core && npx vitest run tests/markdown-writer.test.ts`
- regression checks: core markdown / pipeline tests; slug tags render byte-identically
- [x] implementation and tests accepted

### Chunk 2 — refreshing frontmatter keeps user properties (review F21)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: a paper note with extra keys (`rating: 5`, block `aliases`) and an extra tag keeps them after `refreshPaperNoteFrontmatter` — fails today (whole block replaced)
- Green check: focused vitest
- regression checks: existing `refreshPaperNoteFrontmatter` and manual-fetch tests
- [x] implementation and tests accepted

### Chunk 3 — dates older than the recent listing fail permanently (review F22)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `listForDate` for a date older than the oldest `/recent` bucket returns `failed_permanent`; a date newer than the newest stays `failed_transient` — first fails today
- Green check: focused vitest on the source adapter tests
- regression checks: core scheduler / pipeline tests (`NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1` in packages/core)
- [x] implementation and tests accepted

### Chunk 3b — paper index failures during filtering are retryable (review batch A)

- landed as 7aa19aa before this revision; recorded here for completeness
- [x] implementation and tests accepted

### Chunk 4 — a run interrupted by a crash is retried automatically (review batch B, F25)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core/tests/state-store.test.ts`: a stale `running` entry with attempts below the cap recovers to `failed_transient`; at the cap it becomes `failed_permanent` with a "retries exhausted" message — first fails today (always permanent)
- Green check: `cd packages/core && npx vitest run tests/state-store.test.ts`
- regression checks: core scheduler tests; full core suite
- [ ] implementation and tests accepted

### Chunk 5 — a filter answer in one outer code fence is accepted (review batch A, F26)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core/tests` paper-filter test: a response that is exactly ```` ```json\n{"papers":[…]}\n``` ```` yields the same records as the bare JSON — fails today (`invalid-json`); text around the fence, two fences, or invalid JSON inside the fence still fail as before
- Green check: focused vitest on the paper-filter tests
- regression checks: filter-response-validation tests, pipeline tests
- [ ] implementation and tests accepted

### Chunk 6 — hosts that cannot send automatic email say so (review batch B, F27)

- change kind: bug fix (visibility only; the delivery safety design is unchanged)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: on a storage adapter without exclusive create, (a) a successful test send reports that automatic daily email is unsupported on this system, in the plugin and the CLI; (b) the plugin settings email section shows the same warning; (c) an automatic send refused for `delivery_storage_unsupported` produces a user-visible notice in the plugin (CLI: a warning on stderr) — fail today
- Green check: focused vitest in plugin and CLI
- regression checks: plugin and CLI email tests, core delivery tests
- [ ] implementation and tests accepted

### Chunk 7 — review CLI, node-runtime and email relay (batch C)

- change kind: read-only review, in this session, no subagents
- output: findings added to `docs/reviews/2026-09-24-main-review.md`; each CONFIRMED High finding becomes its own chunk here (L1); lower ones recorded
- [ ] review done and recorded

### Chunk 8 — CLI initialization writes private config files (batch C, F28)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: real-filesystem `runInit` tests in `apps/cli/tests/cli-init-e2e.test.ts` show that fresh, overwritten, and kept config files containing credentials are not mode 0600 on POSIX (fresh file follows umask; existing 0644 files stay readable).
- approach: use the existing Node storage adapter's private atomic-write primitive so credentials are written into a 0600 temporary file before replacing the destination. Create new config directories with mode 0700. Preserve current overwrite / keep-existing content semantics and injected wizard I/O.
- Green check: `npm test --workspace arxiv-daily -- tests/cli-init-e2e.test.ts tests/cli-init.test.ts`
- regression checks: full CLI suite; node-runtime atomic-write tests; root typecheck, lint, build, boundaries, submission checks. Permission assertions are POSIX-only; Windows ACL guarantees are not claimed.
- [ ] implementation and tests accepted

## Phase verification

- Full check set from goal.md constraints

## Abort / reshape triggers

- If quoting changes any existing byte-identical rendering fixture for slug tags, stop and narrow the quoting rule (L1).
- If the scheduler relies on `failed_transient` for old dates in some catch-up path (e.g. a fallback source), stop and reconsider chunk 3.
- If making the email gap visible needs changes to the claim / idempotency design itself, stop — that belongs to the follow-up initiative.
- If batch C turns up more High findings than fit in this phase, split them into a new phase (P7).
