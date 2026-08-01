# P3 — host portability and verification

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

Plugin and CLI activate the same filter checkpoint behavior, Vault data portability carries both checkpoint kinds safely, operators can understand recovery logs and lifecycle, and the complete repository gate supports initiative closure.

## Assumptions

- Existing recursive `.index/**` export/import should carry the new independent filter document without an archive format change.
- Host composition should instantiate Core stores from current output settings and one shared storage adapter; no host-specific reuse policy is needed.
- Normal checkpoint events remain info-level diagnostic logs; corruption and cleanup failures remain warnings.
- Full Core/workspace tests require an 8 GiB heap and one worker in this environment.

## Approach

Wire `DailyFilterCheckpointStore` beside the existing summary store in both composition roots and inject both minimal ports into `ArxivPipeline`. Extend host and real-adapter tests, then prove default/custom Vault export/import restores both documents and exact-compatible filter lookup. Update operator documentation for paths, invalidation, logs, zero-result retention, report authority, and safe cleanup. Run independent adversarial correctness/security review, resolve findings, then execute every repository quality and release gate with resource-safe settings.

## Tasks

- [x] Wire the Core filter store through Plugin and CLI using current storage, output settings, and warning logger.
- [x] Add Plugin/CLI composition and Node/Obsidian real-adapter reconstruction/cleanup tests.
- [x] Extend default/custom-layout data portability tests to carry and reuse both filter and structured-summary checkpoints without relaxing archive safety.
- [x] Document filter/summary checkpoint paths, exact invalidation, hit/miss/persisted logs, zero-result retention, report authority, sensitivity, and safe cleanup.
- [x] Conduct independent adversarial correctness/security review and resolve every real finding.
- [x] Run focused host/portability regressions and the complete lint, boundary, typecheck, test, build, smoke, release, and diff gate.
- [x] Reconcile all goal success criteria, close the initiative, commit P3, and deploy the production plugin assets to the test Vault.

## Verification

- Both hosts create the filter store from the same active output layout as Core and pass it to the pipeline with no duplicated policy.
- A compatible filter checkpoint exported by one host can be imported and reused by the other when effective settings match.
- Cross-layout `.index` skip, traversal, symlink, hardlink, directory-conflict, and secret-handling protections remain intact.
- Operator logs visibly distinguish filter and each summary checkpoint hit/miss/persisted event.
- All repository commands exit zero; expected fault-injection logs are not failures.

## Recorded evidence

- Focused final verification passed: CLI import/runtime 36, Node adapter 13, Plugin composition/adapter 28, and Core filter/checkpoint/pipeline focused suites.
- Resource-safe full workspace run passed 1,369 tests across 94 files: Core 1,009, Node Runtime 13, CLI 59, and Plugin 288.
- Lint passed with 0 errors and 52 pre-existing warnings under the 60-warning limit; boundaries and all workspace typechecks passed.
- Workspace build, production smoke build, release metadata check for `0.3.5`, all 5 release-tool tests, and `git diff --check` passed.
- Adversarial fixes covered immutable request snapshots, readable-corrupt recovery, Node `0600` checkpoint files, raw ZIP structure/count/name validation, bounded streaming expansion, incremental CRC32, and promotion rollback.
- Final independent correctness and security reviews reported no findings within the documented trusted, quiescent Vault and archive boundary.

## Abort / reshape triggers

- If host settings construction can drift from fingerprint inputs, centralize composition rather than accepting host parity by convention.
- If archive format changes become necessary, preserve format-version-1 compatibility and record an explicit migration decision before implementation.
- If any suite must be skipped rather than resource-safely scheduled, stop and fix the invocation; skipped tests do not satisfy closure.
