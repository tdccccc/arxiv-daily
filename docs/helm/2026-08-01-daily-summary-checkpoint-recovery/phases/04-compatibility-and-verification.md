# P4 — compatibility and verification

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

Checkpoint recovery is portable with Vault data, documented and observable for operators, adversarially reviewed across compatibility boundaries, and verified by the complete repository quality suite.

## Assumptions

- Structured-summary checkpoints are Vault data and therefore should follow the CLI's existing `.index` export/import contract without a separate archive format.
- Export and import must derive `.index` from the same configured output layout as core, including custom output roots; stale default directories must not override current configuration.
- Checkpoint JSON is internal bookkeeping, so user documentation should explain behavior, location, invalidation, and safe cleanup without promising schema stability.
- The strict final gate may require bounded test-worker memory because the repository's large pipeline suite can exceed the default V8 heap when combined with other suites.

## Approach

Align CLI data portability with core's `derivePaperInboxPaths` rule and add recursive checkpoint export/import round-trip coverage for default and custom output roots. Add concise operator-facing documentation for resume, invalidation, fallback policy, report authority, and cleanup. Perform an adversarial review of fingerprint inputs, corruption and cancellation windows, host composition, secrets, and partial-report invariants; fix any findings before closure. Run focused regressions followed by every repository lint, boundary, typecheck, test, build, smoke, release-version, and diff check required by the workspace.

## Tasks

- [x] Make CLI export/import derive the active `.index` root from current output settings, preserving nested checkpoint artifacts and ignoring stale default roots.
- [x] Add default/custom-output data export/import tests proving a restored checkpoint is readable by the core store and no path escapes the vault.
- [x] Document checkpoint lifecycle, compatibility invalidation, fallback reuse, storage location, final-report authority, and safe operational recovery.
- [x] Conduct adversarial review across schema/fingerprint changes, secrets, corrupt artifacts, cancellation/crash windows, host parity, and deterministic report output; resolve findings.
- [x] Run focused regression suites for store, summarizer, pipeline, host composition, and data portability.
- [x] Run repository-wide lint, boundary checks, typechecks, tests, builds, smoke build, release tools/version checks, and `git diff --check` with resource-safe test settings.
- [x] Reconcile all goal success criteria and close the Helm initiative only with recorded verification evidence.

## Verification

- Export archives the configured `.index/daily-summary-checkpoints/**` recursively; import restores it to the exact index root core derives for both default and custom output layouts.
- A restored compatible structured/validation-fallback entry is reusable after runtime reconstruction; transport fallback remains a miss.
- No API key, plaintext endpoint host/path/query/userinfo, or raw provider response is persisted. Checkpoints, backups, and export archives are explicitly documented and handled as sensitive Vault data because they can contain paper inputs and model results.
- Documentation clearly distinguishes structured-summary checkpoints from partial daily reports and states that an existing daily report is authoritative.
- Adversarial tests cover mismatched contracts/inputs, corrupt primary/backup/entry state, all commit/cancel/index windows, cleanup failure, and both production hosts.
- Every required repository command exits zero; expected fault-injection logs are distinguished from failures.

## Recorded evidence

- Focused final regressions: Core checkpoint store 74, summarizer 28, Node adapter 12, Obsidian adapter 11, and CLI portability/security 18 tests passed.
- Resource-safe full workspace run (`NODE_OPTIONS=--max-old-space-size=8192`, one test worker): Core 942, Node Runtime 12, CLI 45, and Plugin 287 tests passed — 1,286 tests across 93 files.
- Workspace lint passed with 0 errors and 52 pre-existing warnings under the 60-warning limit; boundary checks and all four workspace typechecks passed.
- Workspace build, production smoke build, release metadata check for `0.3.5`, all 5 release-tool tests, and `git diff --check` passed.
- Final independent correctness and security reviews reported no findings after strict unreadable-primary mutation handling and atomic hardlink-safe CLI import replacement were applied.

## Abort / reshape triggers

- If data portability cannot reuse the current `.index` logical archive without breaking existing archives, preserve format version 1 compatibility and add only a backward-compatible path derivation fix.
- If full tests expose unrelated pre-existing failures, verify them against `main`, record exact evidence, and do not mark affected success criteria complete without a justified waiver.
- If a compatibility input cannot be proven to affect or not affect generation, fail closed by invalidating rather than broadening reuse.
- If strict verification requires skipping a suite rather than resource-safe scheduling, stop and fix the runner invocation; skipped tests do not count as closure.
