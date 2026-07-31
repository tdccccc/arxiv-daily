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

- [ ] Make CLI export/import derive the active `.index` root from current output settings, preserving nested checkpoint artifacts and ignoring stale default roots.
- [ ] Add default/custom-output data export/import tests proving a restored checkpoint is readable by the core store and no path escapes the vault.
- [ ] Document checkpoint lifecycle, compatibility invalidation, fallback reuse, storage location, final-report authority, and safe operational recovery.
- [ ] Conduct adversarial review across schema/fingerprint changes, secrets, corrupt artifacts, cancellation/crash windows, host parity, and deterministic report output; resolve findings.
- [ ] Run focused regression suites for store, summarizer, pipeline, host composition, and data portability.
- [ ] Run repository-wide lint, boundary checks, typechecks, tests, builds, smoke build, release tools/version checks, and `git diff --check` with resource-safe test settings.
- [ ] Reconcile all goal success criteria and close the Helm initiative only with recorded verification evidence.

## Verification

- Export archives the configured `.index/daily-summary-checkpoints/**` recursively; import restores it to the exact index root core derives for both default and custom output layouts.
- A restored compatible structured/validation-fallback entry is reusable after runtime reconstruction; transport fallback remains a miss.
- No API key, endpoint credential/query/fragment, raw provider response, or user-authored daily/paper content is exposed or overwritten by checkpoint lifecycle operations.
- Documentation clearly distinguishes structured-summary checkpoints from partial daily reports and states that an existing daily report is authoritative.
- Adversarial tests cover mismatched contracts/inputs, corrupt primary/backup/entry state, all commit/cancel/index windows, cleanup failure, and both production hosts.
- Every required repository command exits zero; expected fault-injection logs are distinguished from failures.

## Abort / reshape triggers

- If data portability cannot reuse the current `.index` logical archive without breaking existing archives, preserve format version 1 compatibility and add only a backward-compatible path derivation fix.
- If full tests expose unrelated pre-existing failures, verify them against `main`, record exact evidence, and do not mark affected success criteria complete without a justified waiver.
- If a compatibility input cannot be proven to affect or not affect generation, fail closed by invalidating rather than broadening reuse.
- If strict verification requires skipping a suite rather than resource-safe scheduling, stop and fix the runner invocation; skipped tests do not count as closure.
