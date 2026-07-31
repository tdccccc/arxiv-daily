# Daily structured-summary checkpoint recovery

status: done
updated: 2026-08-01
owner: sess_4ab50bf8-a3f3-40cd-bf05-66eb503e2ed9

## Intent

Make daily-report generation resumable so a cancelled, crashed, or otherwise interrupted run can reuse each compatible structured summary already completed, without weakening the daily report's all-or-nothing commit boundary.

## Success criteria

- [x] Each newly completed validated or typed-fallback structured summary is durably checkpointed before the pipeline starts the next paper.
- [x] A rerun reuses only compatible checkpoint entries and invokes the LLM only for selected papers whose entries are missing or stale.
- [x] Recovered and newly generated entries preserve selected-paper order, progress, metrics, cancellation, validation, and fallback semantics.
- [x] The daily report remains a complete deterministic document written once through the existing atomic commit path; checkpoint files never become user-facing partial reports.
- [x] A committed daily report remains the authoritative recovery boundary, and existing report/index repair behavior wins over stale checkpoints.
- [x] Corrupt, incompatible, or unwriteable checkpoint state fails safely without silently attaching a summary to the wrong paper or losing newly paid LLM work.
- [x] Plugin and CLI hosts share the same host-neutral core recovery behavior and pass focused plus repository-wide verification.

## Non-goals

- Resuming an in-flight LLM request or preserving partial model output.
- Parallelizing per-paper summarization or changing provider retry policy.
- Merging partial Markdown, incrementally appending to a daily report, or overwriting an existing daily report.
- Sharing checkpoints concurrently across processes, devices, or vault replicas.
- Reusing checkpoint entries when their compatibility cannot be established from persisted inputs and contract versions.
- Retrofitting checkpoints for historical runs that completed before this feature exists.

## Constraints

- Core remains host-neutral and uses the existing `StorageAdapter`; hosts must not acquire divergent checkpoint rules.
- Checkpoint writes use the storage adapter's atomic path and live as versioned Vault data under the configured index area.
- Reuse identity must cover the paper and every input that can change the structured summary, including source content, summary language, model/provider selection, generation parameters, prompt contract, and result schema contract.
- Checkpoint entries store parsed `DailyPaperResult` values, not untrusted raw model responses or credentials.
- Validated results are reusable when compatible; validation-exhausted fallbacks are reusable only under an exact fingerprint match; transport-exhausted fallbacks may be recorded for diagnostics but are retried on resume by default.
- Existing strict sequential execution, preflight validation, typed fallback behavior, final daily-report atomicity, and post-commit index repair remain intact.
- Cancellation is never converted into fallback, and no new LLM call may start until the preceding entry's checkpoint write succeeds.
- Do not commit or push without explicit user instruction.

## Phases

1. P1 — a versioned host-neutral checkpoint contract and atomic store safely preserve compatible per-paper results — status: done
2. P2 — sequential summarization resumes compatible entries and durably checkpoints each new result before continuing — status: done
3. P3 — daily-report commit, index repair, cancellation, and checkpoint cleanup form one coherent recovery lifecycle — status: done
4. P4 — compatibility scenarios, both hosts, operational documentation, and full regression verification support closure — status: done
