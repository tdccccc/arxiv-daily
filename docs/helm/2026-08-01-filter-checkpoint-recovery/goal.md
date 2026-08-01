# Daily filter checkpoint recovery

status: active
updated: 2026-08-01
owner: sess_4ab50bf8-a3f3-40cd-bf05-66eb503e2ed9

## Intent

Make interrupted daily-report runs reuse an exact-compatible, validated paper-filter classification batch before resuming downstream work, and make filter and per-paper summary checkpoint behavior visible in operator logs.

## Success criteria

- [ ] A validated filter result is durably checkpointed before Paper Index, content, detail, or summary work begins.
- [ ] An interrupted rerun reuses only an exact-compatible filter batch and does not repeat its filter LLM request.
- [ ] Filter and structured-summary checkpoint hit, miss, and persistence events are visible in normal diagnostic logs.
- [ ] Cancellation, invalid responses, transport errors, checkpoint write failures, metrics, progress, and output ordering retain their existing semantics.
- [ ] A committed daily report remains authoritative and triggers best-effort cleanup of all transient daily-generation checkpoints.
- [ ] Plugin and CLI share one host-neutral Core implementation, and filter checkpoints remain portable Vault data without persisting credentials, plaintext endpoints, or raw provider responses.
- [ ] Focused adversarial tests and the complete repository quality suite pass.

## Non-goals

- Reusing source discovery or Atom enrichment responses.
- Changing filter prompt semantics, requiring complete response coverage, or adding filter validation retries.
- Resuming an in-flight filter request or storing partial/raw model output.
- Sharing checkpoints concurrently across processes, devices, or Vault replicas.
- Replacing the final daily report's atomic commit boundary.

## Constraints

- Filter checkpoints use a separate batch document under the configured index area; structured-summary checkpoints remain per paper.
- Compatibility binds the exact rendered filter messages and effective LLM generation identity, with endpoint identity persisted only as a digest.
- Only strictly validated filter records are reusable; errors, cancellation, and malformed responses are never cached.
- Current paper metadata reconstructs recovered `FilteredPaper` values; checkpoint results contain only validated model decisions.
- No downstream pipeline mutation starts until a newly generated filter result is durably persisted.
- Existing daily reports override all checkpoint state; cleanup failure only warns.
- Core remains host-neutral and both hosts use the same stores and lifecycle rules.
- Commit each completed phase separately; do not push without explicit instruction.

## Phases

1. P1 — a strict batch filter contract and durable store preserve only exact-compatible validated decisions — status: done
2. P2 — pipeline filter resume, checkpoint logs, and unified cleanup preserve all cancellation and commit boundaries — status: active
3. P3 — both hosts, Vault portability, operator documentation, adversarial review, and full verification support closure — status: pending
