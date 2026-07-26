# P3 — email-delivery

<!-- Filename 03-email-delivery.md ↔ P3 -->
goal_ref: ../goal.md
status: done

## Outcome

On pipeline `completed`, both Obsidian and CLI can deliver one personal email of the daily digest via Resend HTTPS, with HTML+text body per `email-content-format.md`, vault `delivery-state.json` idempotency, and a test-send path — without failing the pipeline run on delivery errors.

## Assumptions

- P1 paperKey and P2 SourceAdapter are already on this branch.
- Product/engineering contracts in `../email-content-format.md` and `../email-delivery.md` remain authoritative.
- Digest is projected from structured daily assembly / index results, **not** by parsing vault `.md`.
- Dual-host double-send prevention is idempotency-only (no host mutex setting in v1).

## Approach

1. Define `DailyDigest` DTO in core (date, language, categories, topics, papers with five fields / fallback, counts).
2. Project digest at completed daily assembly (or immediately after successful write path).
3. `renderEmailHtml` + `renderEmailText` + subject helper (`arXiv Daily YYYY-MM-DD · N 篇`, including `· 0 篇`).
4. `DeliveryChannel` port + Resend HTTPS implementation (shared logic; hosts supply HTTP + secrets).
5. `delivery-state.json` under vault `.index/`; skip when already `delivered` for date+recipient+channel.
6. Wire auto-send after completed in Obsidian schedule path and CLI run path; settings for enable/To/From; API key via secret/env.
7. Test-send command/settings action; unit tests for renderer, idempotency, zero-day body.

## Tasks

- [x] DailyDigest type + projector from pipeline/daily results.
- [x] HTML + text renderer + subject (zh/en), zero-day and empty-topic rules.
- [x] Delivery state store (load/save/idempotent skip).
- [x] Resend channel (HTTPS) + secret/settings wiring (plugin + CLI).
- [x] Hook after pipeline `completed`; isolate delivery failures from run-state.
- [x] Test send entry points.
- [x] Focused tests + typecheck; update goal/journal when done.

## Verification

- Unit: subject, zero-day body, five-field paper block, HTML escape, skip-if-delivered.
- Integration-style: completed run with mock HTTP records delivered; second call skips; failed then retry succeeds.
- Manual: test send to personal inbox (user).

## Abort / reshape triggers

- If Obsidian cannot call Resend via `requestUrl` for auth shape → L2: CLI-only send first, plugin preview/test later.
- If digest projection requires parsing daily `.md` → stop; fix projector to use structured data only.
