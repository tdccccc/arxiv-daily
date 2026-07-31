# Official email delivery (Beta)

status: done
updated: 2026-07-31

## Intent

Ship **Official delivery (Beta)** end-to-end: users verify an inbox via magic link, then arXiv Daily sends digests through a project Cloudflare Worker + Resend on **arxiv-daily.top**, without pasting a Resend API key into the plugin. **Send yourself** remains the default and fully independent path.

## Success criteria

- [x] Cloudflare Worker exposes verify-start, verify-complete (magic link), and deliver APIs with auth, quota, and idempotency.
- [x] Resend sends from a verified address on arxiv-daily.top (domain already Verified).
- [x] Plugin: mode **Official delivery (Beta)** can be selected when online; user verifies email, stores token, test-send and auto-send work.
- [x] Cross-mode idempotency: same date + recipient not double-sent after self or hosted success.
- [x] Project Resend API key never ships in plugin or git; only Worker secrets.
- [x] Labeled **Beta** in UI and docs; failures never fail pipeline `completed`.

## Non-goals

- Multi-recipient / team broadcast.
- Replacing Send yourself as default.
- Full account system (password login, billing).
- Perfect multi-device token sync UX (token is local settings).

## Constraints

- Stack: **Cloudflare Workers** + KV (or equivalent) + Resend HTTPS.
- DNS: user controls **arxiv-daily.top** (Huawei Cloud); Worker route e.g. `email.arxiv-daily.top`.
- Core stays host-neutral; Worker is a separate deployable under `services/email-relay/`.
- Default mode remains `self`.

## Phases

1. P1 — Plan + Worker API + local/wrangler project skeleton — status: done
2. P2 — Plugin Official delivery (Beta) UI: verify flow, token, base URL, enable send — status: done
3. P3 — Docs, deploy checklist, flag online, smoke notes — status: done

## Current focus

Closed. Deploy Worker + DNS on arxiv-daily.top remains an ops follow-up
outside this initiative (see journal).

## Resolved

- Deploy: Cloudflare Workers.
- Scope: end-to-end Beta this initiative.
- Domain: arxiv-daily.top already **Verified** in Resend.
- Verification: magic link.
- Idempotency: date + recipient across modes.
