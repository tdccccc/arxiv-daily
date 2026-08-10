# P1 — worker-api

goal_ref: ../goal.md
status: done

## Outcome

`services/email-relay` Cloudflare Worker implements verify + deliver with Resend, quota, and idempotency; documented secrets and DNS for `email.arxiv-daily.top`.

## Approach

- Routes under `/v1/*`
- KV for pending verify tokens, device tokens, daily quota, deliver idempotency
- Resend API from Worker secrets only

## Tasks

- [x] Scaffold wrangler project + README deploy steps
- [x] `POST /v1/verify/start` — body `{ email }` → send magic link
- [x] `GET /v1/verify` — query token → bind device token, HTML page to copy token
- [x] `POST /v1/deliver` — Bearer device token + digest/html/text → Resend
- [x] Quota (default 2/day/email) + Idempotency-Key
- [x] Unit-testable pure helpers where practical

## Verification

- `wrangler dev` or documented curl against mock/local
- Typecheck/build of worker package

## Abort

- If Resend From domain not actually verified in production secrets → do not enable public Beta flag until fixed.
