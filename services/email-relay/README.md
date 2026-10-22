# arXiv Daily email relay (Official delivery Beta)

Cloudflare Worker that:

1. Sends a **magic-link** verification email via Resend
2. Issues a **device token** after the user opens the link (once automatic delivery is ready)
3. Accepts **digest deliver** requests from the Obsidian plugin and the CLI (`Bearer` token), records each one in a Durable Object ledger and sends it through Resend at most once

Project Resend API key stays in Worker secrets only.

Production rollout, cutover and releases are described in [`docs/runbook-cutover.md`](docs/runbook-cutover.md); the design is in the Worker section of [`docs/technical-report.md`](../../docs/technical-report.md).

## Prerequisites

- Domain **Verified** in Resend (e.g. `mail.arxiv-daily.top` on `arxiv-daily.top`)
- Cloudflare account (Workers + KV + Durable Objects)
- Node 20+

## Setup

```bash
cd services/email-relay
npm install
npx wrangler login
npx wrangler kv namespace create STORE
npx wrangler kv namespace create STORE --preview
# paste ids into wrangler.toml [[kv_namespaces]]
```

Secrets (all four are required; `npm run preflight` checks that their names exist):

```bash
npx wrangler secret put RESEND_API_KEY
npx wrangler secret put TOKEN_SECRET               # long random string; rotating it invalidates every device token
npx wrangler secret put IDENTITY_SECRET            # at least 16 characters and long-lived; rotating it locks automatic delivery
npx wrangler secret put DELIVERY_V2_CUTOVER_TOKEN  # operator bearer token for the cutover API
```

Vars in `wrangler.toml` (or dashboard):

| Var | Example |
|---|---|
| `PUBLIC_BASE_URL` | `https://mail.arxiv-daily.top` |
| `FROM_EMAIL` | `daily@mail.arxiv-daily.top` |
| `FROM_NAME` | `arXiv Daily` |
| `DAILY_QUOTA` | `5` (public Beta; raise only for private debugging) |

`BUILD_IDENTITY` is a fifth variable, but it is **not** in `wrangler.toml`: it must be `email-relay-v2-<40 lowercase hex characters of a commit SHA>` and `npm run deploy` injects it (see below). Without it automatic delivery stays locked.

DNS / domain:

- Worker **Custom domain**: `mail.arxiv-daily.top` (same host as the Resend sending subdomain is fine).
- Plugin default hosted base URL: `https://mail.arxiv-daily.top`
- Magic links use `PUBLIC_BASE_URL` — keep it aligned with the custom domain.

## Deploy

```bash
npm run deploy                 # checks, then deploys with BUILD_IDENTITY injected
npm run deploy -- --dry-run    # the same checks; prints the command and deploys nothing
```

`npm run deploy` is the only supported way to deploy. A bare `wrangler deploy` replaces every Worker variable with the ones in `wrangler.toml`, which removes `BUILD_IDENTITY`: after it, automatic digests and verification links fail with 503 while **test emails keep working**, so the problem only shows up when the next day's digest does not arrive. The script

- refuses to run unless `services/email-relay` has no uncommitted or untracked files and `HEAD` is a 40-hex commit, because the identity names the commit that is deployed;
- reads `GET /ready` of the live relay and refuses an identity different from the one a cutover is already bound to (the binding is permanent, so that deploy would lock the relay); it then prints the identity to pass as `--identity`;
- runs one `wrangler deploy --var BUILD_IDENTITY:email-relay-v2-<sha>` and nothing else. It needs no secret and never calls a verification, delivery or cutover endpoint.

After a deploy:

- `GET /ready` must return **200** with `"automatic": "ready"`. A successful test email is not enough: test sends do not check cutover readiness. Until the first cutover is finished `/ready` is 503, which is expected; follow the runbook.
- Releasing new code to a relay whose cutover is finished: [`docs/runbook-cutover.md`](docs/runbook-cutover.md), section 9.

Durable Object migrations: `wrangler.toml` carries two append-only tags, `v1-deliver-gate`
(`DeliverGate`) and `v2-verify-rate-limiter` (`VerifyRateLimiter`, binding `VERIFY_RATE_LIMIT`).
`npm run deploy` applies a pending tag together with the new code; there is no separate
migration command. Rollout notes:

- The new class is SQLite-backed (`new_sqlite_classes`), as the free plan requires.
- Until the deploy lands, the old Worker keeps using its KV counters; after it, the old `rl:*`
  KV keys are ignored and expire by TTL (within an hour). Each address and IP starts with a fresh
  allowance, so the first hour after deploy can allow one extra full window per subject.
- If `VERIFY_RATE_LIMIT` is not bound, `/v1/verify/start` fails closed (HTTP 500, no mail).
- Do not edit or reorder existing migration tags; a rollback of the code does not undo a migration.

## API

| Route | Auth | Purpose |
|---|---|---|
| `GET /` , `GET /health` | none | Liveness: `200 {"ok":true,"service":"arxiv-daily-email-relay","beta":true}`, independent of every binding |
| `GET /ready` | none | Readiness of automatic delivery |
| `POST /v1/verify/start` | none | Send the magic link |
| `GET /v1/verify?token=...` | the one-time link token | Show the device token |
| `POST /v1/deliver` | `Bearer` device token | Send a digest or a test email |
| `GET`/`POST /internal/delivery-v2/cutover` | `Bearer` `DELIVERY_V2_CUTOVER_TOKEN` | Operator-only cutover status and actions (404 for anyone else); see the runbook |

### `GET /ready`

`200` with `{ protocolGeneration, buildIdentity, phase: "ready", automatic: "ready", readyGeneration }` only when the cutover is finished and every runtime dependency is valid. Otherwise `503`:

- the same body with `automatic: "locked"` while the cutover is still in progress (`phase` is `locked`, `inventoried`, `observing` or `sealed`, or `ready` while a repair is pending);
- or `{ "automatic": "locked", "reason": "<name>" }` when no cutover state can be read at all. `reason` is a name only, never a value: `build_identity_missing` (BUILD_IDENTITY unset or malformed, e.g. after a bare `wrangler deploy`), `runtime_not_configured` (another binding, secret or var is missing or invalid) or `cutover_locked` (the cutover state is blocked or unreadable, or it is bound to another identity or `IDENTITY_SECRET`).

### `POST /v1/verify/start`

```json
{ "email": "you@example.com" }
```

Sends the magic link. Rate-limited requests get the same "verification email sent" answer.

### `GET /v1/verify?token=...`

Consumes the token; the HTML page shows the **device token** to paste into the plugin. While automatic delivery is not ready it answers `503 Verification unavailable` and keeps the link valid, so retrying later works.

### `POST /v1/deliver`

Headers:

- `Authorization: Bearer <device-token>`
- `Idempotency-Key` (**required**): either `arxiv-daily:auto:<64 lowercase hex>` for the daily digest or `arxiv-daily:test:<32 lowercase hex>` for a test send. Any other value, or none, is `400`.
- `Content-Type: application/json`

Body:

```json
{
  "to": "you@example.com",
  "date": "2026-07-27",
  "subject": "...",
  "html": "...",
  "text": "..."
}
```

`to` must match the email bound to the device token. `date` is `YYYY-MM-DD`, `subject` is at most 500 characters, `html` and `text` at most 2,000,000 characters each, and one of them is required. Other members (the plugin also sends `digest`) are ignored.

The two key kinds take different routes:

| | `arxiv-daily:auto:…` | `arxiv-daily:test:…` |
|---|---|---|
| Durable Object | `recipient-v2:<recipient identity>`: every token of one address shares one ledger, quota and provider key | `device-v2:<device identity>`: per device |
| Logical delivery | recipient + `date` (the client key only marks the kind) | the random key the client generated |
| Cutover | requires `/ready` to be 200 | not required |
| Quota | `DAILY_QUOTA` per UTC day | `DAILY_QUOTA` per UTC day, counted separately |

Answers:

| Status | Meaning |
|---|---|
| `200 {"ok":true}` | Accepted by the provider (or replayed from the ledger) |
| `400` | Invalid JSON, `date`, `subject`/body or `Idempotency-Key` |
| `401` | Missing, unknown, revoked or expired device token |
| `403` | `to` is not the address the token was verified for |
| `409` | The key is bound to a different request, or the same delivery is still in flight (`delivery outcome is pending`, `ambiguous: true`) or reserved (`ambiguous: false`) |
| `413` | Payload too large |
| `422`, `429` `{"error":"provider_definitive_rejection","ambiguous":false}` | Resend refused the mail (`429` when it rate-limited). `429 {"error":"daily quota exceeded (N per UTC day)"}` is the relay's own quota |
| `502 {…,"ambiguous":true}` | Outcome unknown (provider or ledger); the relay does not retry and the client must not blindly resend |
| `503 {"error":"…","ambiguous":false}` | Refused before any provider call: automatic delivery not configured, cutover not ready, ledger unavailable. Installed clients treat exactly this body shape as "nothing was sent" and retry later, so do not add members to it |
| `507` | Ledger capacity reached |

## Local

```bash
npm run dev
npm test
npm run typecheck
npm run preflight        # read-only cutover preflight (see the runbook)
```

## Concurrency (KV has no CAS)

Cloudflare KV cannot do compare-and-swap, so nothing that must be atomic lives there:

- The delivery ledger and the UTC-day quotas live in **Durable Objects** (`DeliverGate`), one per recipient (`recipient-v2:…`, automatic) or per device (`device-v2:…`, test), updated inside storage transactions, so concurrent requests for the same logical send are single-threaded. A cutover singleton (`delivery-cutover:v3`) in the same class holds the permanent build/identity binding and the readiness state.
- `/v1/verify/start` limits use a second Durable Object class (`VerifyRateLimiter`) for the same reason.
- KV only holds the pending verification tokens, the device records, the cutover audit marker and the historical `idemp:*` keys of the first relay generation, which are scanned once during the cutover as evidence of what was already sent.

Deploy requires the Durable Object migrations in `wrangler.toml` (first deploy may take longer).

## Security notes

- Verify-start is rate-limited (3/hour per email address, 10/hour per IP), counted atomically in
  Durable Objects named by a keyed hash (no raw address or IP is stored). The IP limit is checked
  first so a throttled IP cannot spend an address's allowance. Still not a full anti-abuse stack.
- Daily quota per verified email (`DAILY_QUOTA`).
- Do not log full digests in production if avoidable.
- Rotating `TOKEN_SECRET` invalidates all device tokens; the same address maps to the same recipient scope after it verifies again. Rotating `IDENTITY_SECRET` locks automatic delivery until the original value is restored.
