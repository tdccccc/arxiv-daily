# arXiv Daily email relay (Official delivery Beta)

Cloudflare Worker that:

1. Sends a **magic-link** verification email via Resend  
2. Issues a **device token** after the user opens the link  
3. Accepts **digest deliver** requests from the Obsidian plugin (`Bearer` token)

Project Resend API key stays in Worker secrets only.

## Prerequisites

- Domain **Verified** in Resend (e.g. `mail.arxiv-daily.top` on `arxiv-daily.top`)
- Cloudflare account (Workers + KV)
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

Secrets:

```bash
npx wrangler secret put RESEND_API_KEY
npx wrangler secret put TOKEN_SECRET   # long random string
```

Vars in `wrangler.toml` (or dashboard):

| Var | Example |
|---|---|
| `PUBLIC_BASE_URL` | `https://email.arxiv-daily.top` |
| `FROM_EMAIL` | `daily@mail.arxiv-daily.top` |
| `FROM_NAME` | `arXiv Daily` |
| `DAILY_QUOTA` | `2` |

DNS:

- Point `email.arxiv-daily.top` to this Worker (Cloudflare custom domain / CNAME per CF docs).  
- If the zone stays on Huawei Cloud DNS only, use a Worker route after adding the domain to Cloudflare, or a CNAME target Cloudflare gives you.

Deploy:

```bash
npm run deploy
```

## API

### `POST /v1/verify/start`

```json
{ "email": "you@example.com" }
```

Sends magic link email.

### `GET /v1/verify?token=...`

Consumes token; HTML page shows **device token** to paste into the plugin.

### `POST /v1/deliver`

Headers:

- `Authorization: Bearer <device-token>`
- `Idempotency-Key: YYYY-MM-DD|you@example.com` (optional but recommended)
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

`to` must match the email bound to the device token.

## Local

```bash
npm run dev
npm test
npm run typecheck
```

## Security notes

- Rate-limit is basic (daily quota per verified email). Tighten before wide public use.
- Do not log full digests in production if avoidable.
- Rotate `TOKEN_SECRET` invalidates all device tokens.
