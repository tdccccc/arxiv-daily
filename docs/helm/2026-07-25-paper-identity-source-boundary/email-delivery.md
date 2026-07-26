# Email delivery engineering contract

status: accepted (grill 2026-07-25); amended 2026-07-27 (dual-mode product)
implements_in: P3 (self-send shipped); hosted planned
goal_ref: ./goal.md
content_ref: ./email-content-format.md
dual_mode_ref: ./email-dual-mode.md

Host/transport/idempotency/settings for **自己发送** (self Resend BYOK).  
Product dual-mode (自己发送 + 官方代发) lives in `email-dual-mode.md`. Body shape: `email-content-format.md`.

## B1 — Provider

| Decision | Detail |
|---|---|
| First provider | **Resend** (HTTPS API) |
| Settings UX | Clear fixed fields for Resend (not a free-form HTTP template in v1) |
| Future | Optional second provider later; do not block P4 on multi-provider UI |

Suggested settings:

| Field | Quick setup | Advanced |
|---|---|---|
| To | **Required** | Required |
| Resend API key | **Required** | Required |
| Daily auto-send (`enabled`) | Off until test succeeds | On when ready |
| From email | **Empty** → `onboarding@resend.dev` | Custom verified domain address |
| From name | Optional (default `arXiv Daily`) | Optional |

Secret: Resend API key in plugin settings / CLI env (`ARXIV_DAILY_RESEND_API_KEY`).

## B2 — Secrets and config placement

| Item | Where |
|---|---|
| To address | Settings (syncable, non-secret) — primary user field |
| From email / From name | Optional settings; empty From → Resend quick sender |
| Resend API key | Plugin `email.apiKey` (local data.json, like LLM key); CLI env `ARXIV_DAILY_RESEND_API_KEY` |
| Not allowed v1 | API key in vault Markdown or committed config; **project send key never in plugin** |
| UX order | To + API key → **Send test** → enable Daily auto-send |
| Audience | **Personal only** in quick setup: To must be the Resend account email; document in getting-started §8 (EN/中文) |
| User-facing mode name | **自己发送** (default). **官方代发** is planned — see `email-dual-mode.md` |

User-facing guides: `docs/getting-started.md` §8, `docs/getting-started.zh-CN.md` §8.

## B3 — Failure and retry

| Phase | Behavior |
|---|---|
| Within one send attempt sequence | **Limited retries with backoff** (2–3 attempts total, implementer picks exact count/delays) for transient failures |
| After retries exhausted | Record **`failed`** in delivery-state; **do not** mark pipeline run failed |
| Later pipeline **`completed`** for same date | Treat as eligible again if state is missing/`failed` (not `delivered`) — **idempotent success skip only when already `delivered`** |
| Permanent config errors (bad key, 401/403) | Prefer fail fast after limited tries; surface in logs/Notice; leave `failed` for user fix |

User clarification: combine in-run limited retry **and** “wait until next completed, then idempotent retry if not delivered”.

## B4 — Dual-host and cross-mode double-send

| Decision | Detail |
|---|---|
| Policy | **Vault `delivery-state.json` idempotency** — no Obsidian-vs-CLI mutex setting in v1 |
| Skip when delivered | Prefer **`date + recipient`** already `delivered` (**cross-mode**, 2026-07-27): switching 自己发送 ↔ 官方代发 does not auto-send a second mail that day |
| Channel field | Still stored for diagnostics (`email:resend`, future `email:hosted`); not a reason to double-send on mode switch |
| Concurrent race | Accepted residual risk for personal use; no lease/lock in v1 |
| Both hosts may enable auto-send | Yes; first successful writer wins |

## B5 — MVP manual actions

| Capability | P4 MVP |
|---|---|
| Auto-send on pipeline `completed` | Yes (when email enabled + configured) |
| **Test send** one message | Yes — Settings and/or CLI (today’s digest or fixed sample) |
| Full HTML preview pane | Not required for MVP |
| Export-only default-off auto | No — auto remains the primary path when enabled |

## B6 — Zero-paper body structure

When `N = 0` and run `completed`:

1. Subject: `arXiv Daily YYYY-MM-DD · 0 篇` (en: `· 0 papers`)
2. Header: date, count 0, categories
3. **Lead line:** 「今日无相关论文」 / English equivalent
4. **Each configured topic** still listed with empty-topic one-liner
5. Footer: vault daily path

## delivery-state (sketch)

Path: `arxiv-daily/.index/delivery-state.json` (alongside run-state).

**Skip / success identity (auto-send):** `date + recipient` (cross-mode).  

**Stored record** may still key or label by channel for debugging, e.g. last channel `email:resend` | `email:hosted`.

Suggested fields (implementer may refine names):

- `status`: `delivered` | `failed` | (optional `sending` not required in v1)
- `updatedAt`
- `channel` (last attempt / success channel)
- `attempts` / `lastError` (optional, useful for diagnostics)
- `providerMessageId` when delivered (optional)

When implementing cross-mode skip, `shouldSend` must treat any prior **delivered** for the same date+to as skip, regardless of channel.

**Skip send** only when `status === delivered` for that key.  
`failed` or absent → may send on completed hook (after in-run retries as above).

## Pipeline isolation

Delivery runs **after** successful daily completion signaling. Failures never rewrite run-state to failed for the pipeline date.
