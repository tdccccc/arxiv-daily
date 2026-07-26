# Email dual-mode product contract

status: accepted (2026-07-27)
goal_ref: ./goal.md
implements:
  - self-send: **shipped** (P3 Resend BYOK)
  - official-send: **planned** (not in current code)

User-facing names (中文主 UI / 文档):

| Mode | 用户向名称 | English settings label (suggested) |
|---|---|---|
| Self | **自己发送** | Send yourself (Resend) |
| Hosted | **官方代发 (Beta)** | Official delivery (Beta) |

## Intent

Two send paths **coexist** behind one pipeline exit:

```text
daily run completed
  → build DailyDigest
  → already delivered for this date + recipient? (cross-mode)
  → no → deliveryMode
        ├─ 自己发送 → user Resend (API key on device)
        └─ 官方代发 → project API (key only on server)
              requires: verified email + quota + service online
```

Shared: digest body, render, trigger (`completed`), “already sent today”.  
Different: who pays, whether user pastes an API key, dependency on project uptime.

## Defaults

| Setting | Default |
|---|---|
| Active mode | **自己发送** |
| Daily auto-send | Off until test succeeds (self) |
| 官方代发 | Off; user must opt in **and** complete email verification |

Only **one** mode active at a time (no dual auto-send).

## Settings copy (user-facing)

| 选项名 | 副说明（一行） |
|---|---|
| **自己发送** | 使用你的 Resend API Key，发到你填写的邮箱 |
| **官方代发 (Beta)** | 验证邮箱后由 arXiv Daily 发送（需联网；上线后为 Beta） |

Optional microcopy:

- 自己发送：适合希望数据与费用完全自控的用户  
- 官方代发：适合不想申请 API Key 的用户；服务由项目维护，可能有每日限额  

**Implementation:** **自己发送** is live. **官方代发 (Beta)** is scaffolded in code (`mode`, hosted channel, cross-mode idempotency) but **not online** (`OFFICIAL_DELIVERY_AVAILABLE = false`) until the project relay ships. UI labels it Beta and refuses activation.

### 自己发送 — To restriction (must stay visible)

Quick setup (empty From → `onboarding@resend.dev`):

- **To must be the Resend account email** (GitHub login → usually GitHub **primary** email).
- Not a team list; secondary GitHub emails often fail with HTTP 403.
- Custom From after domain verification can widen recipients.

See `docs/getting-started.md` §8 / `getting-started.zh-CN.md` §8.

## Idempotency (cross-mode)

**Decision (2026-07-27):** success is **once per calendar date + recipient**, **across modes**.

- If today already `delivered` for `to@x.com` via 自己发送, switching to 官方代发 does **not** send again.
- Record may still store which channel last succeeded for diagnostics.
- Force/test-send may bypass (existing test-send `force` behavior).

Logical skip key for “already sent”: prefer **`date + recipient`** (not only `date + recipient + channel`).  
Channel remains useful metadata; do not allow a second auto-send solely because the mode changed.

## 官方代发 — planned requirements

When implemented (separate initiative/phase):

| Need | Detail |
|---|---|
| Project domain | Proper From for deliverability |
| Online API | e.g. Worker/Vercel: auth, rate limit, then send |
| Email proof | **Magic link** in email: user enters To → receives link → binds address (when Beta goes live) |
| Quota | Per-user daily cap (e.g. 1–2) |
| Secrets | Project Resend (or other) key **never** in plugin or vault |
| Privacy | Digest content transits project servers — disclose in UI |
| Outage | Self-send users unaffected; hosted users see clear failure |

Not goals for hosted v1: team broadcast, arbitrary multi-recipient, replacing self-send as default.

## Flow (normative)

```text
PipelineResult.kind === completed && digest present
  → if !auto-send enabled for active mode → stop
  → if delivered(date, to) already → skip
  → switch mode:
       自己发送 → Resend with user apiKey (+ optional custom From)
       官方代发 → POST project API with user auth + verified to
  → on success mark delivered(date, to, channelMeta)
  → on failure mark failed; never fail pipeline run
```

## Relationship to prior contracts

- Body: `email-content-format.md` (unchanged).  
- Self engineering: `email-delivery.md` (BYOK Resend).  
- This file: product dual-mode + cross-mode idempotency + hosted magic-link plan.  
- Supersedes any older “never project relay” wording: **relay is optional hosted mode**, not the default, and keys stay off-device.
