## 2026-07-25 — note

- evidence: Product grill (session) locked literature-assistant direction while keeping arXiv daily excellence; email wanted for mobile-readable digest + vault nudge; first structural cut is paper identity + source boundary, not SMTP.
- change: Created initiative `docs/helm/2026-07-25-paper-identity-source-boundary/` with goal.md, P1 phase plan, and this journal. Decisions captured below and mirrored in goal Intent / Constraints / Phases.
- disposition: Keep PLAN.md product narrative as-is until P1–P2 code lands; optional ADR 0002 for durable architecture record. No application code changed yet.
- next: Grill email first-delivery host; fold result into goal P4 + journal (and ADR if warranted). Then execute P1 PaperKey/index work when user asks to implement.

### Grill decisions (2026-07-25)

| Topic | Decision |
|---|---|
| Product identity (6–12 mo) | Research literature assistant; arXiv daily still must be excellent |
| Email motivation | Mobile-readable structured summary + remind user back to vault |
| Email architecture | Abstract delivery: DailyDigest → renderer → channels; host binding later |
| First structural cut | Paper identity + Source boundary (not email UI first) |
| Primary key | Composite `source:externalId` (e.g. `arxiv:2606.12345`) |
| Cross-source duplicates | No merge in v1; future alias/link only |
| Index migration | Read-compat + write-upgrade |
| Disk paths | Keep short external ids; index holds paperKey |
| Adapter depth | Discovery listing + full-text fetch |
| Full-text shape | Normalized `PaperContent` |
| Boundary proof | Fake/fixture second source in tests |
| Phase-1 delivery slice | Contract + arXiv re-home + Fake green tests; no mail send UI |

### Still open after initial grill

- ~~Which host sends email first~~ → resolved in email host grill below.
- paperKey literal normalization details.
- Real second source backlog pick.
- First HTTPS mail provider + settings fields.

## 2026-07-25 — note (email first-delivery host grill)

- evidence: Continued grill after helm frame. User chose dual-host channel interface; HTTPS mail API only; auto-send only on pipeline `completed`; body = daily report content reformatted for email (HTML + text, not Obsidian Markdown); single personal inbox; default no resend on force re-run; delivery state in vault `delivery-state.json`.
- change: Updated goal success criterion (email decisions recorded), Non-goals (SMTP/obsidian-md body), P4 outcome, Open questions. ADR 0002 delivery section filled in. PLAN not rewritten yet.
- disposition: Keep P1–P3 code focused on paper identity/source; P4 implements digest + channel. No send code until P4 starts.
- next: Optional provider pick grill later; execute P1 PaperKey when user asks to implement.

### Email grill decisions (2026-07-25)

| Topic | Decision |
|---|---|
| Host strategy | **Dual host**, same `DeliveryChannel` interface (CLI + Obsidian) |
| Transport | **HTTPS mail API only** (Resend/SendGrid/Mailgun-class); no native SMTP in v1 |
| Trigger | Auto-send **only** when pipeline date/run is **`completed`** |
| Body | **Full daily digest** reformatted for mail: HTML primary + text alternative; topic groups + structured fields + links + vault nudge — **not** raw plugin Markdown/wikilinks |
| Recipients | **Single personal To** address |
| Re-run policy | **Idempotent skip** if already delivered for `date + recipient + channel`; resend only after explicit clear/force later |
| Delivery state | Vault `arxiv-daily/.index/delivery-state.json` (alongside run-state), shared by both hosts |
| Failure isolation | Delivery failure must **not** mark pipeline run failed |
| Out of scope v1 | Multi-recipient, SMTP, in-email mutations, failure-alert emails as separate product |

### Email body note (user clarification)

Obsidian daily Markdown (wikilinks, callouts, vault-relative links) is a poor mobile email body. Renderer must produce **email-safe HTML** (and plain text) from the same structured daily result / `DailyDigest`, not forward the `.md` file bytes.

## 2026-07-25 — note (email content format grill)

- evidence: User wants the **daily report** in mail, reformatted for clients that cannot render plugin Markdown. Grilled data source, per-paper fields, math, layout, links, empty topics, subject, header/footer.
- change: Added `email-content-format.md` content contract; ADR 0002 body subsection; this journal table. Goal P4 still owns implementation timing.
- disposition: No renderer code yet. P1–P3 unchanged.
- next: Execute P1 when asked; P4 implements this contract.

### Email content format decisions (2026-07-25)

| Topic | Decision |
|---|---|
| Render from | Structured **`DailyDigest`** (not parse daily `.md`) |
| Per-paper body | **Isomorphic with vault**: title, source sections line, authors, **arXiv abs + PDF**, five fields |
| Fallback papers | Include with unavailable-summary treatment + abstract when present |
| Empty topics | **Still show** topic + “今日无相关…” |
| Not-selected list | **Omit** from email |
| Math | Keep raw `$...$` / `\(...\)` text; no image math in v1 |
| HTML style | Clean single-column academic; system fonts; email-safe markup |
| Multipart | HTML primary + plain-text alternative |
| Subject | `arXiv Daily YYYY-MM-DD · N 篇` (en variant when language is en) |
| Header | Date, count `N`, categories |
| Footer | Vault path to daily Markdown; **no** generation metrics |
| Links | Absolute https only in paper blocks; vault path only in footer as text |
| Zero papers (`N = 0`) | **Still send** on completed: subject `· 0 篇`; body states **今日无相关论文** (plus empty-topic lines as needed) |

Full contract: `./email-content-format.md`.

## 2026-07-25 — note (zero-day email)

- evidence: User confirmed zero-selection completed days still email: body「今日无相关论文」, subject `arXiv Daily YYYY-MM-DD · 0 篇`.
- change: Updated `email-content-format.md` zero-day row and subject notes; this journal row.
- disposition: Content contract only; no code.
- next: Grill A1–A4 identity details before P1 execute.

## 2026-07-25 — note (P1 identity A1–A4 grill)

- evidence: User locked paperKey literal, schema 4, entry field shape, P1 code scope.
- change: P1 phase plan locked-decisions table; goal open questions resolved section; ADR 0002 key/schema bullets tightened.
- disposition: **P1 ready to implement** from a product/architecture view. Email B-series still open but does not block P1–P3.
- next: Optional B-series grill before P4; or start executing P1 on user request.

### Identity A-series decisions

| ID | Topic | Decision |
|---|---|---|
| A1 | paperKey literal | `source:externalId`, source **lowercase** `[a-z0-9_]+`, arXiv id via modern normalization |
| A2 | schema | **4**; read ≤3 upgrade bare keys; write 4 only |
| A3 | entry fields | `paperKey` + `source` + `externalId` + keep **`arxivId`** alias for arXiv |
| A4 | P1 scope | core index + compat store APIs + tests; Dashboard only if compile breaks |

### Still open (do not block P1)

| ID | Topic | When |
|---|---|---|
| ~~B1–B6~~ | Email engineering | **Resolved** — see entry below + `email-delivery.md` |
| C* | HTML escape, et al., PDF URL helper, delivery-state field names | implementer defaults |
| D* | Real second source, merge, agent/MCP, SMTP, multi-recipient | later initiatives |

## 2026-07-25 — note (email B-series grill complete)

- evidence: User grilled B1–B6 in one pass. B3 refined: in-run limited retries with backoff, then `failed`; later `completed` may send again unless `delivered` (idempotent skip only on success).
- change: Added `email-delivery.md`; updated goal open questions / P4; zero-day structure locked in content format; ADR 0002 delivery bullets aligned.
- disposition: **Product/architecture grill for this initiative is complete enough to implement P1.** P2–P3 follow goal; P4 implements both email contracts. No application code yet.
- next: Execute P1 on user request.

### Email B-series decisions

| ID | Topic | Decision |
|---|---|---|
| B1 | Provider | **Resend** first; fixed settings fields (not generic HTTP template in v1) |
| B2 | Secrets | To/From (name) in settings; **API key via secret port** / CLI env |
| B3 | Failure | **2–3 attempts with backoff** in one send sequence → `failed`; pipeline unaffected; **later completed retries** if not `delivered` |
| B4 | Dual host | **delivery-state idempotency only** (no host mutex setting) |
| B5 | Manual MVP | **Auto-send + test send**; no full preview pane required |
| B6 | Zero body | Lead **「今日无相关论文」** + **each topic empty line** |

Contracts: `./email-content-format.md`, `./email-delivery.md`.

## 2026-07-26 — note (P1 complete)

- evidence: Implemented in worktree `.worktree/paper-identity-source-boundary` on branch `feat/paper-identity-source-boundary`. Added `paper-key.ts`; schema 4 load/save; store lookup accepts paperKey or bare arXiv id; history-sync/daily-selection updated. Verification: core tests 745 passed; core typecheck green; boundaries OK.
- change: P1 phase → done; goal Current focus → P2 (pending, not started). Checked success criteria for paperKey keys + short paths.
- disposition: Changes live only in worktree branch; uncommitted. Prefer this worktree as source of truth vs main-tree untracked helm docs.
- next: Explicit user go before P2 (SourceAdapter + PaperContent + arXiv re-home).

## 2026-07-26 — note (P2 complete)

- evidence: Added `packages/core/src/sources/` (`SourceAdapter`, `NormalizedPaperContent`, `ArxivSourceAdapter`). Pipeline `listForDate` + content fetch go through adapter (default arXiv from fetcher/paperFetcher). Legacy extractor DTO mapped via `mapLegacyPaperContent` / `legacyContentFromNormalized`. Tests: core 753 passed; typecheck green; boundaries OK.
- change: P2 → done; Current focus → P3 (Fake second source). Goal success criteria for adapter + arXiv re-home checked.
- disposition: Uncommitted on `feat/paper-identity-source-boundary` worktree. Manual-fetch still uses PaperContentFetcher directly (acceptable for P2; can route later).
- next: P3 FakeSource green-tests filter → summarize → index when user says go.

## 2026-07-26 — L2 reshape (park second source; email → P3)

- evidence: User decided not to pursue a real second literature source for now (paid full-text sites low value; ADS at most later as enhancement, not daily crawl). Asked to promote email implementation ahead of Fake-source green-tests.
- change: **L2** — Intent narrowed to arXiv identity/adapter foundation + email next. Success criteria: removed Fake-source as initiative gate; email implementation is now the open criterion. Phases: former P3 Fake **parked**; former P4 email **promoted to P3**. Added `phases/03-email-delivery.md` (pending). ADR 0002 notes updated for phase renumber and parked fake-proof.
- disposition: Keep P1–P2 code. Do not implement FakeSource unless later revived. Next code work = email when user says implement.
- next: Current focus P3 email (docs ready; wait for explicit implement go).

## 2026-07-26 — note (P3 complete)

- evidence: Implemented host-neutral `packages/core/src/delivery/` (DailyDigest, email render, delivery-state, Resend, orchestrate). Pipeline Option A: `completed.digest` from slots / zero-paper topics. Scheduler `onDailyCompleted` + CLI/plugin auto-send after completed; failures isolated. Settings `email.*`; CLI env `ARXIV_DAILY_RESEND_API_KEY` / `ARXIV_DAILY_EMAIL_*`. Test send: plugin settings + command, CLI `email-test`. Verification: core delivery 11 tests; CLI 26 passed; core+cli typecheck; plugin tsc; boundaries OK.
- change: P3 phase → done; goal email success criterion checked; Current focus → initiative planned phases complete.
- disposition: Uncommitted on `feat/paper-identity-source-boundary` worktree only. Caches / SourceAdapter untouched except pipeline digest + host hooks.
- next: Manual test-send to personal inbox when Resend key available; optional Fake-source hygiene later (parked).

## 2026-07-26 — note (email quick-setup UX)

- evidence: User rejected project-hosted relay (ties product to operator server). Chose BYOK Resend with simplified UX: To + API key → test send → then enable daily auto-send; empty From uses onboarding@resend.dev.
- change: Core `resolveResendFromEmail` / credentials helpers; defaults fromName; settings tab reordered with wizard copy; email-delivery.md quick-setup table; force test-send no longer requires enabled or custom From.
- disposition: Still uncommitted on feat branch with P1–P3.
- next: Commit full branch after tests green; rebuild plugin for vault hand-test.

## 2026-07-26 — note (email docs: personal-only BYOK)

- evidence: User confirmed Resend test path works only for account email (GitHub primary). Chose to stay on BYOK Resend; document setup and stress personal-only To; no MCP/webhook required for v1.
- change: getting-started EN/中文 §8; settings tab copy; email-delivery.md audience note.
- disposition: Commit as docs+copy follow-up on feature branch.
- next: Optional rebuild plugin for vault copy of settings strings.

## 2026-07-27 — note (email dual-mode docs alignment)

- evidence: User adopted dual-mode product (自己发送 default + 官方代发 optional). Chose docs+naming only for now; cross-mode idempotency date+to; hosted verification = magic link.
- change: Added `email-dual-mode.md`; amended email-delivery B4/state; goal resolved dual-mode; getting-started EN/中文 mode table; settings Delivery mode dropdown (hosted snaps back to self + Notice).
- disposition: No hosted API. Runtime `shouldSend` still channel-keyed until hosted ships — single live channel today so behavior matches “one success per day” in practice.
- next: Implement 官方代发 in a dedicated initiative when domain+Worker ready; optional tighten shouldSend to date+to before then.
