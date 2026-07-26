# Paper identity and source boundary

status: active
updated: 2026-07-27

## Intent

Give arXiv Daily a stable paper identity and a source-adapter boundary so the pipeline is not permanently hard-coded to bare arXiv IDs, then ship **email delivery of the daily digest**. Email product is **dual-mode**: default **自己发送** (user Resend BYOK, shipped); optional **官方代发** (project relay, planned). Real multi-source discovery remains **parked**.

## Success criteria

- [x] Paper index keys are `source:externalId` (e.g. `arxiv:2606.12345`); schema read-path accepts legacy bare arXiv IDs and write-path upgrades them.
- [x] Disk note/PDF paths stay short source-local IDs (e.g. `papers/2606.12345.md`); index stores `paperKey` separately from path stems.
- [x] `SourceAdapter` covers discovery listing and full-text fetch; adapters return normalized content (abstract, optional sections, fallback text, quality, canonical URL).
- [x] Existing arXiv pipeline is re-homed behind that adapter without user-visible path or Dashboard regression for arXiv-only vaults.
- [x] Email/delivery product decisions are recorded (`email-content-format.md`, `email-delivery.md`).
- [x] DailyDigest + DeliveryChannel land; both CLI and Obsidian can send one personal Resend HTTPS email on pipeline `completed`, with vault `delivery-state` idempotency and test-send (see P3).

## Non-goals

- **Real second literature source** (Semantic Scholar, OpenAlex, NASA ADS, RSS, paid publishers, …) in this initiative.
- **Fake/fixture second source** green-tests in this initiative (parked; optional later engineering hygiene, not required to ship email).
- Cross-source paper merging / DOI unification as primary keys.
- Native SMTP inside Obsidian; plugin mail uses HTTPS provider APIs only.
- In-email Star / index mutation, team broadcast / multi-recipient workflows.
- Vector DB, plugin-local citation/Zotero manager, or a chat-shell agent UI.
- Renaming on-disk paper/PDF files to include the `source:` prefix.
- Embedding raw Obsidian wikilink Markdown as the email body (email uses HTML + text alternative).
- Paid full-text scraping.

## Constraints

- Keep `packages/core` host-neutral (no Node/Obsidian/process/Buffer); follow ADR 0001 ports.
- Obsidian remains first host; CLI remains one-shot companion.
- Prefer read-compat + write-upgrade over a mandatory offline migration command.
- Do not break existing daily Markdown, starred state, or run-state for arXiv users.
- Pipeline completed vs delivery status must stay separable.
- Local HTML/abs/TeX caches via `PaperContentFetcher` remain; adapter must not bypass them.
- Helm: thin goal; detail only the active phase; no commit/push unless asked.

## Phases

1. P1 — PaperKey model and index read-compat / write-upgrade are specified and landed for arXiv keys — status: done
2. P2 — SourceAdapter + normalized content exist; arXiv is the first implementation used by the pipeline — status: done
3. P3 — DailyDigest + email contracts implemented: Resend HTTPS, dual host, completed auto-send + test send, delivery-state idempotency — status: done
4. ~~P4 — (old) email~~ — **superseded**: email promoted to P3 (2026-07-26)
5. ~~Former P3 Fake second source~~ — **parked / out of this initiative** (2026-07-26)

## Current focus

P1–P3 code complete (**自己发送**). Docs aligned for dual-mode product (2026-07-27). **官方代发** not implemented — separate future work. Optional later: Fake-source hygiene, real multi-source, agent tooling.

## Open questions

- Hosted service host (Worker/Vercel/…), quotas, and ops when 官方代发 is built.
- Agent tool surface later: CLI subcommands vs MCP (out of this initiative unless promoted).

### Resolved (identity — P1)

- paperKey: `source:externalId`, source lowercase.
- Schema 3 → 4 on load/save upgrade.
- Entry: paperKey + source + externalId + arxivId alias.
- P1 scope: core index + compat APIs + tests.

### Resolved (source boundary — P2)

- SourceAdapter + ArxivSourceAdapter; pipeline re-homed; caches still behind PaperContentFetcher.

### Resolved (email product/engineering — P3)

- B1 Resend; B2 To/From settings + API key secret; B3 in-run 2–3 retries then failed, retry on later completed if not delivered; B4 idempotency; B5 auto + test send; B6 zero-day lead + per-topic empties.
- Contracts: `email-content-format.md`, `email-delivery.md`, **`email-dual-mode.md`**.
- Resend API key: settings `email.apiKey`; CLI env `ARXIV_DAILY_RESEND_API_KEY` (also `ARXIV_DAILY_EMAIL_*` for enable/to/from).
- Option A: `PipelineResult.completed.digest?: DailyDigest` built in `runForDateInner`; repair-only skips email.

### Resolved (email dual-mode — 2026-07-27)

- Modes: **自己发送** (default, shipped) + **官方代发** (optional, planned).
- Docs-only alignment now; no hosted implementation in this pass.
- Idempotency: **date + recipient** success once **across modes**.
- 官方代发 verification: **magic link** in email (when built).
- Only one mode active; project send keys never in plugin.

### Resolved (scope steer 2026-07-26)

- No real second source for now (including ADS as daily source).
- Fake second-source test phase dropped from this initiative’s critical path.
- Email implemented as P3 (自己发送).
