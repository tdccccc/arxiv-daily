# ADR 0002: Paper identity and source boundary

Status: Accepted (2026-07-25); amended 2026-07-26 (email → P3, second source parked)

Related: `docs/helm/2026-07-25-paper-identity-source-boundary/`, ADR 0001 (TypeScript core and hosts).

## Context

The paper index and pipeline are arXiv-shaped: `papers.json` keys are bare modern arXiv IDs, `PaperIndexEntry.source` is the literal `"arxiv"`, and fetch/extract code lives in arXiv-named modules. The near-term product focus is arXiv daily excellence plus email digest delivery. Multi-source literature-assistant expansion remains optional later (with arXiv the depth flagship); real second sources are parked as of 2026-07-26. Hard-coding arXiv identity into every layer would force a second migration when a non-arXiv source lands.

## Decision

1. **Stable paper key** — Index map keys are `paperKey = lowercase(source) + ":" + externalId` (example: `arxiv:2606.12345`). `source` matches `[a-z0-9_]+`. arXiv `externalId` uses the same modern-id normalization as today. Cross-source “same paper” merging is out of scope; future work may add aliases, not a DOI-primary key in v1.

2. **Migration / schema** — Bump paper inbox schema to **4**. On load, accept schema ≤3 bare arXiv ID keys and normalize to `arxiv:<id>`. On save, write schema 4 and only normalized keys (read-compat + write-upgrade). No mandatory offline migrate command for the first landing.

3. **Entry fields** — Each entry stores `paperKey`, `source`, `externalId`. For arXiv, keep **`arxivId`** as a compatibility alias with `arxivId === externalId`.

4. **Paths** — Markdown notes and PDFs keep short source-local stems from `externalId` (e.g. `papers/2606.12345.md`). Path layout does not embed the `source:` prefix. The index remains the source of truth for `paperKey` ↔ paths.

5. **Source boundary** — Introduce a `SourceAdapter` port in core for:
   - discovery listing for a date / recent window;
   - full-text (or best-effort content) fetch.
   Adapters return a normalized **`PaperContent`**: abstract, optional sections, full-text fallback, content quality, canonical URL. arXiv HTML/source extraction becomes the first adapter implementation.

6. **Proof against fake abstraction** — Originally planned as a Fake/fixture second source in tests. **Parked (2026-07-26)** with real multi-source work; not required before email. A real second corpus source remains deferred.

7. **Delivery (product + engineering; implement as initiative **P3** after identity/adapter)**  
   - Project completed daily results to a host-agnostic **`DailyDigest`**, then **`DeliveryChannel`**.  
   - **Both** Obsidian and CLI hosts may register the same email channel interface.  
   - First provider: **Resend** over HTTPS only (no native SMTP). To/From in settings; **API key in secret port** / CLI env.  
   - Send **only** after pipeline **`completed`**. MVP also supports **test send**.  
   - Body contract: `email-content-format.md` (HTML+text from Digest; five fields; empty topics kept; not-selected omitted; TeX as text; subject `arXiv Daily YYYY-MM-DD · N 篇`; **N=0 still sends** with lead「今日无相关论文」+ per-topic empties; no metrics).  
   - Engineering contract: `email-delivery.md`.  
   - **Single** personal To.  
   - In-run: **2–3 attempts with backoff**, then record **`failed`**. Later **`completed`** may try again unless status is **`delivered`**. Force re-run does not resend when already delivered.  
   - Dual-host: **idempotency only** via vault **`arxiv-daily/.index/delivery-state.json`** (no host mutex). Delivery never fails the pipeline run.

## Consequences

- Schema version will bump when on-disk keys change shape; loaders must remain backward compatible with schema 3 bare-id indexes.
- Call sites that assume `papers[arxivId]` need paperKey helpers and temporary bare-id compatibility for UI/commands.
- New sources add adapters + key namespaces; they should not fork summarizer or Dashboard models.
- Email, CLI agent tools, and Obsidian UI should all consume the same `paperKey` / index / digest projections rather than parallel data models.
- Dual-host email implies shared digest rendering in core (or a shared package) and host-only secret + HTTP wiring; provider API keys use existing secret ports.
- A second delivery state file must stay coherent with multi-device vault sync the same way `run-state.json` does.

## Email dual-mode (2026-07-27)

Product: **自己发送** (user Resend BYOK, default, shipped) and **官方代发** (project relay after magic-link verification, planned). Shared digest + cross-mode idempotency on `date + recipient`. Details: `docs/helm/2026-07-25-paper-identity-source-boundary/email-dual-mode.md`.

## Non-decisions

- Exact Resend env var / settings field identifiers (implementer).
- Choice of first real non-arXiv source.
- Agent transport (CLI vs MCP).
- Full HTML preview UI (not MVP); explicit force-resend command beyond clearing failed/delivered state.
