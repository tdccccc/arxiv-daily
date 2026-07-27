# Email content format (product contract)

status: accepted (grill 2026-07-25)
implements_in: P3 (DailyDigest + renderer + channel)
goal_ref: ./goal.md

This document freezes **what the email contains and how it is rendered**.
Transport/host decisions live in `journal.md` and ADR 0002; this file is the body contract only.

## Intent

Deliver the **same daily discovery content** as the vault daily report, reformatted so a phone mail client can read it — not a short nudge, and not a dump of Obsidian Markdown.

## Data source

| Decision | Detail |
|---|---|
| Source of truth | **Structured `DailyDigest`** projected at pipeline success from in-memory daily assembly results (topics, papers, structured/fallback fields, trusted links) |
| Not used as primary | Parsing `daily/YYYY-MM-DD.md` (wikilinks, callouts, metrics markers are host-specific) |
| Language | Follow `summaryLanguage` (`zh` / `en`) for labels and boilerplate, same as vault daily |

## MIME / parts

| Part | Role |
|---|---|
| `text/html` | Primary reading surface (mobile) |
| `text/plain` | Multipart alternative; same information hierarchy, no reliance on CSS |
| Not sent | Raw `.md` file bytes, Obsidian wikilinks as the only link form, external CSS/JS, remote webfonts |

## Subject

```text
arXiv Daily YYYY-MM-DD · N 篇
```

- `N` = count of papers included in the mail body (selected daily papers, including fallback slots), including **`0`**.
- Zero-day example: `arXiv Daily 2026-07-25 · 0 篇`.
- English locale may use an English equivalent (e.g. `arXiv Daily YYYY-MM-DD · N papers`) when `summaryLanguage` is `en`.
- Keep short for mobile list truncation; do not use the full vault H1 (category list) as Subject.

## HTML layout

**Style:** clean single-column academic.

- System / web-safe fonts; clear hierarchy: day title → topic heading → paper block.
- Prefer email-safe markup (simple structure, inline-friendly styles if any).
- No dark-theme “Obsidian card” chrome as default (client breakage risk).
- No external stylesheets or images required for baseline readability.

### Skeleton

1. **Header**
   - Date (and weekday optional)
   - Total related paper count `N`
   - Configured arXiv categories (or future source labels) summary line
2. **Body** — one section per configured topic (see empty topics)
3. **Footer**
   - Vault pointer: full Markdown daily lives at `arxiv-daily/daily/YYYY-MM-DD.md` (or configured `dailyDir`)
   - Optional one-line product name (“arXiv Daily”)
4. **Omit from v1 body**
   - Generation metrics (LLM calls, tokens, wall time)
   - Collapsed “not selected” paper lists from the vault daily

## Per-topic rules

| Case | Mail behavior |
|---|---|
| Topic has ≥1 selected paper | Render topic heading + each paper block |
| Topic has 0 selected papers | **Still render topic heading** + “今日无相关论文更新。” / English equivalent (mirror vault empty-topic line) |
| Not-selected / ignored lists | **Do not** include in email |
| Overall `N = 0` but pipeline `completed` | **Still send.** Subject: `arXiv Daily YYYY-MM-DD · 0 篇` (en: `· 0 papers`). Body: header with count 0; **lead line「今日无相关论文」** / English equivalent; **then each configured topic** with its empty-topic one-liner; footer vault path. Do not omit the zero-day signal. |

## Per-paper block (isomorphic with vault daily)

Aligned with vault fields from `daily-summary-rendering` / sample dailies:

1. **Title** (plain text heading; no vault detail wikilink)
2. **Source sections** — **omitted in email** (often very long section lists; vault daily still has them)
3. **Authors**
4. **Links:** `arXiv` abs **and** `PDF` as absolute `https://` URLs only  
   - arXiv: `https://arxiv.org/abs/<id>`  
   - PDF: `https://arxiv.org/pdf/<id>` (or existing trusted PDF URL helper)  
   - Do **not** put vault-relative note paths in the paper block (unopenable in mail)
5. **Five structured fields** (labels follow language):

| Key | zh | en |
|---|---|---|
| coreProblem | 研究问题 | Research problem |
| keyMethod | 方法设计 | Method design |
| mainResult | 核心结果 | Core results |
| whyRelevant | 研究价值 | Research value |
| limitations | 适用边界 | Scope and limits |

### Fallback papers

If structured summary is unavailable (vault fallback path):

- Keep title / authors / arXiv+PDF links
- Show the same class of warning as vault (“自动摘要不可用” / “Summary unavailable”)
- Include original abstract when available
- Still count toward `N`

## Math / scientific markup

| Decision | Detail |
|---|---|
| Email body | **Soft-normalize** for mail clients: strip `$...$` / `\(...\)` / `\[...\]` delimiters; simplify common macros (`\frac`, Greek, `\leq`, …) to Unicode or plain text via `emailProse` |
| Vault daily | Unchanged (still full scientific Markdown / MathJax-oriented) |
| Not in email | Rasterize formulas to images; full MathJax |

HTML still escapes markup-sensitive characters after prose normalization.

## Plain-text alternative

Mirror hierarchy with plain headings and bullet lines, for example:

```text
arXiv Daily 2026-07-23 · 3 篇
Categories: ...

## Topic name
### Paper title
Authors: ...
arXiv: https://arxiv.org/abs/...
PDF: https://arxiv.org/pdf/...
- 研究问题: ...
- 方法设计: ...
...
```

Empty topics keep the empty-day one-liner.

## Security / trust

- Titles, authors, field prose come from pipeline-trusted assembly data (same trust model as daily Markdown writer).
- URLs for arXiv/PDF from trusted ID helpers only — never from model-emitted freeform URLs.
- HTML-escape all interpolated prose; neutralize raw HTML the same class of care as vault line normalization.

## Explicit non-goals (content)

- Email as interactive triage (star / checkbox)
- Embedding PDF binaries
- Matching Obsidian callout / generation-metrics UX
- Perfect visual parity with every mail client beyond single-column readable HTML

## Implementation notes for P4

- Build `DailyDigest` at completed assembly time (or immediately before write), then `renderEmailHtml(digest)` / `renderEmailText(digest)`.
- Share field labels with `DAILY_SUMMARY_FIELD_LABELS` where possible to avoid label drift.
- Subject and empty-topic strings need zh/en pairs next to existing daily copy helpers.
- Idempotency and send host behavior: see journal email grill + ADR 0002; not redefined here.
