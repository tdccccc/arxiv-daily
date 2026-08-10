# Domain glossary

Language for arXiv Daily. Implementation details live in code and ADRs, not here.

## Products and hosts

| Term | Meaning |
|---|---|
| **Plugin product** | The Obsidian-facing product: settings UI, Dashboard, in-app scheduler while Obsidian is open. |
| **CLI product** | The headless one-shot product for servers, cron, and long-running machines. Same pipeline engine; separate configuration and UX. |
| **Host** | A composition root that wires ports (HTTP, storage, etc.) and invokes core. Plugin and CLI are two hosts, not two business cores. |
| **Core** | Shared pipeline, digest, delivery, index, and validation logic used by both products. |

## Product identity

| Term | Meaning |
|---|---|
| **Personal research companion** | The primary product identity: a continuing assistant for an individual researcher that learns from and works with the researcher's accumulated literature context. Product trade-offs prioritize this user relationship over becoming a general-purpose research agent or developer-first infrastructure. |
| **Research context** | The researcher's accumulated papers, reports, interests, selections, and prior judgments that give continuity to future literature work. |
| **Research lifecycle** | The continuous journey from entering a research direction, building an initial body of literature, and then tracking and refining that direction over time. Newcomers and researchers with an existing library enter at different points in the same lifecycle rather than using separate product modes. |
| **Research understanding** | The companion's evolving, inspectable understanding of the researcher's questions, evidence, claims, uncertainties, and interests. Its purpose is to help the researcher's knowledge evolve as new evidence arrives, not merely to improve recommendations or maintain a hidden model profile. |
| **Knowledge evolution** | The researcher-controlled process by which new evidence supports, refines, challenges, or supersedes existing questions and claims in the knowledge base. Accumulating papers without revisiting prior understanding is not knowledge evolution. |
| **Research signal** | Researcher input that can refine library-guided discovery. In the first initiative, confirmation and correction of proposed directions and representative sets are the authoritative signals. Later reading dispositions may provide additional evidence without requiring full paper notes. |
| **Daily discovery** | The connection between the daily arXiv stream and the researcher's personal literature library: identify potentially valuable new papers, show which manual topic or confirmed library direction selected them, and explain what they appear to add relative to named prior works. |
| **Authoritative research record** | The researcher-owned, readable, editable, and portable local record of research context. If model profiles, embeddings, or indexes are introduced, they remain replaceable, non-authoritative projections rather than the only copy of durable research knowledge. |
| **Personal literature library** | A researcher-chosen local directory containing mostly PDFs and potentially Markdown or other files. It may live inside or outside the Vault; the product must not require migration into arXiv Daily's output layout. The product must not assume that every paper has a note or that the library is already well organized. By default, only files identifiable as papers are included; drafts, notes, and other files are ignored unless the researcher explicitly includes them. |
| **Library processing consent** | The researcher's informed authorization for eligible files from a chosen library to be processed through the named model endpoints (LLM and, when remote embedding is enabled, the embedding endpoint) at the disclosed depth (metadata and abstracts, or full text). The product asks again when an endpoint changes or processing expands from metadata and abstracts to full text. Unrelated files are not implicitly included, and the researcher may revoke authorization. |
| **Progressive library understanding** | The product becomes useful after the researcher selects a library directory and incrementally builds a basic literature catalog from paper-level metadata and abstracts. The researcher is not required to organize the library first. Future capabilities may deepen selected prior works on demand, but the first initiative does not. |
| **Representative set** | A small researcher-approved group of papers that expresses one currently relevant research direction. The product proposes candidate directions and representative papers from the library; the researcher adjusts them instead of selecting manually from the entire library. |
| **Confirmed interest profile** | A researcher-reviewed set of research directions derived from representative sets. Each direction names representative library papers and discovery cues; it becomes an input to daily filtering only after the researcher confirms, corrects, disables, or merges the inferred directions. |
| **Discovery source** | The explicit reason a new paper entered a daily report: a manually configured topic, a confirmed library-derived interest direction, or both. Daily discovery uses the union of these sources and shows the triggering source instead of presenting personalization as a black box. |
| **Personal novelty** | The new task, method, data, evidence, efficiency result, or counter-evidence that a discovered paper appears to add relative to a named representative set. It is expressed as a difference type with its comparison basis and evidence depth, not as an unexplained score. |
| **Library similarity** | Retrieval of the library papers most similar to a given paper or description, over the local full-text knowledge base. Every result carries a similarity score and best-passage evidence so the match is explainable. Distinct from lexical daily-report similarity, which searches papers that already appeared in daily reports. |
| **Reading candidate** | A discovered paper that the researcher chose to keep for later reading. It preserves identity, source, discovery reason, related prior works, and provisional novelty evidence without implying that the researcher has read or endorsed its generated summary. |
| **Direction review** | A periodic review of reading candidates grouped by confirmed research direction. Its primary outcome is a researcher-controlled reading decision: which candidates merit close reading, skimming, or dismissal, and why. |
| **Direction suggestion** | A machine-generated proposal (attach to an existing direction, new direction, split, or merge) about how library papers relate to confirmed directions. It enters a review queue and never takes effect without researcher review — it is a suggestion about the confirmed interest profile, not an automatic change to it. Un-reviewed suggestions may be superseded by newer evidence, with the replacement made visible. |
| **Reading disposition** | Lightweight researcher feedback after reading or reviewing a candidate: relevant or irrelevant, already known or genuinely additive, and keep or dismiss. An optional short judgment may accompany it; a full paper note is not required. |
| **New–prior relationship** | An evidence-bounded explanation of why a newly discovered paper matters in relation to works already in the personal literature library. The explanation states which available sources it used and must not imply knowledge unsupported by those sources. |

## Configuration

| Term | Meaning |
|---|---|
| **Product settings** | User choices that shape discovery and output: categories, topics, summary language, detail policy, email preferences, paths under the vault, LLM endpoint fields, embedding mode fields. |
| **Embedding mode** | Whether full-text chunks are embedded locally (offline, the default) or via a named remote embedding endpoint (fast; requires full-text processing consent; all chunks leave the machine). Chosen when a library is first prepared for indexing; switching modes rebuilds the knowledge base. |
| **Plugin settings store** | Where the plugin product persists product settings and secrets (Obsidian plugin data). Independent of the CLI product. |
| **CLI config** | The CLI product’s single configuration file (TOML). Holds product settings, deployment paths, and secrets for that machine. |
| **Init** | First-run interactive setup for the CLI product that writes CLI config. Required before other CLI commands succeed. |
| **Manual configuration** | Changing product settings by editing the product’s own store (plugin UI / data, or CLI TOML). No automatic cross-product settings sync. |

## Vault data

| Term | Meaning |
|---|---|
| **Vault** | The user-chosen notes root. Daily reports, paper notes, and indexes live here under the configured output layout. |
| **Vault data** | Generated and bookkeeping content under the vault: daily reports, paper notes, paper index, run state, delivery state. Not the CLI config file. |
| **Data export / import** | Manual packaging and restoration of vault data (not product settings) for backup or moving between machines. |
| **Delivery state** | Bookkeeping of whether a digest was already delivered for a date and recipient, shared when both products use the same vault. |
| **Filter checkpoint** | Internal Vault-data bookkeeping that records one validated paper-filter classification batch for a report date. Reuse requires an exact match to the complete rendered request and effective generation identity; it is not a partial daily report. |
| **Structured-summary checkpoint** | Internal Vault-data bookkeeping that records a completed per-paper result until its complete daily report is committed. Reuse requires an exact compatibility match and may depend on result kind; a checkpoint is not a partial daily report or a paper note. |
| **Daily-generation checkpoint** | Collective term for filter and structured-summary checkpoints that remain transient until a complete daily report becomes authoritative. |

## User-facing outputs (README / product copy)

| Term | Meaning |
|---|---|
| **Daily report** | The day’s Markdown file under `daily/` — multi-topic reading list with a short structured summary per selected paper. Prefer this over “daily summary” for the whole file. |
| **Paper note** | A longer per-paper Markdown file under `papers/`. Prefer this over “deep dive” in user-facing docs. Not the same object as a daily-report entry. Chinese user docs: **论文总结** (not 论文笔记). |
| **Structured summary** | The short fields for one paper *inside* a daily report (problem, method, result, …). Not a separate file. |

UI/settings may still say “detail note” or “Detail report”; new user docs should prefer **paper note**.

## Email

| Term | Meaning |
|---|---|
| **Digest** | Mobile-oriented projection of a completed daily run (not raw vault Markdown). |
| **Send yourself** | User-supplied mail provider credentials; project does not send on their behalf. |
| **Official delivery** | Project-operated send path after the user verifies an address; subject to shared capacity limits. |

## Non-terms (avoid)

- Do not call CLI config “plugin settings” or imply one store is shared.
- Do not call vault data “configuration sync.”
- Do not assume a long-running CLI daemon; scheduling for CLI is external (e.g. cron).
- Do not lead README with “not a Zotero replacement”; state what the product does first.
- Avoid “deep dive” as the primary name for paper notes in user-facing overview docs.
