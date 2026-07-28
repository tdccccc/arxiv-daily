# Domain glossary

Language for arXiv Daily. Implementation details live in code and ADRs, not here.

## Products and hosts

| Term | Meaning |
|---|---|
| **Plugin product** | The Obsidian-facing product: settings UI, Dashboard, in-app scheduler while Obsidian is open. |
| **CLI product** | The headless one-shot product for servers, cron, and long-running machines. Same pipeline engine; separate configuration and UX. |
| **Host** | A composition root that wires ports (HTTP, storage, etc.) and invokes core. Plugin and CLI are two hosts, not two business cores. |
| **Core** | Shared pipeline, digest, delivery, index, and validation logic used by both products. |

## Configuration

| Term | Meaning |
|---|---|
| **Product settings** | User choices that shape discovery and output: categories, topics, summary language, detail policy, email preferences, paths under the vault, LLM endpoint fields. |
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

## User-facing outputs (README / product copy)

| Term | Meaning |
|---|---|
| **Daily report** | The day’s Markdown file under `daily/` — multi-topic reading list with a short structured summary per selected paper. Prefer this over “daily summary” for the whole file. |
| **Paper note** | A longer per-paper Markdown file under `papers/`. Prefer this over “deep dive” in user-facing docs. Not the same object as a daily-report entry. |
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
