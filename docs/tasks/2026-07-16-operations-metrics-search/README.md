# Operations, Metrics, Secret Safety, and Local Retrieval

## Status

Completed and merged into the release preparation baseline.

## Context

The TypeScript core workspace is now the single business implementation used by the Obsidian plugin and Node CLI. The next product iteration should improve operational control, cost visibility, secret handling, and local literature discovery without adding a daemon, protocol, database, or second business core.

## Goals

1. Cancel all user-visible long-running tasks: daily pipelines, manual detail summaries, and PDF downloads.
2. Append folded generation metrics to daily reports and detail summaries, including elapsed time and provider-reported token usage.
3. Keep complete API keys out of the settings DOM, logs, diagnostics, URLs, and copied error text while preserving the existing local settings schema.
4. Use one local BM25-style index for relevance-ranked Dashboard search and similar-paper discovery.

## User contracts

- The existing `cancel-current-run` command ID remains stable, while its visible scope expands to all active long tasks.
- `Get Models` remains a short, local UI operation and is not included in global cancellation.
- Obsidian `requestUrl` cancellation remains cooperative: cancellation stops later work but cannot prove that an already-issued remote request stopped or ceased billing.
- Metrics report only usage returned by the configured provider; missing usage is shown as unavailable or incomplete, never as zero.
- The API key remains compatible with the existing `llm.apiKey` field in plugin `data.json` and CLI config/environment variables.
- Search and similar-paper ranking are local, deterministic, and make no network requests.
- Existing Paper Index schema versions and existing Markdown files remain readable without migration.

## Non-goals

- System keyring integration.
- Physical cancellation of Obsidian `requestUrl` connections.
- Incremental token streaming UI.
- Embeddings, vector storage, SQLite, or a search daemon.
- Cost estimation based on model pricing.
- Changes to the VS Code extension.

## Deliverables

- Core operation registry and signal propagation.
- Retry-aware LLM usage/timing metrics and Markdown callouts.
- Core redaction utilities plus safe settings UI behavior.
- Host-neutral BM25 paper index, relevance sorting, and Similar Papers modal.
- Focused regression tests, full workspace validation, and isolated Obsidian verification.
- An execution record in `execution.md`.

## Completion criteria

- [x] All four feature areas are implemented.
- [x] Existing settings, Paper Index, Markdown, CLI, command IDs, and release assets remain compatible.
- [x] API keys are absent from rendered settings inputs, logs, diagnostics, and copied reports.
- [x] Provider-omitted usage is handled without breaking report generation.
- [x] All selected long tasks register and respond to cancellation.
- [x] Search and similar-paper retrieval share one local Core index.
- [x] Full automated gates pass with zero known dependency vulnerabilities.
- [x] The build loads in the isolated `plugin_test` Vault with no captured Obsidian errors.
