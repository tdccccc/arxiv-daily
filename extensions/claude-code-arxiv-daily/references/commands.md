# Product command reference

The plugin bundles the exact arXiv Daily CLI product build, including native storage and its core dependencies:

```sh
node "${CLAUDE_PLUGIN_ROOT}/dist/arxiv-daily-cli.cjs" COMMAND [ARGUMENTS]
```

| Command | Behavior |
|---|---|
| `init` | Human-operated terminal wizard for CLI TOML; needs an interactive terminal |
| `status` | Read-only JSON: paths, topics, categories, model readiness, paper count, recent run state; no credentials or endpoints |
| `ui [--port PORT] [--no-open]` | Local browser reader; opens the browser, prints the full private URL, and stays running until SIGINT/SIGTERM. Port 0 (default) selects an available loopback port |
| `run --today` | Full daily workflow for today in the configured timezone |
| `run --date YYYY-MM-DD` | Full daily workflow for the requested announcement date |
| `run --id ARXIV_ID [--date YYYY-MM-DD]` | Full detailed-note workflow, including Markdown and Paper Index updates |
| `papers [--query TEXT] [--offset N] [--limit N]` | JSON from the existing Paper Index; query uses shared Dashboard lexical search; limit 1–100, default 30 |
| `library connect PATH` | Connect the selected source folder without authorizing model processing |
| `library status` | Disclose current library scope, processing endpoints, authorization status/fingerprint |
| `library authorize --fingerprint HASH` / `library revoke` | Grant the exact reviewed scope or revoke it |
| `library prepare` | Prepare isolated pinned PDF/runtime packages in the product cache |
| `library scan` / `library index` | Reconcile the catalog, then build/reuse title-and-abstract indexes |
| `library propose` / `library directions` | Generate clustered proposals; inspect proposals, configured topics and acceptance receipts |
| `library confirm --candidate ID --proposal-revision N` | Confirm the displayed candidate through the existing coordinator |
| `library search --query TEXT [--mode hybrid\|lexical\|dense] [--limit N]` | Retrieve indexed title-and-abstract evidence; limit 1–50, default 10 |
| `library review --input REQUEST.json` | Edit a candidate or accept proposed topics using the displayed version |
| `schedule show` | Inspect the existing scheduling configuration |
| `schedule install` / `schedule uninstall` | Apply/remove the managed OS schedule, only when requested |
| `data export --out PATH.zip` / `data import PATH.zip [--yes]` | Existing Vault-data portability workflow, only when requested |
| `help` | Complete product command help |

Generation commands use progress messages and a final result line, with status 0 for completed/skipped/done/already-exists and awaiting-announcement, 1 for other pending/runtime failure, and 2 for configuration/argument failure. A background process ID is not a completed result. `status` and `papers` print JSON; they do not start generation or require a valid model key to inspect existing data.

Configuration is `$XDG_CONFIG_HOME/arxiv-daily/config.toml` (default `~/.config/arxiv-daily/config.toml`) on Linux/macOS, or `%APPDATA%/arxiv-daily/config.toml` on Windows. `vault_root` selects the output folder. Settings are not discovered from the current directory and are not automatically copied from Obsidian. The old prototype's `--workspace`, `save`, `read`, filename-inventory `library` interface, and `confirm-direction` command have been removed. The new `library` command group invokes the shared product workflow.

Daily reports, detailed notes, Paper Index, run state and checkpoints follow the normal product output layout. Do not directly edit internal JSON, generate replacement reports in chat, or migrate old `arxiv-daily-agent/` records into it. Read the CLI exit status and preserve existing user-authored notes.

## Reviewing proposed topics

Read `library directions` first. Its output contains `proposal`, `topics` and `acceptances`. Candidates are in `proposal.topics[].directions`; revision values are numbers.

| `operation` | Other required fields |
|---|---|
| `update-candidate` | `candidateId`, `expectedProposalRevision`, `patch`; optional `representativePaperKeys` |
| `accept-topics` | `topicIds`, `expectedProposalRevision`; optional `candidateIds` |

`patch` supports `text` and `discoveryCues`. After the user confirms displayed topics:

```json
{
  "operation": "accept-topics",
  "topicIds": ["ID_FROM_PROPOSAL"],
  "expectedProposalRevision": 3
}
```

Acceptance atomically saves normal research topics and acceptance receipts in the CLI TOML. Repeated acceptance does not recreate a direction the researcher subsequently removed. A stale proposal or config is an error: refresh and reconcile. Accepted directions participate in ordinary topic filtering; later library authorization changes do not erase accepted settings.

The independent interest profile and its enable/disable/lock/unlock/apply/dismiss operations are retired. Edit accepted research directions in topic settings. `library update` is retired; scan, index and propose to prepare new suggestions for review.
