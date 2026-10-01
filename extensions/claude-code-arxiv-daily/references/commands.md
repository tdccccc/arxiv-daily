# Product command reference

The plugin bundles the exact arXiv Daily CLI product build, including native storage and its core dependencies:

```sh
node "${CLAUDE_PLUGIN_ROOT}/dist/arxiv-daily-cli.cjs" COMMAND [ARGUMENTS]
```

| Command | Behavior |
|---|---|
| `init` | Human-operated terminal wizard for CLI TOML; needs an interactive terminal |
| `status` | Read-only JSON: paths, topics, categories, model readiness, paper count, recent run state; no credentials or endpoints |
| `run --today` | Full daily workflow for today in the configured timezone |
| `run --date YYYY-MM-DD` | Full daily workflow for the requested announcement date |
| `run --id ARXIV_ID [--date YYYY-MM-DD]` | Full detailed-note workflow, including Markdown and Paper Index updates |
| `papers [--query TEXT] [--offset N] [--limit N]` | JSON from the existing Paper Index; query uses shared Dashboard lexical search; limit 1–100, default 30 |
| `library connect PATH` | Connect the selected source folder without authorizing model processing |
| `library status` | Disclose current library scope, processing endpoints, authorization status/fingerprint |
| `library authorize --fingerprint HASH` / `library revoke` | Grant the exact reviewed scope or revoke it |
| `library prepare` | Prepare isolated pinned PDF/runtime packages in the product cache |
| `library scan` / `library index` | Reconcile the catalog, then build/reuse full-text indexes and generate applicable incremental suggestions |
| `library propose` / `library directions` | Generate clustered proposals; inspect proposals, confirmed directions and suggestions |
| `library confirm --candidate ID --proposal-revision N --profile-revision N` | Confirm the displayed candidate through the existing coordinator |
| `library update` | Recompute incremental direction suggestions from the indexed library |
| `library search --query TEXT [--mode hybrid\|lexical\|dense] [--limit N]` | Retrieve full-text evidence; limit 1–50, default 10 |
| `library review --input REQUEST.json` | Revise a direction or review a suggestion using its displayed version |
| `schedule show` | Inspect the existing scheduling configuration |
| `schedule install` / `schedule uninstall` | Apply/remove the managed OS schedule, only when requested |
| `data export --out PATH.zip` / `data import PATH.zip [--yes]` | Existing Vault-data portability workflow, only when requested |
| `help` | Complete product command help |

Generation commands use progress messages and a final result line, with status 0 for completed/skipped/done/already-exists, 1 for pending/runtime failure, and 2 for configuration/argument failure. A background process ID is not a completed result. `status` and `papers` print JSON; they do not start generation or require a valid model key to inspect existing data.

Configuration is `$XDG_CONFIG_HOME/arxiv-daily/config.toml` (default `~/.config/arxiv-daily/config.toml`) on Linux/macOS, or `%APPDATA%/arxiv-daily/config.toml` on Windows. `vault_root` selects the output folder. Settings are not discovered from the current directory and are not automatically copied from Obsidian. The old prototype's `--workspace`, `save`, `read`, filename-inventory `library` interface, and `confirm-direction` command have been removed. The new `library` command group invokes the shared product workflow.

Daily reports, detailed notes, Paper Index, run state and checkpoints follow the normal product output layout. Do not directly edit internal JSON, generate replacement reports in chat, or migrate old `arxiv-daily-agent/` records into it. Read the CLI exit status and preserve existing user-authored notes.

## Reviewing directions and suggestions

Read `library directions` first. Its output has `proposal`, `profile`, `suggestions` and aligned `suggestionKeys`. Revision values are numbers. Supported JSON requests:

| `operation` | Other required fields |
|---|---|
| `update-candidate` | `candidateId`, `expectedProposalRevision`, `patch`; optional `representativePaperKeys` |
| `update-direction` | `directionId`, `expectedProfileRevision`, `patch`; optional `representativePaperKeys` |
| `enable`, `disable`, `lock`, `unlock` | `directionId`, `expectedProfileRevision` |
| `apply` | `key`, `expectedSuggestionsRevision`, `expectedProfileRevision`, `expectedProposalRevision` (null when no proposal exists) |
| `dismiss` | `key`, `expectedSuggestionsRevision` |

`patch` uses the existing direction fields: `name`, `description`, `discoveryCues`. For example, after the user asks to disable a displayed direction:

```json
{
  "operation": "disable",
  "directionId": "ID_FROM_PROFILE",
  "expectedProfileRevision": 3
}
```

A stale version is an error. Applying suggestions follows the original sequential persistence semantics; if consuming a suggestion fails after a profile/proposal commit, inspect the current records before retrying. Do not claim a cross-file transaction or overwrite files to hide the error.
