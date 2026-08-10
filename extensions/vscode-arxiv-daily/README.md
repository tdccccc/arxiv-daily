# arXiv Daily Companion for VS Code

Lightweight VS Code companion extension for an existing arXiv Daily vault.

This extension is not a standalone app and does not replace Obsidian, Zotero, or
a PDF reader. VS Code provides the markdown editor, preview, file navigation,
terminal, and command palette; the extension provides arXiv Daily workspace
detection, a reading Dashboard, and current CLI pipeline commands.

## Setup

Install the `arxiv-daily` CLI and run this once before using the companion's
pipeline commands:

```bash
arxiv-daily init
```

Complete the CLI prompts, including its API key and vault location. The CLI TOML
configuration is the only pipeline configuration used by the companion, and its
`vault_root` value is authoritative. Pipeline commands do not require an open
arXiv Daily workspace and do not use the terminal working directory to select a
vault. Open the configured vault as a workspace only when using the companion's
Dashboard and Paper Index editing features.

Set `arxivDaily.cliPath` only when the `arxiv-daily` executable is not available
on the VS Code process path. The companion launches this executable as a VS Code
process task with a separate argv array, without serializing the command through
a shell. A pipeline command reports success only after the CLI process exits with
code 0.

## Current Scope

- Workspace detection for a folder containing `arxiv-daily/`.
- A `workspace.fs`-backed adapter for the shared Paper Index.
- A Webview Dashboard with tabs, search, status/priority filters, resource
  opening, and single-paper status updates.
- `arXiv Daily: Run for Today`, which runs exactly `arxiv-daily run --today` as
  a VS Code process task.
- `arXiv Daily: Summarize by arXiv ID`, which canonicalizes the entered ID and
  runs `arxiv-daily run --id <canonical id>` as a VS Code process task.
- Local scaffold checks through `npm run build`, `npm test`, and `npm run smoke`.

## Planned Scope

- Reuse the host-neutral core storage and Dashboard model from the Obsidian
  plugin.
- Replace the local Dashboard model mirror with a directly shared package once
  the Obsidian plugin core is published in an importable form.
- Replace the CLI process bridge with direct core calls once the extension has
  the same bundled runtime as the Obsidian plugin.
- Launch generated markdown files through VS Code's native editor and preview.

## Release Strategy

The VS Code extension uses an independent version sequence and VSIX artifact.
It is intentionally decoupled from the Obsidian plugin tag series.

```bash
cd extensions/vscode-arxiv-daily
npm run build
npm test
npm run smoke
npm run vsix:package
```

`npm run vsix:package` uses `npx @vscode/vsce package` and writes the VSIX to
`dist/`. Install dependencies or allow `npx` to fetch `@vscode/vsce` before
running that packaging command.
