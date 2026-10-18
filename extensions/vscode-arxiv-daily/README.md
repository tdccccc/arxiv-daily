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
- The shared Paper Index, read and written through the same core
  `PaperIndexStore` and cross-process file lock that the Obsidian plugin and the
  CLI use, so a status change made here cannot overwrite a concurrent change from
  those hosts. The extension reads the vault as a local folder; virtual
  workspaces are not supported.
- A Webview Dashboard with tabs, search, status/priority filters, resource
  opening, and single-paper status updates. A `papers.json` that is damaged or
  written by a newer version is never overwritten; the Dashboard shows the error
  and the file stays as it is.
- `arXiv Daily: Run for Today`, which runs exactly `arxiv-daily run --today` as
  a VS Code process task.
- `arXiv Daily: Summarize by arXiv ID`, which canonicalizes the entered ID and
  runs `arxiv-daily run --id <canonical id>` as a VS Code process task.
- Local checks through `npm run build`, `npm test`, and `npm run smoke`.

## Planned Scope

- Reuse the host-neutral Dashboard model from the Obsidian plugin (the Paper
  Index storage already comes from core).
- Replace the local Dashboard model mirror with a directly shared package once
  the Obsidian plugin core is published in an importable form.
- Replace the CLI process bridge for the pipeline commands with direct core
  calls (the bundled core runtime is already in place for the Paper Index).
- Launch generated markdown files through VS Code's native editor and preview.

## Building

The extension ships as one bundle, `out/extension.js`, which inlines
`@arxiv-daily/core` and `@arxiv-daily/node-runtime` (only `vscode` stays
external). Those packages are TypeScript workspace sources, so install the
repository's root dependencies first; the build also uses the root `esbuild`:

```bash
npm ci                              # repository root
cd extensions/vscode-arxiv-daily
npm ci
npm run build                       # esbuild bundle + package checks
```

The tests run against the built bundle (`npm test` builds first). The bundle is
about 0.9 MB because it carries core.

## Release Strategy

The VS Code extension uses an independent version sequence and VSIX artifact.
It is intentionally decoupled from the Obsidian plugin tag series.

```bash
npm ci                              # repository root, once
cd extensions/vscode-arxiv-daily
npm ci
npm run build
npm test
npm run vsix:package
```

`npm run vsix:package` runs `vsce package`, which rebuilds the bundle through
the `vscode:prepublish` script and writes the VSIX to the current directory.
The VSIX contains `package.json`, this README and `out/extension.js`.
