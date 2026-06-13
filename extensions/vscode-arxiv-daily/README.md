# arXiv Daily Companion for VS Code

Lightweight VS Code companion extension for an existing arXiv Daily vault.

This extension is not a standalone app and does not replace Obsidian, Zotero, or
a PDF reader. VS Code provides the markdown editor, preview, file navigation,
terminal, and command palette; this extension will provide arXiv Daily-specific
workspace detection, Dashboard, pipeline commands, and API-key storage.

## Current Scope

- Independent extension manifest under `extensions/vscode-arxiv-daily/`.
- Command IDs reserved for Dashboard, run, run-pending, summarize-by-ID, and API
  key configuration.
- Minimal activation entrypoint with placeholder command handlers.
- Workspace adapter detects a folder containing `arxiv-daily/` and exposes a
  `workspace.fs`-backed storage interface compatible with the shared core shape.
- Secret adapter stores the LLM API key through VS Code `SecretStorage`.
- Webview Dashboard reads `.index/papers.json` and supports tabs, search,
  status/priority filters, resource opening, and single-paper status updates.
- Pipeline commands run the arXiv Daily CLI in a VS Code terminal with the API
  key injected through terminal environment variables and relative link style
  forced for VS Code-friendly output.
- Local scaffold checks through `npm run build` and `npm test`.
- `npm run smoke` covers the companion workflow locally: open a vault, render the
  Dashboard, change a paper status, open a resource, and send a pipeline command
  to a terminal.

## Planned Scope

- Reuse the host-neutral core storage and Dashboard model from the Obsidian
  plugin.
- Replace the local Dashboard model mirror with a directly shared package once
  the Obsidian plugin core is published in an importable form.
- Replace the terminal CLI bridge with direct core calls once the extension has
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
