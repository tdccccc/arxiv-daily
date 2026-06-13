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
- Local scaffold checks through `npm run build` and `npm test`.

## Planned Scope

- Reuse the host-neutral core storage and Dashboard model from the Obsidian
  plugin.
- Render the Reading Dashboard in a Webview.
- Launch generated markdown files through VS Code's native editor and preview.

## Release Strategy

The VS Code extension uses an independent version sequence and VSIX artifact.
It is intentionally decoupled from the Obsidian plugin tag series.

```bash
cd extensions/vscode-arxiv-daily
npm run build
npm test
npm run vsix:package
```

`npm run vsix:package` uses `npx @vscode/vsce package` and writes the VSIX to
`dist/`. Install dependencies or allow `npx` to fetch `@vscode/vsce` before
running that packaging command.
