# ADR 0001: One TypeScript core with explicit hosts

Status: Accepted (2026-07-15)

Supersedes: the daemon-first roadmap formerly tracked in
`docs/tasks/2026-07-15-daemon-first-roadmap/` (deleted by user request).

## Decision

`packages/core` is the only business core. It depends on explicit ports for HTTP,
storage, markup parsing, progress, secrets, and resource opening, and contains no
Node, Obsidian, `process`, or `Buffer` dependencies. Obsidian composition supplies
the host `DOMParser`; `packages/node-runtime` supplies `linkedom` and the other
Node adapters. `apps/cli` is a one-shot command host. `plugin` contains the
Obsidian host and UI.

The repository does not introduce a wire protocol or long-running daemon. The
Obsidian plugin and CLI consume the TypeScript core directly. Personalization
work remains independent, and `extensions/vscode-arxiv-daily` is outside this decision.

## Consequences

The root npm workspace and lockfile are authoritative. Private workspace packages
publish source-backed root exports for repository builds; they do not claim a
standalone published `dist`. Consumers import only `@arxiv-daily/core` or
`@arxiv-daily/node-runtime` root entrypoints. Boundary checks reject reverse
dependencies, non-allowlisted core dependencies, deep workspace imports, and
legacy duplicate source directories. Obsidian release assets remain
`plugin/manifest.json`, `plugin/main.js`, and `plugin/styles.css`.
