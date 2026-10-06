# DSH research workbench plugin

status: done
created: 2026-10-02T14:56:36+08:00
updated: 2026-10-02T22:36:21+08:00
revision: 6
owner: /root

## Intent

Make the existing research workbench an installable DSH capability with a visible literature entry and a plugin-managed lifetime. Keep paper workflows and data owned by the existing CLI/core so a future Claude Mod can reuse them.

## Success criteria

- [x] A distributable DSH bundle registers a literature button; clicking it opens the workbench in the right Sidebar without a model turn.
- [x] The plugin owns one lazy workbench service, deduplicates concurrent opens, reports setup/start errors, and stops it when unloaded.
- [x] Existing calendar, list, Markdown, persistent marks and explicit generation work through the original bundled CLI and configuration.
- [x] Authenticated DSH RPC controls startup; the browser receives only the capability URL. Exact loopback embedding is opt-in; standalone protection remains unchanged.
- [x] Contract, process, package and actual installed DSH Host checks pass with isolated configuration/data and no paid calls.

- [x] A fixed entry above Settings opens the workbench without a session; the right Sidebar guide provides the same workbench as a dedicated tab.

## Non-goals

- Claude Mod implementation, a second paper store, new LLM pipelines, GUI onboarding/configuration editing, public package publication.
- Replacing the accepted workbench layout or requiring changes to the user's live DSH profile.

## Constraints

- Existing claude-code-research-plugin worktree only; leave the original checkout and real DSH/config/Vault data alone.
- DSH-specific integration stays in extensions/dsh-arxiv-daily; CLI changes are host-neutral and tested.
- Use observed installed DSH contracts: initial Host checks used 0.1.7-alpha.1; P3 uses the now-installed 0.2.0-rc.2. Avoid claiming untested versions.
- User requested no Computer Use. Use HTTP/DOM and real Host tests; do not automate their desktop.

## Phases

1. P1 — Installable literature entry and managed workbench in DSH — status: superseded
2. P2 — Coexist with the DSH gateway during startup and plugin activation — status: done
3. P3 — Global literature entry and dedicated right Sidebar tab — status: done
