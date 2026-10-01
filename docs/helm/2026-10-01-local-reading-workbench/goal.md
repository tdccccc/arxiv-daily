# Local Markdown reading workbench

status: active
created: 2026-10-01T22:53:43+08:00
updated: 2026-10-01T22:53:43+08:00
revision: 1
owner: /root

## Intent

Give the independently running arXiv Daily product a browser reading surface opened from Claude Code CLI. Keep Markdown as the authoritative daily report and paper note; render it as readable HTML and keep agent interaction auxiliary.

## Success criteria

- [ ] A documented CLI / Claude plugin entry opens a loopback-only local workbench from the existing bundled executable.
- [ ] Users can browse and search existing daily reports and paper notes, open readable Markdown with headings, tables, code, images and scientific math, and navigate relative links and existing wikilinks.
- [ ] The workbench shows current non-secret configuration and can invoke the existing daily / manual-detail workflow with visible running and completion states, without duplicating the pipeline.
- [ ] Reading does not modify Markdown, indexes or source PDFs; unavailable documents, empty data and operation failures have usable states.
- [ ] HTTP, renderer and CLI contracts pass focused tests; the actual packaged UI passes browser acceptance including narrow-screen layout and source-link navigation.

## Non-goals

- A full Obsidian replacement: Markdown editing, graph view, community plugins, or annotation management.
- Replacing Markdown with persisted HTML, changing authoritative schemas, or implementing a second discovery engine.
- A desktop installer, hosted service, complete browser settings editor, or graphical personal-library review in this increment.

## Constraints

- Work only in `.worktree/claude-code-research-plugin`; preserve the original checkout and user configuration/data.
- Reuse the CLI configuration and the exact CLI bundle used by the Claude plugin; no automatic model calls on reading.
- Serve only configured output documents and their supported local assets; exclude secrets, internal indexes and paths outside allowed roots.
- Verify with isolated fixture configuration and data; do not call a real paid model or send email for acceptance.

## Phases

1. P1 — Safe local document and rendering APIs over existing product outputs — status: active
2. P2 — Packaged reading UI and Claude launch entry accepted in a real browser — status: pending
