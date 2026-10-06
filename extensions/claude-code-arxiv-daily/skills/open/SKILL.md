---
name: open
description: Open the local arXiv Daily reading workbench in a browser to browse daily reports, read Markdown paper notes and math, follow paper links, or start the existing generation workflow.
argument-hint: "[打开阅读工作台]"
---

# Open the reading workbench

Use the bundled product executable: `${CLAUDE_PLUGIN_ROOT}/dist/arxiv-daily-cli.cjs`.

1. If this session already has a running workbench, reuse its displayed URL instead of starting another server.
2. Run `node "${CLAUDE_PLUGIN_ROOT}/dist/arxiv-daily-cli.cjs" ui` using the Bash tool's `run_in_background: true` option. This is a long-lived local server: do not wait for process exit to decide whether startup succeeded.
3. Read its output. Only report that it is ready after the `Workbench: http://127.0.0.1:PORT/TOKEN/` line appears, and provide that exact URL. The command attempts to open the default browser automatically; if that fails, the URL still works. Do not fabricate a URL or omit its path.
4. If background execution is unavailable, give the same command for the user's terminal. For a terminal without a local browser, use `ui --no-open`; remote-machine port forwarding requires an explicit fixed `--port PORT` and user network setup.
5. If config is missing, guide the user through `node "${CLAUDE_PLUGIN_ROOT}/dist/arxiv-daily-cli.cjs" init` in their interactive terminal. Never ask them to paste API keys into chat. Preserve the same XDG_CONFIG_HOME / APPDATA environment for setup and launch. Working directory does not choose the output folder.

The workbench reads existing Markdown from the configured daily/paper directories. It renders headings, tables, code, images, scientific formulas and existing report links. Reading triggers no model request. The settings dialog shows current non-secret settings; settings changes use the existing terminal workflow and require restarting the workbench.

The daily tab includes a month calendar. Selecting an existing report opens it; other dates show their actual file/run state and, when available, an explicit date-prefilled generation or retry form. Calendar inspection never starts generation. Zero-match completion, missing report files and ungenerated dates have different meanings; do not infer publication availability from an empty date.

Generation buttons invoke the original CLI pipeline, with its own model API, output protection and optional configured email delivery. Do not generate substitute Markdown in the conversation. Personal-library indexing/review remains available through `/arxiv-daily:research` and the documented product commands.

Keep the server running while the user reads. To stop it when asked, send one SIGINT to the owned background process and wait for normal shutdown. Do not delete product files or kill unrelated processes. The current workbench is a reader, not a Markdown editor or a full Obsidian host.
