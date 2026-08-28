# Desktop acceptance harness

Drives a real, isolated Obsidian through the checks that otherwise need a person
clicking: PDF `#page=N` location, the optional parser sidecar staying inert until
enabled, legacy settings migration, and a renderer that logs no errors.

## Running

```sh
npm run build --workspace obsidian-arxiv-daily
OBSIDIAN_TEST_VAULT=/path/to/disposable_vault npm run test:desktop
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OBSIDIAN_TEST_VAULT` | *(required)* | A disposable vault holding at least one PDF |
| `OBSIDIAN_BINARY` | `/opt/Obsidian/obsidian` | Obsidian executable |
| `OBSIDIAN_TEST_PLUGIN_ID` | `arxiv-daily` | Plugin directory name inside the vault |

Exit codes distinguish two different situations:

- **2 — blocked.** The environment cannot run the acceptance. Every blocker is
  listed with the action that fixes it. Nothing was launched.
- **1 — failed.** The acceptance ran and a scenario failed, the renderer logged
  an error, or diagnostics were incomplete.
- **0 — passed.**

## What it does to the vault

The build under test is **deployed and left in place**, matching the ordinary
plugin development loop. State that cannot be regenerated is captured before the
run and restored afterwards on every exit path, including `Ctrl-C`:

- `.obsidian/plugins/<id>/data.json` — hand-configured endpoints, keys, topics
- `.obsidian/workspace.json` — pane layout

Historical `main.js.bak-*` builds are never read or written.

## Isolation

- A throwaway `XDG_CONFIG_HOME` holds a vault list containing **only** the vault
  you named, so your real vault list is never read or rewritten.
- Obsidian runs under `xvfb-run`, so a run never draws over your desktop.
- The debugging port is chosen free at runtime, so a run coexists with an
  Obsidian you already have open.
- The process tree is reclaimed by **process group** (`setsid` + `kill -PGID`).
  Never reclaim by process name or command-line pattern: the first kills the
  user's real Obsidian, and the second also matches the harness script itself.

## Running the checks by hand

If the harness cannot run, these steps are the equivalent record:

1. Build the plugin and copy `plugin/main.js` and `plugin/manifest.json` into
   `<vault>/.obsidian/plugins/arxiv-daily/`, backing up `data.json` first.
2. Open the vault in Obsidian and accept the trust prompt.
3. Open the developer console. It must stay free of errors throughout.
4. Open any PDF with `#page=4` appended to its path. The viewer must land on
   page 4, not merely open the file.
5. In settings, confirm the PDF parser sidecar is **off** after loading settings
   that predate it.
6. Point the sidecar at a loopback port you control and enable it. The probe
   must reach it, fail, and leave PDF.js selected with nothing logged as an
   error. Note that the plugin's HTTP goes out through Obsidian's `requestUrl`
   in the Electron main process, so the developer console's network tab will
   not show it — bind an actual socket.
7. Restore `data.json` and `workspace.json` from your backups.

## Layout

| File | Role |
| --- | --- |
| `acceptance.mjs` | Entry point: preflight, run scenarios, report |
| `preflight.mjs` | Environment checks that produce blockers, not stack traces |
| `session.mjs` | Brackets deploy, launch, CDP attach and reclamation |
| `vault-config.mjs` | Single-vault isolated config, overlap refusal |
| `vault-state.mjs` | Capture and restore, including on signals |
| `build-deploy.mjs` | Deploy the branch build, assert what actually loaded |
| `process-group.mjs` | Detached launch and group-only reclamation |
| `launch.mjs` | Command, isolated env, free port, CDP wait |
| `cdp.mjs` | Dependency-free CDP client and evaluator |
| `diagnostics.mjs` | Console and page-error collection |
| `probe-listener.mjs` | Real loopback socket that observes the plugin's HTTP |
| `trust.mjs` | Trust prompt and plugin readiness |
| `scenarios.mjs` | The four acceptance scenarios |
| `settings-fixture.mjs` | Legacy settings the migration must handle |
| `smoke.mjs` | Minimal session probe, useful when debugging the harness |

Unit tests live in `scripts/tests/desktop-acceptance-*.test.mjs` and run under
`npm run test:release-tools`; they need no Obsidian.
