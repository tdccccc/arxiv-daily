# Desktop acceptance harness

Drives a real, isolated Obsidian through the checks that otherwise need a person
clicking: PDF `#page=N` location, the optional parser sidecar staying inert until
enabled, legacy settings migration, the personal library settings page, and a
renderer that logs no errors.

Two Obsidian sessions run in sequence, because the checks need two different
persisted states: one with settings that predate the sidecar, one with a
personal library already connected. Each session brackets the vault's state on
its own.

## Running

```sh
npm run build --workspace obsidian-arxiv-daily
OBSIDIAN_TEST_VAULT=/path/to/disposable_vault npm run test:desktop
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OBSIDIAN_TEST_VAULT` | *(required)* | A disposable vault holding at least one PDF, and a folder of PDFs for the library page |
| `OBSIDIAN_BINARY` | `/opt/Obsidian/obsidian` | Obsidian executable |
| `OBSIDIAN_TEST_PLUGIN_ID` | `arxiv-daily` | Plugin directory name inside the vault |
| `OBSIDIAN_ACCEPTANCE_SCREENSHOT_DIR` | `.acceptance-out/desktop-acceptance` | Where screenshots are written |

Exit codes distinguish two different situations:

- **2 — blocked.** The environment cannot run the acceptance. Every blocker is
  listed with the action that fixes it. Either nothing was launched, or the
  application turned out not to be showing a vault at all — in which case **no
  assertion result is reported**, because none of them would mean anything.
- **1 — failed.** The acceptance ran and a scenario failed, the renderer logged
  an error, or diagnostics were incomplete.
- **0 — passed.**

## What is checked before anything is believed

Three guards stand between "the harness ran" and "the harness verified
something". They exist because on 2026-08-31 a run reported **seventeen passes
and ten screenshots** while Obsidian was sitting on its own error page with the
vault never opened: the machine's file-watch quota was exhausted, and the only
red was "the renderer logged no error" — so the run exited 1, blaming the
product, rather than 2.

- **Preflight probes the file-watch quota.** Not by reading
  `/proc/sys/fs/inotify/max_user_watches`, which gives the ceiling and says
  nothing about what is left: a machine whose 524288 watches are all spoken for
  reads exactly like an idle one. The probe asks the kernel for
  `WATCH_HEADROOM` real watches on throwaway directories — one directory each,
  because inotify returns the same descriptor for a path it already watches —
  and releases every one it got. `ENOSPC` is a blocker naming how many it
  actually obtained and how to raise the ceiling. Any other refusal is reported
  as *unmeasured*, never as a blocker the harness cannot stand behind.
- **A screenshot is refused rather than written** unless its subject is in the
  document, visible (`display`, `visibility`, `opacity`, and actual client
  rects), laid out with a non-zero size, and at least
  `MIN_VISIBLE_FRACTION` inside the viewport the camera photographs — and
  unless the returned PNG holds more than one colour. Nothing is written until
  every one of those holds, so a refusal leaves no file behind at all. The
  colour rule is the whole image being one flat colour, never a percentage and
  never a comparison with a stored baseline: there is still no pixel diff here.
- **The application is checked before and after every walk.** The criterion is
  the positive capability the acceptance rests on — Obsidian's object graph,
  a mounted workspace with leaves, and the settings entry point — not the
  wording of any error page, so an environment failure nobody anticipated is
  caught along with the one that was. Whatever the page says is *quoted* in the
  blocker so the real cause is visible, and never used to decide. Failing the
  second check discards the results the walk had already produced.

## What it does to the vault

The build under test is **deployed and left in place**, matching the ordinary
plugin development loop. State that cannot be regenerated is captured before the
run and restored afterwards on every exit path, including `Ctrl-C`:

- `.obsidian/plugins/<id>/data.json` — hand-configured endpoints, keys, topics
- `.obsidian/workspace.json` — pane layout

Historical `main.js.bak-*` builds are never read or written.

The personal library session points the library at the first folder of PDFs
inside the vault. It only names that folder in the settings store it restores
afterwards — nothing in it is read, written, or indexed.

## Screenshots

Every run writes PNGs of the states the library assertions only describe in
words, so a person can judge wording, spacing and emphasis — the things no
assertion settles. They land in **`.acceptance-out/desktop-acceptance/`**
(git-ignored; override with `OBSIDIAN_ACCEPTANCE_SCREENSHOT_DIR`) and are
overwritten each run. Nothing compares them: there is no stored baseline and no
pixel diff, and no assertion reads an image back.

A written file therefore has to be worth looking at, since its mere existence
reads as evidence that the state it is named after was reached. Every capture
is refused unless the checks above hold — see *What is checked before anything
is believed*. A refused capture fails the run; it never becomes a file.

| File | What it shows |
| --- | --- |
| `personal-library-section-local-embedding.png` | The whole Personal library section with a folder selected and local embedding: the row's next step, and the section's position between Output & schedule and Email delivery |
| `personal-library-section-remote-embedding.png` | The same section with remote embedding selected and no grant: the extra endpoint rows, and a Library row that still offers no authorization button |
| `remote-full-text-disclosure-modal.png` | The consent dialog raised in place by switching Embedding to remote — folder, file types, depth, and the embeddings endpoint that actually receives the text |
| `library-row-narrow-panel.png` | The Library row alone at the narrow settings panel (window 700 px) |
| `library-row-wide-panel.png` | The Library row alone at the wide settings panel (window 1400 px) |
| `library-row-stacked-panel.png` | The Library row alone below Obsidian's stacking breakpoint (window 560 px), where the row becomes a column and every button is stretched full width |
| `personal-library-section-authorized.png` | The section once remote full text has been granted: the Library row's third button, Revoke |
| `library-row-three-buttons-narrow-panel.png` | The granted three-button row at the narrow panel, where the buttons no longer fit on one line and wrap rather than squeeze the description out |
| `library-row-three-buttons-wide-panel.png` | The same three-button row at the wide panel, where all three still fit on one line |
| `library-row-three-buttons-stacked-panel.png` | The same three-button row below the stacking breakpoint |

The three panel widths are produced by emulating the renderer's device metrics,
so Obsidian's own responsive rules decide the panel width rather than an
injected style. The width each one actually produced is printed with the
geometry result. Each width is judged by the rule that means something there:

- **Two buttons, narrow and wide** — one right-aligned line, inside the control,
  off the description.
- **Stacked** — only that the buttons stay inside the control and the row. One
  line and right alignment stop meaning anything once Obsidian stretches each
  button to the full width.
- **Three buttons, narrow and wide** — everything the two-button rule holds
  except one line, since wrapping is the intended behaviour once they no longer
  fit; plus each line ending flush right, the main call to action laid out
  visibly, and the description staying **readable**. That last one exists
  because the state this assertion was written for satisfied every geometric
  rule above while giving the description six pixels and spelling it out one
  letter per line: nothing overlapped, and nobody could read it. Readable is
  measured as a minimum column width and a minimum number of characters per
  rendered line, counted from the text's own line boxes.

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

1. Build the plugin and copy `plugin/main.js`, `plugin/manifest.json` and
   `plugin/styles.css` into `<vault>/.obsidian/plugins/arxiv-daily/`, backing up
   `data.json` first. The stylesheet matters: the settings-page layout checks
   describe what it lays out.
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
7. Open Settings → arXiv Daily. Personal library must sit between Output &
   schedule and Email delivery. Its Library row must offer at most three
   buttons, none of them an authorization step or a Manage menu, all on one
   line and inside the row. Switching Embedding to remote must ask for
   full-text consent there and then; cancelling must put the dropdown back to
   local and change neither the mode nor the grant. With remote selected and no
   grant, Build index must ask the same question before it starts, and
   cancelling must start nothing.
8. Restore `data.json` and `workspace.json` from your backups.

## Layout

| File | Role |
| --- | --- |
| `acceptance.mjs` | Entry point: preflight, run scenarios, report |
| `preflight.mjs` | Environment checks that produce blockers, not stack traces |
| `app-state.mjs` | Is a vault window mounted at all, before and after a walk |
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
| `scenarios.mjs` | The sidecar, migration and PDF location scenarios |
| `library-settings.mjs` | The personal library settings page walk |
| `screenshots.mjs` | Element-clipped PNGs of the states worth looking at |
| `settings-fixture.mjs` | The persisted states each session starts from |
| `smoke.mjs` | Minimal session probe, useful when debugging the harness |

Unit tests live in `scripts/tests/desktop-acceptance-*.test.mjs` and run under
`npm run test:release-tools`; they need no Obsidian.
