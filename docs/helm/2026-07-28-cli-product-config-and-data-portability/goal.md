# CLI product config and vault data portability

status: done
updated: 2026-07-31

## Intent

Make the CLI a first-class **product** with a simple fixed TOML config and `init`, then finish CLI email setup on that base, then add **vault data** export/import for both plugin and CLI—without automatic settings sync between products.

## Success criteria

- [ ] CLI reads only XDG `arxiv-daily/config.toml`; missing file fails with “run init”; no `ARXIV_DAILY_*` settings env; no config path override flags.
- [ ] `arxiv-daily init` interactive flow writes a runnable config (vault, LLM, optional email, placeholder topic, arXiv fields) with overwrite/merge/cancel if present.
- [ ] CLI email can be configured via TOML; test/status/verify paths work for Send yourself and Official delivery as designed; docs match.
- [ ] Plugin and CLI can export/import zip of logical `daily` + `papers` + `.index` with mtime conflict preview and confirm/`--yes`.
- [ ] Docs state two-product model: settings manual per product; vault data portable.

## Non-goals

- Automatic or bidirectional **settings** sync between plugin and CLI.
- CLI daemon / long-running in-process scheduler (cron remains external).
- PDF inclusion in data packages (v1).
- Multiple CLI config profiles.
- Encrypting secrets at rest.
- Reading legacy cwd `arxiv-daily.config.json`.

## Constraints

- ADR 0001 / 0003: one core; CLI one-shot; fixed XDG TOML; secrets allowed in file.
- ADR 0002: email still host-agnostic digest + delivery-state when vault shared.
- Prefer thin phases; hard cuts documented in release notes when implemented.
- Grill decisions: `docs/adr/0003-two-products-cli-config-and-data-portability.md`, `CONTEXT.md`.
- Command surface: `cli-commands.md`. Schema: `cli-toml-schema.md`.

## Phases

1. P1 — CLI TOML config home + `init` + remove env/JSON config surface — status: done
2. P1b — `[schedule]` defaults in toml + `schedule show|install|uninstall` (cron/managed timer) — status: done
3. P2 — CLI email complete on TOML (status / verify / docs) — status: done (CLI); plugin UI unchanged
4. P3 — Vault data zip export/import — status: done for **CLI**; plugin UI export/import still open

## Current focus

Closed. Success criterion 4 (plugin export/import UI) waived as an optional
follow-up — see journal.

## Open questions

- None blocking P1. Field-level draft: `cli-toml-schema.md` (snake_case TOML ↔ core camelCase); implementer may tweak names only with a schema note.
