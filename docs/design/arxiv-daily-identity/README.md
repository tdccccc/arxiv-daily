> Decision (2026-10-11): user chose wordmark option **06 — Compressed Contrast** from the
> `tmp/wordmark/` exploration — a pure-text wordmark, no icon. Implemented in the browser
> workbench header and the DSH sidebar's own dialog header; see [`wordmark/`](wordmark/) for the
> adopted SVGs. Icon proposals below remain closed design history only.

## Adopted wordmark — 06 Compressed Contrast (2026-10-11)

`arxiv` set heavy (Inter Display Black 900, ink color) + `daily` set smaller and light (Inter
Display Light 300, accent color), very tight tracking — a condensed masthead logotype that stays
legible from ~20px sidebar size up to large display. Replaces the prior plain-text `arxiv-daily`
header (bold, with an accent hyphen).

- Font: **Inter Display** (Black 900 / Light 300) — SIL OFL 1.1. Outlined to static SVG path data
  (`opentype.js`, see `tmp/wordmark/build.mjs`); no font file is shipped, so no OFL notice is
  required in `THIRD_PARTY_NOTICES.md` (the repo's existing notices only cover bundled runtime
  code, e.g. pako/markdown-it/katex — not outlined vector artwork).
- Colors: ink `#242b33` / accent `#2e5fa6` on light surfaces, `#e4e7e8` / `#8eb8ef` on dark
  surfaces — the same `--ink`/`--accent` pair already defined in
  `apps/cli/src/workbench/web/style.css`.
- Files in [`wordmark/`](wordmark/): `06-compressed-contrast.svg` (light bg, two-color),
  `06-compressed-contrast-dark.svg` (dark bg, two-color), `06-compressed-contrast-mono.svg`
  (single `currentColor` fill, used where the two-color split isn't applicable, e.g. the DSH
  sidebar dialog header which follows the host's own text color).
- Full option set and rationale: `tmp/wordmark-report.md` (not copied into the repo; it's a
  working note, not a release artifact).
- Scope: browser workbench header (`apps/cli/src/workbench/web/app.ts`,
  `apps/cli/src/workbench/web/style.css`) and the DSH extension's own dialog header
  (`extensions/dsh-arxiv-daily/src/client.mjs`). The DSH sidebar's main pane embeds the workbench
  in an iframe, so it inherits the header change automatically; the DSH tab-strip label and
  "guide" title are plain text handed to the host's own `sidebarRightTabs` API and aren't
  restyled. Not applied to the Obsidian plugin, VS Code extension, app icons, or release
  metadata.

---

# arxiv daily — identity proposal 01

Status: initial proposal not selected. See [round 2 alternatives](alternatives/comparison.png). No mark is imported by the workbench or any plugin package.

Visual thesis: a calm, substantial lowercase a in deep teal, with a small amber daily-update marker. The mark stays readable without detail or typography at icon sizes.

Preview content: primary mark and wordmark, application tile, dark surface, 24/32/48px and monochrome examples. Static identity; no motion required.

- symbol.svg: primary transparent mark (#145E63 / #ED9A53).
- symbol-dark.svg: light mark for dark surfaces.
- symbol-mono.svg: one-color use.
- app-icon.svg: rounded-square application icon.
- preview.png: review sheet; wordmark uses the local SF Pro font for illustration, font files are not distributed.

Only the header text change is implemented: `arxiv daily`, larger and bold, without the old icon or subtitle. Adopting this mark or replacing the existing DSH plugin icon requires the user's explicit design approval. No release package produced.
