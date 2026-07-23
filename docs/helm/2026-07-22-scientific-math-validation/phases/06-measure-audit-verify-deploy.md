# P6 — measure, audit, verify, and deploy

goal_ref: ../goal.md
status: done

## Outcome

The prompt-primary + thin-script math contract is independently sanity-checked, fully verified, and only the three approved plugin assets are deployed to the isolated test Vault.

## Assumptions

- Isolated test Vault plugin destination is `/home/tiandc/Documents/code/arxiv-daily/plugin_test/.obsidian/plugins/arxiv-daily/` (from prior helm deploy journal).
- Full workspace test/typecheck/build is sufficient automated verification for this initiative.
- Independent audit can be a focused adversarial re-read against the L3 severity rubric (Blocker = byte corruption / mis-rewrite; Acceptable = over-rejection).

## Approach

1. Confirm prompt-primary signals in tests (example-bearing system/correction; thin policy export; reject-without-rewrite).
2. Run full core (and workspace as needed) tests, typecheck, build.
3. Confirm bundled `main.js` contains the new Good/Bad / 正例/反例 wording.
4. Deploy only `main.js`, `manifest.json`, `styles.css` with byte-for-byte cmp; do not touch Vault data.

## Tasks

- [x] Confirm focused contract tests encode prompt-primary + thin fallback.
- [x] Run full `@arxiv-daily/core` tests and typecheck; run plugin build.
- [x] Independent pass against severity rubric (no Blocker: accepted `$...$` stable; only allowed rewrites; reject returns original bytes).
- [x] Confirm `main.js` contains prompt-primary example wording (en literal Good/Bad; zh `正例` / `反例` escapes).
- [x] Deploy only three assets to the isolated test Vault plugin dir and `cmp` them.
- [x] Close goal success criteria and set initiative done.

## Verification

- Full core tests: 56 files / 737 passed.
- Core typecheck passed; plugin production build `main.js` 364.5kb.
- Bundle contains Good/Bad and Chinese 正例/反例 (unicode-escaped) math examples.
- Deployed assets match build outputs byte-for-byte; `data.json` / `.cache` untouched.
- All goal success criteria checked.

## Deploy record

- Destination: `/home/tiandc/Documents/code/arxiv-daily/plugin_test/.obsidian/plugins/arxiv-daily/`
- Assets: `main.js`, `manifest.json`, `styles.css` only
- SHA-256:
  - main.js: `b23a6b6e33434ad2865b18aaa5fe89b21eee8d778c4e69356524b582fd0999d3`
  - manifest.json: `6e86c9b256851abf1479fbafb2ba8987dd6c6e1ff3e4c21a3372c5d335688c04`
  - styles.css: `70bf5a15d85f766fba07cb888afb7251f0211f1088f91f3c7dec7df86e88a51c`
- `cmp` OK for all three.

## Abort / reshape triggers

- If a Blocker (byte corruption / mis-canonicalization) is found, stop deploy and fix before closing.
- If the vault destination is missing or ambiguous, stop before copying.
- If deploy would touch more than the three assets, stop.
