## 2026-07-22 — note

- evidence: The test-vault report retained intact indexed summaries, while emitted report MathJax was corrupted by the reversible `safeMarkdownScalar` escaping layer.
- change: Replaced encode/decode rendering with shared one-physical-line normalization, narrow raw-HTML neutralization, exact standalone parser markers, and direct normalized rescue projection comparison.
- disposition: Keep all existing fallback feature changes and the P1 implementation; closed Helm initiatives remain untouched.
- next: P2 is active; begin by tracing writer and storage-adapter boundaries and add user-visible regression coverage only.

## 2026-07-22 — P2 complete

- evidence: Writer, Obsidian adapter, pipeline repair, history sync/search, and legacy marker regressions pass; focused core reported 212 tests, focused plugin 16 tests, both typechecks and `git diff --check` pass.
- decision: Exact standalone watch/highlight marker matching is required and backward compatible with emitted legacy and `selection:` controls; inline marker-like prose must remain inert.
- disposition: Keep P1 and fallback changes unchanged; P2 added regression coverage only and did not expose or copy plugin data/settings.
- next: P3 is active for combined audit, final verification, and test-vault deployment; no P3 task has been executed yet.

## 2026-07-22 — P3 independent-audit L1 adjustment

- evidence: Independent audit found overly broad angle-bracket neutralization, lost historical H3 identity, inconsistent historical checkbox whitespace/CRLF handling, and a rescue contract whose raw transported scalars disagreed with byte-exact normalized output instructions.
- change: Adjust P3 in place to repair those compatibility contracts and rerun focused rendering/parser/rescue/selection/pipeline/writer/history verification before any deployment decision.
- constraint: Automation must never modify or regenerate `/home/tiandc/Documents/code/arxiv-daily/plugin_test/arxiv-daily/daily/2026-07-21.md`; deployment is allowed only when separately authorized, existing report inspection/rescan is read-only, and rerun is manual user action only.
- disposition: Keep the scientific-Markdown goal and P3 active; preserve all current changes, do not commit/push/deploy, and do not close P3 after this adjustment.
- next: Implement the four audited contracts, verify exact behavior, and leave P3 active for full verification/audit/deployment.

## 2026-07-22 — P3 audit adjustment verified

- evidence: Final CommonMark/raw-HTML, parser identity, selection-marker, rescue-contract, pipeline/writer/history focused core tests passed 217/217; focused plugin tests passed 16/16; core and plugin typechecks plus `git diff --check` passed.
- evidence: The protected report SHA-256 remained `a2689e9a9bfd79525aca1702e720dbd685b6c1af9cf34871f96f73826c72c318`; no report generation, deployment, commit, or push occurred.
- decision: Keep P3 active. The audit adjustment is verified, but combined diff audit, separately authorized deployment, read-only test-vault inspection/rescan, and manual user rerun remain outstanding.
- next: Full verification/audit/deployment checkpoint under the protected-report boundary; do not close P3 yet.

## 2026-07-22 — P3 code-span L1 blocker verified

- evidence: Maximal backtick-run scanning and adversarial normal/emergency/rescue raw-output coverage passed 92/92 focused assembler, rescue, parser, and writer tests; core typecheck and `git diff --check` passed.
- decision: A code-span closer must be a maximal backtick run exactly equal to its opener; ordinary backslashes do not disable backtick delimiters, and unmatched runs remain plain text so outside raw HTML is neutralized.
- evidence: Scientific Markdown/autolink regressions remained green and the protected report SHA-256 remained `a2689e9a9bfd79525aca1702e720dbd685b6c1af9cf34871f96f73826c72c318`.
- next: Keep P3 active for the remaining combined audit/deployment checkpoint; no commit, push, deployment, report generation, or protected-report modification occurred.

## 2026-07-22 — P3 history abstract precedence L1 verified

- evidence: History reconstruction previously forwarded every marker-confirmed fallback display abstract through ordinary index upserts, allowing a normalized/HTML-neutralized report value to overwrite an exact canonical Atom abstract and allowing later reports to replace an earlier recovery.
- change: History sync now treats report abstracts as fill-only fallback data: undefined, null-normalized, empty, and whitespace-only stored values are missing; the earliest sorted marker-confirmed fallback repairs them, while any meaningful canonical or recovered abstract wins unchanged.
- evidence: Focused history tests passed 12/12 and parser/search tests passed 27/27, including exact multiline/raw-HTML canonical preservation, English and Chinese recovery, whitespace policy, multiple-report ordering, absent markers, no generated five-field summary, and indexed/legacy search. Core and plugin typechecks plus `git diff --check` passed.
- evidence: The protected report SHA-256 remained `a2689e9a9bfd79525aca1702e720dbd685b6c1af9cf34871f96f73826c72c318`.
- next: Keep P3 active for the remaining combined audit/deployment checkpoint; no commit, push, deployment, report generation, or protected-report modification occurred.

## 2026-07-22 — P3 and initiative closed

- evidence: Full repository verification passed from the exact feature worktree: lint completed with 0 errors and the existing configured 134 warnings; typecheck passed all four workspaces; 76 test files passed with 919 tests total (core 55/655, node runtime 1/8, CLI 3/26, plugin 17/230); workspace boundaries, production build, build smoke, and `git diff --check` all passed.
- evidence: The combined status and diff audit found only intended core source/tests plus preserved fallback and scientific-Markdown Helm documents. No lockfile, `data.json`, settings, cache, production vault, or tracked generated artifact changed; `plugin/main.js` remained an ignored build artifact.
- evidence: A production plugin build was copied only as `main.js`, `manifest.json`, and `styles.css` to `/home/tiandc/Documents/code/arxiv-daily/plugin_test/.obsidian/plugins/arxiv-daily/`; all three passed byte-for-byte `cmp`. Their SHA-256 values are `2c1c30717561a612a8c45cb9f2c6e706de8baddb818cc4498d3e08cfcfa938be`, `6e86c9b256851abf1479fbafb2ba8987dd6c6e1ff3e4c21a3372c5d335688c04`, and `70bf5a15d85f766fba07cb888afb7251f0211f1088f91f3c7dec7df86e88a51c`, respectively.
- evidence: Target `data.json` and `.cache` snapshots were unchanged. The protected report SHA-256 was `a2689e9a9bfd79525aca1702e720dbd685b6c1af9cf34871f96f73826c72c318` before actions and after deployment; automation did not run Obsidian, report generation, regeneration, or rescan.
- decision: All automated, audit, deployment, and preservation gates passed, so P3 and the scientific Markdown rendering initiative are done. Source/rendered inspection and any rerun remain explicit manual user steps.
- disposition: Preserve all feature work without commit or push; closed Helm initiatives and production vault remain untouched.
