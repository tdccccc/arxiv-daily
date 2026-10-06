# Journal

## 2026-10-01 — P1 accepted, start P2

The user approved Markdown-backed browser reading and confirmed work remains in the existing Claude integration worktree. P1 renderer and scoped HTTP service passed their expected Red/Green contracts; the original generation and storage contracts remain unchanged. Full CLI tests (156), CLI typecheck, boundaries and product inventory pass. A Node Fetch Host override was ignored, so the Host check uses a real node:http request instead. KaTeX 0.17.0 keeps build dependencies compatible with Node 20; renderer and HTTP tests were rerun after pinning.

Next: the browser reading surface and portable `ui` launch. Configuration editing remains the existing terminal wizard/TOML workflow; show current non-secret settings in the workbench and require restart after changes.

## 2026-10-01T23:27:43+08:00 — Workbench accepted and closed

All goal criteria are met without a waiver. The workbench uses original Markdown as the authoritative content, the exact CLI artifact for generation, and an auxiliary Claude open skill. A navigation regression found during packaging was corrected with observed Red/Green: links from chat can load the static landing page while cross-origin data APIs remain blocked. UI tests also cover stale response ordering and keyboard focus.

Accepted evidence: 168 CLI tests, 7 independent plugin tests, copied-bundle Node 20.19 workflow, full workspace typechecks/build, boundaries, product inventory, submission check, strict plugin validation; lint has 21 pre-existing warnings. Real Chromium reading and a fixture-backed generation flow passed; existing Markdown remained unchanged. CSS was visually verified rather than given implementation-mirroring tests.

The domain glossary now names the optional reading workbench; this adds an entry surface without changing the business core or authoritative record. Current limitations are explicit: no Markdown editor, full Obsidian ecosystem, graphical settings writer, or graphical library review. Native Windows/macOS browser opening and real provider output quality remain outside this Linux fixture acceptance. Changes are local to the existing worktree, committed and not published.
