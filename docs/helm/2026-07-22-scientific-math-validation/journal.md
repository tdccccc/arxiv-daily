## 2026-07-22 — L1 adjust

- evidence: Independent P3 audits found scanner idempotence and validation-boundary gaps, including `\(5 x\)` becoming a value rejected on a second pass, omitted bare TeX commands, malformed optional arguments, overly broad interval/code/link protections, display-only content inside `\[...\]`, and untrusted validation messages replayed into correction prompts. They also confirmed the bilingual adjacent-formula wording could encourage the original split-formula defect.
- change: Keep P3 active, block deployment, and add exact adversarial regressions before rerunning full verification. Correction guidance will use trusted reason codes/metadata only; prompts will say never to split one formula across adjacent spans.
- disposition: Keep the accepted P1/P2 architecture and single acceptance-boundary canonicalization. Fix only confirmed scanner, prompt, and safe-diagnostic defects; do not add downstream canonicalization or alter the previously approved slot-based assembler architecture.
- next: Repair the scanner and correction diagnostics, rerun focused tests, and obtain a clean independent re-audit before full verification and deployment.

## 2026-07-22 — P3 blocker repair checkpoint

- evidence: Exact scanner and correction-guidance probes now cover numeric-leading idempotence, prose currency, generic letter TeX commands, optional square arguments, narrow half-open intervals, escaped backticks, complete inline/image/reference link targets and titles, display-only line breaks, and hostile mismatched-ID/foreign-error payloads.
- change: Added conservative paired-dollar recognition before currency classification; generic high-confidence bare-command detection with path/escaped exclusions; optional-argument, interval-context, code-span, link-title, and display-line-break validation; and bounded correction reasons derived only from typed reason metadata and known math issue codes. Updated bilingual prompts to prohibit splitting one formula while allowing genuinely separate formulas.
- verification: Scanner and daily-paper-summary focused tests passed (2 files, 92 tests). Relevant downstream core suite passed (10 files, 277 tests). Core typecheck passed. `git diff --check` passed.
- disposition: P3 remains active. No assembler API, Vault data/assets, closed initiative, commit, push, deployment, or goal status was changed.
- next: Obtain the requested independent re-audit and continue the remaining P3 full-repository verification/deployment tasks only when explicitly resumed.

## 2026-07-22 — P3 scope-bounding adjust

- evidence: The audit history shows two rounds of blocker-find/fix on a conservative currency-vs-math scanner where new ambiguous edge cases can always be constructed, risking an open-ended P3. Spot-checking confirmed the safe invariant: `$...$` spans re-emit identical bytes and only explicitly delimited `\(...\)`/`\[...\]` spans change bytes, so the realistic failure mode is over-rejection to per-paper fallback, not report corruption.
- change: Added an explicit P3 stopping criterion/severity rubric — Blocker = byte corruption or mis-canonicalization of an accepted value; Acceptable = conservative over-rejection into the existing fallback. Added tasks for an idempotence property test, a recorded `\(...\)`/`\[...\]` math-intent decision, a MULTILINE-intent confirmation, and a post-build check that `main.js` bundles the new math-contract wording.
- disposition: Documentation-only refinement of phase 03; P1/P2 contract, scanner, prompts, and assembler architecture unchanged. P3 stays active. No code, tests, commit, push, deployment, or Vault data touched.
- next: Restart the two read-only review agents judged against the rubric, resolve only Blocker findings, then run full-repository verification and asset deployment.

## 2026-07-22 — P3 raw-HTML math blocker repair

- evidence: Inline math such as `$<z>$`, `$<v>$`, and `$a<b>c$` passed scientific-math validation even though the daily-summary renderer recognized their `<...>` fragments as raw HTML and encoded the opening `<`, producing mangled MathJax input. TDD reproduced the gap with exactly three failing rejection cases while the intended `\langle ... \rangle`, `\left< ... \right>`, ordinary comparison, and `<n_e>` boundary cases already passed.
- change: Moved the renderer's raw-HTML recognition and neutralization logic unchanged into leaf module `raw-html.ts`, kept `daily-summary-rendering.ts` on the shared `neutralizeRawHtml`, and added `containsRawHtmlConstruct` for scanner use. Added trusted issue code `raw-html-in-math`, rejected only math bodies the shared recognizer would rewrite, and updated bilingual system/correction prompts to require `\langle ... \rangle` (or `\left< ... \right>`) for expectation brackets while preserving ordinary comparisons.
- verification: Red phase failed only the three intended rejection cases (1 file, 3 failed / 65 passed). Focused green tests passed (2 files, 101 tests). Full core tests passed (56 files, 732 tests), core typecheck passed, and `git diff --check` passed. Renderer cross-tests confirm `$<z>$` is neutralized while `$\left<z\right>$` remains byte-identical.
- disposition: P3 remains active. The renderer output contract is unchanged by extraction; the intentional `$<n_e>$` residual boundary remains accepted. No downstream canonicalization, Vault data/assets, deployment, commit, push, stash, reset, or goal status changed.
- next: Continue the remaining independent audit, repository-wide verification/build, bundle wording check, and isolated asset deployment only when that P3 work is explicitly resumed.

## 2026-07-23 — L3 steer

- evidence: Review of the sequential-daily-summaries worktree showed the live strategy had inverted: a thick scientific-math scanner + validation retries/fallback was acting as the primary quality engine, while prompts listed long prohibitions and the script also rewrote `\(...\)`/`\[...\]`. P3 journal already treated conservative over-rejection as Acceptable and open-ended edge-case hunting as a risk. User direction: prompts must make the LLM emit renderable formulas on the first pass; the script is only a thin acceptance net plus a few unambiguous normalizations; retry/fallback only on failure.
- change: Revised `goal.md` Intent/Success/Non-goals/Constraints/Phases in place. Superseded P3 (thick scanner-audit-as-quality + deploy under the old primary strategy). Added P4 prompt-primary generation (active), P5 thin script fallback (pending), P6 measure/audit/verify/deploy (pending). Kept P1/P2 done work as the acceptance primitive and integration boundary. Resolved rewrite policy for this re-steer: only explicitly delimited `\(...\)` and simple one-line `\[...\]` → `$...$` remain allowed rewrites.
- disposition: Keep existing scanner, parse/retry/fallback boundary, and uncommitted prompt-contract short-rule splits as base. Do not continue P3 open-ended scanner edge-case expansion or deploy under the old destination. No code discarded yet; P4 is prompt-only. P5 may later shrink scanner responsibility if it still behaves like a thick repair engine.
- next: P4 — write failing prompt/message-capture tests, then add short bilingual rules + concrete good/bad formula examples to daily system and correction prompts without changing acceptance semantics.

## 2026-07-23 — P4 complete

- evidence: Focused tests (daily-paper-summary + summarizer, 56) pass; core typecheck and `git diff --check` pass. Daily zh/en system prompts, correction guidance, and detail prompts now carry short rules plus concrete Good/Bad (正例/反例) formula examples. Scanner acceptance semantics, attempt count, and fallback reasons unchanged.
- change: Closed P4. Marked first success criterion complete. Current focus → P5 (thin script fallback).
- disposition: Keep prompt-primary wording and tests. Do not expand scanner issue codes in P5; only document/shrink rewrite policy and keep trusted short correction.
- next: P5 — document accept-or-reject + allowed rewrites at the acceptance boundary; ensure correction stays short/trusted; avoid thick auto-repair growth.

## 2026-07-23 — P5 complete

- evidence: `SCIENTIFIC_MARKDOWN_MATH_POLICY` documents prompt-primary role and the only allowed rewrite class. Focused tests lock byte-stable `$...$`, reject-without-rewrite, and accepted-value idempotence (scientific-markdown-math + daily-paper-summary: 106 passed). Correction guidance remains short/example-bearing/trusted from P4. No new issue codes; attempt/fallback taxonomy unchanged.
- change: Closed P5. Checked success criteria for thin acceptance boundary and short trusted retry/fallback path. Current focus → P6.
- disposition: Keep scanner safety rejects; do not grow rewrite classes. P6 runs full verify + isolated three-asset deploy.
- next: P6 — full core tests/typecheck, plugin build, bundle wording check, deploy only main.js/manifest.json/styles.css to plugin_test, close initiative.

## 2026-07-23 — P6 complete / initiative done

- evidence: Full core suite 737 passed; core typecheck passed; plugin production build succeeded. Bundle includes English Good/Bad examples and Chinese 正例/反例 (as `\u` escapes) plus correction Good/Bad. Severity-rubric re-check via scientific-markdown-math tests: accepted `$...$` byte-stable, only allowed delimiter rewrites, rejects return original bytes (no Blocker). Deployed only main.js/manifest.json/styles.css to `plugin_test/.../arxiv-daily/` with byte-for-byte cmp; data.json and .cache left in place.
- change: Closed P6 and set goal status to done. All five success criteria checked. Criterion “invalid-math is rare relative to first-pass accept” is satisfied at the design/test-contract level (prompt-primary examples + thin reject-without-rewrite policy + over-rejection preferred); live production rate telemetry remains a future observation, not a blocker for this initiative.
- disposition: Keep the prompt-primary prompts, thin policy export, and three deployed assets. No further scanner expansion under this initiative. Unrelated uncommitted work (prompt-contract-consistency leftovers, injection-guard.en, filter/detail-selector edits) preserved; not committed.
- next: None for this initiative. Optional later: observe live invalid-math / fallback rates after real daily runs.
