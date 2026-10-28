# Journal — summary quality evaluation

## 2026-10-28 — note

- evidence: Chunk 3's real-vault run flagged `$6:1$`, `$8:1:1$` and `$g \in (0, 0.15]$`, which render fine in Obsidian. A probe of core's `canonicalizeScientificMarkdownMath` confirms the two forms. A digit-leading colon ratio gives `unclosed-delimiter`, because a `$` before a digit is read as currency. A half-open interval gives `unbalanced-parentheses`. `$0.5$`, `$10^{10}\,M_{\odot}$`, `$0.8$–$0.95$`, `$1.2\pm0.3$` and `$3\sigma$` pass. Daily summaries use the same validator, so a summary that writes either form is rejected, retried and may fall back.
- change: none to this initiative. The offline report labels these patterns as known core-rule false positives and does not count them as defects.
- disposition: product follow-up, outside this initiative: relax the validator for ratios and half-open intervals, with tests.
- next: P1 chunk 4 (report and command).
