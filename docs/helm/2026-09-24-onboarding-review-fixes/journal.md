# Journal

## 2026-09-24 — note

- evidence: P5's three planned chunks landed (frontmatter YAML quoting, frontmatter refresh keeps user properties, dates older than /recent fail permanently). The last one changes a pinned contract: two pipeline tests expected `failed_transient` for such dates; with the submittedDate fallback long gone nothing can fetch them, so they now expect `failed_permanent`.
- change: P5 blocked while the review agent's core / CLI / relay batches are still outstanding; P6 (low-priority plugin items) started meanwhile.
- disposition: the `extra_body` thinking-parameter finding (review) is not changed without real provider calls; it goes to the user as a decision. The idle-timeout and surrogate-truncation findings are recorded in the review and not fixed here.
- next: P6 chunk 1 (enable + Run today does not block); resume P5 when the review agent reports.
