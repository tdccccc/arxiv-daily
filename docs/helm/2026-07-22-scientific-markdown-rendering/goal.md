# Scientific Markdown rendering

status: done
updated: 2026-07-22

## Intent

Keep generated daily reports structurally safe while preserving scientific and user-authored inline Markdown as readable Markdown. Remove the reversible escaping scheme that corrupted emitted MathJax despite intact indexed summaries in the test-vault report.

## Success criteria

- [x] All report paths use one shared physical-line normalization that preserves scientific Markdown punctuation while preventing interpolated values from creating blocks, machine markers, comments, or raw HTML.
- [x] Parsers recognize machine markers only in their exact standalone structural positions and return stored scientific Markdown directly after intended whitespace compacting.
- [x] Core focused tests, core typecheck, and `git diff --check` pass, including direct raw-output MathJax assertions and hostile structure cases.
- [x] User-visible writer, adapter, pipeline, and history regressions are covered before the initiative closes.

## Non-goals

- Changing canonical arXiv ID validation or `safeDetailLink` policy.
- Altering assembly preflight or structured/fallback separation.
- Modifying closed Helm initiatives.

## Constraints

- Preserve all existing uncommitted fallback feature changes; no commit or push.
- Keep the goal thin, maintain exactly three outcome phases, and detail only the active phase.
- Keep exact rescue `requiredMarkdown` postflight separate from rescue-contract delimiter protection.
- Never automate modification or regeneration of `/home/tiandc/Documents/code/arxiv-daily/plugin_test/arxiv-daily/daily/2026-07-21.md`; deployment is allowed only when separately authorized, existing report inspection/rescan is read-only, and rerun is manual user action only.

## Phases

1. P1 — Core rendering, parsing, rescue, and summarizer paths preserve scientific Markdown without structural injection — status: done
2. P2 — User-visible writer, adapter, pipeline, and history behavior has regression coverage — status: done
3. P3 — End-to-end scientific report behavior is validated and the initiative is ready to close — status: done

## Current focus

Done
