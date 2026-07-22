# P1 — core Markdown safety

goal_ref: ../goal.md
status: done

## Outcome

Core normal, rescue, and emergency report paths preserve readable scientific inline Markdown while hostile interpolated values cannot create Markdown structure, parser classifications, comments, or executable raw HTML.

## Assumptions

- Folding line and separator whitespace to spaces is sufficient to keep every interpolated scalar on one physical line without escaping ordinary Markdown punctuation.
- Neutralizing only actual HTML comments and tag-shaped raw HTML can leave ordinary `<`, `>`, and `&` comparison prose byte-for-byte unchanged.
- The test-vault report diagnosis is an emission problem: indexed summaries were intact, but `safeMarkdownScalar` corrupted emitted MathJax.

## Approach

Replace the generic encode/decode pair with one shared one-line Markdown normalization helper, anchor parser-significant markers to exact standalone lines, and make rescue projection validation compare normalized source prose directly. Keep rescue transport delimiter handling and exact postflight validation as separate controls.

## Tasks

- [x] Implement shared physical-line Markdown normalization with narrow HTML comment/tag neutralization.
- [x] Reuse normalization in normal, rescue, fallback, and emergency rendering without changing link or preflight contracts.
- [x] Remove production decode dependence and anchor parser-significant markers to exact standalone lines.
- [x] Harden rescue projection/topic markers and synchronize rescue prompts.
- [x] Add focused assembler, parser, rescue, and summarizer regressions including direct raw MathJax assertions.
- [x] Run focused core tests, core typecheck, and `git diff --check`; fix failures.

## Verification

- `npm test -w @arxiv-daily/core -- --run packages/core/tests/daily-summary-assembler.test.ts packages/core/tests/daily-summary-parser.test.ts packages/core/tests/daily-summary-rescue.test.ts packages/core/tests/summarizer.test.ts`
- `npm run typecheck -w @arxiv-daily/core`
- `git diff --check`
- Raw generated Markdown retains representative `$...$`, backslashes, braces, underscores, carets, parentheses, `+`, `|`, `%`, ordinary `&`, `<`, and `>`.

## Abort / reshape triggers

- If narrow HTML neutralization requires global entity encoding of ordinary comparison punctuation, stop and reshape rather than restoring the corrupting encoder.
- If exact standalone marker recognition breaks the supported writer contract, stop and separate writer compatibility into a new phase before proceeding.
