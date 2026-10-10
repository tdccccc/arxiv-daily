# Verification — 2026-10-10

Branch: `test/ui-regression`, based on `main` at `42e8335`.

## Observed results

| Check | Result |
|---|---|
| `npm run test:acceptance` (default build plus both hosts) | 20 passed: 12 real workbench journeys, 8 real Obsidian journeys; exit 0 |
| Real model API exploration | 3 tasks passed; 19 API calls / 19 actions; 70.329 seconds; 105,066 input + 640 output tokens |
| Controlled model with real browser | The same 3 tasks passed; 21 decisions; 13.390 seconds |
| Full workspace regression | Core 2,456, Node runtime 73, CLI 454, Plugin 1,022 passed: 4,005 total |
| Optional real-corpus checks | 2 skipped: real knowledge-base migration and retrieval require separately configured corpus inputs |
| `npm run test:release-tools` | 488 passed, including acceptance infrastructure and CI contracts |
| `npm run typecheck` | All workspaces passed |
| `npm run lint` | 0 errors, 16 existing warnings |
| Boundaries / product inventory / diff checks | Passed |

The final fixed acceptance report was opened in a real browser: 20 scenario sections rendered, and all 42 evidence artifacts exist.

## Local evidence

- [Final combined fixed acceptance](../../../output/playwright/acceptance/run-a5964k/report.html)
- [Final report JSON](../../../output/playwright/acceptance/run-a5964k/report.json)
- [Rendered report preview](../../../output/playwright/acceptance/run-a5964k/report-preview.png)
- [Real model exploration](../../../output/playwright/acceptance/run-LKWQzP/report.html)
- [Controlled browser exploration](../../../output/playwright/acceptance/controlled-exploration/exploration.json)
- [Real Obsidian component diagnosis](../../../output/playwright/acceptance/obsidian-button-debug-WNZIgu/evidence.json)

Evidence is intentionally git-ignored and remains on this machine. The runnable scenarios and their contracts are versioned.

## Product regressions found and fixed

1. A permanently failed date could not be manually retried. Regression tests failed before the fix; the retry decision now uses authoritative state under the vault lock. Completed and running dates remain protected.
2. A search input's blur/change event could schedule a late list refresh that replaced an opened paper. Both event-order regressions failed before the fix; repeated values are ignored and entering reading cancels the pending search.
3. Obsidian's date Run button looked enabled while the host component remained disabled. A real listener probe and targeted component enable confirmed the cause; click/Enter regressions now use a faithful mock and the product calls the component's `setDisabled` method.

Each product fix is a separate commit from the test infrastructure.

## Scope and checks not run

- No research relevance or summary-faithfulness evaluation was performed.
- Full library indexing, every topic-management combination, email delivery, and import/export were not added as real UI journeys in this first version. Existing tests remain in place; see the [coverage table](../../../scripts/acceptance/README.md#当前覆盖范围).
- No actual email was sent and no user research library was used. Model exploration read only the authorized workbench `[llm]` configuration.
- The new GitHub Actions workflow was checked locally. Remote CI was not run: the branch has not been pushed or merged.
- Other operating systems and Obsidian versions were not exercised in this session.
