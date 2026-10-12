# Journal — model-driven release exploration

## 2026-10-12 — note (P4 chunk 1 results and triage decisions)

- evidence: first real-model release runs (deepseek / gpt-6-luna). Workbench: 6/6 goal tasks passed, 3 personas explored, 568,828 input tokens, 80 calls, 4.7 min (/tmp/p4c1-workbench-1791719806). Obsidian: 13/14 passed, 1 blocked, 3 personas explored, 696,135 input tokens, 106 calls, 8.9 min (/tmp/p4c1-obsidian-1791720467). add-arxiv-category was blocked: the model clicked "Add category" 16 times and never used the select, although cs.LG was in the options. Each click immediately saved the next unused category (astro-ph, …), and the growing settings page exceeded the 100,000-character observation budget. A first-time-user finding: the workbench banner "配置已改变，请重启工作台后再操作。" implies every action is blocked while reading still works; the state leaked from the preceding external-config-conflict task. The Obsidian first-time-user persona dead-ended after opening a plain note outside the plugin surface.
- change: user decisions (2026-10-12): (1) "Add category" adds an unselected row and saves only after a category is chosen; (2) reword the workbench config-changed banner to say which actions need a restart. Owner decisions: fix three harness gaps — task isolation after an external config edit, a way back to the plugin surface from non-plugin Obsidian views, and compact repeated select options in observations.
- disposition: keep all P1–P3 work; product changes are separate commits with tests.
- next: P4 chunk 2 — the two product changes in parallel, then the harness fixes, then a scripted full-library regression.

## 2026-10-12 — note (close)

- evidence: confirmation real-model release runs after the P4 chunk 2 fixes. Workbench: 6/6 passed, 3 personas explored, 0 findings, 80 calls, 574,869 input tokens, 6.0 min (/tmp/p4c3-workbench-1791813963). Obsidian: 14/14 passed, 3 personas explored, 0 findings, 105 calls, 514,059 input tokens, 8.1 min (/tmp/p4c3-obsidian-1791814427). add-arxiv-category passed in 5 steps; the first-time-user persona used home 4 times; the config-changed state no longer leaks between tasks.
- change: all success criteria met. Obsidian has 14 goal tasks, above the 10–12 in the criterion; this is intentional (the reorder task was replaced and reading tasks were added). Release defaults calibrated; README documents the release workflow.
- disposition: keep everything; branch test/model-exploration is unpushed pending user approval.
- next: none in this initiative; ask the user about pushing and a PR.
