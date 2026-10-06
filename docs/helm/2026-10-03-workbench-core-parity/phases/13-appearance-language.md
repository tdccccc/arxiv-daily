# P13 — Appearance and interface language

goal_ref: ../goal.md
created: 2026-10-04T17:39:35+08:00
updated: 2026-10-04T18:11:35+08:00
revision: 2

## Outcome

设置新增Appearance/外观，集中主题和界面语言下拉；用户选择中文或English后整个工作台控件一致，不改变论文内容或summary_language。

## Assumptions

- 默认中文；主题支持light/dark/system。保存设置后应用，避免切换时丢失草稿。
- 研究内容、用户输入、模型标识和原始诊断不作为UI文案翻译。
- UI偏好持久化独立于CLI研究配置，侧栏保存不能覆盖语言/主题。

## Chunks

- Preference persistence: strict Red→Green，patch merge/并发保存/旧格式兼容，保持sidebar behavior。
- Localization: strict Red→Green，中英文设置与主要浏览/日历/状态/控件，保留user-content；Appearance下拉、theme system和locale persistence。
- Integrated flow: strict Red→Green，保存设置切语言不改变summaryLanguage与draft；移除header快捷theme按钮，测试初始加载和重启。

## Verification

- 相关DOM/API测试、全量CLI、DSH host、typecheck/boundaries/inventory/build/pack。无Computer Use和真实模型调用。

## Abort / reshape triggers

- 不能用全文替换翻译研究Markdown/论文标题，不把UI语言写入论文总结语言。

## Acceptance

- 4e57cc8: core appearance type/default/validation and zh/en catalogs; host prefs locked merge with legacy defaults. Observed schema/concurrent update Red, then Green.
- a99271e: settings Appearance group, full workbench chrome localization, saved language/theme remount, system-theme listener lifecycle and unchanged user content/summaryLanguage. UI/localization cases observed Red then Green.
- Full CLI:35files292tests pass; actual isolated DSH Host/component suite20tests pass. Core/CLI typechecks, boundaries, inventory, diff check and build/pack pass. After catalog review found Topic/Theme ambiguous source, added failing assertion and corrected context; targeted23tests pass.
- Installed DSH0.1.11 in Web profile, matching bundle hash. Existing DSH session not stopped. No Computer Use/live model/email calls.
- Core now owns reusable appearance definitions and UI dictionaries. Obsidian still owns native settings rendering and plugin-data persistence, and was not migrated to these new translations in this phase. CLI and DSH use the same workbench/business configuration.
- Translation is limited to UI messages. Research titles, Markdown, user text, IDs, and unknown raw provider diagnostics remain unchanged. Existing static common workbench errors are localized.
