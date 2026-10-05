# P18 — Reading context and navigation

goal_ref: ../goal.md
created: 2026-10-05T11:24:56+08:00
updated: 2026-10-05T11:35:27+08:00
revision: 2

## Outcome

Summary sources及后续附加信息与正文分隔；阅读末尾显示真实token/耗时/具体生成时间；阅读区提供前进后退并恢复跳转上下文。

## Assumptions

- core已有token/LLM耗时callout，补持久化生成时间与结构化读取，历史缺失显示未记录，不使用文件mtime冒充生成时间。
- 日报统计属于整次日报；单篇详细总结属于该详细总结。概览只有来源日报统计时明确标明范围，不冒充单篇消耗。
- 安全解析并保持原文；不使用Computer Use/真实模型/邮件。

## Chunks

1. [x] Strict Red→Green：core生成统计的写入/读取/兼容与时间语义。
2. [x] Strict Red→Green：附加信息分隔、末尾统计展示，中文/英文和缺失值，报告/概览范围明确。
3. [x] Strict Red→Green：前进后退、列表/概览/文档/锚点恢复、滚动位置、无历史时禁用；不能离开工作台回退到DSH父页。

## Verification

定向core metrics/writer测试、工作台DOM/HTTP与导航测试，typecheck/boundaries/build和隔离DSH Host。保留现有返回列表快捷入口。

## Abort / reshape triggers

不可靠历史数据不补造统计；不把浏览器全局history.length当作当前工作台可回退范围；不覆盖用户文献。

## Accepted implementation

- Core writes generatedAt at the Markdown persistence boundary and exact versioned metrics inside the existing folded callout. splitGenerationMetrics reads structured/legacy records, tolerates CRLF, ignores fenced examples and preserves unknown markers/later user notes. Missing provider usage stays incomplete, never invented as zero. Detail wall time covers its own generation; manual detail includes fetch-to-write preparation.
- Reader projects saved statistics separately from body; Summary sources/总结依据 headings receive a divider. Overview moves source sections and following references into an appendix, with the generation footer last. Input/output/total tokens, elapsed generation and cumulative LLM durations and exact UTC timestamp appear in zh/en. Overview uses associated daily statistics with explicit whole-report scope; detailed-note statistics have separate scope. No mtime inference or user-file rewriting.
- Header back/forward retains the return-list shortcut. Workbench session indexes bound traversal without using history.length; scroll records, query/filter state, document links/anchors and forward-branch truncation are covered. Unknown browser entries reset the local boundary; stale async results cannot replace the current view. Same-document anchors preserve filters.
- Observed Red→Green: absent metrics API/generated time/detail walltime/CRLF, missing appendix and metadata projection, missing history controls, same-document filter loss. Core332 tests/9 files passed; CLI full358/40; Obsidian full1007/50; DSH20, zero skipped, including actual configured and first-run Host. Core/CLI/plugin typechecks, boundaries, inventory and builds pass.
- DSH0.1.16 linux/x64 packaged at extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.16.tgz; sha1 1099965614447a7162e0a495b47681f9a469e3b3. No production install, remote push, user-vault edits, real models/email or Computer Use. Electron visual review and the entire core/native cross-platform suites were not run.

Commits: core ade91ff; workbench52384ad. Existing files missing timestamps remain explicitly unrecorded; newly generated files carry durable metadata. P2/P3/P4 remain pending outside this reading request.
