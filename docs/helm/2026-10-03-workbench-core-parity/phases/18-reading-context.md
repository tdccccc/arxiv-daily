# P18 — Reading context and navigation

goal_ref: ../goal.md
created: 2026-10-05T11:24:56+08:00
updated: 2026-10-05T11:24:56+08:00
revision: 1

## Outcome

Summary sources及后续附加信息与正文分隔；阅读末尾显示真实token/耗时/具体生成时间；阅读区提供前进后退并恢复跳转上下文。

## Assumptions

- core已有token/LLM耗时callout，补持久化生成时间与结构化读取，历史缺失显示未记录，不使用文件mtime冒充生成时间。
- 日报统计属于整次日报；单篇详细总结属于该详细总结。概览只有来源日报统计时明确标明范围，不冒充单篇消耗。
- 安全解析并保持原文；不使用Computer Use/真实模型/邮件。

## Chunks

1. Strict Red→Green：core生成统计的写入/读取/兼容与时间语义。
2. Strict Red→Green：附加信息分隔、末尾统计展示，中文/英文和缺失值，报告/概览范围明确。
3. Strict Red→Green：前进后退、列表/概览/文档/锚点恢复、滚动位置、无历史时禁用；不能离开工作台回退到DSH父页。

## Verification

定向core metrics/writer测试、工作台DOM/HTTP与导航测试，typecheck/boundaries/build和隔离DSH Host。保留现有返回列表快捷入口。

## Abort / reshape triggers

不可靠历史数据不补造统计；不把浏览器全局history.length当作当前工作台可回退范围；不覆盖用户文献。
