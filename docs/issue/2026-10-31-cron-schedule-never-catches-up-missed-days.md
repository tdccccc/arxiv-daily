# cron 定时只跑当天，机器关机错过的日子永远不补

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：中 · 置信度：代码推断

## 问题与依据

- `apps/cli/src/schedule-cmd.ts:57-66`：`schedule install` 为每个时段生成 `… run --today`。
- `apps/cli/src/main.ts:292-300`：`run --today` 只调用 `scheduler.runForDateNow(今天)`，不看其他日期。
- 5 天回看补跑逻辑在 `scheduler.tick()`（`packages/core/src/services/scheduling/scheduler-driver.ts:174-239`，`constants.ts` 的 `LOOKBACK_DAYS = 5`），只由 `run --scheduled` 触发；而 `main.ts:240-243` 把 `run --scheduled` 的启用绑定在 `[workbench_schedule].enabled`（默认 false），`tick()` 还要求当前时刻在该表的时间窗内——cron 用户摸不到。
- `interval_hours` 只是让当天多试几次，不涉及其他日期。
- `apps/cli/README.md:118` 没有说明这一点；`docs/getting-started.md:123` 对 Obsidian 插件承诺"错过的工作日可以之后补跑"，CLI 没有对应能力。

用户看到的是：笔记本某天没开机（或 cron 触发时正在休眠），那一天的日报永远缺席，日历上空着，没有任何提示；只能自己想起来手动 `run --date`。

## 原因

cron 行和补跑逻辑是分开实现的；`run --scheduled` 的启用条件是为工作台内置调度设计的。

## 建议和验收

- 让 cron 行走补跑路径：例如新增 `run --today --catch-up`（或让 `run --scheduled` 以 `[schedule]` 意图作为启用条件和时间窗），执行与 `tick()` 相同的回看逻辑。
- 补跑仍受 isDone / 已完成保护，不会重写已有日报；每个 pending 日期一次。
- README 的 schedule 小节说明补跑范围（5 天）。

## 修复后应补的测试

- `schedule-cmd` 测试：生成的 cron 行使用补跑模式。
- `main` 测试：补跑模式对回看窗口内每个 pending 日期各调用一次，已完成 / 已跳过 / 永久失败的日期不重跑；周末跳过。
