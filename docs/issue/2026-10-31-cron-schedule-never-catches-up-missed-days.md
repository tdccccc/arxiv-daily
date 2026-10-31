# cron 定时只跑当天，机器关机错过的日子永远不补

状态：不改行为——ADR 0003 记录为有意设计；README 已说明（分支 `fix/2026-10-31-review`）

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：中 · 置信度：代码推断

## 问题与依据

- `apps/cli/src/schedule-cmd.ts:57-66`：`schedule install` 为每个时段生成 `… run --today`。
- `apps/cli/src/main.ts:292-300`：`run --today` 只调用 `scheduler.runForDateNow(今天)`，不看其他日期。
- 5 天回看补跑逻辑在 `scheduler.tick()`（`packages/core/src/services/scheduling/scheduler-driver.ts:174-239`，`constants.ts` 的 `LOOKBACK_DAYS = 5`），只由 `run --scheduled` 触发；而 `main.ts:240-243` 把 `run --scheduled` 的启用绑定在 `[workbench_schedule].enabled`（默认 false），`tick()` 还要求当前时刻在该表的时间窗内——cron 用户摸不到。
- `interval_hours` 只是让当天多试几次，不涉及其他日期。
- `docs/getting-started.md:123` 对 Obsidian 插件承诺"错过的工作日可以之后补跑"，CLI 没有对应能力。

用户看到的是：笔记本某天没开机（或 cron 触发时正在休眠），那一天的日报缺席，日历上空着，没有提示；只能自己想起来手动 `run --date`。

## 处理情况

复核时发现这是记录在案的产品决定，不是遗漏：

- ADR 0003（`docs/adr/0003-two-products-cli-config-and-data-portability.md:88`）："CLI does not batch lookback days (`run-pending` retired); missed days use `run --date`."
- 0.3.4 发布说明（`docs/releases/0.3.4.md:87`）：没有多日 `run-pending`，cron 应调用 `run --today`，错过的日子用 `run --date …`。
- `main.ts:463-467` 至今保留着 `run-pending` 的退役提示。

因此本次**不改变行为**，只补上审查发现的文档缺口：`apps/cli/README.md` 的 schedule 小节现在明确写出"cron 条目只生成当天日报；机器关机错过的工作日不会补，用 `arxiv-daily run --date YYYY-MM-DD`，并指向 ADR 0003"。

如果将来要让 CLI 补跑，应先修订 ADR 0003，再考虑复用 `scheduler.runAllPending()`（它已按 arXiv `/recent` 列出的日期过滤、受 isDone 保护）；届时 `run --scheduled` 的启用条件也要从 `[workbench_schedule]` 解耦。

## 修复后应补的测试

若日后改行为：`schedule-cmd` 测试断言 cron 行使用补跑模式；`main` 测试断言补跑对回看窗口内每个 pending 日期各调用一次，已完成 / 已跳过 / 永久失败的不重跑，周末跳过。
