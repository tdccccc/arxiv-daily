# Dashboard 日历把"日报被删掉的已完成日期"画成可运行（绿色播放键），点击后却只提示"skipped: already done"，什么也不会生成

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-02
> 范围：Obsidian 插件 Dashboard 日历 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch2`，提交 a7094cb） · 置信度：已复现

## 问题

用户删掉了最近几天里某一天的日报文件（或把它移走了），那一天在运行状态里仍是 `completed`。设计意图（`docs/helm/2026-07-22-daily-rerun-consistency`，成功标准"A recent completed date becomes runnable when its daily file is missing"）是：这个日期在日历上变成绿色可运行格，点一下就重新生成。

实际：日历确实把它画成可运行（播放图标，无障碍标签"run daily report"），但点击后调用的是 `scheduler.runForDateNow(date, { trigger: "calendar" })`，调度器按普通模式处理，看到状态是 `completed` 就直接返回"skipped: already done"，流水线一次都没有调用。用户看到一条提示"arXiv Daily 2026-10-20: skipped: already done"，日报没有回来，格子也还是可运行，可以无限点下去。

绕过办法是 Dashboard"More → Force run for date…"（先清掉该日状态），但日历格本身给出的承诺是错的。

实测（临时测试，已删除）：状态里放入 `2026-10-20` 为 `completed`（papersWritten=10），用真实的 `SchedulerService` 调 `runForDateNow("2026-10-20", { trigger: "calendar" })`，返回 `{"kind":"skipped","reason":"already done"}`，注入的流水线调用次数为 0。日历侧的"可运行"判断由现有单元测试 `plugin/tests/dashboard/calendar-state.test.ts`（"makes a missing completed report runnable…"）覆盖，两边合起来就是上面的现象。

## 原因

- `plugin/src/dashboard/calendar.ts:92-118`、`200-207`：`isCalendarRunWhitelisted` 有意放行"非 0 篇的 completed 且无日报"，`resolveCalendarCellState`（同文件 81 行）随后判为 `runnable`。
- `plugin/src/dashboard/view.ts:1255-1278` `runDateFromCalendar`：只调用 `runForDateNow(date, { trigger: "calendar" })`，没有在运行前清掉该日期的 `completed` 状态。
- `packages/core/src/services/scheduling/scheduler-driver.ts:635-641`：`runMode === "normal"` 且 `store.isDone(date)` 时返回 `skipped: already done`；`trigger: "calendar"` 只影响运行历史，不改变模式（514 行只有 `clearDateBeforeRun` / `retryFailed` 会改变模式）。
- 当时的 helm 阶段文档（`phases/02-missing-report-rerun.md`）写明了"如果调度器的手动运行仍然拒绝 completed 状态，改成在日历执行前清掉该日状态"这条备用方案，但当前代码里没有实现。

## 可能的解决方案

推荐：日历对"completed 但日报文件缺失"的格子走强制路径。`runDateFromCalendar` 判断该日期当前没有日报（即 `calendarDailyReports` 里没有）且运行状态是 `completed` 时，调用 `scheduler.forceRunForDate(date)`（它清状态再跑，且仍不会覆盖已存在的日报文件），而不是 `runForDateNow`。

备选：给 `runForDateNow` 增加一个"日报缺失时允许重跑 completed"的显式选项（由日历传入），在调度器里做同样的清状态。

## 修复后应补的测试

- 日历点击"completed 且日报缺失"的日期：流水线被调用，结果不是"already done"。
- 日历点击"completed 且日报存在"的日期（有报告的格子本来就不是可运行格）：行为不变。
- 点击"pending / failed_transient"的可运行格：行为不变。

## 修复情况

日历上“已完成但日报文件被删掉”的日期，点击现在会真的重新生成日报，不再只提示“skipped: already done”：点击时先清掉这一天的已完成状态再运行（和“Force run for date”相同，仍不会覆盖已存在的日报文件）。待运行、失败的日期，以及日报还在的已完成日期（本来就不是可运行格），行为不变。代价：这种重跑在运行历史里的触发方式记为 `force`，不再是 `calendar`。

测试：`dashboard-calendar-rerun.test.ts`（用真实调度器：日报被删的已完成日期会调用流水线；没完成过的日期仍走普通运行；日报还在的日期不会被清状态；正在运行的日期不受影响）；`calendar-state.test.ts` 里已有的点击用例补上了点击现在要读取的运行状态。
