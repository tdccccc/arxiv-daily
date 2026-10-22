# 开启调度器时选"Skip today"，会把今天已经完成的日报运行记录改写成"已跳过"，向导退回"还没生成第一份报告"

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-01
> 范围：Obsidian 插件的"启用调度器"流程（设置页开关、引导页第 5 步、Dashboard"More"菜单） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 69f88f5） · 置信度：已复现

## 问题

按引导页的顺序操作是最自然的触发路径：第 4 步"Generate first report"跑完（今天的运行状态是 `completed`，日报文件也在），接着点第 5 步"Turn on daily reports"。插件弹出对话框"Scheduler will check for new papers daily. Run today's summary right now?"，三个按钮是 Cancel / Skip today / Run today。刚刚才生成过今天的日报，用户多半会选"Skip today"。

选了之后，今天的运行状态从 `completed` 被改写成 `skipped`（错误信息写成"user opted out at enable time"）。可见的后果：

- 引导页的"Generate your first report"一项重新变成未完成（"首份报告完成"按 `completed` 状态判断，`skipped` 不算），引导显示"4 of 5 complete"并且不会退场。再点"Generate first report"得到的是"skipped: already done"（`skipped` 也算终态，普通运行会被拒绝），所以当天没有办法让它变回完成，只能等明天的运行，或用"Force run for date…"。
- Dashboard 顶部状态行变成"Last run: never"，状态栏变成"arXiv: idle"（不再显示 last 日期）。
- 日历格仍因日报文件存在而显示为有报告，但运行历史、诊断报告里这一天变成 skipped，与实际不符。

不选"Skip today"、而是在今天尚未运行时选它，是它本来的用途（今天不想再跑）。问题只在于它不检查今天是否已经有终态记录，直接覆盖。

实测（临时测试，已删除）：状态里放入今天 `completed`（papersWritten=7），走 `setScheduleEnabled(true)` 并选 skip，之后状态为 `{"status":"skipped","papersWritten":7,"error":"user opted out at enable time"}`；`getSetupStatus(...).firstReportComplete` 从 `true` 变成 `false`。

## 原因

- `plugin/main.ts:2447-2452`：`choice === "skip"` 时直接 `await this.stateStore.setSkipped(today, "user opted out at enable time")`，没有先读 `this.stateStore.get(today)`。
- `packages/core/src/services/state-store.ts:79-91`：`setSkipped` 无条件把条目改成 `status: "skipped"` 并清掉 `outcome`，对 `completed`、`failed_permanent`、`running` 也一样。
- `plugin/main.ts:2475-2486` 的对话框无论今天处于什么状态都提供这三个按钮，文案"Run today's summary right now?"也没有区分"今天已经有日报"的情况。
- `plugin/src/onboarding.ts:76-86` 的 `firstReportComplete` 只认 `completed`（`isCompletedDiscovery`），所以被改写后立刻回退。

## 可能的解决方案

推荐：`setScheduleEnabled` 在弹对话框之前先看今天的状态。今天已经是 `completed` / `skipped` / `failed_permanent`（`stateStore.isDone(today)`）时不弹"Skip today / Run today"这一问，直接启用调度器；今天是 `running` 时同样不问。对话框只在今天仍是 `pending` / `failed_transient` / 无记录时出现。

备选：保留对话框，但 `choice === "skip"` 分支只在今天没有终态记录时才调用 `setSkipped`，已完成时什么都不写。

## 修复后应补的测试

- 今天已 `completed` 时选 skip：状态仍是 `completed`，`firstReportComplete` 仍为 `true`。
- 今天已 `completed` 时开启调度器：不弹对话框（或弹但不改状态），调度器正常启动。
- 今天无记录或 `pending` 时选 skip：仍写入 `skipped`（原有行为）。

## 修复情况

开启调度器时，如果今天已经是 `completed`、`failed_permanent`、`skipped` 或 `running`，不再弹"Skip today / Run today"这一问，直接启用调度器，今天的记录原样保留，引导页的"首份报告完成"不会再被改回未完成。只有今天没有记录、`pending` 或 `failed_transient` 时才会问。选 Skip today 也改走新的 `StateStore.skipIfOpen`，在持久化的状态上判断：弹窗期间另一个进程（CLI 定时任务等）刚好把今天跑完，也不会被改写，提示会说今天已有结果、原样保留。

测试：`plugin/tests/schedule-enable-today.test.ts`（9 例，旧代码下 5 例失败，其余是原有行为的保护用例）、`state-store.test.ts` 的 `skipIfOpen`（6 例）。
