# 调度器在后台跑完日报后，已打开的 Dashboard 不会刷新：顶部一直显示"Running…"，列表和日历还是旧的

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-05
> 范围：Obsidian 插件 Dashboard 视图 · 严重程度：中 · 状态：未修复 · 置信度：已复现

## 问题

Dashboard 只在自己发起的动作结束后重读数据（点"Refresh"、"Run today"、日历运行、收藏等），以及命令面板里的运行命令结束后（`refreshOpenDashboardViews`）。由调度器自己发起的运行不在其中，而这是插件最主要的使用方式：

- Obsidian 启动时恢复了上次的工作区布局，Dashboard 标签页被还原并 `onOpen()`；几乎同时 `onload` 里的 `tickTodayScheduled()` 开始补跑今天的日报。Dashboard 打开时那个后台运行还在进行，顶部状态行渲染成"Running…"。
- 运行结束后，状态行依然是"Running…"（它只在整页重绘时重新计算），新论文不出现，日历里今天那一格仍是旧状态（可能还是"可运行"的播放键），"Last run" 也不更新。需要用户自己点"Refresh"才会正确。
- 定时轮次（每 20 分钟一次的计时器）在 Dashboard 打开期间完成日报，也是同样的结果。

实测（临时测试，已删除）：用真实的 `registerDashboardView` 创建视图和真实的操作注册表，先开始一个 `daily-run` 操作再 `onOpen()`，然后结束该操作并把今天的运行状态写成 `completed`；等待 50 ms 后，视图的 `render` 只在打开时调用过 1 次，`reloadIndex` 只调用过 1 次，之后没有任何重绘。

## 原因

- `plugin/src/dashboard/view.ts:315-331`：视图只在 `onOpen` 和显式的 `refreshFromVault()` 里读取数据；全文没有订阅 `plugin.operations`、`vault` 事件或调度器完成事件（整个插件代码没有任何 `registerEvent`）。
- `plugin/src/dashboard/view.ts:764-781` `renderHeader`：状态行的"Running…"取决于渲染那一刻的 `operations.snapshot().length > 0`，之后不会更新。
- `plugin/main.ts:502-519`：构造 `SchedulerService` 时 `onDailyCompleted` 只接了邮件投递（`deliverCompletedDigest`），没有调用 `refreshOpenDashboardViews`。该函数只被 `plugin/src/commands.ts`（命令面板的运行命令）和设置页的"生成首份报告"调用。

## 可能的解决方案

推荐：让视图订阅 `plugin.operations.subscribe`。操作列表发生变化时只更新顶部状态行；当一个 `daily-run` 操作从有变无（运行结束）时，用 `refreshFromVault()` 重读数据（视图关闭后取消订阅，注册到 `ItemView.register` 以便卸载时自动清理）。

备选：在 `main.ts` 构造调度器时，让 `onDailyCompleted` 之外再增加一个"运行批次结束"回调，里面调用 `refreshOpenDashboardViews(this)`。覆盖面略小（取消、失败的批次要单独处理）。

## 修复后应补的测试

- Dashboard 打开后，一个 `daily-run` 操作结束：视图重新读取 papers.json 并重绘，状态行不再是"Running…"。
- 视图关闭后，订阅被取消，操作结束不再触发重绘。
- 非日报类操作（PDF 下载等）结束时只更新状态行，不触发整页重载。
