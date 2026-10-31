# 工作台页面看不到服务端自动调度启动的日报任务，跑完也不刷新

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

- 工作台服务端在 `workbench_schedule.enabled` 时按 tick 周期自己启动 "Automatic daily report check"（`apps/cli/src/workbench/server.ts:90-92`，`apps/cli/src/workbench/schedule.ts`）。
- 前端 `apps/cli/src/workbench/web/app.ts:698-724`：只有 `acceptRun` 看到某个任务处于 running 时才安排下一次轮询；`initializeWorkspace`（785-805 行）只在页面加载、重连、保存设置时查一次 `api/runs/current`。

于是页面空闲时，服务端自己启动的任务不会出现在任务托盘里；它完成后列表和日历也不会刷新。用户下次随手点一下才发现日报早已生成（或失败了却没看到原因）。这与 Obsidian 侧已修的 OBS-05（调度器跑完 Dashboard 不刷新）是同一类问题。

## 原因

轮询只由"页面已知的任务"驱动，没有空闲心跳。

## 建议和验收

开启 `workbench_schedule` 时加低频空闲轮询（30–60 秒一次 `api/runs/current`），或改用 SSE 推送任务状态；任务出现/结束时沿用现有的托盘和刷新逻辑。

## 修复后应补的测试

`apps/cli/tests` 的工作台 DOM 测试：页面无交互、服务端 run 从 null → running → completed，托盘出现并在完成后刷新列表/日历。
