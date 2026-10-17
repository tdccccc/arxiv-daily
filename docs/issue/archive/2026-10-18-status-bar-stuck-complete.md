# 文献库扫描、索引、方向生成、详情摘要结束后状态栏一直停在"Complete"/"Failed"；调度器空闲轮次会覆盖正在运行的文献库任务

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 P6
> 范围：插件状态栏 · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 a4667f4） · 置信度：已复现（覆盖部分为代码推断）

## 问题

非日报任务以 `progress.setComplete()` / `setError()` 结束，从不调用 `setIdle()`。`setComplete` / `setError` 把字面文本"Complete"/"Failed"写进状态栏，只让浮动面板自动隐藏（4 秒 / 10 秒），状态栏文本本身从不恢复。调度器关闭时（或直到下一次调度轮次），文献库扫描/索引、方向生成、诊断或手动详情摘要之后，状态栏会一直显示"Complete"或"Failed"，看起来像一个早已被遗忘的任务永久失败了。两者还保留 `task`，之后只更新阶段的进度会显示在过期的任务标题下。

反过来，每次调度轮次都无条件以 `setIdle()` 结束；长时间的文献库索引/扫描运行期间，下一次轮次会把它的进度替换为"arXiv: idle"。

## 原因

- `plugin/src/services/status-bar.ts:101-130`：`setComplete` 设置 `el.textContent = 'Complete'` 并安排隐藏面板，之后没有任何地方重置文本或 `task`；`setError` 同理。
- 文献库方法（`plugin/main.ts:1332-1337`、`1462-1467`、`1876-1905`）和 `ManualFetchService`（`packages/core/src/services/manual-fetch.ts:302`）以 `setComplete`/`setError` 结束。
- 调度轮次（`packages/core/src/services/scheduling/scheduler-driver.ts:171-183` 等）调用 `setIdle` 时不检查其他活动操作。

## 可能的解决方案

推荐：
1. `StatusBarController` 在 complete/error 的自动隐藏计时器里同时恢复状态栏文本（清空 task/stage/batch 后回到 idle/disabled 文本），并在 `setComplete`/`setError` 时清空 `task`。"Failed"仍保留 10 秒，但不再永久显示。
2. 宿主提供的进度对象在 `operations.snapshot()` 非空时忽略调度器的 `setIdle`。

## 修复后应补的测试

- `setComplete`/`setError` 并过了自动隐藏延迟后，文本恢复为 idle/disabled，task 被清空。
- 文献库索引进行中时，调度器空闲轮次不覆盖任务文本。

## 修复情况

complete / error 面板隐藏后，状态栏恢复 idle / disabled 文本并清空已结束的任务；调度器的进度对象在有活动操作时丢弃 idle / disabled 更新。
