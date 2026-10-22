# 状态栏的"arXiv: disabled / idle"文字不跟调度器的真实开关走：开启后还写着 disabled，关闭时手动跑一次反而变成 idle，清空运行状态后仍显示旧的 last 日期

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-03
> 范围：Obsidian 插件状态栏 · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch2`，提交 fcac15e） · 置信度：已复现

## 问题

状态栏文字由零散的事件（`setIdle` / `setDisabled` / `setTask`…）改写，而不是由设置里"调度器是否启用"和运行状态直接推出，于是有三种对不上的情形：

1. **启用后仍显示 disabled。** 设置页打开调度器并在对话框里选"Skip today"（或任何不会立刻跑一轮的启用方式）：设置已保存、调度器计时器已启动，但状态栏仍是"arXiv: disabled"，要等到第一次计时器触发（默认 20 分钟后，`tickIntervalMin`）才会被调度器的 `setIdle` 改掉。这 20 分钟里用户看到"disabled"，会以为开关没生效。
2. **关闭时手动运行后显示 idle。** 调度器是关的，用户用命令"Run today"、Dashboard 的"Run today"或日历运行了一次，结束时调度器无条件调用 `setIdle`，状态栏变成"arXiv: idle · last 2026-10-21"，并把"静止时回到 disabled"的记忆也改成 false（`restingDisabled`），此后一直显示 idle，直到重启 Obsidian 或再切换一次调度器开关，而设置里调度器仍是关的。
3. **清空运行状态后仍显示旧日期。** 命令"Clear run state…"清空后调用 `progress.setIdle(undefined)`，但 `setIdle` 只在参数非空时才更新 `lastCompletedDate`，状态栏继续写"arXiv: idle · last 2026-10-20"，而 Dashboard 顶部已经是"Last run: never"。修改 Daily/Papers 输出目录后切换到一个没有任何完成记录的新位置（`installOutputStores` 同样传 `undefined`）也是如此。

实测（临时测试，已删除，用真实的 `StatusBarController` 与 `SchedulerService`）：

- 情形 1：`setScheduleEnabled(true)` 选 skip 后，`settings.schedule.enabled = true`，状态栏文字 `arXiv: disabled`。
- 情形 2：`schedule.enabled = false` 时 `runForDateNow(today)` 完成后，文字为 `arXiv: idle · last 2026-10-21`。
- 情形 3：清空后 `runState` 条目数为 0，文字仍是 `arXiv: idle · last 2026-10-20`。

## 原因

- `plugin/main.ts:2496-2503` `applyScheduleEnabledRuntime(true)` 只调用 `scheduler.start()`；`packages/core/src/services/scheduling/scheduler-driver.ts:129-140` 的 `start()` 只装计时器，不碰 `progress`。关闭分支才调用 `setDisabled()`，两个方向不对称。
- `plugin/src/services/status-bar.ts:147-160` `setIdle` 无条件把 `disabled`、`restingDisabled` 置为 false，所以任何一次运行（包括调度器关闭时的手动运行，`scheduler-driver.ts:519` 的 `setIdle(latestCompleted())`）都会把"关闭"的外观清掉。
- 同一函数 155 行 `if (lastCompletedDate) this.lastCompletedDate = lastCompletedDate;`：传 `undefined` 表示"没有"，却被当成"不更新"。调用点：`plugin/src/commands.ts:173`、`plugin/main.ts:2349`。

## 可能的解决方案

推荐：让状态栏的"静止文字"由当前设置和运行状态决定。`StatusBarController` 增加 `setResting({ enabled, lastCompletedDate })`（或让 `setIdle` 接收显式的 `enabled`），宿主在三处调用：启用/停用调度器后、清空运行状态后、切换输出目录后；调度器自己的 `setIdle` 也带上当前 `schedule.enabled`，关闭时回到"disabled"而不是"idle"。`setIdle(undefined)` 应当清掉 `lastCompletedDate`。

备选：只修最小的三点——`applyScheduleEnabledRuntime(true)` 之后立即 `setIdle(latestCompleted)`；`setIdle` 的第一个参数显式传入时（包括 `undefined`）总是覆盖；`withBusyGuard` 的 `setIdle` 在设置里调度器为关时改为转发 `setDisabled`。

## 修复后应补的测试

- 启用调度器并选 skip 后，状态栏不再是"disabled"。
- 调度器关闭时手动运行完成后，状态栏仍是"disabled"（或明确区分"idle (scheduler off)"）。
- 清空运行状态、切换到无记录的输出目录后，状态栏不再带旧的"last <date>"。

## 修复情况

状态栏的静止文字现在跟着调度器开关走：开启调度器后（包括在对话框里选“Skip today”）马上变成“arXiv: idle · last …”；调度器关着时手动运行一次，结束后仍是“arXiv: disabled”，不会变成 idle；清空运行状态，或切换到没有完成记录的输出目录后，不再带旧的“last <日期>”。有任务正占着状态栏时切换开关，进度不会被冲掉，等任务结束后再按新开关显示。

测试：`status-bar.test.ts`（控制器的静止文字和开关变化）；`status-bar-wiring.test.ts`（真实的 `onload` 接线、真实调度器和状态栏：启用后、调度器关着时手动运行、“Clear run state…”命令、切换输出目录）。
