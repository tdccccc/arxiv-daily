# 插件卸载只取消当前任务，不阻止新任务：卸载后的实例仍会启动日报运行、详情摘要或 PDF 下载

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 P5
> 范围：插件生命周期 · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 9ee0c13、6580fb4） · 置信度：已复现（注册表部分），调度器窗口为代码推断

## 问题

插件被禁用（或 Obsidian 重新加载插件）后，下列路径仍可能在已卸载的实例上调用 `operations.begin()` / `runForDate`：正在进行的调度轮次、onload 里不等待的 `scheduler.tickTodayScheduled()`、`setScheduleEnabled` 排队的 `scheduler.tickToday()`、面板或设置里的异步后续操作。这些操作创建时未被中止（`onunload` 里的 `cancelAll` 已经执行过），于是卸载后仍可能开始一次日报运行、详情摘要或 PDF 下载：继续调用 LLM 和 arXiv、写入库，再向已销毁的状态栏报告。快速禁用再启用时，新旧两个实例可能同时跑同一天（文件锁保护日期，但旧实例持有过期的设置服务和存储）。

## 原因

- `plugin/main.ts:561-569` `onunload()` 设置 `unloading`，调用 `scheduler.stop()` 和一次 `operations.cancelAll()`。
- `main.ts:253-289` `SettingsOperationRegistry.begin()` 只检查 `outputTransitionActive`。
- `main.ts:489-491` 调度器的 `runForDate` 回调不检查 `this.unloading`。
- `cancelAll` 只中止当时已存在的操作。

## 可能的解决方案

推荐：卸载时让注册表进入终止状态——增加 `close(reason)`，之后 `begin()` 抛错（或返回已中止的句柄），在 `onunload()` 中先于 `cancelAll` 调用。`runForDate` 回调、`setScheduleEnabled` 的延迟 tick、onload 的初始 tick 都先检查 `this.unloading`。

## 修复后应补的测试

- `onunload()` 后 `operations.begin()` 抛错或返回已中止句柄。
- `runForDateNow()` 以取消结束，不调用流水线。
- `onunload` 时正在等待的 tick 不会启动运行。

## 修复情况

卸载时调用 `SettingsOperationRegistry.close()`，之后 `begin()` 返回已中止的句柄（经正常取消路径退出）；`runForDate` 回调、`withOutputOperation`、启用调度的意图与延迟的运行今天都检查 `unloading`。
