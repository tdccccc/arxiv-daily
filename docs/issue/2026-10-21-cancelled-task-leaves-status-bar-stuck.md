# 取消文献库扫描/索引/方向生成、PDF 下载、详情摘要后，状态栏和右下角进度浮层会一直停在"Cancelling active tasks · 1 unwinding"

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-04
> 范围：Obsidian 插件状态栏、任务取消 · 严重程度：中 · 状态：未修复 · 置信度：已复现

## 问题

用户在设置页点文献库索引行的"Cancel"，或用命令"Cancel active tasks"/Dashboard"More → Cancel active tasks"取消一个正在进行的非日报任务（文献库扫描、文献库索引、方向生成、PDF 下载、详情摘要、论文标记/笔记写入）。任务本身会正常退出，但状态栏文字停在"arXiv: Cancelling active tasks · 1 unwinding"，窗口底部还留着一个带 0% 进度条的浮层（标题"Cancelling active tasks"，详情"1 unwinding"），没有任何自动隐藏。

只有两种情况会把它清掉：调度器开着并且下一次计时器触发（默认 20 分钟后调用 `setIdle`）；或者之后有别的任务写状态栏。调度器关闭时（只用文献库功能的用户）会永远停在那里，直到重启 Obsidian。看上去像一个取消不掉、卡死的任务。

日报运行不受影响：取消后调度器自己会 `setIdle`。

实测（临时测试，已删除）：真实 `onload` 接线（状态栏 + 操作注册表订阅），调度器关闭，启动 `scanPersonalLibrary()`，再 `operations.cancelAll()`；扫描以取消结束，注册表为空。120 秒后：状态栏文字 `"arXiv: Cancelling active tasks · 1 unwinding"`，浮层可见（没有 `is-hidden`），浮层文字 `Cancelling active tasks0%1 unwinding`。

## 原因

- `plugin/main.ts:486-491`：操作注册表订阅回调在"所有活动操作都已请求取消"时调用 `progress.setTask("Cancelling active tasks", "N unwinding")`。`setTask` 会清掉隐藏计时器并显示浮层（`plugin/src/services/status-bar.ts:102-109`）。这个回调只在 `active.length > 0` 时工作，操作结束后（`active` 为空）没有任何"收尾"动作。
- 这些任务取消后都不会再碰状态栏：`indexPersonalLibraryFullText` 的 catch 先 `throwIfCancelled(operation.signal)` 再抛出，跳过后面的 `setError`（`plugin/main.ts:1897-1905`）；`scanPersonalLibrary` 与 `generatePersonalLibraryDirections` 明确写了"已取消则不调用 `setError`"（`main.ts:1463-1467`、`1333-1337`）；PDF 下载没有进度汇报。
- 先前的修复（归档 `2026-10-18-status-bar-stuck-complete.md`）只处理了 `setComplete`/`setError` 的收尾，没有覆盖这条"取消提示"路径。

## 可能的解决方案

推荐：订阅回调在操作列表变空时收尾。凡是由这个回调写过"Cancelling active tasks"的，列表变为空后调用 `progress.setIdle(...)`（或按设置调用 `setDisabled()`，同时参考 OBS-03）；或者让回调只更新浮层文字而不占用 `task`，操作结束时恢复原状。

备选：让每个会被取消的任务在自己的 `catch` 里按取消结果收尾（`setIdle`），但入口多（索引、扫描、方向生成、PDF、详情摘要），容易遗漏。

## 修复后应补的测试

- 取消一个文献库扫描（或索引）后，操作注册表清空时状态栏恢复为 idle/disabled 文字，浮层隐藏。
- 取消 PDF 下载后同上。
- 取消日报运行的现有行为保持不变。
