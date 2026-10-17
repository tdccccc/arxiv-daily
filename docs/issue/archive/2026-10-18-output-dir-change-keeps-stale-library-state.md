# 在设置中修改输出目录后，文献库状态不会刷新（`reloadStateStoreForOutputPaths()` 在生产代码中无人调用）

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 P4
> 范围：插件设置、文献库 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 81840b5） · 置信度：已复现

## 问题

文献库目录、方向提案、索引状态（"last indexed N papers"）、新论文提醒计数以及所有版本守卫（`libraryOutputRevision`）都和输出目录绑定（存储按 `settings.output` 构建，`.index` 根目录由 dailyDir/papersDir 推出）。用户在设置里修改 Daily/Papers 文件夹后，日报相关存储被切换，但内存里的文献库状态仍描述旧位置：

- 设置行仍声称新路径下并不存在的索引；
- 审阅弹窗和提醒显示旧的提案和新论文计数；
- 下一次审阅操作或搜索读取新的（空）位置，出现"search index not ready"或"Scan and load the personal library catalog first"之类的错误，直到重启 Obsidian 或重新选择文件夹。

接受/生成/预览等比较 `libraryOutputRevision` 的守卫也察觉不到切换。

## 原因

- 输出路径切换由 `SettingsChangeService`（`plugin/src/settings/change-service.ts:165-206`）调用 `plugin.prepareOutputStores` / `installOutputStores` 完成。`installOutputStores`（`plugin/main.ts:2314-2344`）只切换 `stateStore` / `runHistoryStore` 和进度文字。
- 负责文献库部分的 `reloadStateStoreForOutputPaths()`（`main.ts:2346`：递增 `libraryOutputRevision`、取消文献库操作、重新加载目录与提案）在生产代码中没有调用方，只有测试调用。测试直接调这个孤立方法，没走真实的设置路径。

## 可能的解决方案

推荐：把文献库切换并入 `installOutputStores`（提交后、仍在输出切换期间，没有活动操作）：递增 `libraryOutputRevision`，清空文献库目录/提案/新论文/已索引论文，然后排队重新加载目录、配置文档和索引状态，错误作为提交后错误报告。删除孤立方法，测试改走 `settingsChanges.changeValue('output.papersDir', ...)`。

## 修复后应补的测试

- 通过 `settingsChanges.changeValue('output.papersDir', ...)` 修改后，`libraryOutputRevision` 递增，文献库状态从新位置重新加载，索引状态重新计算。

## 修复情况

`installOutputStores` 重置文献库状态、递增 `libraryOutputRevision` 并从新位置重新加载；孤立的 `reloadStateStoreForOutputPaths` 已删除，测试改走真实设置路径。
