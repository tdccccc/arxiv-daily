# 面板"下载 PDF"、"刷新"、运行日报后不重新读取 papers.json，"打开 PDF"一直打开在线版本

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 P1
> 范围：插件面板 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 1ffd8db） · 置信度：已复现

## 问题

在面板某一行点击"Download PDF"后，PDF 已保存、`papers.json` 也记下了 `pdfPath`，但面板仍显示内存里的旧条目（`pdfPath` 为空）。"Open PDF"依旧打开 arXiv 网页 PDF 而不是本地文件，相似论文弹窗也仍写着"PDF online"。直到受管目录下增删某个 Markdown 文件或重开插件/视图才更新。

同样的问题出现在所有不重置 `lastSyncedHistoryPaths` 的 `reloadIndex()` 调用：工具栏"Refresh"、"Retry"、运行今天/选定日期/全部待处理/重试失败、日历运行、面板命令。凡是只改 `papers.json` 而不增删 daily/papers 目录下 .md 文件的变化（PDF 路径、命令或其他窗口改的状态/优先级、只更新索引字段的强制重跑），面板都看不到，包括用户主动点"Refresh"。

## 原因

- `plugin/src/dashboard/view.ts:324-395` `reloadIndex()` 在 `shouldSkipDashboardHistorySync`（`plugin/src/dashboard/files.ts:42-53`）判断 Markdown 路径集合不变时跳过 `syncDashboardHistory`，并直接用旧的 `this.entries` 渲染（354-359 行）；条目只在 `syncDashboardHistory` 内部从 `store.load()` 重新加载（361-371 行）。所以"跳过"不仅跳过了库到索引的对账，也跳过了把 `papers.json` 读回内存。
- `PdfService.downloadById` 写的是 `<root>/pdfs/<id>.pdf`（不是 Markdown）并调用 `setPdfPath`，路径集合不变。

## 可能的解决方案

推荐：路径集合不变时只跳过昂贵的库扫描/同步，但仍执行 `store.load()`，重建条目、搜索索引、详情引用和日报列表。`store.load()` 相比读取所有笔记很便宜。

备选：在下载 PDF 和各运行入口里先把 `lastSyncedHistoryPaths` 置空，"Refresh"/"Retry"改调 `refreshFromVault()`。改动更小，但给以后的调用方留下同样的坑。

## 修复后应补的测试

- 路径集合不变时调用 `store.setPdfPath()` / `setStatus()` 后，`reloadIndex()` 让 `view.entries` 反映新值。
- 现有 `shouldSkipDashboardHistorySync` 单测保持通过。

## 修复情况

Markdown 文件集合未变时只跳过库扫描，仍重新读取 `papers.json`。下载 PDF、刷新、其他地方设置的标记都能立即显示。
