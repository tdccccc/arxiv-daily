# 没连接文献库的用户每次搜索都看到"Library search unavailable"错误块，且每次重绘都重新嵌入查询

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 P2
> 范围：插件面板搜索 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-18-review` 提交 9253311，取消余项见 `fix/followup-library-abort`） · 置信度：已复现

## 问题

个人文献库是可选功能，但只要搜索词达到 2 个字符，`refreshLibrarySearch()` 就会运行。没有连接文献库时 `searchPersonalLibraryFullText()` 抛出"Choose a personal library first"，面板在结果下方渲染一个错误样式的"Library matches / Library search unavailable: Choose a personal library first"块。所有没设置文献库的用户每次搜索都会看到这个错误。

另外 `render()` 每次都调用 `refreshLibrarySearch()`，而收藏/取消收藏、批量操作、切换日历月份、刷新最近日期、每次 `reloadIndex()` 都会触发 `render()`，同一个查询被反复嵌入：嵌入模式为 remote 时每次都是一次网络请求，本地模式则是一次模型推理。过期结果会按令牌丢弃，但工作本身从未取消。

## 原因

- `plugin/src/dashboard/view.ts:2343-2392` `refreshLibrarySearch()` 不检查文献库连接或索引就绪状态，把"未连接"当成失败。
- `view.ts:551` `render()` 无条件调用，没有记住上一次查询。
- `plugin/main.ts:2046-2073`、`plugin/src/dashboard/library-search-block.ts:46-52`。

## 可能的解决方案

推荐：
1. 未连接文献库时直接清空容器、不调用搜索；已连接但未建索引时显示一行中性提示，不用错误样式。
2. 记住上一次提交的查询和索引版本，查询未变时重绘不再重新搜索。
3. 可选：给搜索传入 `AbortSignal`，让新搜索取消旧搜索（需要 core 改动时另行处理）。

## 修复后应补的测试

- 未连接文献库时 3 字符搜索不调用 `searchPersonalLibraryFullText`，不渲染错误块。
- 查询不变时收藏切换/批量操作不再触发搜索。

## 修复情况

未连接文献库时不调用搜索、不显示任何内容；已连接但未建索引时显示一行中性提示；结果按查询、索引版本和嵌入模式缓存，重绘共享进行中的搜索。

**取消余项已修复**（分支 `fix/followup-library-abort`）：输入新搜索词后，上一个还在进行的文献库搜索会被立即取消，远程嵌入的网络请求随之中止，本地模型则在批次之间停止；面板关闭时同样取消。同一个查询的重复重绘（收藏、批量操作等）仍共享同一个进行中的搜索，不会被取消。被取消的搜索既不会写入缓存，也不会显示成错误块；回到被取消的查询时会重新搜索。搜索词变短（少于 2 个字符）、文献库断开或索引不可用时，进行中的搜索也一并取消。核心搜索在信号已取消时直接以"已取消"拒绝，且即使嵌入模型忽略信号，取消后也不会继续做检索。
