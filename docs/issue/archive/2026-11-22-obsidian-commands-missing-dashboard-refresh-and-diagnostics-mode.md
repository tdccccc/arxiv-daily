# Obsidian 命令面板与诊断三处遗漏：状态/标记命令不刷新 Dashboard、远程嵌入诊断强行拉取本地模型、打开今日日报未规范化路径

状态：已修复（分支 `fix/2026-11-22-review`）

> 来源：2026-11-22 main `2557b90a` 代码审查（Obsidian 命令与诊断）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

在 `plugin/src/commands.ts` 与 `plugin/main.ts` 中存在三处独立但均落在命令/诊断路径上的缺陷：

1. **`clear-run-state` 与 `set-paper-status` / `mark-current-*` 命令执行后不刷新已打开的 Dashboard，且 `clear-run-state` 不检查活跃任务**：
   - `plugin/src/commands.ts:161-175`（`clearRunState`）：清空 `plugin.stateStore.clearAll()` 后只发了一条 Notice，**没有检查 `plugin.operations.snapshot()` 是否有正在运行的日报任务**（若在日报跑一半时执行会擦除当前 `running` 状态导致状态错乱），也没有调用 `refreshOpenDashboardViews(plugin)`（导致打开的 Dashboard 日历仍显示旧的 completed/failed 标记）。
   - `plugin/src/commands.ts:222-251`（`setPaperMark`，被 `set-paper-status`、`mark-current-saved`、`mark-current-read`、`mark-current-archived`、`mark-current-starred` 等命令调用）：更新 `papers.json` 后同样没有调用 `refreshOpenDashboardViews(plugin)`（对比第 288 行 `openArxivIdPicker` 正确调用了刷新）。用户从笔记用命令收藏/归档论文后，已打开的 Dashboard 列表仍显示旧标记，直到手动刷新。
2. **`diagnoseEmbedding()` 忽略 `embedding.mode === "remote"`，强行初始化并下载 ~130 MB 本地模型**：
   - `plugin/main.ts:2291-2319`：
     ```ts
     private async diagnoseEmbedding(): Promise<EmbeddingDiagnostics> {
       const embedding = createTransformersEmbeddingModel();
       ...
       await embedding.embed(["diagnostic probe"]);
     ```
   - 当用户配置了远程嵌入（尤其是因网络无法访问 Hugging Face 才切到远程嵌入的用户）并运行 `Run personal library indexing diagnostics` 命令排查问题时，诊断方法完全无视 `this.settings.embedding.mode`，硬编码构造本地 `createTransformersEmbeddingModel()` 去拉 Hugging Face ~130 MB 模型，导致诊断报告反映的根本不是当前生效的嵌入后端，还可能在离线/受限网络下长时间卡住或报无关的 Hugging Face 下载失败。
3. **`openTodayDaily()` 手工拼接路径而非使用 `MarkdownWriter.dailyPath()`**：
   - `plugin/src/commands.ts:297-305`：`const path = \`${plugin.settings.output.dailyDir}/${today()}.md\`;` 直接传给 `vault.getAbstractFileByPath(path)`，没有通过 `normalizePath` 或 `buildMarkdownWriter().dailyPath(today())` 规范化；若 `dailyDir` 带有尾斜杠或反斜杠（或路径规范化与 Vault 索引不一致），`getAbstractFileByPath` 返回 `null` 误报找不到今日日报。

## 原因

- 各命令在后来陆续补充 `refreshOpenDashboardViews` 和 `operations` 保护时遗漏了 `clearRunState` 与 `setPaperMark`。
- `diagnoseEmbedding` 早于 ADR 0008（远程嵌入模式）编写，新增 `buildEmbeddingModel()` 后未同步更新诊断入口。
- `openTodayDaily` 未复用 `MarkdownWriter.dailyPath()`。

## 建议和验收

- `clearRunState()`：若有活跃日报任务运行（或通过 `withOutputOperation` 串行门控）时拒绝清空并提示先取消运行；清空成功后调用 `await refreshOpenDashboardViews(plugin)`。
- `setPaperMark()`：更新 `papers.json` 成功后调用 `await refreshOpenDashboardViews(plugin)`。
- `diagnoseEmbedding()`：当 `this.settings.embedding.mode === "remote"` 时，先校验远程配置并通过当前 `buildEmbeddingModel()` 探测（或在未授权时仅汇报配置与授权状态、不悄悄下载本地模型），清楚标明当前诊断的是 `remote` 还是 `local` 模式。
- `openTodayDaily()`：改用 `plugin.buildMarkdownWriter().dailyPath(today())`。

## 修复后应补的测试

- `plugin/tests/commands.test.ts`：
  - 调用 `set-paper-status` / `mark-current-*` 与 `clear-run-state` 后，断言已打开的 Dashboard 视图收到刷新；有活跃日报任务时 `clear-run-state` 被安全拦截；
  - `dailyDir` 含尾斜杠（如 `"arxiv-daily/daily/"`）时 `open-today-daily` 仍能找到并打开今日日报。
- `plugin/tests/fulltext-runtime-diagnostics.test.ts`：`embedding.mode === "remote"` 时运行诊断不调用本地 `transformers` 模型下载。

## 修复情况

`clear-run-state` 在存在活跃任务时拒绝清理，清理成功后刷新 Dashboard；`set-paper-status`/`mark-current-*` 更新后刷新 Dashboard；`open-today-daily` 使用 writer 规范化的 `dailyPath()`；`diagnoseEmbedding` 在远程模式下校验配置与授权并通过共享嵌入后端探测，不再下载本地模型，诊断报告标注 local/remote。测试：`commands`、`fulltext-runtime-diagnostics` 等；插件全量 1310 项通过。
