# 撤销文献库远程处理授权时未取消正在进行的全文索引，远程嵌入请求继续外发并提交索引

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（Obsidian 文献库授权与索引生命周期）
> 严重程度：高 · 置信度：代码推断

## 问题与依据

当用户启用远程嵌入模式（`embedding.mode === "remote"`）开始为个人文献库建立索引后，若中途点击撤销远程处理授权（`revokeLibraryProcessing()`），插件并不会中止正在运行的索引任务：

- `plugin/main.ts:742-763`（`revokeLibraryProcessing`）：注释明确要求 `"Consent becomes ineffective synchronously at invocation"`，但第 747、752 行只调用了 `this.cancelPersonalLibraryDirectionGeneration(...)`，**完全没有取消 `personal-library-fulltext-index` 操作**。
- `plugin/main.ts:1696-1836`（`indexPersonalLibraryFullText`）：只在索引开始前（第 1696 行）调用了一次 `this.assertRemoteEmbeddingReady()`；在 `indexFullTextKnowledgeBase(...)` 逐篇调用远程 embedding API 期间既没有因撤销而中止 `AbortSignal`，也没有在 `beforeManifestCommit` 中再次检查当前授权状态。
- 直到所有剩余论文的标题和摘要全部发送到远程 API、并且 `replaceManifest` 已经写入磁盘之后，第 1832 行才检查 `if (this.libraryConnection !== connection)` 抛错。

这直接违背了 ADR 0008（远程嵌入必须受用户显式授权约束、撤销立即生效）的隐私保证：用户已经撤销授权，插件仍在后台把剩余论文元数据持续发送到第三方远程端点。

## 原因

`revokeLibraryProcessing()` 编写时只覆盖了方向生成（LLM）路径，遗漏了后来接入远程授权约束的 `personal-library-fulltext-index` 操作（contrast `disconnectLibrary()` / `selectLibraryRoot()` 或其他设置变更路径中对全文索引的取消处理）。

## 建议和验收

- 在 `revokeLibraryProcessing()`（`plugin/main.ts:747,752`）中，同步取消正在进行的远程全文索引（可新增或复用 `this.cancelPersonalLibraryFullTextIndex("library processing authorization revoked")`，或在 `operations` 中按 `kind === "personal-library-fulltext-index"` 取消，至少在 `embedding.mode === "remote"` 时必须立即 abort）。
- 在 `indexPersonalLibraryFullText()` 传给 `indexFullTextKnowledgeBase` 的 `beforeManifestCommit` 回调中，增加连接与远程授权状态复检（`this.libraryConnection !== connection || (this.settings.embedding.mode === "remote" && this.getLibraryConnectionStatus().kind !== "authorized")` 时抛取消错误），防止竞态下在撤销后仍提交 manifest。
- 验收：远程嵌入模式下启动多篇论文索引，在第 1 篇发出后立即调用 `revokeLibraryProcessing()`，断言索引的 `AbortSignal` 立即触发、后续论文不再发起远程 HTTP 嵌入请求、且 manifest 不被推进。

## 修复后应补的测试

- `plugin/tests/fulltext-index-lifecycle.test.ts` 或 `plugin/tests/library-connection-lifecycle.test.ts`：
  - 模拟远程嵌入模式索引 3 篇论文，在第 1 篇嵌入回调内调用 `revokeLibraryProcessing()`，断言第 2、3 篇不再调用远程 `http.request`，索引任务以取消结束且不提交新 manifest。
  - 本地嵌入模式下撤销（仅影响 LLM 方向生成的）授权时，确认预期行为与 ADR 0008 一致（不误杀本地离线索引，或按明确产品策略测试）。
