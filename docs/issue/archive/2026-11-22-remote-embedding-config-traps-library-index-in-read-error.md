# 远程嵌入配置尚未填完时被误报为“文献库索引无法读取”，且随后补填 embedding.apiKey 不会恢复状态

状态：已修复（分支 `fix/2026-11-22-review`）

> 来源：2026-11-22 main `2557b90a` 代码审查（Obsidian 文献库设置与索引状态刷新）
> 严重程度：高 · 置信度：代码推断

## 问题与依据

已连接个人文献库的用户在设置页将 Embedding mode 切换为 `Remote (OpenAI-compatible)`，并按界面从上到下的顺序依次填写 Base URL、API Key、Model、Dimension 时，会进入一个误导性错误并卡死：

1. **把未填完的远程嵌入配置误报成“索引文件读失败”**：
   - `plugin/main.ts:1990-2044`（`refreshLibraryIndexTrace`）在读取 manifest 之后、第 1996 行位于 `try` 块内部调用 `const expected = this.buildEmbeddingModel()`。
   - 当 `embedding.mode === "remote"` 且 `baseUrl`、`apiKey`、`model`、`dimension` 任意一项为空或不合法时，`buildEmbeddingModel()` -> `createRemoteEmbeddingModel()` 直接抛 `Error`（例如 `"Remote embedding API key is required"`）。
   - 该配置异常被第 2036-2044 行的 `catch` 捕获，设置 `preparationError = LIBRARY_INDEX_READ_FAILED`（UI 显示 `"Could not read the library index. Retry preparation to repair it."`），同时清空 `this.libraryIndexedPapers = []` 与 `this.libraryNewPaperKeys = []`，并把错误日志记成 `"could not read the index manifest for the settings row"`。
2. **填完 `embedding.apiKey` 后永远不会重新刷新该错误状态**：
   - `plugin/src/settings/tab.ts:366-370`：
     ```ts
     private async refreshLibraryAfterEmbeddingChange(keys: readonly string[]): Promise<void> {
       if (keys.some((key) => ["embedding.mode", "embedding.baseUrl", "embedding.model", "embedding.dimension"].includes(key))) {
         await this.plugin.refreshLibraryIndexTrace();
       }
     }
     ```
   - 监听列表漏掉了 `"embedding.apiKey"`！如果用户先填了 Base URL / Model / Dimension（或原本保留着上次的 Base URL / Model / Dimension），最后填写 `embedding.apiKey`，`refreshLibraryIndexTrace()` 根本不会被调用。设置页继续显示 `"Could not read the library index. Retry preparation to repair it."`，并且因为 `libraryIndexedPapers` 仍为空、处于错误态，相关操作受阻，直到用户重启 Obsidian 或再去无意义地改动另一个嵌入字段。

## 原因

1. `refreshLibraryIndexTrace()` 没有先校验远程嵌入配置是否完整（或把配置不完整与 manifest 磁盘读取失败拆开处理），直接调用会抛异常的 `buildEmbeddingModel()` 构造客户端。
2. `refreshLibraryAfterEmbeddingChange()` 的键名白名单遗漏了 `"embedding.apiKey"`。

## 建议和验收

- 在 `refreshLibraryIndexTrace()` 中，对比期望模型前先检查 `validateEmbeddingConfig(this.settings)`（或仅从 `settings.embedding` 提取期望的 `modelId` / `dimension`，不构造需要 `apiKey` 的 HTTP 客户端）；若远程嵌入配置尚未完整，不将其标记为 `LIBRARY_INDEX_READ_FAILED`（也不清空 `libraryNewPaperKeys`），而是呈现明确的配置待完善状态或沿用现有配置校验提示。
- 在 `plugin/src/settings/tab.ts:367` 的数组中添加 `"embedding.apiKey"`，确保补填或清空密钥后索引状态立即重新评估。
- 验收：已连接文献库后切到 Remote embedding（此时密钥为空），设置页不再出现 `"Could not read the library index"` 误报；填完 Base URL / Model / Dimension 后最后补填 `embedding.apiKey`，状态行和索引就绪状态立即刷新恢复。

## 修复后应补的测试

- `plugin/tests/embedding-mode-selection.test.ts` 或 `plugin/tests/settings-declarative-tab.test.ts`：
  - 连接并建好本地索引后切到 `remote` 模式（缺少 `apiKey`），断言 `libraryIndexStatus.snapshot().preparationError` 不为 `LIBRARY_INDEX_READ_FAILED`。
  - 已有完整 `baseUrl`/`model`/`dimension` 但 `apiKey` 为空时，通过设置页仅修改 `embedding.apiKey`，断言 `refreshLibraryIndexTrace` 被调用且状态立即更新。

## 修复情况

`refreshLibraryIndexTrace` 不再构造会抛异常的嵌入客户端：先用 `validateEmbeddingConfig` 校验，配置不完整时进入独立的“配置待完善”状态并保留 manifest 派生的已索引论文与新论文计数；期望模型身份直接由设置推导（本地固定值，远程 `remoteEmbeddingModelId`）。`embedding.apiKey` 已加入设置页刷新白名单（`f480bb8a`）。测试：`fulltext-index-lifecycle` 33、`settings-declarative-tab`/`embedding-mode-selection` 191 项通过。
