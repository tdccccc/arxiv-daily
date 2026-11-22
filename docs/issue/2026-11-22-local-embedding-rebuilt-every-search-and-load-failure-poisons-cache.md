# 本地嵌入模型每次搜索都重建 ONNX WASM 实例且未释放，且首次加载失败会永久毒化实例缓存

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（Obsidian 文献库搜索与本地嵌入运行时）
> 严重程度：中 · 置信度：部分已复现（core `pipelinePromise` 失败毒化已离线复现；插件侧未缓存实例为代码推断）

## 问题与依据

这里包含同一路径上的两个缺陷：

1. **Obsidian 插件每次搜索/索引/诊断都新建本地嵌入模型实例并泄漏 ONNX WASM 会话**：
   - `packages/core/src/library/fulltext/embedding-model.ts:16-17` 明确约定：“ONNX session is created once and reused”。
   - 然而 `plugin/main.ts:1605-1616` 的 `buildEmbeddingModel()` 在本地模式下每次都调用 `return createTransformersEmbeddingModel(options);`，没有任何缓存或复用。
   - 每次调用 `searchPersonalLibraryFullText()`（`plugin/main.ts:2095`，包括 Dashboard 搜索框与相似论文弹窗）、`indexPersonalLibraryFullText()`（`plugin/main.ts:1734`）以及 `diagnoseEmbedding()`（`plugin/main.ts:2292`），都会重新拉起一遍完整的 `transformers.js` `feature-extraction` ONNX WASM 流水线、重新加载 ~130 MB 模型权重并跑一次 `"dimension probe"`，且从未对旧的 ONNX session 调用 `dispose()`。
   - 用户在 Dashboard 连续输入几个关键词搜索，每次查询不仅延迟飙升，而且桌面端 Electron 渲染进程内存持续累积直至卡顿或 OOM。
2. **core 层 `createTransformersEmbeddingModel` 在 `load()` 首次失败后永久毒化 `pipelinePromise`**：
   - `packages/core/src/library/fulltext/transformers-embedding-model.ts:84`：
     ```ts
     const loading = pipelinePromise ?? (pipelinePromise = load());
     ```
   - 如果 `load()` 因网络临时断开（首次下载失败）或短暂环境错误 reject，`pipelinePromise` **永远不会被重置为 `undefined`**。
   - 一旦未来按约定复用同一个本地模型实例，该实例后续所有 `embed()` 调用都会永久复用那个已 rejected 的 Promise，即使网络恢复也无法再加载模型。

## 原因

- `plugin/main.ts` 为每次操作即时构造 `EmbeddingModel` 而不区分“轻量 HTTP 封装的远程模型”和“持有重型 WASM 推理会话的本地模型”，并且 `EmbeddingModel` 缺少生命周期释放挂钩（如插件 `onunload()` 时释放）。
- `transformers-embedding-model.ts` 缓存了 `pipelinePromise`，但没有加 `.catch` 清理失败的 Promise。

## 建议和验收

- 在 `packages/core/src/library/fulltext/transformers-embedding-model.ts:84`：当 `load()` reject 时在 `catch` 中清空 `pipelinePromise = undefined`（并确保不会误清已被替换的 promise），让后续重试能够重新加载。同时可选提供 `dispose?()` 用于卸载时释放底层 pipeline 资源。
- 在 `plugin/main.ts`：为本地模式缓存单一 `EmbeddingModel` 实例并在 `searchPersonalLibraryFullText()` / `indexPersonalLibraryFullText()` 间复用（若需要传递每次调用的 `onProgress`，可通过 `embed()` 参数或回调注册支持，而不是重建整个 ONNX session）；在插件 `onunload()` 或切换离开本地模式时释放缓存实例。
- 验收：
  - 同一 `createTransformersEmbeddingModel` 实例首次 `loadPipeline` 抛错后，第二次调用 `embed()` 能重新触发 `loadPipeline` 并成功返回向量；
  - 在插件中连续发起两次本地 `searchPersonalLibraryFullText()`，只创建并加载一次本地 ONNX pipeline。

## 修复后应补的测试

- `packages/core/tests/transformers-embedding-model.test.ts`：注入第 1 次 reject、第 2 次 resolve 的 `loadPipeline`，验证第 1 次 `embed()` 失败后第 2 次 `embed()` 能恢复成功（`loadPipeline` 被调用 2 次）。
- `plugin/tests/embedding-model-runtime.test.ts` 或 `plugin/tests/dashboard-library-search.test.ts`：连续两次执行本地 `searchPersonalLibraryFullText()`，断言底层本地模型加载只发生一次。
