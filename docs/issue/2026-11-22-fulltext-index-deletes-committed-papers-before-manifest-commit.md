# 文献库索引在提交新 manifest 之前提前删除旧论文文档，切换模型失败或中途取消会破坏已提交索引

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（个人文献库索引编排）
> 严重程度：高 · 置信度：代码推断

## 问题与依据

`indexFullTextKnowledgeBase`（`packages/core/src/library/fulltext/index-orchestration.ts`）设计上宣称在 manifest CAS 提交成功之前不破坏已提交的旧索引——例如第 423-425 行在切换嵌入模型且所有论文都建索引失败时明确报错：

> `"No papers could be indexed with the selected model; the previous index was preserved. Check the model and retry."`

但实际上，编排流程在 `store.replaceManifest(next, loaded.revision)`（第 428 行）**之前**就多次调用了 `store.removePaper(paperKey)`：

- `packages/core/src/library/fulltext/index-orchestration.ts:217-219`：单篇论文预检失败（`unit.preparationError`）时若 `previous?.status === "ready"`，立即调用 `discardOrphanedPaperDocument(store, paperKey, log)` -> `store.removePaper(paperKey)`。
- `packages/core/src/library/fulltext/index-orchestration.ts:386-388`：单篇论文抽取/嵌入失败时若 `previous?.status === "ready"`，立即调用 `discardOrphanedPaperDocument(store, paperKey, log)` -> `store.removePaper(paperKey)`。
- `packages/core/src/library/fulltext/index-orchestration.ts:418-420`：清理已移出 catalog 的论文时，在 `replaceManifest` 之前立即调用 `await store.removePaper(paperKey)`。

而 `FullTextKnowledgeBaseStore.removePaper(paperKey)`（`packages/core/src/library/fulltext/knowledge-base-store.ts:231-244`）是根据**当前已提交的 `manifest`** 计算路径并删除文件的！

因此出现两类真实数据损坏：
1. **切换嵌入模型全部失败时，“保留旧索引”实际把旧索引文件删光**：已用 `model-a` 建好索引后，切换到不可用的 `model-b`（如远程 API 500 或配置错误），每篇论文都在第 387 行触发 `store.removePaper(paperKey)`——删除的是 `model-a` 命名空间下已提交的论文 `.json`；随后第 424 行抛错声称旧索引已保留，但此时旧 manifest 仍标所有论文 `status: "ready"`，而 `loadPaper(paperKey)` 全部返回 `null`，后续搜索代构建或增量复用直接损坏。
2. **同模型重跑中途取消或 manifest CAS 冲突**：若前面几篇重建失败或进入 pruning 循环后用户取消任务（第 402/426 行）或 `replaceManifest` 发生 revision 冲突（第 428 行），旧 manifest 保持未变，但磁盘上对应论文文档已被提前删掉，出现 manifest 与磁盘文件不一致的悬空引用。

## 原因

`store.removePaper(paperKey)` 的删除动作被直接放在了逐篇处理循环和 pruning 循环里，而不是推迟到 `replaceManifest()` 成功提交之后执行；且跨模型切换时 `removePaper` 删除的是已提交旧模型的命名空间文件，而不是当前未提交新模型的孤儿文件。

## 建议和验收

- 将所有待删除的 `paperKey` 收集到列表（例如 `keysToRemoveAfterCommit`），**仅在 `await store.replaceManifest(next, loaded.revision)` 成功之后**才执行清理，且不得因为后置清理失败而回滚已提交的 manifest（记录 warning 即可）。
- 当 `modelChanged === true` 时，切勿删除旧模型命名空间中的论文文档（ADR 与现有设计均已明确切换模型时不删除旧模型逐篇文档，且旧 manifest 仍依赖它们）；失败路径只需确保不写入新模型下的坏文件，或仅清理本次运行在新模型命名空间中刚写入的临时对象。
- 验收：
  - `model-a` 建好索引后切换至必然报错的 `model-b` 运行索引，抛错后 `manifest.modelId` 仍为 `model-a` 且所有旧论文 `kb.loadPaper(paperKey)` 均可读、无文件丢失。
  - 同模型索引在单篇失败或 pruning 之后、manifest 提交之前取消（`AbortSignal`）或遭遇 CAS 冲突，旧 manifest 引用的所有磁盘论文文档保持完整。

## 修复后应补的测试

- `packages/core/tests/fulltext-index-orchestration.test.ts`：
  - 切换模型且新模型 embedding 全失败时，断言抛出保留旧索引错误后，原模型的 `loadPaper` 全部非空且磁盘文件数未减少。
  - 同模型下某篇原 `ready` 论文重建失败或某篇被 prune，在 `beforeManifestCommit` 触发取消或并发修改造成 CAS 冲突，断言旧 manifest 下原 `ready` 论文的 `.json` 文件均仍存在于存储中。
