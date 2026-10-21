# 文献库里有两份内容完全相同的未识别 PDF 时，索引在最后一步失败，且每次重试都失败

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 LIB-02
> 范围：个人文献库索引构建（插件、CLI 共用的 core 逻辑） · 严重程度：高 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 e9c0784） · 置信度：已复现

## 问题

同一份 PDF 放在文献库的两个文件夹里（例如 `Zotero/x.pdf` 和 `arXiv/x.pdf`，或 `Reviews/a.pdf` 和 `cosmology/a.pdf`），这两个文件字节完全相同、又都没能识别出 arXiv 编号时，索引会按 PDF 内容哈希把它们合并成一个条目。条目里的两个路径需要按 UTF-16 码元顺序排列，而构建条目时用的是 `localeCompare`。两个路径的首个不同字符一大一小写时（`Z` 对 `a`、`B` 对 `a`、`R` 对 `c`），两种顺序不一致，于是：

1. 这篇论文保存失败（`cannot persist invalid full-text paper document`）。逐篇失败本来只会让这一篇记为 `failed`；
2. 但失败记录里写入的仍是不符合码元顺序的路径数组，整个 manifest 在提交时校验不通过，所有论文都处理完之后才抛出 `cannot persist invalid or identity-mismatched full-text knowledge base manifest`；
3. 这一轮所有已嵌入的论文全部作废，索引没有提交；每次"Retry preparation"结果相同，用户看到的只有笼统的"准备失败"。要恢复只能删掉其中一份 PDF。

没有 arXiv 编号的出版社 PDF 占库里的大多数，同一篇论文存在多个项目文件夹里很常见，所以这不是边角情形。有 arXiv 编号的论文不受影响（目录里的路径用默认排序）。

复现：三种路径组合，字节相同的两份未识别 PDF。`["Zotero/x.pdf","arXiv/x.pdf"]` 和 `["a/x.pdf","B/x.pdf"]` 都以 `cannot persist invalid or identity-mismatched full-text knowledge base manifest` 抛出；`["a.pdf","b.pdf"]`（两种顺序一致）正常提交。

## 原因

- `packages/core/src/library/fulltext/index-orchestration.ts:525-530`：`collectIndexUnits` 把合并后条目的路径按 `left.path.localeCompare(right.path)` 排序。
- `packages/core/src/library/fulltext/knowledge-base.ts:511-527`：`isLogicalPathArray` / `isOptionalLogicalPathArray` 要求数组等于 `[...value].sort()`（默认的码元顺序）。解码 paper 文档（第 217 行）和 manifest 记录（第 291、294 行）都用它。
- `packages/core/src/library/fulltext/index-orchestration.ts:723-749`：`recordFailed` 把同一个乱序的 `unit.filePaths` 写进 `failed` 记录，所以失败记录也让 manifest 校验不通过。
- `packages/core/src/library/fulltext/knowledge-base-store.ts:124-128`：`replaceManifest` 先 `decodeFullTextKnowledgeBaseManifest`，不通过就整体拒绝。
- 同文件里其它排序（`paperKey`、迁移源键）同样用了 `localeCompare` 或默认排序，目前只有路径数组受解码器约束。

## 可能的解决方案

推荐：在 `collectIndexUnits` 里改用与解码器一致的码元比较（`a < b ? -1 : a > b ? 1 : 0`），并让 `unit.filePaths`、`unit.observationFingerprints` 按这个顺序成对排列。

备选：

- 让 `recordFailed` 在写失败记录前先排序，使一篇论文保存失败不会拖垮整个 manifest 提交。这也能消除"一篇论文的格式问题让整轮索引失败"这类放大效应，建议一并做。
- 把"manifest 提交前校验失败"从整轮失败降级为逐篇记 `failed`。

## 修复后应补的测试

- 两份字节相同的未识别 PDF，路径 `Zotero/x.pdf` 与 `arXiv/x.pdf`：索引成功提交，manifest 里该条目的 `filePaths` 按码元顺序排列，`failed` 为 0。
- 同一场景下再跑一轮：全部 `reused`。
- `recordFailed` 写出的记录总是能通过 `decodeFullTextKnowledgeBaseManifest`（对路径乱序的输入做一次防御性排序的单元测试）。

## 修复情况

文献库里有两份内容完全相同、又都没能识别出 arXiv 编号的 PDF（例如 `Zotero/x.pdf` 与 `arXiv/x.pdf`）时，索引不再在最后一步失败：合并后条目的路径按 UTF-16 码元顺序排列，与存储解码器的要求一致，每个路径仍和自己的观测指纹成对。失败记录写出前同样做这一排序，所以即使某一篇论文保存失败，也只让这一篇记为 `failed`，不会再让整个 manifest 提交被拒、整轮嵌入作废。两种顺序本来一致的路径不受影响。

测试：`fulltext-index-orchestration.test.ts`：四组两种顺序不一致的路径（大写对小写、带重音的字符）索引成功、`failed` 为 0，路径与指纹按码元顺序成对排列，第二轮全部 `reused`；目录条目的路径乱序、保存失败时，只有该篇记为 `failed`，失败记录的路径与指纹成对排序，其余论文照常提交。
