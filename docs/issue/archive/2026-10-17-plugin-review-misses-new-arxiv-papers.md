# 插件审阅页概览漏数新增的 arXiv 论文

> 发现日期：2026-10-17 · 来源：main `e6eaed5`（#93–#95 合并后）代码审查
> 范围：插件文献库方向审阅页的 "Library overview" 页 · 严重程度：低（提示文字前后不一致，不影响数据和更新操作） · 状态：未修复 · 置信度：代码确认，未在真实 Obsidian 中复现

## 问题

插件审阅页 "Library overview" 页的第一行是 "N papers analyzed on <日期>. … X currently indexed papers were not in this analysis."。其中 X 只统计没有 arXiv 编号、需要索引自己提取标题的本地文件。新增的 arXiv 论文，也就是标题记在目录（catalog）里的论文，不会被计入。

结果是，用户加入几篇 arXiv 论文的 PDF 并重建索引后：

- "Topics from your library" 页的新论文提示会说 "N new papers in your library aren't reflected in your direction suggestions yet."，并给出更新按钮。
- "Library overview" 页却不提这些论文。X 为 0 时，整句都不显示。
- Workbench 概览的"分析之后新增的论文"会列出这些论文。

同一个文献库状态，插件的两个页面之间、插件和 Workbench 之间给出了不同的说法。而 `plugin/src/library/interest-profile-modal.ts:532` 的注释声称这里与 Workbench 概览用的是同一集合，与实际不符。

影响：只影响这一句提示文字。更新按钮、索引后的新论文提醒、增量更新实际发送的论文，都按 `newPaperKeys` 计算，不受影响。

## 原因

- 插件交给审阅页的 `indexedPapers` 只包含"索引自己命名的论文"。`plugin/main.ts:2010-2015` 只保留 manifest 记录里带 `title` 的 ready 论文，注释写明 arXiv 论文的标题以 catalog 为准。快照类型 `plugin/main.ts:228` 的注释也写着 "Fallback-indexed papers, empty when every indexed paper is in the catalog"。"arXiv 论文的 manifest 记录不带 title"是根据这两处注释和过滤条件得出的，没有逐条检查真实的 manifest。
- 概览的计数在 `plugin/src/library/interest-profile-modal.ts:528-536`：`outside = [...indexedKeys].filter((key) => !accountedKeys.has(key) && scannable.has(key)).length`。它以 `snapshot.indexedPapers` 为全集，再和 `snapshot.newPaperKeys` 取交集。全集里没有 arXiv 论文，交集里自然也没有。
- 对照 Workbench：它的 `indexedPapers` 来自 `LibraryWorkflow.review()`（`packages/core/src/library/library-workflow.ts:295-302`），对每条 ready 记录优先取 catalog 标题，没有时再用 manifest 标题，因此包含 arXiv 论文。`addedKeys()`（`apps/cli/src/workbench/web/library-review.ts:173-180`）因此能数到它们。
- 这个口径不是 #93 才有的。改动前，同一行也以 `snapshot.indexedPapers` 为全集，算的是"已索引减已分析"。#93 改写了这一行，并加上"与 Workbench 是同一集合"的注释，但没有更换全集。

## 复现

本条由读代码确认，没有在真实 Obsidian 中复现。推导过程：

1. 在一个已有方向建议的插件文献库里，加入 3 个文件名能识别出 arXiv 编号的 PDF，然后运行索引。
2. `recomputeLibraryNewPaperKeys` 结合 manifest 和 catalog，把这 3 篇算进 `newPaperKeys`。"Topics from your library" 页显示 "3 new papers … aren't reflected …"，索引结束时还会弹出新论文提醒。
3. `libraryIndexedPapers` 只含 manifest 带标题的论文，这 3 篇不在其中。概览的 X 为 0，"… were not in this analysis." 整句不出现。

## 可能的解决方案

### 方案 A（推荐）：概览直接用 `newPaperKeys` 计数

把 `interest-profile-modal.ts:536` 改为 `const outside = snapshot.newPaperKeys.length;`，并改正第 529-534 行的注释。`newPaperKeys` 本身就是还没被任何分析统计过的论文，不需要再和 `accountedKeys` 求差。

- 代价最小，只改一处。
- 数字会与 "Topics from your library" 页的提示、新论文提醒、增量更新实际发送的篇数一致。
- 也与 Workbench 的 `addedKeys()` 一致。后者要求论文在 `indexedPapers` 里有标题；`listUnaccountedLibraryPapers` 在传入 catalog 时，也只保留有 catalog 记录或有 manifest 标题的论文。
- 例外：插件在读不到 catalog 时，`newPaperKeys` 会退回只按 manifest 计数（`plugin/main.ts` 的 `recomputeLibraryNewPaperKeys`）。这时概览会和其他地方一起多数，口径仍然一致。

### 方案 B：让插件的 `indexedPapers` 包含全部已索引论文

把 `plugin/main.ts:2012-2015` 改成和 `LibraryWorkflow.review()` 一样，对每条 ready 记录用 catalog 标题兜底，把 arXiv 论文也包括进来。

代价：

- 改变了快照字段的含义，`plugin/main.ts:228` 的类型注释也要改。
- 审阅弹窗里还有别处依赖它是"catalog 之外的论文"：
  - `proposablePaperKeys`（`interest-profile-modal.ts:1491-1495`）本来就会与 catalog 合并，改了也无害。
  - `paperTitle`（第 1498-1504 行）优先用 catalog，也无害。
  - `canGenerateMissingProposal`（第 1461-1464 行）用 `indexedPapers.length > 0` 判断没有提案时能否自动生成。纯 arXiv 文献库原来这个条件恒为假，改后变为真，行为会变。这可能本来就该修，但没有核实首次自动生成的完整流程，属于推断，需要单独确认。
- `refreshLibraryIndexTrace` 需要额外读一次 catalog。

推荐方案 A。方案 B 牵涉面更大，如果要做，应作为单独的 issue，连同 `canGenerateMissingProposal` 的语义一起处理。

## 修复后应补的测试

- 插件：在 `plugin/tests/personal-library-interest-profile-modal.test.ts` 里构造与真实插件一致的快照：`indexedPapers` 只含非 arXiv 论文，`newPaperKeys` 含 arXiv 论文。断言 "Library overview" 页出现 "… currently indexed papers were not in this analysis."，数字等于 `newPaperKeys.length`，并与 "Topics from your library" 页的提示一致。
- 现有用例的快照如果把 arXiv 论文放进了 `indexedPapers`，就与真实插件不符，建议一并调整，以免掩盖这类口径问题。这一点是推断，没有逐条检查现有用例。
