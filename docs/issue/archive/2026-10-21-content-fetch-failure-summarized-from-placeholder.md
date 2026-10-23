# arXiv 正文和摘要页都取不到时，模型拿到的是“[获取失败]”占位符而不是已有的摘要，结果被当成正常摘要写进日报

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 PIPE-03
> 范围：日报流水线的逐篇摘要（`wantFullText` 内容获取阶段） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch2`，提交 1bed407） · 置信度：已复现

## 问题

日报运行里，每篇入选论文都要依次尝试渲染版 HTML、源码包、`abs` 页面三条路来取内容。三条路都失败时（例如运行中途 arXiv 抛出 503/超时，或 arxiv.org 被限流到重试用尽），`PaperContentFetcher` 返回 `abstractConclusion = "[获取失败] arXiv ID: <id>"`。

此时论文的真实摘要其实已经在内存里：列表阶段已经用 Atom 接口取到并放进 `FilteredPaper.abstract`。但逐篇摘要的提示词只用 `abstractConclusion` 和 `fullSections` 拼装，**完全不用 `paper.abstract`**。结果（用假的 arXiv 服务复现）：

- 发给模型的用户消息只有标题、作者，以及
  `Source sections: Abstract` / `## Abstract` / `[获取失败] arXiv ID: 2605.08080`。
- 模型只能凭标题写“研究问题 / 方法 / 结果 / 研究价值 / 适用边界”，要么大量填“原文未说明”，要么编造。
- 这个结果被当成 `structured`（正常摘要）提交：日报里没有“使用回退内容”的计数，没有回退警告，“信息来源”还写着 `Abstract`，摘要检查点也把它保存为可复用结果；日报文件一旦存在就不再重跑，用户不知道哪几篇是这样生成的。

一旦 arXiv 在运行中途出问题，整份日报里可能有一整段论文都是这种无依据的摘要。摘要页能打开、但页面里没有摘要块时，`fetchAbsAbstract` 也会把摘要写成字符串 `N/A`（`paper-content.ts:202`），同样被模型当作摘要内容。

## 原因

- `packages/core/src/pipeline/paper-content.ts:196-210`：三条路全失败时返回占位符 `[获取失败] arXiv ID: …`（第 208 行），并带 `fullTextFailure`。
- `packages/core/src/sources/arxiv-source-adapter.ts:340-355` `legacyContentFromNormalized`：占位符是非空文本，被包装成 `## Abstract\n[获取失败] …`，于是后面的逻辑都认为“摘要在”。流水线自己的 `catch` 分支（`packages/core/src/pipeline/pipeline.ts:395-411`）同样塞进占位符。
- `packages/core/src/pipeline/daily-paper-summary.ts:296-318` `buildPaperData`：只用 `abstractConclusion`、`fullSections`，不使用 `paper.abstract`。
- `daily-paper-summary.ts:275-285` `derivePaperSourceSections`：占位符被包成 `## Abstract` 之后，`startsWith("[获取失败]")` 判断失效，来源仍显示 `Abstract`。

## 复现

用真实的 `PaperContentFetcher` 加三个都抛 503 的假 `ArxivFetcher` 方法，再把结果交给 `summarizeDailyPaperWithValidationRetry`（一次性测试，已删除）。得到：`legacy.abstractConclusion = "## Abstract\n[获取失败] arXiv ID: 2605.08080"`；模型收到的用户消息就是上面那一段；返回 `kind: "structured"`；传入的真实摘要字符串完全没出现在提示里。

## 可能的解决方案

推荐：

1. 内容获取失败时，摘要提示词改用已有的 `paper.abstract`（标题加摘要），并把来源标成“仅摘要”；`paper.abstract` 也为空时才进入回退（原始摘要不可用）。
2. 占位符不要再包装成 `## Abstract`，用显式的“无内容”状态向下传递，`derivePaperSourceSections` 据此显示“获取失败”。
3. 三条路全因暂时性错误（503、超时、冷却中）而失败时，可以让这一天按暂时失败重试（保留已完成论文的检查点），而不是带着无依据的摘要提交。

备选：只做第 1 条，成本最小。

## 修复后应补的测试

- 内容全部获取失败但列表阶段有摘要时，模型收到的是真实摘要，而不是占位符。
- 没有摘要也没有内容时，该论文走回退块，不产生 `structured` 结果。
- 摘要检查点不会把“无依据的摘要”保存为可复用结果。

## 修复情况

渲染版 HTML、源码包和摘要页都取不到时，逐篇摘要现在用列表阶段已经拿到的摘要来写（信息来源标为 Abstract），不再把"[获取失败]"占位符交给模型；摘要页能打开却没有摘要块（原来写成 `N/A`）也一样当作没有内容。摘要和正文都没有的论文不再询问模型：日报里显示为回退块，写明"无法获取这篇论文的内容"，信息来源标为"获取失败"，并计入"其中 N 篇使用回退内容"。摘要检查点按这份实际输入保存，之后内容能取到时输入不同，会重新生成。

取舍：没有照“可能的解决方案”第 3 条在 arXiv 暂时不可用时让整天重试。有了列表摘要，摘要是有依据的；少数一篇都没有的论文直接显示为不可用即可。

测试：`daily-paper-summary.test.ts`（模型收到的是真实摘要、没有摘要时不调用模型）、`sources/arxiv-source-adapter.test.ts`（占位符不再被包成 `## Abstract`）、`pipeline/content-fetch-failure.test.ts`（真实 `PaperContentFetcher` 加三条路都返回 503，整条流水线）、`daily-summary-assembler.test.ts`、`daily-summary-checkpoint-store.test.ts`。
