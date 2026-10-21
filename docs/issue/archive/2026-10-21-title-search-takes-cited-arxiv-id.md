# 按标题识别文献库 PDF 时，arXiv 条目摘要或评论里引用的另一篇论文编号会被当成这份 PDF 的编号

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 LIB-01
> 范围：个人文献库扫描（插件、CLI、Workbench 共用的 core 识别逻辑） · 严重程度：高 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 9a1d0da） · 置信度：已复现

## 问题

文献库里绝大多数 PDF 的文件名没有 arXiv 编号（代码注释引用的实测语料里只有 1.4% 的 PDF 自带编号），扫描时靠"读 PDF 标题 → 到 arXiv 按标题检索"来识别。检索返回的 Atom 条目里，只要摘要（`<summary>`）、作者评论（`<arxiv:comment>`）或期刊引用里写着 `arXiv:2101.01234` 这样的文字，解析出来的编号就是被引用的那篇，而不是这个条目自己的编号。

后果是静默的识别错误：

- 这份 PDF 在目录里变成另一篇论文：`reconcilePersonalLibraryCatalog` 之后会按错误编号去取元数据，文件记为 `ready`，目录里的标题、摘要、分类都是被引用论文的。审阅页、代表论文列表、方向提案送给模型的摘要用的都是这份错误元数据。
- 如果用户同时保有被引用论文本身的 PDF，两个文件会被合并成同一篇论文（`filePaths` 里有两个文件）。索引只读 `filePaths[0]`（按路径排序的第一个）来提取标题和摘要，索引里这篇论文的文本可能来自另一份 PDF。
- 错误的标题检索结果还会覆盖 PDF 自带的可靠编号：`pdf-library-file-identifier.ts` 里只要检索返回了编号就直接返回它，`directId`（Info 字典 /arXivID 等）只在检索失败时才用。

复现：用一个 Atom 条目（`<id>` 为 `2403.19236v2`，标题与查询一致，摘要里含 "We extend the analysis of arXiv:2101.01234 to larger scales."）调用 `matchBestTitle`，返回 `arxivId: "2101.01234"`；把同一句放进 `<arxiv:comment>` 结果相同。已有测试 `arxiv-title-search.test.ts` 的 Atom 夹具只含 `<id>` 和 `<title>`，没有覆盖这个情形。

## 原因

- `packages/core/src/library/arxiv-title-search.ts:78`：`parseAtomEntries` 对整个 `<entry>` 正文调用 `extractArxivIdsFromText(body)[0]`，取"正文里出现的第一个 arXiv 编号"，而不是条目自己的 `<id>`。
- `packages/core/src/library/pdf-text-utils.ts:30-32`：`extractArxivIdsFromText` 的注释写"按文档顺序返回"，实际是按匹配规则分三轮追加：先所有 `arXiv:NNNN.NNNNN` 文字，再旧式编号，最后才是 `arxiv.org/abs/...` URL。条目自己的 `<id>` 是 URL 形式，所以排在摘要里的文字引用之后。
- `packages/core/src/library/pdf-library-file-identifier.ts:34-44`：标题检索返回非空就压过 `directId`。

## 可能的解决方案

推荐：

1. `parseAtomEntries` 只读条目自己的 `<id>…</id>`（用 `indexOf` 取出元素文本，再交给 `modernArxivResources`），不要在整个条目正文里搜编号。
2. 让 `extractArxivIdsFromText` 真正按位置排序（把三轮匹配合并后按 `match.index` 排序），使"文档顺序"的承诺成立；`pdf-identification-evidence.ts` 的头部文字取第一个编号也依赖这个顺序。
3. 可选：标题检索命中后，再用返回的编号取元数据，核对元数据标题与 PDF 标题是否一致，不一致就保持未识别。

备选：只做第 1 点，最小改动，已经能消除本问题。

## 修复后应补的测试

- `matchBestTitle`：条目 `<id>` 为 A，摘要或 `<arxiv:comment>` 含 `arXiv:B`，返回 A。夹具要带 `<summary>`、`<arxiv:comment>` 等真实 Atom 字段。
- `extractArxivIdsFromText("见 https://arxiv.org/abs/2403.19236 以及 arXiv:2101.01234")`：先返回 2403.19236（按出现位置排序）。
- `createPdfLibraryFileIdentifier`：检索结果的编号与 PDF 自带编号冲突时的取舍有明确断言。

## 修复情况

标题检索现在只认 Atom 条目自己的 `<id>`：摘要、作者评论、期刊引用甚至标题里写到的其他论文编号，不会再被当成这份 PDF 的编号；条目自己的编号不是现代编号时视为没有匹配，而不是被引用的编号顶替。`extractArxivIdsFromText` 也真正按文字出现的先后返回，PDF 内容流头部取"第一个编号"依赖这个顺序。PDF 自带编号与标题检索命中冲突时，取舍沿用原有设计并写明：严格且无歧义的标题命中优先于流内文字或 Info 字典里的编号（技术报告记载的行为，`library-workflow` 既有测试已依赖它）；检索无结果、有歧义或失败时保留 PDF 自带编号；只复述 arXiv 戳记的标题不做检索。识别规则版本从 3 升到 4，已被误认的文件会在下一次扫描时重新识别。

测试：`arxiv-title-search.test.ts`（摘要、评论、期刊引用、标题本身引用他人编号，旧式编号条目，缺少 `<id>`，两个条目引用同一论文仍判歧义）、`pdf-identification-evidence.test.ts`（编号按出现位置排序，流头部取最先出现的编号）、新增的 `pdf-library-file-identifier.test.ts`（冲突取舍和各种失败情形都有明确断言）。
