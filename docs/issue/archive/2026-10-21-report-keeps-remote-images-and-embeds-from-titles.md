# 论文标题和模型摘要里的 Markdown 图片、嵌入会原样写进日报，打开日报就会向作者指定的网址发请求

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 PIPE-09
> 范围：日报 Markdown 渲染（`### 标题`、五个摘要字段、作者、信息来源） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch4`，提交 e529a8b） · 置信度：已复现

## 问题

日报把论文标题、作者和模型生成的摘要字段放进 Markdown 时，只做两件事：把换行压成一行，并把 CommonMark 的原始 HTML 构造（`<img>`、注释等）的 `<` 编码掉。注释里明确写着链接、强调、行内代码等“保持可读，不做全局转义”。图片语法不在处理范围里。

arXiv 论文标题由作者填写，可以包含 `![](https://…)`；模型生成的摘要字段则可能被论文摘要里的提示注入诱导写出同样的语法（摘要字段通过了数学校验，没有对图片做任何检查）。这些文本会原样进入日报。用 Obsidian 打开日报时，阅读视图和实时预览默认加载远程图片，于是作者可以知道“有人在某个时间、从某个 IP 打开了这份日报”（读取回执），而用户完全看不到异常。同样，`![[某篇私人笔记]]`、`[[…]]`、`%%…%%` 这类 Obsidian 语法也会生效，例如把用户自己的另一篇笔记嵌进日报里，或让一段文字被隐藏。

提示词里没有任何密钥或私人数据（过滤请求只输出受校验的 JSON），所以不存在“把私人内容拼进图片网址带出去”的路径，影响限于读取回执和排版被篡改，评为低。

复现（一次性测试，已删除）：标题 `![](https://attacker.example/t.gif) Hostile ![[Private Note]] %% hidden`，摘要字段 `Normal method ![](https://attacker.example/px.gif?m=1).` 和 `Result [[Private Note]] and ![[Another Private Note]].` 都通过 `validateStructuredPaperSummaryValue` 的校验，`assembleDailySummary` 输出里分别出现：

```
### ![](https://attacker.example/t.gif) Hostile ![[Private Note]] %% hidden
- **核心结果**: Result [[Private Note]] and ![[Another Private Note]].
> - **方法设计**: Normal method ![](https://attacker.example/px.gif?m=1).
```

## 原因

- `packages/core/src/pipeline/daily-summary-rendering.ts:52-55` `normalizeMarkdownLine`：只压缩空白并调用 `neutralizeRawHtml`，不处理图片、嵌入、Obsidian 注释语法。
- `daily-summary-rendering.ts:159`、`:187`、`:249-255` 把标题、作者、摘要字段都经由它写入。
- `packages/core/src/pipeline/scientific-markdown-math.ts` 的校验只关心公式，不检查这类语法。

## 可能的解决方案

推荐：对不可信文本（标题、作者、来源章节名、模型摘要字段）在 `normalizeMarkdownLine` 里额外转义：`![` 改成 `!\[`（让图片变成普通文本）、`[[` 改成 `\[\[`、`%%` 改成 `\%\%`，链接可以保留。数学片段 `$…$` 和行内代码里的内容保持原样。

备选：只拦截 `![` 和 `![[`（影响最大的两类），其余不动。

## 修复后应补的测试

- 标题或摘要字段里的 `![](https://…)`、`![[…]]`、`[[…]]`、`%%…%%` 在日报输出里不再是图片、嵌入、双链或注释。
- 正常标题里的方括号、感叹号、公式、行内代码保持原样；回退块的原始摘要同样处理。

## 修复情况

标题、作者、来源、摘要字段及回退摘要中的图片、Obsidian嵌入/双链和隐藏注释以普通文字保存，打开日报不会因此加载外部图片或嵌入私人笔记。公式、行内代码、普通链接和系统生成的论文链接保持原样。三项回归先失败后通过；日报组装、应急报告、解析及邮件往返共179项通过，中英文补发邮件仍与新生成邮件的HTML和纯文本一致。
