# 没有相关论文那天的邮件，页脚仍然写着“完整 Markdown 日报路径”，但那天根本没有写日报文件；英文正文还会写成 “1 papers”

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-08
> 范围：日报邮件正文（HTML 与纯文本）的页脚和头部说明 · 严重程度：低 · 状态：未修复 · 置信度：已复现

## 问题

1. **页脚指向不存在的文件。** 当天没有论文通过筛选（`no_matches`）时，流水线明确不写空日报（`pipeline.ts:309,352` 的注释“Don't write empty file - show "0" in calendar”），但仍然为这一天构造一个零论文 digest 并发邮件，digest 的 `dailyPath` 是 `writer.dailyPath(dateStr)`（`pipeline.ts:623-631`）。邮件页脚固定写“Full Markdown daily report: arxiv-daily/daily/2026-07-27.md”（中文：“完整 Markdown 日报路径：…”）。用户照着去 Vault 里找，文件不存在。已复现（一次性测试，已删除）：对 `emptyDailyDigest` 渲染纯文本，正文是 `No relevant papers today.`，页脚是 `Full Markdown daily report: arxiv-daily/daily/2026-07-27.md`；HTML 里同样有这一行。
2. **英文头部不分单复数。** `headerMetaLine` 的英文分支写死 `${n} papers`（`email-render.ts:85`），1 篇论文时写成 “1 papers”；标题里的 `renderEmailSubject` 是区分单复数的（`1 paper`），同一封邮件里自相矛盾。已复现：一篇论文的样例邮件，主题是 `arXiv Daily 2026-07-27 · 1 paper`，头部是 `Date: 2026-07-27 · 1 papers · Categories: …`。

另外页脚路径是 Vault 相对路径，对于“手机版投影”的读者没有直接用处，这一点属于产品取舍，不在本问题范围内。

## 原因

- `packages/core/src/delivery/email-render.ts:94-98` `footerLine`：不看 `digest.paperCount`，一律输出日报路径。
- `packages/core/src/pipeline/pipeline.ts:623-631` `buildZeroDigest`：为零论文日构造的 digest 仍带 `dailyPath`。
- `email-render.ts:83-88` `headerMetaLine`：英文分支没有单复数处理。

## 可能的解决方案

推荐：`footerLine` 在 `paperCount === 0` 时省略路径行（或改写为“今天没有生成日报文件”）；`headerMetaLine` 英文分支复用 `renderEmailSubject` 里的单复数逻辑（抽成一个小函数）。

备选：零论文日也让 `buildZeroDigest` 不带 `dailyPath`（类型改成可选），渲染端按有无决定是否输出页脚。

## 修复后应补的测试

- 零论文 digest 的 HTML 与纯文本都不包含日报路径页脚（`email-render.test.ts` 已有零论文用例，补充断言）。
- 英文 digest 的头部：1 篇为 `1 paper`，0 篇与多篇为 `papers`；中文不变。
