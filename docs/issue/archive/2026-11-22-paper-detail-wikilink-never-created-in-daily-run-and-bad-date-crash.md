# 日报流水线先写精读笔记再写日报导致 published 日报双向链接永远缺失，且非法日期字符串会让写笔记崩溃

状态：已修复（分支 `fix/2026-11-22-review`）

> 来源：2026-11-22 main `2557b90a` 代码审查（日报流水线与 MarkdownWriter）
> 严重程度：中 · 置信度：部分已复现（非法日期抛 `Error("Invalid date: invalid-date")` 已离线复现；流水线先后顺序为代码推断）

## 问题与依据

`MarkdownWriter.writePaperDetail()`（`packages/core/src/pipeline/markdown-writer.ts:276-284`）设计上希望在单篇论文精读笔记的 frontmatter 中把 `published:` 写成指向当日日报的双向链接（例如 `published: "[[arxiv-daily/daily/2026-06-09|2026-06-09]]"`，见 `packages/core/tests/markdown-writer.test.ts:406-437`），其判断依据是目标日报文件是否已存在：

```ts
private async publishedReportLink(
  published: string | undefined,
): Promise<string | undefined> {
  if (!published) return undefined;
  const path = this.dailyPath(published);
  return (await this.opts.storage.exists(path))
    ? dailyReportLink(path, published)
    : undefined;
}
```

这在两条路径上都存在问题：

1. **正常日报流水线中永远生成不了日报双向链接**：
   - `packages/core/src/pipeline/pipeline.ts:555`：步骤 7 先循环调用 `this.deps.writer.writePaperDetail(paper, dateStr, detail, ...)` 写出精读笔记；
   - `packages/core/src/pipeline/pipeline.ts:640`：步骤 8 之后才调用 `this.deps.writer.writeDaily(dateStr, dailySummary, ...)` 写出当日日报；
   - 写完日报后，`pipeline.ts` 从未调用 `refreshPaperNoteFrontmatter()` 回填链接（全仓仅有 `manual-fetch.ts` 调用过 `refreshPaperNoteFrontmatter`）。
   - 因此在每一次正常日报运行中，执行 `writePaperDetail` 时 `storage.exists(dailyPath(dateStr))` **必然为 `false`**，所有由日报自动生成的精读笔记都退化为纯文本 `published: YYYY-MM-DD`，无法从精读笔记的 frontmatter 点击回到当日日报。
2. **非 `YYYY-MM-DD` 的 `published` 字段会导致 `writePaperDetail` / `writePaperNote` / `refreshPaperNoteFrontmatter` 直接抛异常崩溃**：
   - `markdown-writer.ts:279-280` 将 `dateOnly(...)` 提取的 `published` 直接传入 `this.dailyPath(published)`；
   - `dailyPath()` 内部严格要求匹配 `/^\d{4}-\d{2}-\d{2}$/`，否则抛 `Error("Invalid date: ...")`。
   - 若上游条目（例如受损或手工编辑的 `papers.json`、异常 Atom 元数据）中 `published` 为非标准日期字符串（如 `"invalid-date"` 或 `"2026/06/09"`），`publishedReportLink()` 直接抛出未捕获异常，导致整篇笔记写入失败而不是安全回退为无链接。

## 原因

1. 流水线为了在日报正文里填入已落盘精读笔记的链接而把 `writePaperDetail` 放在 `writeDaily` 之前，但 `writePaperDetail` 的 `publishedReportLink` 又反向要求日报已经落盘，形成鸡生蛋死锁。
2. `publishedReportLink` 未先校验 `published` 是否符合 `YYYY-MM-DD` 格式再构造 `dailyPath`。

## 建议和验收

- 对于每日流水线回链：在 `pipeline.ts:640` 写完日报且更新完 `paperIndex` 之后，对本轮新写出的 `confirmedPaths` 调用 `writer.refreshPaperNoteFrontmatter(...)`（或允许 `writePaperDetail` 通过选项知晓当前日报即将写出/回填）；优先在 `writeDaily` 成功后刷新 frontmatter，这样中途取消或日报失败时也不会留下悬空死链。
- 对于 `publishedReportLink`（`markdown-writer.ts:276-284`）：先校验 `/^\d{4}-\d{2}-\d{2}$/.test(published)`，不符合则返回 `undefined`，避免因非法日期字符串阻断笔记生成。
- 验收：
  - 跑完一次包含精读论文的完整 `DailyPipeline`，检查生成的精读笔记 frontmatter 包含 `published: "[[<dailyDir>/<date>|<date>]]"`；
  - 对 `published: "invalid-date"` 的条目调用 `writePaperNote()`，不抛 `Invalid date` 异常。

## 修复后应补的测试

- `packages/core/tests/pipeline.test.ts`：完整跑一次包含 detail 论文的 `DailyPipeline`，断言写出精读笔记的 `published` frontmatter 为指向当日日报的 wikilink（且在 `writeDaily` 失败时不产生悬空日报链接）。
- `packages/core/tests/markdown-writer.test.ts`：向 `writePaperNote` / `writePaperDetail` 传入非法 `published` 字符串，断言安全写出笔记且不抛 `Invalid date`。

## 修复情况

`writeDaily` 成功且索引记录日报后，对本次 `confirmedPaths` 调用 `refreshPaperNoteFrontmatter` 回填日报 wikilink；日报失败或取消时不执行，避免悬空链接。`publishedReportLink` 先校验 `YYYY-MM-DD`，非法值退化为无链接。测试：`pipeline` + `markdown-writer` 93 项通过。
