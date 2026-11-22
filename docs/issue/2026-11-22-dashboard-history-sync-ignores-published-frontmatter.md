# Dashboard 历史索引同步忽略 MarkdownWriter 写入的 published frontmatter，重建后单篇笔记丢失日期

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（Dashboard 索引重建与笔记格式契约）
> 严重程度：中 · 置信度：代码推断

## 问题与依据

`MarkdownWriter` 为单篇论文笔记（精读笔记 `writePaperDetail`、轻量笔记 `writePaperNote`）生成的 frontmatter（`packages/core/src/pipeline/markdown-writer.ts:370-384`）写入的是 `published:` 字段（可能为 `published: 2026-06-09` 或 `published: "[[arxiv-daily/daily/2026-06-09|2026-06-09]]"`），**从不写入**旧字段 `date:` 或 `daily_report:`（这一点在 `packages/core/tests/markdown-writer.test.ts:424-432` 中也有明确断言）。

然而，当 Dashboard 执行历史扫描重建 `papers.json` 时，`packages/core/src/dashboard/history-sync.ts:298-308`（`collectPaperCandidates`）却只读取了已废弃的 `frontmatter.date` 和 `frontmatter.daily_report`：

```ts
const dailyReport = dailyReportPathFromLink(frontmatter.daily_report);
const candidates = candidatesById.get(arxivId) ?? [];
candidates.push({
  arxivId,
  title: frontmatter.title || firstH1(markdown) || arxivId,
  authors: frontmatter.authors || "",
  date: frontmatter.date || dateFromDailyReport(dailyReport) || undefined,
  topic,
  path,
  detail: true,
});
```

完全没有读取 `frontmatter.published`（对比 `packages/core/src/pipeline/paper-note-classifier.ts:110-124`，那里正确解析了 `published:` 字段及其中包含的日报 wikilink）。

影响：任何通过手工命令（`Summarize arXiv paper by ID…`）创建、或所属日报已被删除、或 `papers.json` 丢失后通过 Vault 扫描重建的精读笔记，在 `syncDashboardHistory` 后：
- `entry.published` 变成空字符串 `""`；
- `entry.seenDates` 变成空数组 `[]`；
- 在 Dashboard 列表里缺少日期、无法按日期范围（7 天/30 天）筛选，排序也落到末尾。

## 原因

论文笔记的 frontmatter schema 从 `date` / `daily_report` 迁移为统一的 `published`（含可选日报 wikilink）时，更新了 `MarkdownWriter` 和 `paper-note-classifier.ts`，但漏掉了 `history-sync.ts` 的 `collectPaperCandidates`。

## 建议和验收

- 在 `packages/core/src/dashboard/history-sync.ts:297-308` 中补充解析 `frontmatter.published`：
  - 若 `frontmatter.published` 为日报链接（如 `[[arxiv-daily/daily/2026-06-09|2026-06-09]]`），提取其中的日期（必要时也可兼容解析日报路径）；
  - 若为纯日期字符串（`YYYY-MM-DD` 或 ISO 时间戳），截取日期部分；
  - 按 `frontmatter.published -> frontmatter.date -> dateFromDailyReport(frontmatter.daily_report)` 的优先级填充 `candidate.date`（兼容历史旧笔记）。
- 验收：用 `MarkdownWriter.writePaperDetail()` 写出一条精读笔记（不含对应日报文件），清空 `papers.json` 后运行 `syncDashboardHistory()`，重建的索引条目包含正确的 `published` 日期与 `seenDates`。

## 修复后应补的测试

- `plugin/tests/dashboard-history-sync.test.ts`（或 core 对应测试）：
  - 分别测试 `published: 2026-06-09` 和 `published: "[[arxiv-daily/daily/2026-06-09|2026-06-09]]"` 两种现行 `MarkdownWriter` 格式的论文笔记，在无 `papers.json` 和无日报文件时经 `syncDashboardHistory` 重建，断言 `entry.published === "2026-06-09"` 且 `entry.seenDates` 包含 `"2026-06-09"`；
  - 保留旧版 `date:` / `daily_report:` 笔记的兼容回归测试。
