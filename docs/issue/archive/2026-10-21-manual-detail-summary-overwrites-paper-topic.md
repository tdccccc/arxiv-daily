# 给已入库的论文生成详细总结后，它的研究主题被换成 arXiv 分类，主题筛选与列表对不上

> 发现日期：2026-10-21 · 来源：模型探索（gpt-6.1-sol，high）+ 人工复核 · 编号 EXP-01
> 范围：核心（手动详细总结 `ManualFetchService`、论文索引、面板历史同步），Obsidian 面板、阅读工作台、CLI `summarize` 共用 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch2`，提交 32327fa） · 置信度：已复现

## 问题

用户对一篇已经在日报里命中研究主题的论文（比如主题 `research`）点面板行上的“Summarize by arXiv ID”（或工具栏“Summarize by ID”、命令、CLI 单篇总结）生成详细总结之后：

- 面板 Topic 列里这篇论文的主题从 `research` 变成 `cs-ai`（arXiv 分类 `cs.AI` 的连字符写法），它不是用户配置的任何研究主题。
- Topic 下拉里多出 `cs-ai` 和 `cs.AI` 两个选项，是同一个 arXiv 分类的两种写法，和研究主题混在一起。
- 论文笔记 frontmatter 的 `primary_topic` 和标签也被写成 `cs.AI`。
- 此时把 Topic 选成 `research`，列表仍然包含这篇论文（索引的 `topics` 里还留着 `research`），但它的 Topic 列写着 `cs-ai`：选中的筛选值和列表里看到的主题互相矛盾。想按 `cs-ai` 筛时，只会筛出被这样改过的论文。

2026-10-21 的探索里，impatient-clicker 报告的就是这个现象（“Topic 筛选值与显示论文不一致”）。模型把原因猜成“连点两次 Run today”，实际无关：那两次 Run today 只是各自得到“等待公告发布”，没有改任何论文。真正的触发点是前一个角色 first-time-user 在步骤 20 点了 2610.10001 那一行的“Summarize by arXiv ID”，生成的状态留在了 Vault 里。

纯手动加入（没出现在任何日报里）的论文也有一半同样的问题：索引先记下原始分类 `cs.AI`，下一次面板重载时历史同步又把它改写成 `cs-ai`，两种写法同时留在 `topics`，主题下拉同样出现两个选项。

数据变化（用真实的 `ManualFetchService`、`MarkdownWriter`、`PaperIndexStore`、`syncDashboardHistory`、`queryDashboard` 复现，网络和模型用桩）：

| 阶段 | primaryTopic | topics | 面板 Topic 列 | 主题下拉 |
|---|---|---|---|---|
| 日报写入后 | `research` | `[research]` | research | research |
| 手动详细总结后 | `cs.AI` | `[research, cs.AI]` | — | — |
| 面板重载（历史同步）后 | `cs-ai` | `[research, cs.AI, cs-ai]` | `cs-ai` | `cs-ai`、`cs.AI`、`research` |

没有数据丢失（`topics` 仍保留旧主题），属于“界面和笔记里的主题被悄悄改写并污染筛选项”。

## 证据

- 运行目录 `output/playwright/acceptance/2026-10-21-review-obsidian-personas2/exploration/`（本地，不入库）：
  - `exploration-events.jsonl`：first-time-user 步骤 14-20 的观察里 Topic 下拉只有 `Any topic / research`，两篇论文的 Topic 列都是 `research`；步骤 20 点击 `Summarize by arXiv ID`；接下来 impatient-clicker 第一个观察（s23）里下拉变成 `Any topic / cs-ai / cs.AI / research`，2610.10001 的 Topic 列变成 `cs-ai`，通知里有 `summarizing 2610.10001…`、`done → arxiv-daily/papers/2610.10001.md`。
  - `impatient-clicker-step-27.png`：Topic 选 `research` 后，列表仍显示 2 篇，其中一篇 Topic 列是 `cs-ai`。
- 确定性复现：用一次性 vitest（已删除，不入库）依次执行索引写入、`ManualFetchService.fetchAndSummarize`、`syncDashboardHistory`、`queryDashboard`，得到上表的数值，与探索里记录的观察一致。结果文件在 `/tmp/arxiv-review-2026-10-21-t6i7WE/exp/o6-manual-fetch-result.json`，纯手动论文的变体在同目录的 `o6-new-manual-result.json`。

## 原因

- `packages/core/src/services/manual-fetch.ts:274-285`：详细总结写完后调用 `reconcileManualDetail`，传入的 `primaryTopic: category` 是 Atom 元数据里的 arXiv 分类（`cs.AI`），而不是这篇论文已有的研究主题。
- `packages/core/src/services/paper-index.ts:849,881-882`：`upsertEntry` 只要传入的主题非空，就覆盖 `primaryTopic`，并追加到 `topics`。已入库论文原有的研究主题因此被顶掉。
- 同一个函数里更早的一步是对的：`manual-fetch.ts:218-221,261-266` 先读出已有索引条目并交给 `writePaperDetail`，`packages/core/src/pipeline/markdown-writer.ts:130` 用 `indexEntry?.primaryTopic ?? paper.category`，已有主题优先。随后 `manual-fetch.ts:291` 的 `refreshPaperNoteFrontmatter(indexed.entry, path)` 用已被覆盖的条目重写 frontmatter，把 `primary_topic`、标签改成 `cs.AI`。
- `packages/core/src/dashboard/history-sync.ts:278,833-840,858-866,948-956`：面板重载时 `syncDashboardHistory` 读到笔记里的 `primary_topic: cs.AI`，`topicFromHeading` 在配置的主题里找不到，就用 `slugTopic` 变成 `cs-ai`；`history-sync.ts:569` 的 `needsSync` 看到 `topics` 里没有 `cs-ai`，再 upsert 一次（`history-sync.ts:517` 把 `primaryTopic` 设成 `cs-ai`）。`topicKey`（`history-sync.ts:944-946`，去掉标点后比较）本可以认出 `cs.AI` 与 `cs-ai` 相同，但 `needsSync` 用的是精确字符串比较。
- 显示口径不一致：`plugin/src/dashboard/actions.ts:112-121`（`topicOptions`）把每条的 `primaryTopic` 和全部 `topics` 都放进下拉；`packages/core/src/dashboard/model.ts:511-516`（`matchesTopic`）按 `[primaryTopic, ...topics]` 匹配，而 `model.ts:298,548-550`（`displayTopic`）、`plugin/src/dashboard/view.ts:1554` 的 Topic 列只显示 `primaryTopic`（没有时取 `topics[0]`）。阅读工作台的 `apps/cli/src/workbench/papers.ts:49,101` 同样把 `primaryTopic` 和 `topics` 合并展示、合并列进主题下拉，会得到同样的污染（代码推断，未在工作台里复现）。
- 测试缺口：`packages/core/tests/paper-index.test.ts:1174-1240` 和 `manual-fetch.test.ts` 里关于 `reconcileManualDetail` 的用例，两边用的都是同一个主题（`photo-z`），覆盖不到“分类顶掉已有主题”。

## 可能的解决方案

推荐：

1. `ManualFetchService` 在已有索引条目时不改主题：传 `primaryTopic: existingIndexEntry?.primaryTopic || category`（和 `markdown-writer.ts:130` 的取值保持一致）；更干净的做法是让 `reconcileManualDetail` 只在条目没有主题时才用传入的分类做兜底，并且已有主题时不把分类追加进 `topics`。
2. 历史同步的 `needsSync` 和 `topics` 去重改用 `topicKey` 比较（大小写、标点不敏感），这样纯手动论文不会同时留下 `cs.AI` 和 `cs-ai`。

备选：手动论文干脆不把 arXiv 分类写进 `primaryTopic`/`topics`（留空，面板显示 `(none)`），分类只放在 `category`/`categories`。改动更小，但会改变纯手动论文目前显示分类的习惯，需要产品确认。

## 修复后应补的测试

- 核心：`manual-fetch.test.ts` 里对已用 `research` 入库的论文执行 `fetchAndSummarize`（分类是 `cs.AI`），断言索引 `primaryTopic` 仍是 `research`、`topics` 不含 `cs.AI`、笔记 frontmatter 的 `primary_topic` 仍是 `research`。
- 核心：`paper-index.test.ts` 里 `reconcileManualDetail` 对已有不同主题的条目，断言不覆盖已有主题；条目不存在时才使用传入的分类。
- 核心：`dashboard-history-sync.test.ts` 里纯手动论文（`primary_topic: cs.AI`）同步后，`topics` 里只出现一种写法，`topicOptions` 里没有 `cs-ai` 与 `cs.AI` 并存。
- 面板：对一篇已入库论文生成详细总结后，Topic 列仍显示它原来的研究主题，选中该主题筛选得到的行和 Topic 列一致。

## 修复情况

给已入库的论文生成详细总结后，它的研究主题不再被 arXiv 分类顶掉：索引里的主要主题、笔记 frontmatter 的 `primary_topic` 和标签，都保持日报给它的研究主题（例如 `research`），面板的 Topic 列、主题下拉和主题筛选重新一致；只有完全没有主题的论文（纯手动加入）才用 arXiv 分类当主题。面板重载时的历史同步也不再给已记下的 `cs.AI` 再追加一个 `cs-ai` 写法，也不再把论文的 arXiv 分类改写成 `cs-ai`；笔记里的主题和索引真的不同时，仍然以笔记为准。没有采用写作方案里的“让 `reconcileManualDetail` 永远不覆盖已有主题”：已有详细总结的修复路径要用笔记自己的主题，必须能覆盖。阅读工作台只是如实展示索引里的主题，没有改动。代价：已经被旧行为改过的数据不会自动改回，修复只保证以后不再发生。

测试：`manual-fetch.test.ts`（用真实的笔记写入器和论文索引：研究主题保留、两个日报主题保留且主要主题不变、没有主题时才用分类）；`dashboard-history-sync.test.ts`（真实的按 ID 总结之后做真实的历史同步和查询：Topic 列、筛选、下拉选项和 `category` 都不变，纯手动论文只有一种写法，笔记主题真的不同时仍被跟随）。
