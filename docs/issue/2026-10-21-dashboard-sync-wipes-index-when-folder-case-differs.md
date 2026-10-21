# 设置里的日报文件夹大小写与磁盘上不一致时，面板同步会把所有报告当作"已丢失"，并删除全部未标记论文的索引条目

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 STATE-03
> 范围：Obsidian 插件面板的历史同步（`syncDashboardHistory`），macOS / Windows 等大小写不敏感磁盘 · 严重程度：低 · 状态：未修复 · 置信度：已复现（核心同步逻辑；前提是设置与磁盘大小写不一致）

## 问题

在大小写不敏感的磁盘上（macOS 默认、Windows），`research/daily` 和 `Research/Daily` 指向同一个文件夹，写文件时系统不会报错。如果"设置里的日报文件夹"和"磁盘上实际的文件夹"只有大小写不同，例如：

- 文件夹早先就以 `Research/Daily` 存在，用户在设置里填了 `research/daily`；
- 用户在文件管理器里只改了文件夹名的大小写，而设置没跟着改；

日报仍然能正常写入，但面板同步会把所有历史报告判成"不存在"：

1. 插件用区分大小写的前缀判断筛出要扫描的 Markdown 文件，磁盘上的 `Research/Daily/...` 全被过滤掉；
2. 同步把论文索引里记录的 `research/daily/2026-10-20.md`（流水线按设置的拼写记录）逐个判成"报告文件缺失"，清掉 `dailyReports` 和 `seenDates`；
3. 对没有被标记过的论文（状态 `inbox`、优先级 `normal`、无 PDF 和项目关联），条目被整条删除。

实测：索引里 2 篇论文、对应日报在 `Research/Daily/` 下、设置为 `research/daily`，一次面板同步后索引剩 0 篇。被标星、改过状态或有 PDF 的条目会保留，但引用的日报全部丢失。

影响是"面板突然变空 / 日历没有报告"，而 Markdown 日报本身没有任何变化；把设置改成与磁盘一致后，下一次同步会从日报重建索引，所以不是永久丢失，但中间被删掉的条目的 `seenDates` 等细节无法恢复，而且界面完全没有提示原因。面板的日报列表（`loadDailyReports`）用的是同一个区分大小写的判断，所以列表也是空的。

## 原因

- `packages/core/src/dashboard/history-sync.ts:937-942` `dailyDateFromPath` 和 `931-935` `isDirectChildMarkdown` 用区分大小写的 `startsWith` 判断路径是否在日报/论文目录内；`collectDailyReportPaths`（320-332 行）据此得到"磁盘上存在的日报"集合。
- `history-sync.ts:627-706` `pruneStaleDailyReports`（`reportMissing` 在 652 行）：集合里没有的报告一律视为缺失，并按 `isSafeToDeleteWholeEntry` 删除条目。没有"整个目录一份报告都没扫到"这种明显异常的保护。
- `plugin/src/dashboard/files.ts:8-22`（`filterDashboardMarkdownFiles`）、`24-41`（`dashboardHistoryPathSet`）、`72-79`（`dailyDateFromPath`，被 `plugin/src/dashboard/view.ts:446-470` 的 `loadDailyReports` 使用）同样用区分大小写的前缀判断。
- 代码库其他地方已经承认大小写不敏感：`packages/core/src/settings/validation.ts` 的 `portablePathCollisionKey` 用于判断日报/论文目录是否冲突，但同步和过滤没有使用它。

## 复现

临时 vitest（`packages/core/tests/zz-review-state-case.test.ts`，用后已删除）：设置 `research/daily`，索引里两篇论文的 `dailyReports` 为 `research/daily/2026-10-20.md`，磁盘文件为 `Research/Daily/2026-10-20.md`，按插件的方式过滤文件列表后调用 `syncDashboardHistory`：

```
before sync: 2 papers
after sync (plugin-style filtered list): 0 papers
```

## 可能的解决方案

推荐：两处一起改。

1. 目录判断统一用 `portablePathCollisionKey`（NFC + 大小写折叠）比较前缀和文件名，`filterDashboardMarkdownFiles`、`dashboardHistoryPathSet`、`dailyDateFromPath` 以及 `history-sync.ts` 里的同名函数都用同一套规则；写入索引的 `dailyReports` 尽量使用 Vault 报告的实际拼写。
2. 给破坏性清理加一道总量保护：当某个目录在这次扫描里一份受管文件都没有、而索引里有很多条目引用它时，视为"目录不可见"，跳过清理并写一条 warning，而不是逐条删除。这同时覆盖文件夹被移动、Vault 列表尚未加载完等相同的症状。

备选：只做第 2 条。不改变大小写判断，但至少不会在这种配置下清空索引。

## 修复后应补的测试

- 设置 `research/daily`、磁盘 `Research/Daily`：同步后索引条目和 `dailyReports` 保持不变，日历仍能找到这些报告。
- 目录里一份受管报告都没扫到、而索引引用了很多条目：不删除任何条目，日志有 warning。
- 正常删除单个报告文件：仍然清理对应的引用（现有行为保持）。
