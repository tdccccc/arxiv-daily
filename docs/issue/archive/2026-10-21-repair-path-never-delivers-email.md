# 日报已经写出、但后面的索引步骤失败或被取消后，重试走“修复”分支，当天的邮件永远不会发送

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 PIPE-08
> 范围：日报流水线的提交后恢复；邮件自动发送（插件、CLI） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch3`，提交 4fb0bc5） · 置信度：已复现

## 问题

流水线在写出日报文件（这是持久的提交点）之后，还要更新论文索引：`addDailyReports`、出处投影、`setSummaries`。这些步骤失败时返回 `failed_transient`（“paper index daily report update failed: …”）；如果恰好此时用户取消、插件卸载，则返回 `cancelled`。无论哪种，这一次的结果都不带 `digest`。

之后的重试看到日报文件已存在，进入“修复现有日报”分支：补索引、返回 `completed`，但这个分支从不构造 `digest`。两个宿主（插件 `plugin/main.ts:2878`，CLI `apps/cli/src/runtime.ts:189-192`、`apps/cli/src/main.ts:293-300`）都只在结果带 `digest` 时才自动发邮件，于是日志里只有“no digest … skip auto-send (repair path)”，这一天的邮件再也不会发出，也没有任何界面提示。邮件开启时，用户会发现那天只有日报文件、没有邮件。

这与邮件投递的设计不一致：ADR 0002 写的是“之后的 `completed` 可以再试，除非状态已经是 `delivered`”，投递状态按“日期 + 收件人”做了幂等，补发本来是安全的；目标文档（`docs/helm/2026-07-25-paper-identity-source-boundary/goal.md:78`）只说“repair-only skips email”，没有考虑“第一次提交后失败，修复才是第一次成功的 completed”这种情形。

触发需要提交后的索引写入失败（磁盘满、索引文件锁超时、插件与 CLI 同时写同一个库）或恰好在这几百毫秒内取消，概率不高，所以评为低。

## 原因

- `packages/core/src/pipeline/pipeline.ts:214-217` 与 `:737-786` `repairExistingDaily`：返回 `{ kind: "completed", papersWritten }`，没有 `digest`、`outcome`。
- `pipeline.ts:607-620`：只有新生成路径才调用 `buildDailyDigest`。
- 宿主侧的自动发送只认 `result.digest`（见上）。
- 复现（一次性测试，已删除）：第一次运行 `addDailyReports` 抛出 `index busy` → `{"kind":"failed_transient","reason":"paper index daily report update failed: index busy"}`，日报已写出；第二次运行 → `kind: completed`、`papersWritten: 1`、`hasDigest: false`。

## 可能的解决方案

推荐：修复分支在 `completed` 时也构造 `digest`——从已存在的日报文件重建（`extractPaperSummaries` 已能解析出每篇的结构化字段和回退块），或在第一次提交时把 digest 作为待发送状态写进投递状态文件，修复时取出。投递状态已按日期和收件人去重，重复调用不会重复发送。

备选：修复分支在日志里给出一条 warn 并在运行结果里标记“邮件未发送”，让用户可以手动补发。

## 修复后应补的测试

- 第一次运行索引更新失败、第二次修复成功：第二次的结果带 `digest`（或触发了投递），邮件只发送一次。
- 日报文件由旧版本写出、邮件已经 `delivered` 的日期再次运行，不会重复发送。

## 修复情况

日报已写出、索引步骤失败或被取消之后，重试走的“修复”分支现在也会带着 digest：它从已写出的日报重建邮件内容（`digestFromDailyReport`，复用现有的块、字段和回退摘要解析器，保持日报里的论文顺序与主题分组，语言以日报自己的标题为准），测试用真实的日报渲染器验证，重建出的 digest 渲染出的主题、HTML 和纯文本与新生成时逐字节一致（中英文）。宿主照旧在结果带 digest 时调用投递，按“日期 + 收件人”的 claim 决定是否真的发送：已 `delivered`、结果不明和明确拒绝的日期被跳过，从未尝试过或被释放为“未发送”的日期会发出，所以邮件只发一次。读不到日报或解析不出任何论文时，修复照常完成、没有 digest，不会失败。

这同时给 MAIL-02 留下的缺口一条出路：一个已 completed、邮件被释放为 `failed / provider_not_invoked` 的日期，重新运行该日期（日报已存在，只走修复）会带着 digest 重新 claim 并补发，不需要另外的设计。代价：重新运行一个过去的日期时，如果那天的邮件从未尝试过（例如当时没开邮件），现在会发出那天的邮件；调度器的回溯窗口只有 5 天，所以最坏情况是清空运行状态后补发最近几天的邮件，仍受每日额度约束。不需要部署。

测试：`packages/core/tests/pipeline.test.ts`（第一次索引失败、第二次修复：digest 与新生成的一致；配真实 claim 存储时邮件只发一次，重复修复被 `already_delivered` 跳过；被释放的日期补发；读不到论文的日报、重建失败都不影响修复。前三个用例在旧代码下失败）、`packages/core/tests/delivery/digest-from-report.test.ts`（中英文往返、顺序与空主题、改名或删除的主题、旧布局、紧急报告、CRLF、线性时间）。

- PR 复核补强：读取日报标题时只匹配固定长度的标题前缀，再直接截取正文，避免长空白与 Unicode 分隔符触发反复回溯；新增十万制表符标题的正确性及耗时回归。邮件与流水线 253 项测试、核心类型检查通过。
