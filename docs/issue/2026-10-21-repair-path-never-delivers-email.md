# 日报已经写出、但后面的索引步骤失败或被取消后，重试走“修复”分支，当天的邮件永远不会发送

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 PIPE-08
> 范围：日报流水线的提交后恢复；邮件自动发送（插件、CLI） · 严重程度：低 · 状态：未修复 · 置信度：已复现

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
