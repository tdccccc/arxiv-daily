# `schedule install` 写入的 cron 行通常跑不起来（没有 node 路径、没有日志、临时文件名可预测）

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CLI-4
> 范围：CLI `schedule install`（Linux / macOS cron） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 35f0294） · 置信度：代码推断

## 问题

安装的 cron 行形如 `M H * * 1-5  <argv[1]> run --today  # arxiv-daily-managed`。可执行文件是带 `#!/usr/bin/env node` 的 .cjs。cron 的 PATH 只有 `/usr/bin:/bin`，node 来自 nvm / fnm / volta / Homebrew（Apple Silicon 上是 `/opt/homebrew/bin`）的机器会得到 `env: node: No such file or directory`，日报永远不会运行；输出也没有重定向，用户看不到任何日志。

附带：`writeUserCrontab` 把临时文件写到 `os.tmpdir()/arxiv-daily-cron-<pid>-<Date.now()>`，用默认 `fs.writeFile`（跟随符号链接、权限取决于 umask）。在共享 /tmp 上，预先放置在可猜测路径的符号链接能让它覆盖用户可写的文件。

## 原因

- `apps/cli/src/schedule-cmd.ts:11-18` `buildCronLines` 只生成 `${executable} run --today`，没有使用 `process.execPath`、`PATH=` 或输出重定向。
- `resolveBinaryPath` 返回 `argv[1]`。
- `schedule-cmd.ts:195` `writeUserCrontab` 使用可预测临时文件名，未用 `wx`。

## 可能的解决方案

推荐：
1. 生成 `<process.execPath> <脚本绝对路径> run --today >> <配置/缓存目录>/schedule.log 2>&1`，沿用现有的转义逻辑。
2. crontab 通过标准输入写入（`crontab -`），或用 `fs.mkdtemp` + `wx`、0600 权限。
3. 代价：`execPath` 固定了 node 版本，升级 node 后需重新 `schedule install`；在命令输出与文档里说明。

## 修复后应补的测试

- 生成的 cron 行包含 node 绝对路径、脚本路径与日志重定向，路径含空格时正确转义。
- crontab 写入不再使用可预测的临时文件。

## 修复情况

cron 行改为 `<node 绝对路径> <脚本绝对路径> run --today >> <配置目录>/schedule.log 2>&1`，沿用原转义逻辑；crontab 经 `crontab -` 标准输入安装，不再使用临时文件。安装输出与两份 README 提示更换 Node 后需重新 `schedule install`。
