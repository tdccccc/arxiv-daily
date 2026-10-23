# 设置了自定义 `XDG_CONFIG_HOME` 的用户：`schedule install` 装好的 cron 任务每天都找不到配置，日报永远不会生成

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 CLI-03
> 范围：CLI `schedule install` / `schedule show`（Linux / macOS cron） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch3`，提交 e766d93） · 置信度：已复现

## 问题

CLI 的配置位置由环境变量决定（ADR 0003、两份 README 都写明：`$XDG_CONFIG_HOME/arxiv-daily/config.toml`，缺省 `~/.config/arxiv-daily/config.toml`），而且是唯一的发现路径，没有 `--config` 可以覆盖。

很多人只在 `~/.profile`、`~/.zshenv`、`~/.bashrc` 里 `export XDG_CONFIG_HOME=...`（例如 `~/dotfiles/config`、`~/.local/etc`）。`arxiv-daily init`、`schedule install` 在他们的终端里运行时读写的是自定义位置。但 cron 用 `/bin/sh -c` 执行任务，不读这些启动文件，环境里没有 `XDG_CONFIG_HOME`。`schedule install` 写出的 cron 行没有把它带上：

```
30 9 * * 1-5  /usr/bin/node /…/arxiv-daily-cli.cjs run --today >> /custom/xdg/arxiv-daily/schedule.log 2>&1  # arxiv-daily-managed
```

于是 cron 里的 `run --today` 去读 `~/.config/arxiv-daily/config.toml`，不存在，退出码 2。实测（沙箱里把配置放在自定义 `XDG_CONFIG_HOME`，用 `env -i HOME=… PATH=/usr/bin:/bin /bin/sh -c '<cron 行的命令部分>'` 模拟 cron 环境）：

```
sh exit=2
# schedule.log（写在自定义配置目录旁边）：
CLI config not found: <HOME>/.config/arxiv-daily/config.toml
Run: arxiv-daily init
```

结果是 `schedule install` 报告"installed 1 cron line(s)"、`schedule show` 也看不出问题，但之后每个工作日 cron 都失败一次，只在 `schedule.log` 里留下一行错误。用户通常是几天后发现日报没有生成才去查。如果用户恰好也在默认位置留着一份旧配置（比如以前跑过 `init`），cron 会悄悄用那份旧配置运行——不同的主题、不同的 vault，更难发现。

Windows 不受影响（`schedule install` 本来就不支持）。

## 原因

- `apps/cli/src/schedule-cmd.ts:15-19` `defaultCronRuntime` 只记录 `nodePath` 和 `logPath`；`schedule-cmd.ts:41-43` 的 cron 行模板只有 `<node> <script> run --today >> <log> 2>&1`，没有把解析出的配置位置传给子进程。
- `apps/cli/src/config-path.ts:13-16`：配置目录取自运行时环境变量 `XDG_CONFIG_HOME`，cron 环境里这个变量不存在。
- 同一份设计已经为"cron 的 PATH 很短"做了处理（用 `process.execPath` 指名 node，见已归档的 `2026-10-18-schedule-install-cron-unrunnable.md`），但没有对"cron 的环境变量很少"做同样处理。日志路径已经跟着配置目录走（`logPath` 是 `dirname(configPath)/schedule.log`），所以日志和配置在自定义目录，而任务读的是默认目录。

## 可能的解决方案

推荐：安装时，如果当前解析出的配置目录不是默认的 `~/.config/arxiv-daily`，在每条 cron 命令前加一个 shell 环境变量赋值：

```
30 9 * * 1-5  XDG_CONFIG_HOME=/custom/xdg /usr/bin/node … run --today >> … 2>&1  # arxiv-daily-managed
```

取值用 `dirname(dirname(config.configPath))`，沿用现有的 `cronExecutable` 转义（`%` 必须转义）。默认位置时不加，保持现有输出不变。`schedule show` 与 `schedule install` 共用 `buildCronLines`，两处自动一致。安装成功的提示里可以顺带写出"此任务固定使用 <配置路径>"。

备选：只在安装时检测到 `XDG_CONFIG_HOME` 已设置时发出警告，要求用户自行在 crontab 顶部加 `XDG_CONFIG_HOME=` 行。代价是 crontab 变量对该用户的所有任务生效，工具无法安全代写。

## 修复后应补的测试

- `config.configPath` 位于自定义目录时，`buildCronLines` 的每一行都带 `XDG_CONFIG_HOME=<目录>` 前缀（路径含空格、`%`、非 ASCII 时转义正确）。
- 配置在默认位置时，输出与现在逐字相同。
- 用 `env -i` 模拟 cron 环境执行生成的命令，能读到同一份配置。

## 修复情况

`schedule install` 和 `schedule show` 生成的 cron 行，在配置文件不在默认位置（`~/.config/arxiv-daily`）时，会在命令前加 `XDG_CONFIG_HOME=<目录>`：取配置文件所在 `arxiv-daily` 目录的上一级，用与 node、脚本、日志路径相同的转义（空格、`%`、引号、非 ASCII 都安全），相对路径会写成绝对路径。这样 cron 用精简环境启动的 `/bin/sh` 也读到同一份配置，不再悄悄找不到配置或改用默认位置里的旧配置。配置在默认位置、路径不符合 XDG 布局、或原生 Windows 时，输出与以前逐字相同。安装成功后多一行说明这个任务读取哪份配置，移动配置后需要重新运行 `schedule install`。README 和技术报告已补充。

测试：`apps/cli/tests/cli-schedule.test.ts`：自定义目录时每一行（含多个触发时间）都带前缀；默认位置和非 XDG 路径的输出不变；相对路径转为绝对路径；安装输出写出配置路径；并用 `env -i /bin/sh -c` 按 cron 的方式执行生成的命令，路径含空格、`%`、`$`、引号、反斜杠、反引号、分号、非 ASCII 时子进程拿到的 `XDG_CONFIG_HOME` 与读到配置的那个一致，默认位置时子进程环境里没有这个变量。
