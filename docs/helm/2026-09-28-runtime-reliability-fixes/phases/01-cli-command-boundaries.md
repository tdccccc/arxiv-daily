# P1 — cli-command-boundaries

goal_ref: ../goal.md
created: 2026-09-28T22:22:37+08:00
updated: 2026-09-28T22:22:37+08:00
revision: 1

## Outcome

CLI 生成的 cron 命令准确保留可执行路径，非法时间窗不会改写已有任务，异步子命令的错误和信号生命周期经过统一边界。

## Assumptions

- cron 通过 POSIX shell 执行命令，并在 shell 解析之前处理未转义 `%`；仅 shell quoting 不够。
- 当 intervalHours > 0 且 until < on 时视为错误，而非隐式跨天；单次运行不要求 until 晚于 on。
- 退出码 2 表示配置/输入错误，1 表示运行错误；CLI 的现有输出脱敏继续生效。

## Approach

保留普通安装路径的输出兼容性，对特殊路径进行 shell 与 cron 两层编码并拒绝控制字符。验证失败必须发生在读写 crontab 前。所有异步命令在统一 try/catch 中 await，信号处理器在对应工作完成后才移除。

## Chunks

### Chunk 1 — cron 可执行路径正确编码（F29）

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新建 `apps/cli/tests/cli-schedule.test.ts`；生成含空格、单引号、反斜线、美元符、反引号和百分号路径的 cron 行，经 cron 命令预处理和真实 `/bin/sh` 调用临时可执行文件，记录的参数须恰为 run、--today。原实现失败；含换行的路径须在 writeCrontab 前拒绝。
- Green check: `npm test --workspace arxiv-daily -- tests/cli-schedule.test.ts`
- regression checks: CLI 现有 schedule/config/main 测试、CLI typecheck。
- [ ] implementation and tests accepted

### Chunk 2 — 非法时间窗不安装空任务（F30）

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 倒置时间窗时 scheduleInstall 返回 2、说明原因且 read/writeCrontab 都不被调用；有效的单次和重复时间窗照常执行。原实现返回 0 并写入空任务。
- Green check: CLI schedule/config focused tests。
- regression checks: 完整 CLI suite、typecheck。
- [ ] implementation and tests accepted

### Chunk 3 — await 子命令并保留统一错误边界（F31）

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `runCli` 的 init/update/schedule/data/email 子命令 Promise 拒绝时解析为非零退出码，输出脱敏；等待中的 email 命令保留信号处理器，完成后移除。原实现部分 Promise 越过 catch/finally。
- Green check: `npm test --workspace arxiv-daily -- tests/cli-main.test.ts tests/cli-email.test.ts`
- regression checks: 完整 CLI suite、root typecheck/build/boundaries；本地假 HTTP、临时文件与注入任务，不发送真实邮件。
- [ ] implementation and tests accepted

## Phase verification

- 全 CLI 测试、类型检查；每个 chunk 的 Red/Green 与回归结果记录在 journal。
- 真实 shell 验证不修改用户 crontab；测试临时文件清理。

## Abort / reshape triggers

- cron 特殊字符处理无法通过真实 shell 行为验证时继续缩小复现，不仅用字符串快照宣称修复。
- 若统一 await 暴露已有初始化交互的不同退出语义，保留明确的取消行为并单独检验。
