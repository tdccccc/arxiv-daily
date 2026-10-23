# 在运行工作台的终端按 Ctrl+C，正在生成的日期会卡在"正在运行"，一小时内无法重新生成

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 CLI-01
> 范围：CLI 阅读工作台（生成任务的取消与退出） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 e79a7ba） · 置信度：已复现
> 相关：STATE-01（`2026-10-21-crash-left-running-state-blocks-rerun.md`）是核心侧的"陈旧 running 恢复"策略；本条只处理工作台 Ctrl+C 这个会主动制造残留状态的触发点，两条可以分别修。

## 问题

工作台启动后提示"Press Ctrl+C to stop."。如果此时有一个生成任务（日报或单篇详细总结，也包括工作台自己的定时检查）正在运行，直接在终端按 Ctrl+C 退出，会留下一个"假的运行中"状态：

- `run-state.json` 里该日期保持 `status: "running"`，不会变成可重试的 `pending`。
- 之后一小时内（`STALE_RUNNING_RECOVERY_MS = 60 * 60 * 1000`）再次运行同一天，不管用 `arxiv-daily run --date`、`run --today` 还是工作台的"生成"按钮，都只得到 `run 2026-10-20: skipped (already running)`，没有任何运行。
- 工作台日历里这一天显示"正在生成 / 该日期的日报任务正在运行"，而且 `canGenerate: false`，没有"重试生成"按钮。
- `arxiv-daily status` 的 `recentRuns` 也一直显示 `running`。

对照实验（同一个沙箱，同一个被挂起的 arXiv 请求）：

| 取消方式 | 子进程收到的信号 | 结束后该日期的状态 |
|---|---|---|
| 工作台里点"取消任务"（`api/runs/cancel`） | 1 次 SIGINT | `pending`，`error: "cancelled by SIGINT"`，可重试 |
| 只给 workbench 进程发 SIGINT（`kill -INT <pid>`） | 1 次 SIGINT | 同上 |
| 终端 Ctrl+C（SIGINT 发给整个前台进程组） | 2 次 SIGINT | 仍是 `running`，一小时内无法重新生成 |

Windows 上 `child.kill("SIGINT")` 没有 POSIX 信号语义，Node 会直接强制结束子进程（文档：进程会被"forcefully and abruptly"终止）。按代码推断，Windows 上连"取消任务"按钮也会走到同样的"卡住"结果。这一点没有在 Windows 上验证。

## 原因

- `apps/cli/src/workbench/launch.ts:41`：生成任务通过 `spawn(process.execPath, [executable, ...args], …)` 启动，没有 `detached`，子进程和 workbench 在同一个进程组。终端按 Ctrl+C 时，SIGINT 会发给整个前台进程组，子进程先收到第 1 次。
- `launch.ts:59-70`、`launch.ts:68`：workbench 同时也收到 SIGINT，`stop()` → `app.close()` → `controller.abort()`，触发 `launch.ts:49` 的 `cancel`，对子进程再发第 2 次 `child.kill("SIGINT")`。
- `apps/cli/src/main.ts:527-552` `installSignalHandlers`：第一次信号做优雅取消，第二次信号直接 `process.exit(130)`（`main.ts:544`）。第 2 次信号只比第 1 次晚几毫秒，优雅取消还没来得及把日期写回 `pending`，进程就被强退了。
- 之后的一小时由 `packages/core/src/services/state-store.ts:188` 的 `recoverStaleRunning` 兜底，它只在新建调度器或启动时执行，且只回收 `lastAttempt` 早于一小时的记录；`apps/cli/src/workbench/calendar.ts:81` 把 `running` 状态直接显示成"正在运行"并且不给重试入口。

## 复现

在一个只含本机回环的网络命名空间（`unshare -rnm`）里，把 `arxiv.org` 指向本地一个只接收请求、从不应答的 HTTPS 桩，用本地 CA 信任它：

1. 启动 `arxiv-daily ui --no-open`（`setsid`，自成进程组），`POST api/runs {"kind":"daily","date":"2026-10-20"}`，等子进程停在 `stage fetch-recent`，此时 `run-state.json` 为 `running`。
2. `kill -INT -- -<workbench pgid>`（等价于终端 Ctrl+C）。workbench 打印 `Stopping workbench…` 后退出，`run-state.json` 仍为 `{"status":"running",…}`。
3. `arxiv-daily run --date 2026-10-20` → `skipped (already running)`；`api/calendar` 该日 `state: "running"`、`canGenerate: false`。
4. 对照：同样的步骤，第 2 步改为 `kill -INT <workbench pid>` 或调用 `api/runs/cancel`，状态变成 `pending` / `cancelled by SIGINT`。

## 可能的解决方案

推荐：让 workbench 成为唯一把取消信号发给子进程的一方，并且只发一次。

1. POSIX 上用 `detached: true` 启动子进程（新进程组），终端的 Ctrl+C 就只会到达 workbench，由 `cancel` 转发唯一一次 SIGINT。workbench 异常退出时要保证子进程不会成为孤儿（退出前 `await running` 已经覆盖了正常路径）。
2. Windows 上不要依赖 `kill("SIGINT")`：通过已有的 IPC 通道（`stdio` 里已经有 `ipc`）给子进程发 `{ type: "arxiv-daily/cancel" }`，子进程收到后调用与 SIGINT 相同的 `operations.cancelAll`，子进程退出后才算取消完成。

备选：子进程侧把"收到 IPC 通道的父进程管理时的第二次 SIGINT"当作重复信号处理（比如首次信号后 2 秒内的第二次信号只记录不退出）。这样不改 spawn 参数，但会让交互式 `run` 在终端里"连按两次 Ctrl+C 立即退出"的行为和有 IPC 的情况不一致。

## 修复后应补的测试

- 子进程在 workbench 同进程组时（模拟终端 Ctrl+C，对进程组发一次 SIGINT），任务结束后 `run-state` 为 `pending` 而不是 `running`。
- `cancel` 只给子进程发一次取消（POSIX 信号或 Windows IPC）。
- 取消后，同一天的日历状态不是"正在运行"，`canGenerate` 为 true。

## 修复情况

工作台现在通过子进程已有的 IPC 通道发一条 `arxiv-daily/cancel` 消息来取消生成任务（`runCliChild`），不再给子进程转发第二个 SIGINT。终端 Ctrl+C 已经把 SIGINT 发给整个前台进程组，之前工作台又补发一次，子进程把第二次当成"强制退出"，来不及把取消写回就走了；现在子进程只收到一次信号，优雅取消后日期回到 `pending`。连按两次终端 Ctrl+C 仍是强制退出，Windows 上也不再依赖会直接终止进程的 `kill("SIGINT")`。CLI 入口在第一个 `await` 之前就挂好这条消息的监听（`WorkbenchCancel`），启动阶段收到的取消不会丢，而是在运行开始前就退出 130。没有采用 write-up 的 `detached: true`，因为那样第二次 Ctrl+C 就无法强制结束卡住的子进程。

测试：`apps/cli/tests/workbench-run-child.test.ts`（真实子进程，3 例，旧代码下全部失败，终端同时发信号的用例在旧代码下得到退出码 130）、`cli-main.test.ts` 新增 5 例。用真实构建的 CLI 在隔离网络命名空间里验证：对整个进程组发 SIGINT，修复前日期停在 `running`，修复后是 `pending`；单独给工作台发 SIGINT、点"取消任务"同样得到 `pending`。
