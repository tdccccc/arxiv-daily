# DSH 被强制结束或崩溃后，插件启动的文献工作台进程不会退出，会一直在后台运行

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 DSH-01
> 范围：`extensions/dsh-arxiv-daily` 工作台子进程的生命周期（DSH Web 与 DSH Desktop，Linux/macOS/Windows） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch3`，提交 8e3925e） · 置信度：已复现

## 问题

插件声明的生命周期是“插件拥有一个懒启动的工作台服务，卸载时停止它”（`docs/helm/2026-10-02-dsh-research-plugin/goal.md` 的成功标准，README 也这样描述）。这只在 DSH 正常卸载插件或正常退出时成立：宿主调用 `manager.dispose()`，向子进程组发送 SIGTERM。

如果 DSH 宿主进程被 SIGKILL、崩溃、被 OOM 杀掉、被系统关机时强杀，或者 DSH Desktop 被强制退出，`dispose()` 不会执行。工作台子进程是用 `detached: true`（非 Windows，自成进程组）、`stdio: ['ignore','pipe','pipe']` 启动的，不依赖父进程存活：没有 IPC 通道，没有检测 stdin/管道关闭，`arxiv-daily ui` 自身也只响应 SIGINT/SIGTERM（`apps/cli/src/workbench/launch.ts:68-69`）。所以它会继续监听 `127.0.0.1:<随机端口>`，一直运行到用户手动 kill 或重启机器。

实际后果：

- 每次异常退出的 DSH 会话都留下一个孤儿 `arxiv-daily ui` 进程，用户看不到它，也没有任何提示。
- 如果用户在 CLI 配置里开启了工作台自动调度（`workbench_schedule`，由 `WorkbenchSchedule` 定时器驱动），孤儿进程会继续在后台自动生成日报并发邮件，而新一次 DSH 会话又会再启动一个新的工作台，于是两个工作台同时按计划运行（日报运行锁和邮件 claim 能避免重复结果，但仍是用户不知情的两路后台任务）。
- 孤儿进程继续持有旧的插件版本代码（README “Restart after update” 一条只覆盖正常升级，没有提到它）和已加载的配置、API 密钥内存副本。
- 它的 `frame-ancestors` 只允许最初那个 DSH 来源，所以不会被误用，也不是安全漏洞；这是资源与行为的泄漏。

已复现（一次性脚本，已删除；在脚本里用与生产相同的 `WorkbenchProcess` 启动测试夹具子进程，打印子进程 PID 后对自己发送 SIGKILL）：宿主进程以 137 退出，子进程 `ps` 显示仍在运行（`PPID` 变为 1，自己是进程组组长，`ELAPSED` 持续增长），需要手动 `kill` 才会结束。

## 原因

- `extensions/dsh-arxiv-daily/src/workbench-process.mjs:22`：`spawn(..., { stdio: ['ignore','pipe','pipe'], detached: process.platform !== 'win32' })`，子进程与父进程之间没有存活绑定。
- `workbench-process.mjs:55-79`：只有 `stop()`/`dispose()`（由插件卸载的 effect 清理触发）会终止子进程；没有 `process.on('exit')` 兜底（对 SIGKILL 也无效），也没有让子进程自己监测父进程。
- `apps/cli/src/workbench/launch.ts:55-71`：`runWorkbench` 只在 SIGINT/SIGTERM 时 `app.close()`，没有“父进程消失就退出”的逻辑。

## 可能的解决方案

推荐：让子进程自己监测父进程，并保持现有的 SIGTERM 路径。

1. 工作台子进程增加一个只在被插件启动时启用的“父进程存活监测”：插件改用 `stdio: ['pipe', 'pipe', 'pipe']` 并保持 stdin 管道打开，`arxiv-daily ui` 增加内部参数（例如 `--exit-on-stdin-close`，不写进公开 USAGE 或仅在 `--frame-origin` 同时出现时启用），读到 stdin 的 `end`/`close` 就走与 SIGTERM 相同的 `stop()`。父进程无论怎么死，管道都会被内核关闭，子进程随之退出，Windows 与 POSIX 行为一致。
2. 插件启动时若发现端口文件/锁里记录了上一个未退出的工作台（可以把子进程 PID 与随机路径写到插件自己的状态目录里），先尝试终止旧进程再启动新的。

备选：只在 POSIX 上用 `prctl(PR_SET_PDEATHSIG)`（需要原生模块，不推荐）；或在 README 的 Limitations 里明确写出“DSH 异常退出后需手动结束遗留的 `arxiv-daily ui` 进程”。

## 修复后应补的测试

- 进程测试：启动 `WorkbenchProcess` 后强杀宿主（子进程里启动一个小宿主脚本，再对它发送 SIGKILL），断言工作台子进程在限定时间内退出。
- 正常卸载路径不回归：`dispose()` 仍然 SIGTERM 进程组，`stopTimeoutMs` 后升级到 SIGKILL。
- `arxiv-daily ui` 的单元测试：stdin 关闭触发与 SIGTERM 相同的关闭流程，已有的 `run` 子任务被取消。

## 修复情况

插件启动的文献工作台现在会随 DSH 一起退出，即使 DSH 被强制结束、崩溃或被系统杀掉。`arxiv-daily ui` 新增参数 `--exit-on-stdin-close`（写在 USAGE 里，供嵌入它的宿主使用）：stdin 结束或关闭时，工作台走与 SIGTERM 相同的关闭流程（取消进行中的 run 子任务、关闭服务）。插件用 `stdio: ['pipe', …]` 启动子进程，保持 stdin 管道打开、从不写入；宿主一死，内核就关闭这根管道，Linux、macOS、Windows 行为一致，也不受宿主怎么死的影响。不带这个参数时不监听 stdin，所以手动在终端里启动的工作台不会因为 Ctrl+D 或 stdin 关闭而退出。正常卸载时 `dispose()` 仍先对进程组发 SIGTERM、`stopTimeoutMs` 后升级到 SIGKILL。DSH 包版本和发布标签没有改动，README 已说明工作台随 DSH 一起停止。

测试：`extensions/dsh-arxiv-daily/tests/process.test.mjs` 新增端到端用例，用真实打包的 CLI 通过 `WorkbenchProcess` 启动工作台，对宿主进程发 SIGKILL，等待工作台进程退出且地址不再可访问（修复前该用例失败，工作台一直存活；用例结束时会清理残留进程），原有用例补充了参数和 stdin 管道的断言；`apps/cli/tests/workbench-lifetime.test.ts`（stdin 结束触发关闭并移除监听；不带参数时不监听）和 `cli-workbench.test.ts`（参数解析，含配置缺失的启动路径）。
