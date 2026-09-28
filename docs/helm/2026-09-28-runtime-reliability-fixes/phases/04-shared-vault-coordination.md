# P4 — shared-vault-coordination

goal_ref: ../goal.md
created: 2026-09-28T23:16:55+08:00
updated: 2026-09-28T23:28:27+08:00
revision: 2

## Outcome

同机多个 CLI/Plugin 进程使用同一 Vault 时，每次日运行以及论文索引的读改写事务有跨进程互斥；崩溃不会造成永久的普通争用或错误抢占活跃持有者。

## Assumptions

- 本地文件系统提供同目录 hard-link 的原子不覆盖发布；不支持该能力时明确失败，不声称支持网络文件系统。
- 使用 OS PID 存活检查，仅明确确认持有者已退出时恢复；PID 被重用但无法证明旧持有者已退出时保守等待，不靠墙钟抢占。
- 协调记录在用户私有的机器本地目录，以 canonical Vault root 和资源名散列分区；不进入 Vault 数据导出，移动研究目录不会复制一把活跃的锁。
- 不删除最新世代以避免 ABA；竞争恢复用一个原子决策文件仲裁，只有完整写入的记录才发布。

## Approach

给 StorageAdapter 增加可选的共享锁端口。Node 提供可移植锁实现，Plugin desktop 通过受限 node-runtime 子路径复用；不让 Core 依赖 Node。RunLock 保留原有同步接口，在 withLock 中获取共享锁；PaperIndexStore 在完整 load/mutate/save 外获取等待式锁。启动时的临时 Markdown 清理使用与日运行相同的锁，避免删除另一进程的在写文件。

## Chunks

### Chunk 1 — 本地共享锁与崩溃恢复

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新端口/模块的类型契约缺失；最小可编译表面后，真实子进程争用、崩溃、双恢复者和活跃持有者等待测试未满足。
- Green check: node-runtime real-filesystem/child-process tests；无同时进入临界区，退出后可重新取得锁，wait=false 返回忙，取消/超时不泄漏占位。
- regression checks: node-runtime suite 与 typecheck。
- [x] implementation and tests accepted (missing API baseline then four behavioral Reds; real subprocess Green; node-runtime 49/typecheck passed)

### Chunk 2 — 日运行与论文索引接入共享锁

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 独立实例的 RunLock 不共享临界区；真实两个子进程同时更新索引丢记录。新增回归先失败再接入 StorageAdapter 锁。
- Green check: Core RunLock/PaperIndexStore tests 与 Node 多进程索引验证；Plugin/CLI composition tests 证明选中真实锁实现。
- regression checks: scheduler/pipeline/index suites、完整 node-runtime/CLI/plugin suites、root typecheck/boundaries/build。
- [ ] implementation and tests accepted

### Chunk 3 — 启动清理不干扰运行中进程

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 在另一实例持有日运行锁时构造 host，不得调用 Markdown 临时文件清理；现有启动路径无条件调用。
- Green check: CLI runtime / Plugin lifecycle tests。
- regression checks: 全 workspace tests、lint/typecheck/build/boundaries/submission。
- [ ] implementation and tests accepted

## Phase verification

- 所有并发关键结论使用真实临时目录和独立 OS 子进程，不只依赖 Promise.all 或 mock。
- 记录实际平台 Linux；Mac/Windows 原生验证留 P6，但 P4 的代码保持可移植。

## Abort / reshape triggers

- 回收协议可能删除/抢占新活跃持有者时停止，先证明唯一仲裁点。
- 锁文件不可读、PID 状态未知、文件系统不支持原子发布时明确失败，不将其当作空锁。
