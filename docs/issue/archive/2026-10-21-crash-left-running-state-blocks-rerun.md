# 进程被强杀或崩溃后留下的"运行中"状态：一小时内谁都不能重跑，长期开着的 Obsidian 甚至永远不会自动恢复

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 STATE-01
> 范围：核心调度器（Obsidian 插件、CLI、工作台、DSH 共用） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 e79a7ba） · 置信度：已复现
> 相关：CLI-01（`2026-10-21-workbench-ctrl-c-leaves-date-running.md`）修的是工作台 Ctrl+C 这一个触发点；本条是核心侧的恢复策略，覆盖其余所有"没来得及写回状态就死掉"的路径。

## 问题

日报运行开始时，调度器把该日期在 `run-state.json` 里记为 `running`；正常结束会改成 `completed`、`pending` 或 `failed_*`。如果进程在中间被 SIGKILL、被 OOM 杀掉、断电、终端关闭（SIGHUP）、Obsidian 崩溃或被强制退出，这条记录就留在 `running`。

此后调度器对这个日期的所有入口都把它当成"别人正在跑"：

- 定时 tick：`checkTickGate` 返回 `running`，直接跳过；
- "运行今天 / 指定日期"（`arxiv-daily run`、插件命令、面板日历）：返回 `skipped (already running)`；
- "强制重跑"：同样返回 `skipped (already running)`，连强制也过不去；
- "重试失败 / 全部待处理"：`running` 不是失败状态，也被跳过。

唯一的兜底是 `recoverStaleRunning`，它有两个限制：

1. **只回收 `lastAttempt` 早于一小时的记录。** 崩溃发生在 09:05，用户 09:20 重新打开 Obsidian 或重新执行命令，记录才 15 分钟，不回收。
2. **只在新建调度器或调用 `start()` 时运行一次。** 插件里调度器在 `onload` 创建，之后只有改调度设置、开关调度时才会再次 `start()`。所以"崩溃后很快重启 Obsidian、然后一直开着"的用户，这条记录会一直是 `running`：过了一小时也没人再检查，当天（以及之后每次 tick）都不会再生成报告，面板日历一直显示"运行中"，直到下次重启 Obsidian 或手动切换调度开关。CLI 单次进程同理：崩溃后一小时内再执行 `run` 只会得到 `already running`。

实测（见"复现"）：09:05 崩溃、09:20 重启，`tickTodayScheduled`、`runForDateNow`、`forceRunForDate` 全部返回 skipped；时间推进到 14:00 仍然如此，流水线一次都没被调用；只有再新建一个调度器（模拟又一次重启）才被回收为 `failed_transient`。

这个"一小时"的前提其实不必要：只要本进程已经拿到了整个 Vault 共用的"日报运行锁"，就能证明没有任何活着的进程在跑日报（`running` 只会在持锁期间写入；持锁者一旦死亡，下一个获取者就能回收这把锁，见 `file-lock.ts` 按进程是否存活的判断）。在这种情况下看到的 `running` 一定是残留。

## 原因

- `packages/core/src/services/state-store.ts:7,188-218`：`recoverStaleRunning` 用固定的一小时 `STALE_RUNNING_RECOVERY_MS` 判断，不看锁。
- `packages/core/src/services/scheduler.ts:113,117,186-201`：恢复只在构造函数和 `start()` 里各触发一次，`tick`、`runForDateNow`、`forceRunForDate` 都不会再触发。
- `packages/core/src/services/scheduling/scheduler-driver.ts:642-652`（普通运行）、`653-663`（强制运行）、`624-634`（全部待处理）和 `packages/core/src/services/scheduling/run-gate.ts:19-21`（定时 tick）：在已经持有跨进程锁、且已从磁盘重新读取权威状态之后，仍把 `running` 当成"别人在跑"而放弃。
- `packages/core/src/services/run-lock.ts:27-40`：`withLock` 在拿到共享锁之后才执行回调，调用方因此本可以知道"此刻没有其他活着的运行"，但没有任何地方利用这一点。

## 复现

临时 vitest（`packages/core/tests/zz-review-state-stale-running.test.ts`，用后已删除）：用内存 `StateStore` 放一条 `{status:"running", lastAttempt: 09:05}`，共享锁总是可获得（模拟持锁进程已死），时间设为 09:20：

```
tickTodayScheduled@09:20 -> {"kind":"skipped","reason":"guarded"}  state: running
runForDateNow@09:20      -> {"kind":"skipped","reason":"already running"}
forceRunForDate@09:20    -> {"kind":"skipped","reason":"already running"}
(时间推进到 14:00，同一个调度器 tick + runForDateNow)
runForDateNow@14:00      -> {"kind":"skipped","reason":"already running"}  state: running
该日期的流水线调用次数：0
新建调度器（模拟再次重启）后 state: failed_transient  "recovered stale running state after startup"
```

## 可能的解决方案

推荐：在持有共享锁时直接回收残留的 `running`，不再等一小时。

1. `tryRun` / `confirmDurablyDone` 在持锁并 `loadAuthoritative()` 之后，如果该日期的状态是 `running`，且本进程没有它的待提交完成记录（`pendingCompletions`），并且这把锁是真正的跨进程锁（`RunLock` 构造时传入了 `acquireShared` 且返回了非空租约），就把它按 `recoverStaleRunning` 的同样规则改成 `failed_transient`（计入重试上限，错误文案保留"recovered stale running state"），然后继续按原模式处理（定时 tick 受退避保护，手动运行和强制重跑立即执行）。
2. 没有跨进程锁的宿主（例如无文件系统的移动端）保持现有一小时规则；同时让长期运行的调度器在每次 tick 开头也执行一次 `recoverStaleRunning`，而不只在启动时。

备选：只做第 2 条（每次 tick 回收超过一小时的记录）。能解决"长期开着的 Obsidian 永远卡住"，但崩溃后一小时内手动运行仍会被拒绝，CLI 一次性进程也得不到改善。

## 修复后应补的测试

- 持有共享锁、状态为 `running`、`lastAttempt` 仅 15 分钟前：`runForDateNow`、`forceRunForDate`、定时 tick 都会执行流水线，状态不再停在 `running`。
- 共享锁被另一个活着的持有者占用（`acquireShared` 返回 `null`）：`running` 状态保持不动，返回 `lock held`。
- 本进程自己有待提交的完成记录（`setCompleted` 失败后保留）时，不把该日期当作残留。
- 无跨进程锁的宿主：一小时内的 `running` 保持原行为；长期运行的调度器过了一小时之后的下一次 tick 会回收它。

## 修复情况

`running` 只会在持有日报运行锁期间写入，所以现在 `tryRun` 在拿到锁并重读磁盘状态后，如果该日期仍是 `running` 且本进程没有它的待提交结果，就把它当作已死运行的残留，按 `failed_transient` 回收（计入重试上限，保留崩溃那次尝试的时间，定时 tick 的退避从那时算起）。`RunLock.withLock` 为此新增 `crossProcess` 提示，只有真正持有跨进程的 Vault 锁时才这样做。崩溃 15 分钟后再点"运行今天 / 强制重跑 / 重试失败"会立即执行，常开的 Obsidian 下一次 tick 也会自己恢复；另一个活着的进程持锁时仍返回 `lock held`，记录不动。没有跨进程锁的宿主（如移动端）保持一小时规则，但改为每次 tick 访问到的日期都检查，而不只在启动时。工作台日历是只读展示，在恢复发生（下一次运行或 tick）之前仍会把这样的残留显示为"运行中"。

测试：`packages/core/tests/scheduler-stale-running.test.ts`（10 例，其中 7 例在旧代码下失败）、`state-store.test.ts` 的 `recoverRunningDate`、`run-lock.test.ts`。另外用真实构建的 CLI 在隔离网络命名空间里验证：同一份残留，修复前 `run --date` 输出 `skipped (already running)`，修复后会真正执行。
