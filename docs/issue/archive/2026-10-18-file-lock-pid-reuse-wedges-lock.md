# 文件锁只按 PID 判断持有者是否存活：崩溃或重启后 PID 被复用，日报运行锁永久卡死

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CORE-09
> 范围：`packages/node-runtime` 主机锁（日报运行锁、论文索引锁） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 7a7662b） · 置信度：已复现

## 问题

主机锁位于 `~/.arxiv-daily/host-locks`，跨重启持久存在，持有者记录只有 `{pid, token}`。只有 `process.kill(pid, 0)` 返回 ESRCH 时才会回收锁。

CLI 或插件在持锁时被杀（SIGKILL、OOM、断电、容器停止），记录会留下。重启后只要有任意无关进程碰巧用了这个 PID（开机后 PID 很小且复用很快），`pidIsAlive()` 就返回 true，这个库的 `tryAcquire` 永远返回 null。调度器每次都记"锁被占用 / 跳过"；论文索引锁（`wait:true`）30 秒后抛出"shared storage lock timed out"，表现为反复的索引失败。没有自动恢复，没有指出是哪个陈旧锁的日志，也没有命令能解锁，用户只能手动删除一个哈希命名的目录。EPERM 也被当作"存活"。

CORE-01 的 OOM 正是留下陈旧锁的一个现实途径。

## 原因

- `packages/node-runtime/src/file-lock.ts:87`：`if (pidIsAlive(previous.pid)) return null;`，没有进程身份校验（启动时间、开机 ID、主机名）。
- `file-lock.ts:146-153` `pidIsAlive`；持有者记录（7-12 行）缺少身份字段。

## 复现

审查脚本伪造一条 PID 为当前某个无关存活进程的锁记录，`tryAcquire` 永远返回 null。

## 可能的解决方案

推荐：持有者记录增加进程身份并校验——Linux 读 `/proc/<pid>/stat` 第 22 字段（启动时间）和 `/proc/sys/kernel/random/boot_id`；其他平台至少记录主机名和开机时间（由 `os.uptime()` 推算）。开机 ID / 开机时间或进程启动时间不一致即视为持有者已死。旧格式记录（无身份字段）保持现有行为。回收陈旧锁时记录日志。

保持"不按时间抢占存活持有者"的原则。

## 修复后应补的测试

- 记录中开机 ID 与当前不同 → 锁被回收。
- 同一开机、PID 存活但启动时间不同 → 锁被回收。
- 真正存活的持有者 → 不回收。
- 旧格式记录 → 行为不变。

## 修复情况

锁记录增加进程身份（Linux：boot_id + /proc 启动时间；其他平台：主机名 + 开机时间）。PID 存活但身份不符时视为陈旧锁，回收并记录日志。没有身份字段的旧记录保持仅按 PID 判断。
