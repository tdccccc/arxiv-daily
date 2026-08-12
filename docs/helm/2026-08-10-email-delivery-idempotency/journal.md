## 2026-08-10 — L2 reshape

- evidence: 独立安全复核确认 `DataAdapter.copy` 只承诺目标存在时失败，没有跨进程线性化保证；relay 按客户端幂等 key 分片会让同设备不同 key 并发绕过 KV 配额，且裸 key ledger 可被跨租户抢占。复核还发现孤儿锁/claim、408/409/5xx 分类和 eventual-consistent KV rollout 不满足 phase outcome。
- change: P1 标记为 superseded，启动 P1b；phase 改用桌面 `FileSystemAdapter` + OS exclusive create，非桌面 fail closed，并将 relay 串行边界改为认证设备作用域，纳入请求指纹、配额预占和 ledger retention。
- disposition: 保留严格状态读取、v1 阻断表示、稳定且无 PII 的 provider key、`delivered_unrecorded`/`ambiguous` 结果、缺少 DO binding fail closed 及其有效测试；重写 Obsidian exclusive-create、锁/claim 恢复、HTTP 分类和 relay DO/KV quota/rollout 测试与实现。旧客户端静态读取兼容继续保留，但不宣称未升级旧进程并发时 exactly-once。
- next: 先观察新的桌面/非桌面 claim 和租户/配额并发 Red，再以最小安全实现修复 blocker；任何真实投递和生产部署继续禁止。

## 2026-08-11 — L2 clarify compatibility boundary

- evidence: 最终安全复核要求本地持久化完全移除明文 recipient，但基线 v1 reader 只按 `record.recipient` 明文相等识别 delivered 阻断，遇到哈希 identity、未知 schema、无效记录或 marker 会退化为空状态并允许发送；不存在收件人无关的 v1 阻断表示。
- change: 保留唯一一份旧 reader 必需的兼容主状态 recipient，并要求 Node primary/tmp/backup 私有写入与 `0600`；不可变 claim/decision/result sidecar、provider key、relay ledger、日志与结果改用哈希 recipient identity 和稳定安全错误码。Provider 状态使用明确拒绝白名单；OS claim 改为 descriptor-anchored 路径解析；v2 Worker 增加 quiesced legacy KV 导入与 readiness barrier。
- disposition: 拒绝“哈希新 schema + marker 可阻断未升级旧客户端”的不实方案；继续满足 goal 中的 v1 阻断兼容和 provider key 无 PII 约束，同时将不得不保留的本地 PII 限定在私有兼容文件。单版本 cutover 不承诺新旧 Worker 并存或安全回滚。
- next: 先用旧 reader、文件 mode、sidecar 内容、legacy KV、非白名单 4xx 与父目录交换测试观察 Red，再恢复最小实现；若 descriptor-anchored create 或 quiesced migration 无法 fail closed，则禁用对应宿主/阻止 Worker v2 流量，不退回不安全路径。
