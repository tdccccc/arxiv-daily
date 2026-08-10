## 2026-08-10 — L2 reshape

- evidence: 独立安全复核确认 `DataAdapter.copy` 只承诺目标存在时失败，没有跨进程线性化保证；relay 按客户端幂等 key 分片会让同设备不同 key 并发绕过 KV 配额，且裸 key ledger 可被跨租户抢占。复核还发现孤儿锁/claim、408/409/5xx 分类和 eventual-consistent KV rollout 不满足 phase outcome。
- change: P1 标记为 superseded，启动 P1b；phase 改用桌面 `FileSystemAdapter` + OS exclusive create，非桌面 fail closed，并将 relay 串行边界改为认证设备作用域，纳入请求指纹、配额预占和 ledger retention。
- disposition: 保留严格状态读取、v1 阻断表示、稳定且无 PII 的 provider key、`delivered_unrecorded`/`ambiguous` 结果、缺少 DO binding fail closed 及其有效测试；重写 Obsidian exclusive-create、锁/claim 恢复、HTTP 分类和 relay DO/KV quota/rollout 测试与实现。旧客户端静态读取兼容继续保留，但不宣称未升级旧进程并发时 exactly-once。
- next: 先观察新的桌面/非桌面 claim 和租户/配额并发 Red，再以最小安全实现修复 blocker；任何真实投递和生产部署继续禁止。
