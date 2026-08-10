# Paper Index Durability

status: active
updated: 2026-08-10
owner: zcode-main-session

## Intent

让 Paper Index 能从有效备份可靠恢复，并确保同一进程内所有读改写事务都在同路径队列中使用最新状态，避免空索引和陈旧快照覆盖。

## Success criteria

- [ ] primary missing/corrupt 时能读取有效 `.bak`，真实 I/O 不可读或所有现有副本无效时 fail closed。
- [ ] 保存流程不会用损坏 primary 覆盖唯一有效备份，提升失败后仍保留可恢复内容。
- [ ] daily selection、history sync 和所有领域 mutation 在队列内重新读取并合并最新索引。
- [ ] schema 1–4、legacy 路径和现有 Paper Index 用户语义保持兼容。
- [ ] 定向及完整门禁通过，diagnostics 和技术报告反映真实恢复语义。

## Non-goals

- 不引入仓库级数据库或抽取全仓 durable JSON 框架。
- 不声称在当前 Storage contract 下实现 Plugin/CLI 跨进程 serializability 或掉电级 fsync 保证。
- 不升级 Paper Index schema。

## Constraints

- `.bak` 是敏感内部索引，恢复和保留不得泄露内容。
- 生产调用方不得继续以仓储外的陈旧 `load → save` 完成 read-modify-write。
- 改动限定在 Paper Index、selection/history 调用方、diagnostics 及对应测试。

## Phases

1. P1 — Paper Index 具备有效副本恢复和排队读改写语义 — status: active
