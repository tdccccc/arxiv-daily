# Workbench core parity

status: active
owner: /root
created: 2026-10-03T00:22:59+08:00
updated: 2026-10-03T12:38:19+08:00
revision: 2

## Intent

让 DSH 中的独立工作台承接 Obsidian 产品的核心文献流程，Agent 对话仍是辅助入口。用户已授权依次补设置与首次使用、个人文献库、方向审核、定时与运行管理。

## Success criteria

- 无配置也能打开工作台，在图形设置中完成首次配置并运行日报；已有配置可修改，密钥不回显，冲突不覆盖。
- 可连接、授权、扫描、索引与检索个人文献库，复用现有共享业务。
- 可审核并确认方向与增量建议，未经确认不参与个性化发现。
- 可查看并配置定时任务、查看运行结果与执行恢复操作。

## Non-goals

- Claude Mod 实现、Obsidian 专属编辑体验、另一套数据库或模型管线。

## Constraints

- 仅当前隔离 worktree；保留已有 0.1.3 名称修改。
- 不使用 Computer Use，不修改真实文献、定时任务，不调用付费模型或发送邮件进行测试。
- Markdown 为研究记录；共享 CLI/core 与既有 consent、revision、原子写入机制。

## Phases

1. P1 — 图形设置与首次使用闭环 — status: done
2. P2 — 个人文献库管理与检索 — status: active
3. P3 — 方向与建议审核 — status: pending
4. P4 — 定时与运行管理 — status: pending
