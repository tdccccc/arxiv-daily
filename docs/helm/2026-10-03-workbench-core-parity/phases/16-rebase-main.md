# P16 — Rebase onto main

goal_ref: ../goal.md
created: 2026-10-05T01:53:35+08:00
updated: 2026-10-05T01:53:35+08:00
revision: 1

## Outcome

Rebase当前隔离分支到origin/main 7e1774b，保留主线文献库/设置/可靠性改进及DSH共享设置/P15状态，验证后提供重新构建的DSH包。

## Assumptions

- 用户明确授权rebase。备份backup/claude-code-research-plugin-pre-rebase指向eafe972。
- 仅修改当前worktree；不改main、不push、不安装生产插件、不用Computer Use。
- 行为保持型整合：Green基线→冲突语义整合→Green；整合暴露的实际缺陷用已有失败回归或新增Red→Green。

## Chunks

1. 保存备份并确认基线：CLI配置/业务设置/日历20、Obsidian设置/onboarding54、core源/状态30，104项已通过。
2. 逐提交rebase；检查重叠设置、源适配器、宿主服务和测试，不能用全局ours/theirs替代整合。
3. 回归各端设置/状态/文献库，typecheck、boundaries、inventory、构建和DSH隔离Host；检查main是当前HEAD祖先与工作区干净。

## Abort / reshape triggers

- 不确定行为取舍时先查双方提交/测试，不丢弃功能；若出现目标冲突则记录并报告。
- 备份保留；不使用skip绕过未解决的业务冲突。
