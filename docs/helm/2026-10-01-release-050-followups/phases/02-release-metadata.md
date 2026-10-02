# P2 — 0.5.0版本、发行说明与发布前检查

goal_ref: ../goal.md
created: 2026-10-02T15:03:26+08:00
updated: 2026-10-02T15:03:26+08:00
revision: 1

## Outcome

在fix/review-followups同步0.5.0元数据，提交准确发行说明与本地发布检查结果，停在push授权点。

## Assumptions

- 用户已确认Retry preparation完成，并明确选择A：生成、接受、保存和窗口操作均正常；Linux核心流程手测通过。
- CLI设置编辑新命令不纳入0.5.0，仍是后续版本范围。
- 版本同步是机械元数据变更，无需人为制造Red；以sync/check-release-version、完整测试和实际构建验证。

## Chunks and checks

1. 运行sync:release-version与check:release-version，逐文件检查版本/内部依赖/版本映射/锁文件。
2. 发行说明按最终实现核对，修正Paper Index schema5、标签隐藏、配置目录权限、普通日报上限、旧资料库恢复与Linux手测范围。
3. 为避免影响其他工作树，在/tmp独立源码副本执行npm ci、audit、完整测试、lint/typecheck/build、release-tools、boundaries、submission、smoke:build和smoke:install；比较被测源文件与当前分支一致。
4. 展示发行说明，单独本地提交版本与说明；记录证据和后续授权点。不自动push或更新PR。

## Abort / reshape triggers

- 依赖安装/测试/构建或版本检查失败不得当通过；未验证平台只标明现有证据。
- 不重置main，不清理分支/stash，不触碰其他worktree。发布仍经已有PR51与逐项远程授权。
