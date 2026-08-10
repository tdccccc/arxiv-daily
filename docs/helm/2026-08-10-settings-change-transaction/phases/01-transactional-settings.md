# P1 — transactional-settings

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

任何受支持的设置变更要么同时提交持久配置和对应 runtime effect，要么保持原状态且向用户报告失败。

## Assumptions

- output store 可以拆成先 prepare/load、后无失败 install 的两阶段操作。
- scheduler/logger effect 在持久化成功后同步应用不会失败或可以被显式处理。
- 普通 declarative scalar 可由 flat key 映射到候选 settings，而无需替换 live 根对象。

## Approach

新增 plugin-local SettingsChangeService 串行构造候选、验证并准备资源，持久化成功后原位提交和安装 effect；declarative 与 legacy renderer 只负责输入与显示回退。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增 declarative output collision/store reload、save failure、logger/timer effect 和无效 timezone 测试，确认当前实现保存了错误值或未更新运行时。
- Green / regression checks: change service、declarative、legacy、lifecycle 测试通过，Obsidian typecheck/lint、边界与完整 release-equivalent gate 通过。

## Tasks

- [ ] 以 Red 测试锁定 declarative 直接 mutation、output store、runtime effect 与 timezone 缺口。
- [ ] 实现串行 SettingsChangeService 和候选 settings 的 validate/prepare/persist/commit 协议。
- [ ] 将普通 declarative scalar 与关键 custom rows 接入统一服务。
- [ ] 将 legacy output、timezone、interval、log level 和 schedule enabled 接入同一服务。
- [ ] 增加无效持久化 timezone fallback 和 renderer 控件回滚测试。
- [ ] 运行定向、完整回归和独立代码复核。
- [ ] 完成独立技术报告交接、提交和 Helm 收口。

## Verification

- `npm test --workspace obsidian-arxiv-daily -- --maxWorkers=1 tests/settings-change-service.test.ts tests/settings-declarative-tab.test.ts tests/settings-tab.test.ts tests/settings-lifecycle.test.ts`
- `npm run typecheck --workspace obsidian-arxiv-daily && npm run lint && npm run check:boundaries`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`
- `npm run typecheck && npm run build && npm run smoke:build && git diff --check`

## Abort / reshape triggers

- 如果统一服务要求替换 settings 对象 identity 或重写全部 topic/category 编辑器，停止并缩小事务边界。
- 如果活跃 operation 无法可靠判定，输出根切换必须保守拒绝而非静默交错。
- 如果需要改变 CLI timezone 语义，停止并将共享 Core 校验拆为独立兼容决策。
