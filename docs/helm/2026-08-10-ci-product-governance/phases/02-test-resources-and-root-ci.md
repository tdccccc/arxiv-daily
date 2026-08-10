# P2 — test-resources-and-root-ci

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

默认根测试在标准 Node heap 下稳定完成，根 PR workflow 覆盖与发布前相同的代码质量、测试、构建和 smoke 检查，且不为同一 PR 重复运行完整门禁。

## Assumptions

- Core 测试的资源峰值来自单个长期 Vitest 进程累积，分批独立进程可以在批次之间释放 heap。
- focused test 调用是开发与验收路径，必须继续把测试文件和 Vitest 参数传给现有 runner。
- 根 release group 不包含 email relay 和 VS Code companion；它们在 P3 进入独立 CI。

## Approach

先在默认 heap 下复现普通 `npm test` 的资源失败并保留诊断证据。为 Core 增加确定性分批 runner：无显式测试目标时按互斥批次启动短生命周期 Vitest 子进程；有显式目标或参数时保持直接调用兼容。随后更新根 PR workflow，使 release-tool tests、boundaries、lint、typecheck、默认 tests、build 和 smoke build 都是必经门禁，并将完整 push 触发限制为 `main`。

## Test strategy

- change kind: CI/runtime fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 标准 Node heap 下普通根 `npm test` 在 Core 阶段耗尽 heap；现有根 PR workflow 缺少 tests、boundaries、build、smoke 与 release-tool tests。
- Green / regression checks: 默认 heap 根测试完成；Core focused 文件和额外 Vitest 参数仍生效；workflow invariant 测试锁定事件去重和 release-equivalent steps；8 GiB 单 worker 完整门禁继续通过。

## Tasks

- [ ] 复现并记录默认 `npm test` 的 Core heap 失败。
- [ ] 以 Red 测试锁定分批 runner 的完整覆盖、失败传播和 focused invocation 兼容性。
- [ ] 实现 Core 短生命周期测试批次并在标准 heap 下验证根测试。
- [ ] 升级根 PR workflow 并加入 workflow invariant 测试。
- [ ] 运行 release-equivalent 根门禁、完成独立复核与技术报告交接。

## Verification

- `npm test -- --maxWorkers=1`
- Core focused test invocation with an explicit file and Vitest arguments
- `npm run test:release-tools`
- `npm run check:boundaries && npm run lint && npm run typecheck`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`
- `npm run build && npm run smoke:build && git diff --check`

## Abort / reshape triggers

- 如果必须改变 Vitest 断言环境或测试隔离语义才能降低 heap，停止并重新划分测试进程，而非降低覆盖。
- 如果分批无法保证每个 Core test file 恰好执行一次，停止并改用 manifest 驱动的显式分组。
- 如果 workflow 改动需要把 relay/companion 纳入根 lockfile 或版本组，保持 P2 边界并留到 P3 的独立 workflow。
