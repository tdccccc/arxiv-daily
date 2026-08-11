# P3 — operator-tool-runbook-and-governance

goal_ref: ../goal.md
updated: 2026-08-11

## Outcome

Relay 独立产品提供默认只读、可中断恢复的 cutover operator 工具和完整 runbook；只有交互式 `apply` 经逐步精确确认才能触发生产 mutation，CI 永远只执行测试与 Wrangler dry-run。

## Assumptions

- 生产 cutover 只用于当前单用户、单版本、一次性切换，不需要 mixed-version rollout、自动 rollback 或 unattended apply。
- Wrangler 可以只读确认 account、binding 与 secret 名称；工具不读取 secret 值，operator bearer 只通过隐藏输入驻留内存。
- 首次 inventory 前部署的精确 source SHA 会成为永久 cutover build binding；工具不能静默替换已绑定 build/protocol/identity。
- 验证邮件、device token 更新和真实测试邮件继续由 operator 在工具外手工完成。

## Approach

把流程拆成可注入依赖的纯编排库和很薄的 CLI。默认命令只输出 plan/preflight；`apply` 要求 TTY、clean worktree、精确 HEAD、通过的本地 dry-run、正确 account/binding/secret 名称，并在每个生产 mutation 前要求包含安全资源后缀的精确确认。部署和 action 的不明结果只通过 authenticated GET status 恢复；私有 journal 使用 `0600` 原子保存安全状态，不保存 bearer、邮箱、token、KV 值、API key 或正文。

## Test strategy

- change kind: safety-critical operator workflow
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 注入 fake Wrangler/fetch/clock/filesystem 的测试先证明当前没有默认 plan、TTY-only apply、精确确认、私有 journal、断点恢复和 no-email/no-deploy 边界。
- Green / regression checks: 非 TTY、脏 worktree、HEAD/account/build/binding/secret-name mismatch、部署或 mutation 结果不明全部停止；同 operation 通过 status 恢复；journal 始终 `0600` 且无敏感内容；CI 和默认命令不执行 deploy、KV mutation、验证或投递 endpoint。

## Tasks

- [ ] 实现纯 operator 编排库与默认 plan/preflight，使用 fake dependencies 完成严格 Red-Green。
- [ ] 实现 TTY-only `apply`、隐藏 bearer 输入、逐步精确确认和 `0600` 原子 journal/恢复，并覆盖中断与结果不明路径。
- [ ] 接入 Wrangler 与 authenticated cutover status/action，显式注入 full-SHA `BUILD_IDENTITY`，并证明工具不会调用验证/投递或直接操作 KV。
- [ ] 增加 relay package scripts、CI/no-deploy invariant 与独立产品治理测试。
- [ ] 编写 relay operations、部署/cutover runbook 和 0.4.2 release note，明确 `TOKEN_SECRET` 轮换、`IDENTITY_SECRET` 长期稳定及永久 build binding。
- [ ] 运行 relay、root release-equivalent、workflow/product invariant、build/smoke 与 diff 门禁，完成技术报告 handoff 并接受 P3。

## Verification

- `npm --prefix services/email-relay run test:cutover`
- `npm --prefix services/email-relay test -- --maxWorkers=1`
- `npm --prefix services/email-relay run typecheck`
- `npm exec --prefix services/email-relay -- wrangler deploy --dry-run --config services/email-relay/wrangler.toml`
- `npm run test:release-tools`
- `node --test scripts/tests/independent-product-workflows.test.mjs`
- root release-equivalent gates、`git diff --check`

## Abort / reshape triggers

- 若 Wrangler 只能通过读取 secret 值而非名称证明 preflight，停止并保留该项为人工检查点。
- 若部署或 mutation 的不明结果无法由安全的只读 status 判定，停止自动恢复并进入人工 incident/fix-forward。
- 若实现要求把 operator bearer、邮箱、token、KV 原值或邮件正文写入 argv、日志、journal、PR 或确认短语，停止并重划边界。
- 若必须自动调用 `/v1/verify/start`、`/v1/deliver`、CLI/Plugin test send 才能完成流程，停止；这些动作不属于工具职责。
