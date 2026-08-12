# P3 — operator-tool-runbook-and-governance

goal_ref: ../goal.md
updated: 2026-08-12

## Outcome

Relay 独立产品提供默认只读的 cutover preflight/status 脚本、明确的人工 runbook 和 CI 只读确认；所有生产 mutation（部署、凭据撤销/更新、cutover action）由 operator 逐项授权执行，脚本永不部署、不写 KV、不调用验证/投递端点。

## Assumptions

- 生产 cutover 只用于当前单用户、单版本、一次性切换，不需要编排库、自动 apply、私有 journal 或断点恢复状态机。
- Wrangler 可以只读确认登录、binding 与 secret 名称；工具不读取 secret 值，operator bearer 只通过隐藏输入驻留内存。
- 首次部署的精确 source SHA 会成为永久 cutover build binding；preflight 报告本地 HEAD 与部署 build identity，不静默替换已绑定 build/protocol/identity。
- 验证邮件、device token 更新和真实测试邮件继续由 operator 在工具外手工完成。

## Approach

交付一个无编译的只读 Node 脚本 `scripts/cutover-preflight.mjs`：本地检查（git HEAD、wrangler.toml 必需 binding/var 名称、wrangler 登录、secret 名称清单、Wrangler dry-run）与远程只读检查（`GET /health`、`GET /ready` 状态报告）。脚本带 `--check-readonly` 静态自检模式，供 CI 证明其只读边界；真实 cutover 动作（deploy、凭据撤销、`/internal/delivery-v2/cutover` action）不在脚本内，由 runbook 指引 operator 手工执行。

## Test strategy

- change kind: safety-critical operator workflow（简化版）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 注入 fake 命令/fetch/文件依赖的测试先证明当前没有 preflight 脚本、脚本缺失时无检查输出，且 `--check-readonly` 能识别含 deploy/写 KV/投递端点调用的源码。
- Green / regression checks: 脚本对缺失 secret 名、未登录、dry-run 失败、远程不可达均报告 FAIL 且退出码非零；`/ready` 返回 503 locked 时报告 locked 而非失败；输出不含 secret 值；`--check-readonly` 通过；CI 不执行 deploy、KV mutation、验证或投递端点。

## Tasks

- [x] 将 P3 从编排库方案重划为只读 preflight + runbook（本文件）。
- [x] 实现只读 `scripts/cutover-preflight.mjs`（本地检查 + 远程只读 + `--check-readonly`），依赖注入便于测试。
- [x] 编写 preflight 单测与只读源码检查测试（vitest，fake 依赖）。
- [x] 编写 cutover runbook（部署、凭据撤销、cutover action、readiness 确认、人工验证与测试邮件）。
- [x] 增加 relay package script、CI 只读确认步骤。
- [x] 运行 relay 测试/typecheck/diff 门禁，完成技术报告 handoff 并接受 P3。

## Verification

- `npm --prefix services/email-relay test`（含 preflight 单测）
- `npm --prefix services/email-relay run typecheck`
- `node services/email-relay/scripts/cutover-preflight.mjs --check-readonly`
- `npm exec --prefix services/email-relay -- wrangler deploy src/index.ts --dry-run --config services/email-relay/wrangler.toml --outdir /tmp/x`
- `git diff --check`

## Abort / reshape triggers

- 若 Wrangler 只能通过读取 secret 值而非名称证明 preflight，停止并保留该项为人工检查点。
- 若脚本必须调用 `deploy`（非 dry-run）、KV 写、`/v1/verify/start`、`/v1/deliver` 或 `/internal/delivery-v2/cutover` 才能完成任务，停止并重划边界。
- 若必须自动发送验证/测试邮件才能完成流程，停止；这些动作不属于脚本职责。
