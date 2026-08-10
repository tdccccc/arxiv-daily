# P1 — cli-contract

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

VS Code companion 保留的流水线命令均使用当前 CLI 语法，CLI 与扩展通过同一命令契约防止再次漂移。

## Assumptions

- `run --today` 和 `run --id` 覆盖 companion 仍有价值的两种流水线入口。
- 用户 TOML 配置与 `vault_root` 可以取代 extension 生成 JSON、参数覆盖和 secret 注入。
- 无需恢复 run-pending 才能保留 companion 的核心价值。

## Approach

先将 companion 测试改成当前 CLI 契约并观察 Red，再移除失效命令与配置面；以共享 fixture 同时驱动 CLI parser 断言和 extension argv 断言。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: companion 测试要求 `run --today`、`run --id` 且禁止旧命令/flags，当前实现因输出 `run-pending`、`summarize` 和配置覆盖而失败。
- Green / regression checks: CLI focused parser test、extension package/build/test/smoke 全绿，Dashboard、Vault 检测和 Paper Index 编辑命令仍在。

## Tasks

- [x] 以 Red 测试和共享 fixture 锁定允许的 companion CLI argv。
- [x] 将 today/ID 命令迁移到当前 CLI 并移除失效命令、flags、secret/config UI。
- [x] 更新 manifest、README、package checks 和 smoke tests，保留非流水线功能。
- [x] 运行 CLI 与 companion 定向和仓库兼容回归。
- [x] 完成独立技术报告交接并提交 P1。

## Verification

- `npm test --workspace arxiv-daily -- --maxWorkers=1 tests/cli-main.test.ts`
- `npm --prefix extensions/vscode-arxiv-daily run build`
- `npm --prefix extensions/vscode-arxiv-daily test`
- `npm run check:boundaries && npm run typecheck && git diff --check`

## Abort / reshape triggers

- 如果当前 CLI 无法表达 today 或 ID 运行，停止并重新决定 companion 产品面，不恢复旧命令绕过。
- 如果共享契约必须引入新的运行时依赖，优先改用纯 JSON/ESM fixture；仍不可行时再执行 L2 reshape。
- 如果移除 secret/config bridge 会破坏已发布兼容承诺，先取得发布证据并记录迁移方案。
