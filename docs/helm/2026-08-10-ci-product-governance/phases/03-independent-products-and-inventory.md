# P3 — independent-products-and-inventory

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

Email relay 与 VS Code companion 使用各自的依赖锁和产品契约进入独立 PR 验证；仓库通过显式 inventory 识别根 release group、独立 service 与 extension，拒绝未经分类的新产品单元。

## Assumptions

- Email relay 与 VS Code companion 保持独立版本、lockfile 和 CI，不加入根 npm workspace 或根发布版本组。
- Relay dry-run 只验证构建和 Wrangler 配置，不部署 Worker、不读取生产 secret。
- Companion 的 VSIX 只在临时目录或已忽略的 `dist/` 中验证，不发布扩展、不把产物提交到 Git。
- P1 锁定的 CLI command contract 继续是 companion 与 CLI 的共享契约。

## Approach

先用 invariant tests 描述三个产品单元的 manifest、lock/version 策略、所属验证 workflow 和路径覆盖。新增 relay 与 companion 的独立 GitHub Actions：各自在自己的目录执行 `npm ci` 和产品测试，relay 额外执行 typecheck 与 Wrangler dry-run，companion 执行 build/test/smoke 和无发布的 VSIX 打包验证。最后让 inventory checker 扫描 `packages/*`、`apps/*`、`plugin`、`services/*`、`extensions/*`，任何新增含 manifest 的产品目录必须显式分类并拥有匹配治理策略。

## Test strategy

- change kind: CI/product governance
- strategy: strict Red-Green-Refactor
- Red / baseline signal: relay 与 companion 没有专属 workflow；仓库没有 product-unit inventory，新增 service/extension/package 可静默绕过治理。
- Green / regression checks: inventory mutation fixtures 拒绝未知单元、缺 lockfile/manifest/workflow/version policy 和路径漏触发；两份 workflow 的结构化 invariant 锁定事件、路径、固定 action SHA、最小权限与命令；本地执行 relay 和 companion 自身门禁及无部署 dry-run/临时 VSIX 验证。

## Tasks

- [ ] 以 Red 测试锁定 root release group、email relay、VS Code companion 的 inventory 与治理策略。
- [ ] 新增 email relay 独立 PR workflow，覆盖自身 lockfile、Core hosted-delivery contract、tests、typecheck 与 Wrangler dry-run。
- [ ] 新增 VS Code companion 独立 PR workflow，覆盖自身 lockfile、CLI contract、build、tests、smoke 与临时 VSIX 验证。
- [ ] 实现 product-unit inventory checker，拒绝未经分类或治理信息不完整的新 package/app/plugin/service/extension。
- [ ] 运行根、relay、companion 三组门禁，完成独立复核、技术报告交接和远端 PR CI 验证准备。

## Verification

- `npm run test:release-tools`
- Root release-equivalent gate
- Email relay: `npm ci`, typecheck, tests, Wrangler dry-run
- VS Code companion: `npm ci`, build, tests, smoke, temporary/ignored VSIX package verification
- Workflow and inventory invariant tests
- `git diff --check`

## Abort / reshape triggers

- 如果独立 workflow 必须把 relay 或 companion 加入根 workspace/lockfile/version group，停止并保留独立产品边界。
- 如果验证需要真实部署、生产凭据或发布 VSIX，停止并改为本地 dry-run/临时产物。
- 如果 inventory 必须扫描测试夹具或第三方目录才能工作，收紧产品根和 manifest 判定，而不是引入误报白名单堆积。
