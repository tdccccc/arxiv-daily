# P2 — ci-hardening

<!-- Status lives in goal.md's phase index, not here. -->
goal_ref: ../goal.md
updated: 2026-08-23

## Outcome

PR 和发布 CI 持续验证依赖安全、Node 20.19/22 兼容性、CLI 安装产物、topic settings renderer 回归、CodeQL 和所有独立产品 lockfile。

## Assumptions

- GitHub Actions runner 可使用 Node 20.19.0 和 Node 22.17.0，并允许 CodeQL security-events 写入。
- Obsidian 没有可在 GitHub-hosted runner 中稳定启动的官方 headless GUI；现有 happy-dom renderer contract 是 settings UI 的可执行回归边界。

## Approach

在现有 root verification 和独立 Relay workflow 上增加 audit 与安装 smoke；root 增加 Node compatibility matrix 和 topic UI 定向测试；新增固定 SHA 的 CodeQL、Dependabot 配置和 invariant tests。

## Test strategy

- change kind: CI/product governance
- strategy: strict Red-Green-Refactor
- Red / baseline signal: root audit 为 1 moderate / 4 high，Relay audit 为 2 moderate / 5 high；Node 20.11.0 兼容 job 暴露 Vite/Clack 的实际 engine 约束；workflow invariant 不包含 audit、matrix、CodeQL、Dependabot 或 package-install contract。
- Green / regression checks: 58 个 release/CI invariant、root/Relay audit 0、完整 workspace suite、topic UI 定向测试、Relay 141 tests、typecheck、build、Obsidian submission、build smoke 和 CLI install smoke 全部通过。
- exception: 未增加真实 Obsidian GUI E2E，因为仓库没有可复现的 headless Obsidian runtime；以 production settings renderer 的 happy-dom contract 定向测试和发布 submission/build checks 作为补偿验证。

## Tasks

- [x] 新增 root/Relay audit gate，修复并锁定 root 与 Relay 漏洞依赖。
- [x] 新增 Node 20/22 compatibility job、CLI tarball install smoke 和 topic settings UI 定向门禁。
- [x] 新增固定 SHA CodeQL、Dependabot 和 workflow invariant tests。
- [x] 运行 root、Relay、构建和发布相关回归。

## Verification

- `npm run test:release-tools`: 58 tests passed; Product unit inventory OK
- `npm audit --audit-level=moderate`: 0 vulnerabilities
- `npm --prefix services/email-relay audit --audit-level=moderate`: 0 vulnerabilities
- `NODE_OPTIONS=--max-old-space-size=8192 npm run test:workspaces -- --maxWorkers=1`: full workspace suites passed
- Relay: typecheck, 141 tests, Wrangler 4.125.0 dry-run passed
- `npm run build`, `npm run check:obsidian-submission`, `npm run smoke:build`, `npm run smoke:install`: all passed

## Abort / reshape triggers

- If GitHub rejects CodeQL permissions for fork PRs, keep analysis on trusted push/schedule and document the permission boundary rather than granting broad write access.
- If a future official Obsidian headless runtime becomes supported, replace the renderer-contract exception with a real settings-flow job.
