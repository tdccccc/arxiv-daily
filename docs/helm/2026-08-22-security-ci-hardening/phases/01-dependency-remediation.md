# P1 — dependency-remediation

<!-- Filename must be 01-dependency-remediation.md with 01 = P1. -->
<!-- Status lives in goal.md's phase index, not here. -->
goal_ref: ../goal.md
updated: 2026-08-22

## Outcome

Root release group 和独立 Email relay 使用的锁定依赖不再产生 moderate 或 high `npm audit` 漏洞，且所有升级经过兼容性验证。

## Assumptions

- 当前 root `package-lock.json` 的 1 moderate、4 high 报告可由兼容锁文件更新消除。
- Relay 的额外 2 moderate、5 high 报告来自 `wrangler@3.114.0` 开发工具链，升级到修复版本可保留现有 dry-run 契约。
- 依赖升级可通过最小直接依赖或锁文件更新解决，不需要更改产品公开配置或协议。

## Approach

先记录完整 audit JSON 和依赖路径，针对可升级的直接依赖增加失败预期的 audit 回归检查，再用最小兼容版本更新 root 和 Relay lockfile；验证根发布组与 Relay dry-run 后再进入 CI 阶段。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `npm audit --audit-level=moderate` 返回非零，并列出当前 vulnerable dependency paths。
- Green / regression checks: 同一审计命令退出 0；release tools、typecheck、workspace tests、build 和 smoke 通过。

## Tasks

- [x] 记录 root 与 Relay audit 的漏洞、受影响范围和可用修复版本，并新增可重复的 audit 门禁测试。
- [x] 以最小依赖升级清除漏洞，验证 root/Relay lockfile 与发布边界一致。
- [x] 运行完整 release-equivalent 与 Relay 回归并接受该依赖修复提交。

## Verification

- `npm audit --audit-level=moderate`
- `npm --prefix services/email-relay audit --audit-level=moderate`
- `npm run test:release-tools && npm run typecheck`
- Observed: root and Relay audit both report 0 vulnerabilities; Relay typecheck, 141 tests, and Wrangler 4.125.0 dry-run pass.
- `NODE_OPTIONS=--max-old-space-size=8192 npm run test:workspaces -- --maxWorkers=1`
- `npm run build && npm run smoke:build && git diff --check`

## Abort / reshape triggers

- 如果唯一修复需要不兼容的主版本升级，停止并将兼容性迁移单列为后续 phase。
- 如果 Relay 的 v4 upgrade 改变现有 preflight 或 dry-run 契约，停止并将迁移和兼容性调整单列为后续 phase。
