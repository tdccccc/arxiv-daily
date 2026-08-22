# P1 — dependency-remediation

<!-- Filename must be 01-dependency-remediation.md with 01 = P1. -->
<!-- Status lives in goal.md's phase index, not here. -->
goal_ref: ../goal.md
updated: 2026-08-22

## Outcome

Root release group 使用的锁定依赖不再产生 moderate 或 high `npm audit` 漏洞，且所有升级经过兼容性验证。

## Assumptions

- 当前 1 moderate、4 high 报告来自 root `package-lock.json`，而不是独立 Relay 或 Companion 的 lockfile。
- 依赖升级可通过最小直接依赖或锁文件更新解决，不需要更改产品公开配置或协议。

## Approach

先记录完整 audit JSON 和依赖路径，针对可升级的直接依赖增加失败预期的 audit 回归检查，再用最小兼容版本更新 lockfile 和相关测试；验证完整 release group 后再进入 CI 阶段。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `npm audit --audit-level=moderate` 返回非零，并列出当前 vulnerable dependency paths。
- Green / regression checks: 同一审计命令退出 0；release tools、typecheck、workspace tests、build 和 smoke 通过。

## Tasks

- [ ] 记录 root audit 的漏洞、受影响范围和可用修复版本，并新增可重复的 audit 门禁测试。
- [ ] 以最小依赖升级清除漏洞，验证 lockfile 与 release metadata 一致。
- [ ] 运行完整 release-equivalent 回归并接受该依赖修复提交。

## Verification

- `npm audit --audit-level=moderate`
- `npm run test:release-tools && npm run typecheck`
- `NODE_OPTIONS=--max-old-space-size=8192 npm run test:workspaces -- --maxWorkers=1`
- `npm run build && npm run smoke:build && git diff --check`

## Abort / reshape triggers

- 如果唯一修复需要不兼容的主版本升级，停止并将兼容性迁移单列为后续 phase。
- 如果漏洞只存在于独立产品 lockfile，保持产品边界并调整该产品的专属 CI，而不是篡改 root release group。
