# Security and CI Hardening

status: active
updated: 2026-08-22
owner: root

## Intent

清除 root release group 当前报告的依赖漏洞，并让 PR 与发布工作流持续验证依赖安全、支持的 Node 版本和可执行的产品回归路径。

## Success criteria

- [ ] `npm audit` 对 root release group 不再报告 moderate 或 high 漏洞，升级保持锁文件和发布边界一致。
- [ ] PR CI 自动检查依赖安全、Node 20 最低支持版本与 Node 22 当前运行时，并在不部署生产资源的情况下验证 plugin、CLI 与独立产品契约。
- [ ] CI 配置有结构化测试，且本地和远端发布门禁通过。

## Non-goals

- 不修改 hosted relay 的生产 cutover 状态或执行真实 Worker 部署。
- 不将 Relay 或 VS Code companion 并入 root workspace 或同步发布版本组。
- 不以降低 audit 阈值、忽略漏洞或跳过测试代替修复。

## Constraints

- 根 release group 的最低 Node 支持版本继续是 `20.11.0`；CI 保留 Node 22 发布环境。
- GitHub Actions 继续使用固定 commit SHA 和最小权限。
- UI 回归必须可在 CI 无 Obsidian GUI 的环境中执行，真实宿主集成仍由 Obsidian submission check 与发布资产检查覆盖。

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. Outcomes only — no steps. The active line is the current focus. -->
1. P1 — 清除 root release group 报告的依赖漏洞并锁定无漏洞审计结果 — status: active
2. P2 — 将安全扫描、Node 支持矩阵和可执行产品回归纳入可验证 CI — status: pending

## Open questions

- 审计结果中的漏洞是否全部来自 root release group，还是包含独立 Relay 或 Companion 的依赖树？
