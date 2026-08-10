# CI Product Governance

status: active
updated: 2026-08-10
owner: zcode-main-session

## Intent

修复 VS Code companion 与当前 CLI 的公开契约，并让根产品、email relay 和 companion 都进入可重复、资源可承受的 PR 验证与显式治理清单。

## Success criteria

- [ ] Companion 只生成当前 CLI 接受的 `run --today` 和 `run --id` 命令，失效命令、flags 和 secret bridge 被移除并由共享契约测试锁定。
- [ ] 默认 `npm test` 在标准 Node heap 下通过，同时 focused-test 调用保持兼容。
- [ ] 根 PR gate 覆盖 release tools、boundaries、lint、typecheck、tests、build 和 smoke build，避免 branch push 与 PR 重复昂贵门禁。
- [ ] Email relay 与 VS Code companion 各有独立 CI，使用自身依赖/版本策略验证构建、测试和部署 dry-run。
- [ ] 显式 product-unit inventory 能拒绝未经分类的新 package/service/extension，所有门禁和技术报告同步。

## Non-goals

- 不恢复 `run-pending`、`summarize`、`--config`、`--vault-root` 或环境 API key 等旧 CLI API。
- 不把 relay 或 companion 纳入根 workspace、根 lockfile或 `0.4.1` 同步版本组。
- 不把未锁定 VSIX 发布或真实部署作为 PR 门禁。

## Constraints

- Companion 用户需先运行 `arxiv-daily init`，TOML `vault_root` 继续是权威工作目录。
- CI 使用固定 action commit 和可重复依赖安装；relay dry-run 不访问生产 secret 或部署资源。
- P1、P2、P3 逐阶段接受，当前只维护一个 active phase 文件。

## Phases

1. P1 — Companion 命令与当前 CLI 契约一致 — status: active
2. P2 — 默认测试资源问题解决且根 PR 门禁覆盖 release-equivalent 检查 — status: pending
3. P3 — Relay、companion 与全部产品单元进入独立 CI 和显式治理 — status: pending
