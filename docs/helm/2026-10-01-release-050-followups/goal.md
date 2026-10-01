# 0.5.0 中断修复收尾与发布准备

status: active
created: 2026-10-01T23:06:37+08:00
updated: 2026-10-01T23:20:17+08:00
revision: 3
owner: codex-root-2026-10-01

## Intent

保留并收尾上一位 agent 的首次建索引修复，扩大研究方向审核窗口，交付可手测的构建。用户手测通过后，在当前分支完成 0.5.0 版本同步与发行说明。

## Success criteria

- [x] 首次建索引按需扫描；已有 catalog 不自动重扫；取消与整体失败不继续索引（自动测试通过，真实宿主待手测）。
- [x] arXiv 查询失败不阻断可读 PDF 全文索引；零结果与功能覆盖提示准确（自动测试通过）。
- [ ] 审核对话框大屏有足够空间，小窗口可以滚动与操作。
- [x] 两个修复独立提交，要求的检查有实际记录，备份后部署手测文件且不碰 data.json。
- [ ] 用户手测通过后，完成版本同步、发行说明核对和发布前本地检查与提交。

## Non-goals

- 不自行 push、修改 PR、合并、打 tag、发布或清理分支/stash。
- 不触碰任何已有 .worktree，包括 claude-code-research-plugi 与 agent-product-strategy。
- CLI 设置新功能明确放后续版本；先完成 0.5.0 收尾与手测，届时已授权独立本地提交。

## Constraints

- 分支 fix/review-followups；接手 HEAD f0662a2；保留全部现有未提交修改。
- 子任务不得继续派生；文件编辑范围不重叠；子任务不提交。
- 本轮修复已授权逐项本地提交；提交前看 staged diff；Why / What / Validation 使用多个 -m。
- 发行说明草稿留到版本同步提交；部署只复制三件资产，不启动 Obsidian。

## Phases

1. P1 — 两个中断修复通过本地验证、独立提交并部署手测 — status: done
2. P2 — 手测通过后完成 0.5.0 元数据与发行说明、本地发布前验证 — status: pending

## Open questions

- 等待用户重启 Obsidian 手测与 Scan library 识别数量反馈。
